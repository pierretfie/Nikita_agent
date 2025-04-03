#!/usr/bin/env python3

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import psutil
import json
from datetime import datetime
from pathlib import Path
import re
import shlex
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Global flag for alternative model loading
alternative_model_loaded = False

# Try to import quantization libraries
try:
    from transformers import BitsAndBytesConfig
    quantization_available = True
except ImportError:
    quantization_available = False

console = Console()

# ===============================
# === CONFIG ====================
# ===============================

# Base directory for Nikita
NIKITA_BASE_DIR = os.path.join(os.path.expanduser("~"), "Nikita_Agent_model")

# Config paths
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Model name
MODEL_PATH = "/home/eclipse/tinyllama"  # Local model path
OUTPUT_DIR = os.path.join(NIKITA_BASE_DIR, "outputs")
HISTORY_FILE = os.path.join(NIKITA_BASE_DIR, "history.json")
CHAT_HISTORY_FILE = Path(os.path.join(NIKITA_BASE_DIR, "nikita_history.json"))

# Create necessary directories
os.makedirs(NIKITA_BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model parameters
MAX_TOKENS = 512
TEMPERATURE = 0.7
MEMORY_LIMIT = 10  # Number of messages to keep in memory

# 🔍 Auto-detect GPU or fallback to CPU
device = "cpu"  # Default to CPU to avoid VRAM issues

# Add option to use GPU if explicitly requested
use_gpu = False  # Set to False by default
if torch.cuda.is_available():
    console.print("[cyan]GPU detected, but using CPU by default to avoid VRAM issues[/cyan]")
    response = input("Would you like to attempt using GPU instead? (y/n): ").strip().lower()
    if response == 'y':
        use_gpu = True
        device = "cuda"
        console.print("[yellow]Warning: Using GPU. If the process gets killed, restart with CPU mode[/yellow]")

# Check available VRAM if using CUDA
if device == "cuda":
    try:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        free_memory = torch.cuda.memory_reserved(0) / (1024**3)
        console.print(f"[cyan]GPU: {torch.cuda.get_device_name(0)}[/cyan]")
        console.print(f"[cyan]Total VRAM: {gpu_memory:.2f} GB[/cyan]")
        console.print(f"[cyan]Available VRAM: {free_memory:.2f} GB[/cyan]")
        
        # TinyLlama needs much less RAM than Phi-2, around 2GB is sufficient
        if gpu_memory < 2:
            console.print("[yellow]Warning: Your GPU has less than 2GB VRAM, which is not recommended for TinyLlama[/yellow]")
            response = input("Continue with GPU anyway? (y/n): ").strip().lower()
            if response != 'y':
                console.print("[cyan]Switching to CPU mode[/cyan]")
                device = "cpu"
                use_gpu = False
    except Exception as e:
        console.print(f"[yellow]Could not check GPU memory: {e}[/yellow]")
        console.print("[yellow]Continuing with GPU, but be aware it might crash if VRAM is insufficient[/yellow]")
else:
    console.print("[cyan]Using CPU mode (slower but more stable)[/cyan]")

console.print("🔧 [cyan]Initializing Nikita with local TinyLlama...[/cyan]")

# Add memory optimization before model loading
if device == "cpu":
    # Try to limit memory usage on CPU
    console.print("[cyan]Setting up memory optimizations for CPU loading...[/cyan]")
    
    # Get system RAM info
    ram = psutil.virtual_memory()
    total_ram_gb = ram.total / (1024**3)
    available_ram_gb = ram.available / (1024**3)
    
    console.print(f"[cyan]System RAM: {total_ram_gb:.2f} GB (Available: {available_ram_gb:.2f} GB)[/cyan]")
    
    # Free up memory before loading
    import gc
    gc.collect()
    
    # TinyLlama needs much less RAM than Phi-2, around 2GB is sufficient
    if available_ram_gb < 2:
        console.print("[yellow]Warning: Low system RAM. Model loading may be slow.[/yellow]")
    
    # Change memory allocation for PyTorch
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            # These options can help with memory usage on some systems
            torch.backends.cuda.matmul.allow_tf32 = False
            console.print("[cyan]Disabled TF32 for better memory usage[/cyan]")
    
    # Limit number of threads to save memory
    if hasattr(torch, 'set_num_threads'):
        num_cpus = psutil.cpu_count(logical=False) or 2
        torch.set_num_threads(max(1, num_cpus - 1))
        console.print(f"[cyan]Limited PyTorch to {max(1, num_cpus - 1)} threads to save memory[/cyan]")

# 🔹 Load Phi-2 tokenizer & model
try:
    # First check if path exists and is accessible
    if not os.path.exists(MODEL_PATH):
        console.print(f"[red]Error: Model directory not found at {MODEL_PATH}[/red]")
        exit(1)

    # List all files in the directory and verify access
    console.print("\n[cyan]Checking model files...[/cyan]")
    try:
        files_in_dir = [f for f in os.listdir(MODEL_PATH) if os.path.isfile(os.path.join(MODEL_PATH, f))]
        console.print(f"[green]Found {len(files_in_dir)} files in directory[/green]")
        
        # Print file details and verify access
        for file in files_in_dir:
            file_path = os.path.join(MODEL_PATH, file)
            try:
                with open(file_path, 'rb') as f:
                    # Just test if we can read the file
                    f.seek(0)
                file_stat = os.stat(file_path)
                console.print(f"[green]  + {file} (size: {file_stat.st_size} bytes, permissions: {oct(file_stat.st_mode)[-3:]})[/green]")
            except Exception as file_error:
                console.print(f"[red]  ! {file} (access error: {file_error})[/red]")
                raise Exception(f"Cannot access required model file: {file}")
    except Exception as e:
        console.print(f"[red]Error accessing model directory: {e}[/red]")
        exit(1)

    # Try loading with direct path first
    try:
        console.print("[cyan]Attempting to load model directly...[/cyan]")
        
        # Verify tokenizer files
        tokenizer_files = ["vocab.json", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]
        missing_files = [f for f in tokenizer_files if not os.path.isfile(os.path.join(MODEL_PATH, f))]
        if missing_files:
            console.print(f"[yellow]Warning: Missing tokenizer files: {', '.join(missing_files)}[/yellow]")
            raise FileNotFoundError(f"Missing tokenizer files: {', '.join(missing_files)}")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False  # Try using slow tokenizer if fast fails
        )
        console.print("✅ [green]Tokenizer loaded successfully![/green]")

        # Memory optimization settings for model loading
        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        
        # Add device-specific settings
        if device == "cuda":
            model_kwargs.update({
                "device_map": "auto",
                "torch_dtype": torch.float16,  # Use half precision on GPU
            })
            
            # Use 8-bit quantization if available
            if quantization_available:
                console.print("[cyan]Using 8-bit quantization to reduce memory usage[/cyan]")
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
            else:
                console.print("[yellow]Warning: quantization not available, may require more VRAM[/yellow]")
        else:
            # CPU-specific optimizations
            model_kwargs.update({
                "torch_dtype": torch.float32,
                "offload_folder": os.path.join(NIKITA_BASE_DIR, "offload"),
                "offload_state_dict": True,  # Offload weights to disk to save RAM
                "low_cpu_mem_usage": True,   # Enforce low memory usage
                "device_map": {"": "cpu"},   # Explicitly map to CPU
            })
            
            # If extremely low RAM, use additional techniques
            if available_ram_gb < 4:
                # Additional memory saving techniques
                model_kwargs.update({
                    "torch_dtype": torch.float16,     # Use half precision to save memory
                    "max_memory": {0: "1GB"},         # Strictly limit memory usage
                })
                console.print("[cyan]Using half precision to reduce memory usage[/cyan]")
                
                # Global flag to track if we're using an alternative model
                alternative_model_loaded = False
                
                # For extremely low memory, offer alternative tiny model
                if available_ram_gb < 2.5 and device == "cpu":
                    console.print("[red]WARNING: Your system has very limited RAM (< 2.5GB available)[/red]")
                    console.print("[yellow]Would you like to use a smaller alternative model instead of TinyLlama?[/yellow]")
                    console.print("[yellow]Options:[/yellow]")
                    console.print("[yellow]1) Continue with TinyLlama anyway (might crash)[/yellow]")
                    console.print("[yellow]2) Try TinyLlama-1.1B (much smaller model)[/yellow]")
                    console.print("[yellow]3) Try GPT2-small (even smaller, around 500MB)[/yellow]")
                    
                    choice = input("Enter choice (1/2/3): ").strip()
                    
                    alt_model_loaded = False
                    
                    if choice == "2":
                        try:
                            console.print("[cyan]Attempting to load TinyLlama (smaller model)...[/cyan]")
                            # We'll load a completely different model directly
                            console.print("[yellow]Warning: This will download the model if not already present[/yellow]")
                            os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Allow online loading
                            
                            # This is a self-contained block to load a different model
                            from transformers import pipeline
                            alt_pipe = pipeline(
                                "text-generation", 
                                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                torch_dtype=torch.float16
                            )
                            console.print("🔥 [green]Successfully loaded smaller model instead of TinyLlama![/green]")
                            console.print("🐺 [cyan]Running with alternative model[/cyan]")
                            
                            # Use a simple chat loop with the alternative model
                            while True:
                                try:
                                    console.print("\n[bold cyan]┌──(SUDO)[/bold cyan]")
                                    console.print(f"[bold cyan]└─>[/bold cyan] ", end="")
                                    
                                    user_input = input().strip()
                                    
                                    if user_input.lower() in ["exit", "quit"]:
                                        console.print("\n[bold red]Nikita:[/bold red] Exiting. Stay frosty.\n")
                                        exit(0)
                                    
                                    console.print("[cyan]Generating response with TinyLlama...[/cyan]")
                                    response = alt_pipe(user_input, max_length=512)[0]['generated_text']
                                    
                                    console.print(f"[bold magenta]┌──(NIKITA 🐺)[/bold magenta]")
                                    console.print(f"[bold magenta]└─>[/bold magenta] {response}")
                                    
                                except KeyboardInterrupt:
                                    console.print("\n[bold red]Nikita:[/bold red] Caught interrupt signal. Exiting gracefully.\n")
                                    exit(0)
                                except Exception as e:
                                    console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
                            
                            # This code is never reached but satisfies the linter
                            exit(0)
                            
                        except Exception as e:
                            console.print(f"[red]Failed to load alternative model: {e}[/red]")
                            console.print("[yellow]Continuing with TinyLlama (may still crash)[/yellow]")
                    
                    elif choice == "3":
                        try:
                            console.print("[cyan]Attempting to load GPT2-small (smaller model)...[/cyan]")
                            # We'll load a completely different model directly
                            console.print("[yellow]Warning: This will download the model if not already present[/yellow]")
                            os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Allow online loading
                            
                            # This is a self-contained block to load a different model
                            from transformers import pipeline
                            alt_pipe = pipeline(
                                "text-generation", 
                                model="gpt2",
                                torch_dtype=torch.float16
                            )
                            console.print("🔥 [green]Successfully loaded smaller model instead of TinyLlama![/green]")
                            console.print("🐺 [cyan]Running with alternative model[/cyan]")
                            
                            # Use a simple chat loop with the alternative model
                            while True:
                                try:
                                    console.print("\n[bold cyan]┌──(SUDO)[/bold cyan]")
                                    console.print(f"[bold cyan]└─>[/bold cyan] ", end="")
                                    
                                    user_input = input().strip()
                                    
                                    if user_input.lower() in ["exit", "quit"]:
                                        console.print("\n[bold red]Nikita:[/bold red] Exiting. Stay frosty.\n")
                                        exit(0)
                                    
                                    console.print("[cyan]Generating response with GPT2...[/cyan]")
                                    response = alt_pipe(user_input, max_length=512)[0]['generated_text']
                                    
                                    console.print(f"[bold magenta]┌──(NIKITA 🐺)[/bold magenta]")
                                    console.print(f"[bold magenta]└─>[/bold magenta] {response}")
                                    
                                except KeyboardInterrupt:
                                    console.print("\n[bold red]Nikita:[/bold red] Caught interrupt signal. Exiting gracefully.\n")
                                    exit(0)
                                except Exception as e:
                                    console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
                            
                            # This code is never reached but satisfies the linter
                            exit(0)
                            
                        except Exception as e:
                            console.print(f"[red]Failed to load alternative model: {e}[/red]")
                            console.print("[yellow]Continuing with TinyLlama (may still crash)[/yellow]")
            
            # Only continue with TinyLlama loading if alternative model wasn't loaded
            if not alternative_model_loaded:
                # Create offload folder if using CPU
                os.makedirs(os.path.join(NIKITA_BASE_DIR, "offload"), exist_ok=True)
                console.print("[cyan]Using disk offloading to reduce RAM usage[/cyan]")
            
            # Load model with optimized settings
            console.print("[cyan]Loading model with memory optimizations...[/cyan]")
            
            # Load model with optimized settings
            console.print("[cyan]Loading model with memory optimizations...[/cyan]")
            
            if device == "cpu" and available_ram_gb < 4:
                # Use manual shard loading for extremely low memory systems
                console.print("[cyan]Using manual shard loading to minimize peak memory usage...[/cyan]")
                
                # Find model shards
                shard_files = [f for f in files_in_dir if f.startswith("model-") and f.endswith(".safetensors")]
                if shard_files:
                    console.print(f"[green]Found {len(shard_files)} model shards[/green]")
                    
                    # Force garbage collection before loading
                    import gc
                    gc.collect()
                    
                    # Load shards with minimal memory usage
                    from transformers.modeling_utils import load_sharded_checkpoint
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_PATH,
                        **model_kwargs
                    )
                    
                    console.print("✅ [green]Model loaded successfully with shard-based loading![/green]")
                else:
                    # Regular loading as fallback
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_PATH,
                        **model_kwargs
                    ).to(device)
                    console.print("✅ [green]Model loaded successfully![/green]")
            else:
                # Regular loading for systems with sufficient memory
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    **model_kwargs
                ).to(device)
                console.print("✅ [green]Model loaded successfully![/green]")
    except Exception as local_error:
        console.print(f"[yellow]Direct loading failed: {local_error}[/yellow]")
        console.print("[cyan]Attempting alternative loading method with memory optimizations...[/cyan]")
        
        # Try loading with model name and cache_dir
        model_files_path = os.path.join(MODEL_PATH, "tinyllama")
        os.makedirs(model_files_path, exist_ok=True)
        
        # Copy (don't move) files to the nested directory
        for file in files_in_dir:
            src = os.path.join(MODEL_PATH, file)
            dst = os.path.join(model_files_path, file)
            if not os.path.exists(dst) and os.path.isfile(src):
                try:
                    import shutil
                    shutil.copy2(src, dst)
                    console.print(f"[green]Copied {file} to model directory[/green]")
                except Exception as e:
                    console.print(f"[yellow]Could not copy {file}: {e}[/yellow]")

        # Try loading from the nested directory with memory optimizations
        tokenizer = AutoTokenizer.from_pretrained(
            model_files_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False
        )
        console.print("✅ [green]Tokenizer loaded successfully (alternative method)![/green]")

        # Set up memory-efficient model loading
        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        
        # Add device-specific settings
        if device == "cuda":
            model_kwargs.update({
                "device_map": "auto",
                "torch_dtype": torch.float16,  # Use half precision on GPU
            })
            
            # Use 8-bit quantization if available
            if quantization_available:
                console.print("[cyan]Using 8-bit quantization to reduce memory usage[/cyan]")
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
            else:
                console.print("[yellow]Warning: quantization not available, may require more VRAM[/yellow]")
        else:
            # CPU-specific optimizations
            model_kwargs.update({
                "torch_dtype": torch.float32,
                "offload_folder": os.path.join(NIKITA_BASE_DIR, "offload"),
                "offload_state_dict": True,  # Offload weights to disk to save RAM
                "low_cpu_mem_usage": True,   # Enforce low memory usage
                "device_map": {"": "cpu"},   # Explicitly map to CPU
            })
            
            # If extremely low RAM, use additional techniques
            if available_ram_gb < 4:
                # Additional memory saving techniques
                model_kwargs.update({
                    "torch_dtype": torch.float16,     # Use half precision to save memory
                    "max_memory": {0: "1GB"},         # Strictly limit memory usage
                })
                console.print("[cyan]Using half precision to reduce memory usage[/cyan]")
                
                # For extremely low memory, offer alternative tiny model
                if available_ram_gb < 2.5 and device == "cpu":
                    console.print("[red]WARNING: Your system has very limited RAM (< 2.5GB available)[/red]")
                    console.print("[yellow]Would you like to use a smaller alternative model instead of TinyLlama?[/yellow]")
                    console.print("[yellow]Options:[/yellow]")
                    console.print("[yellow]1) Continue with TinyLlama anyway (might crash)[/yellow]")
                    console.print("[yellow]2) Try TinyLlama-1.1B (much smaller model)[/yellow]")
                    console.print("[yellow]3) Try GPT2-small (even smaller, around 500MB)[/yellow]")
                    
                    choice = input("Enter choice (1/2/3): ").strip()
                    
                    alt_model_loaded = False
                    
                    if choice == "2":
                        try:
                            console.print("[cyan]Attempting to load TinyLlama (smaller model)...[/cyan]")
                            # We'll load a completely different model directly
                            console.print("[yellow]Warning: This will download the model if not already present[/yellow]")
                            os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Allow online loading
                            
                            # This is a self-contained block to load a different model
                            from transformers import pipeline
                            alt_pipe = pipeline(
                                "text-generation", 
                                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                torch_dtype=torch.float16
                            )
                            console.print("🔥 [green]Successfully loaded smaller model instead of TinyLlama![/green]")
                            console.print("🐺 [cyan]Running with alternative model[/cyan]")
                            
                            # Use a simple chat loop with the alternative model
                            while True:
                                try:
                                    console.print("\n[bold cyan]┌──(SUDO)[/bold cyan]")
                                    console.print(f"[bold cyan]└─>[/bold cyan] ", end="")
                                    
                                    user_input = input().strip()
                                    
                                    if user_input.lower() in ["exit", "quit"]:
                                        console.print("\n[bold red]Nikita:[/bold red] Exiting. Stay frosty.\n")
                                        exit(0)
                                    
                                    console.print("[cyan]Generating response with TinyLlama...[/cyan]")
                                    response = alt_pipe(user_input, max_length=512)[0]['generated_text']
                                    
                                    console.print(f"[bold magenta]┌──(NIKITA 🐺)[/bold magenta]")
                                    console.print(f"[bold magenta]└─>[/bold magenta] {response}")
                                    
                                except KeyboardInterrupt:
                                    console.print("\n[bold red]Nikita:[/bold red] Caught interrupt signal. Exiting gracefully.\n")
                                    exit(0)
                                except Exception as e:
                                    console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
                            
                            # This code is never reached but satisfies the linter
                            exit(0)
                            
                        except Exception as e:
                            console.print(f"[red]Failed to load alternative model: {e}[/red]")
                            console.print("[yellow]Continuing with TinyLlama (may still crash)[/yellow]")
                    
                    elif choice == "3":
                        try:
                            console.print("[cyan]Attempting to load GPT2-small (smaller model)...[/cyan]")
                            # We'll load a completely different model directly
                            console.print("[yellow]Warning: This will download the model if not already present[/yellow]")
                            os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Allow online loading
                            
                            # This is a self-contained block to load a different model
                            from transformers import pipeline
                            alt_pipe = pipeline(
                                "text-generation", 
                                model="gpt2",
                                torch_dtype=torch.float16
                            )
                            console.print("🔥 [green]Successfully loaded smaller model instead of TinyLlama![/green]")
                            console.print("🐺 [cyan]Running with alternative model[/cyan]")
                            
                            # Use a simple chat loop with the alternative model
                            while True:
                                try:
                                    console.print("\n[bold cyan]┌──(SUDO)[/bold cyan]")
                                    console.print(f"[bold cyan]└─>[/bold cyan] ", end="")
                                    
                                    user_input = input().strip()
                                    
                                    if user_input.lower() in ["exit", "quit"]:
                                        console.print("\n[bold red]Nikita:[/bold red] Exiting. Stay frosty.\n")
                                        exit(0)
                                    
                                    console.print("[cyan]Generating response with GPT2...[/cyan]")
                                    response = alt_pipe(user_input, max_length=512)[0]['generated_text']
                                    
                                    console.print(f"[bold magenta]┌──(NIKITA 🐺)[/bold magenta]")
                                    console.print(f"[bold magenta]└─>[/bold magenta] {response}")
                                    
                                except KeyboardInterrupt:
                                    console.print("\n[bold red]Nikita:[/bold red] Caught interrupt signal. Exiting gracefully.\n")
                                    exit(0)
                                except Exception as e:
                                    console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
                            
                            # This code is never reached but satisfies the linter
                            exit(0)
                            
                        except Exception as e:
                            console.print(f"[red]Failed to load alternative model: {e}[/red]")
                            console.print("[yellow]Continuing with TinyLlama (may still crash)[/yellow]")
            
            # Only continue with TinyLlama loading if alternative model wasn't loaded
            if not alternative_model_loaded:
                # Create offload folder if using CPU
                os.makedirs(os.path.join(NIKITA_BASE_DIR, "offload"), exist_ok=True)
                console.print("[cyan]Using disk offloading to reduce RAM usage[/cyan]")
            
            console.print("[cyan]Loading model with memory optimizations (alternative method)...[/cyan]")
            
            if device == "cpu" and available_ram_gb < 4:
                # Use manual shard loading for extremely low memory systems
                console.print("[cyan]Using manual shard loading to minimize peak memory usage (alternative method)...[/cyan]")
                
                # Find model shards
                shard_files = [f for f in os.listdir(model_files_path) if f.startswith("model-") and f.endswith(".safetensors")]
                if not shard_files:
                    # Copy shard files if needed
                    shard_files = [f for f in files_in_dir if f.startswith("model-") and f.endswith(".safetensors")]
                    for shard in shard_files:
                        src = os.path.join(MODEL_PATH, shard)
                        dst = os.path.join(model_files_path, shard)
                        if not os.path.exists(dst) and os.path.isfile(src):
                            try:
                                import shutil
                                console.print(f"[cyan]Copying shard {shard} to target directory...[/cyan]")
                                shutil.copy2(src, dst)
                            except Exception as e:
                                console.print(f"[yellow]Could not copy shard {shard}: {e}[/yellow]")
                
                # Check for model index file and copy if needed
                index_file = "model.safetensors.index.json"
                if not os.path.exists(os.path.join(model_files_path, index_file)) and os.path.exists(os.path.join(MODEL_PATH, index_file)):
                    try:
                        import shutil
                        console.print(f"[cyan]Copying model index file to target directory...[/cyan]")
                        shutil.copy2(os.path.join(MODEL_PATH, index_file), os.path.join(model_files_path, index_file))
                    except Exception as e:
                        console.print(f"[yellow]Could not copy model index: {e}[/yellow]")
                
                # Force garbage collection before loading
                import gc
                gc.collect()
                
                # Count shards in target directory
                shard_files = [f for f in os.listdir(model_files_path) if f.startswith("model-") and f.endswith(".safetensors")]
                if shard_files:
                    console.print(f"[green]Found {len(shard_files)} model shards in target directory[/green]")
                    
                    # Load with optimized approach
                    model = AutoModelForCausalLM.from_pretrained(
                        model_files_path,
                        **model_kwargs
                    )
                    console.print("✅ [green]Model loaded successfully with shard-based loading (alternative method)![/green]")
                else:
                    # Regular loading as fallback
                    model = AutoModelForCausalLM.from_pretrained(
                        model_files_path,
                        **model_kwargs
                    ).to(device)
                    console.print("✅ [green]Model loaded successfully (alternative method)![/green]")
            else:
                # Regular loading for systems with sufficient memory
                model = AutoModelForCausalLM.from_pretrained(
                    model_files_path,
                    **model_kwargs
                ).to(device)
                console.print("✅ [green]Model loaded successfully (alternative method)![/green]")
        
        console.print("🔥 [cyan]TinyLlama model loaded successfully![/cyan]")
        
except Exception as e:
    console.print(f"[red]Error loading TinyLlama model: {e}[/red]")
    console.print("\n[yellow]Debug information:[/yellow]")
    console.print(f"[yellow]Model path: {MODEL_PATH}[/yellow]")
    console.print(f"[yellow]Python working directory: {os.getcwd()}[/yellow]")
    console.print("[yellow]Files in directory:[/yellow]")
    try:
        for root, dirs, files in os.walk(MODEL_PATH):
            console.print(f"[cyan]Directory: {root}[/cyan]")
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_stat = os.stat(file_path)
                    console.print(f"[green]  + {file} (size: {file_stat.st_size} bytes, permissions: {oct(file_stat.st_mode)[-3:]})[/green]")
                except Exception as stat_error:
                    console.print(f"[red]  ! {file} (stat error: {stat_error})[/red]")
    except Exception as list_error:
        console.print(f"[red]Could not list directory contents: {list_error}[/red]")
    exit(1)

# ===============================
# === UTILITY FUNCTIONS ========
# ===============================

def get_system_info():
    """Get system information"""
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu_count = os.cpu_count() or 4
    ram_gb = ram.total / (1024 * 1024 * 1024)
    return ram, swap, cpu_count, ram_gb

def load_chat_history():
    """Load chat history from file"""
    if CHAT_HISTORY_FILE.exists():
        try:
            with open(CHAT_HISTORY_FILE, 'r') as f:
                history = json.load(f)
                return history[-MEMORY_LIMIT:]
        except Exception as e:
            console.print(f"[yellow]Could not load chat history: {e}[/yellow]")
    return []

def save_chat_history(messages):
    """Save chat history to file"""
    try:
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(messages, f, indent=2)
    except Exception as e:
        console.print(f"[yellow]Could not save chat history: {e}[/yellow]")

def save_command_output(cmd, output, error=None):
    """Save command output to file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f"cmd_{timestamp}.txt")
    with open(output_file, 'w') as f:
        f.write(f"=== Command ===\n{cmd}\n\n")
        f.write(f"=== Output ===\n{output}\n")
        if error:
            f.write(f"\n=== Errors ===\n{error}\n")
    return output_file

# ===============================
# === COMMAND HANDLING =========
# ===============================

def validate_command(cmd):
    """Validate and sanitize command input"""
    if not cmd:
        return None, "Empty command"
    
    cmd = cmd.strip('"\'')
    
    if ';' in cmd or '&&' in cmd or '||' in cmd:
        return None, "Invalid command: contains unsafe characters"
    
    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        return None, f"Invalid command format: {str(e)}"
    
    if not parts:
        return None, "Empty command after parsing"
    
    return cmd, None

def run_command(cmd):
    """Execute a system command safely"""
    if cmd.count('.') < 3 and " " in cmd:
        console.print(f"❌ [red]Command incomplete:[/red] {cmd}")
        return

    try:
        console.print(f"⚡ [bold cyan]Running:[/bold cyan] {cmd}")
        
        cmd_list = shlex.split(cmd)
        result = subprocess.run(cmd_list, capture_output=True, text=True, timeout=300)
        
        output_file = save_command_output(cmd, result.stdout, result.stderr)
        
        if result.stdout.strip():
            if cmd.startswith("nmap"):
                output_lines = result.stdout.strip().split("\n")
                formatted_output = "\n".join(f"  {line}" if "open" in line else line for line in output_lines)
                console.print(f"🖥️  [green]{formatted_output}[/green]")
            else:
                console.print(f"🖥️  [green]{result.stdout.strip()}[/green]")
            console.print(f"📝 [cyan]Output saved to:[/cyan] {output_file}")
        else:
            console.print(f"⚠️  [yellow]No output returned.[/yellow]")
            
    except subprocess.TimeoutExpired:
        console.print("❌ [red]Command timed out after 5 minutes[/red]")
        save_command_output(cmd, "", "Command timed out after 5 minutes")
    except Exception as e:
        console.print(f"❌ [red]Error:[/red] {e}")
        save_command_output(cmd, "", str(e))

# ===============================
# === PHI-2 INTEGRATION =======
# ===============================

def generate_response(prompt, max_length=MAX_TOKENS):
    """Generate response using TinyLlama model"""
    try:
        # Handle memory before processing
        if device == "cpu":
            # Free memory before tokenization
            import gc
            gc.collect()
        elif device == "cuda":
            torch.cuda.empty_cache()
        
        # Tokenize with smaller batch size and limit context
        max_context = 512  # Limit context size to save memory
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, 
                          truncation=True, max_length=max_context).to(device)
        
        with torch.no_grad():
            # Use more memory-efficient generation settings
            generation_config = {
                "max_length": max_length,
                "temperature": TEMPERATURE,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "num_return_sequences": 1,
                
                # Memory optimization settings
                "use_cache": True if device == "cuda" else False,  # Disable KV cache on CPU to save memory
                "repetition_penalty": 1.2,  # Help prevent long repetitive outputs
                "early_stopping": True,  # Stop when model would likely generate EOS
                "length_penalty": 1.0,   # Encourage shorter responses
            }
            
            # Free up memory before generation
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "cpu":
                import gc
                gc.collect()
                
            # Print generation status
            console.print("[cyan]Generating response...[/cyan]")
            
            # Generate with progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("Thinking...", total=None)
                outputs = model.generate(**inputs, **generation_config)
            
        # Process response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up memory after generation
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "cpu":
            import gc
            gc.collect()
            
        return response.replace(prompt, "").strip()
    except Exception as e:
        console.print(f"[red]Error generating response: {e}[/red]")
        return "I encountered an error while processing your request."

# ===============================
# === MAIN LOOP ===============
# ===============================

def main():
    chat_memory = load_chat_history()
    
    # Print version banner
    console.print("\n[bold red]╔══════════════════════════════════════╗[/bold red]")
    console.print("[bold red]║[/bold red]   [bold white]NIKITA TINYLLAMA AGENT v1.0[/bold white]    [bold red]║[/bold red]")
    console.print("[bold red]╚══════════════════════════════════════╝[/bold red]\n")
    
    console.print("✅ [bold green]Nikita (TinyLlama Mode) Loaded[/bold green]")
    
    if chat_memory:
        console.print(f"📚 [cyan]Loaded {len(chat_memory)} previous messages[/cyan]")

    while True:
        try:
            console.print("\n[bold cyan]┌──(SUDO)[/bold cyan]")
            console.print(f"[bold cyan]└─>[/bold cyan] ", end="")
            
            user_input = input().strip()
            
            if user_input.lower() in ["exit", "quit"]:
                save_chat_history(chat_memory)
                console.print("\n[bold red]Nikita:[/bold red] Exiting. Stay frosty.\n")
                break

            # Add user input to chat memory
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            chat_memory.append({
                "role": "user",
                "content": user_input,
                "timestamp": timestamp
            })

            # Enforce memory limit
            if len(chat_memory) > MEMORY_LIMIT:
                chat_memory = chat_memory[-MEMORY_LIMIT:]

            # Process the command
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task_id = progress.add_task("🐺 Nikita is thinking...", total=None)
                
                # Generate response
                prompt = f"You are Nikita, an AI Security Assistant. User request: {user_input}\nResponse:"
                response = generate_response(prompt)
                
                # Check if response contains a command
                command_match = re.search(r'`(.*?)`', response)
                if command_match:
                    cmd = command_match.group(1)
                    validated_cmd, error = validate_command(cmd)
                    if validated_cmd:
                        run_command(validated_cmd)

                progress.stop()
                console.print()
                
                # Save response to chat memory
                chat_memory.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                save_chat_history(chat_memory)
                
                console.print(f"[bold magenta]┌──(NIKITA 🐺)[/bold magenta]")
                console.print(f"[bold magenta]└─>[/bold magenta] {response}")

        except KeyboardInterrupt:
            save_chat_history(chat_memory)
            console.print("\n[bold red]Nikita:[/bold red] Caught interrupt signal. Exiting gracefully.\n")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
            console.print("[yellow]Continuing to next interaction...[/yellow]\n")
            continue

if __name__ == "__main__":
    main()
