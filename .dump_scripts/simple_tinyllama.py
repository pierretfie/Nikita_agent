#!/usr/bin/env python3

import os
import gc
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

# Initialize console for pretty output
console = Console()

# Path to the model - using the specific path where files have been found
MODEL_PATH = "/home/eclipse/tinyllama/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"

console.print("[bold green]🔧 Loading TinyLlama (minimal version)[/bold green]")

# Perform aggressive memory cleanup
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Get system RAM info
ram = psutil.virtual_memory()
available_ram_gb = ram.available / (1024**3)
console.print(f"[cyan]Available RAM: {available_ram_gb:.2f} GB[/cyan]")

# Set PyTorch to use minimal resources
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

try:
    # Load tokenizer with minimal settings
    console.print("[cyan]Loading tokenizer...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )
    console.print("[green]✓ Tokenizer loaded successfully[/green]")

    # Set up extreme memory optimization for model loading
    console.print("[cyan]Loading model with memory optimization (16-bit precision)...[/cyan]")

    # Force garbage collection
    gc.collect()

    # Calculate memory to reserve for offloading
    memory_to_reserve = max(0.5, available_ram_gb - 0.8)  # Leave at least 800MB free
    console.print(f"[cyan]Reserving {memory_to_reserve:.2f}GB for model loading[/cyan]")

    # Create a memory map to limit memory usage
    memory_map = {0: f"{int(memory_to_reserve * 1024)}MB"}

    # Use offloading folder for temporary files
    os.makedirs("offload_folder", exist_ok=True)

    # Load model with memory efficient settings
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,      # Use half precision
        low_cpu_mem_usage=True,         # Low memory usage
        device_map="cpu",               # Force CPU
        local_files_only=True,          # Don't try to download
        offload_folder="offload_folder",# Use disk offloading
        offload_state_dict=True,        # Offload weights to disk
        max_memory=memory_map,          # Limit memory usage
    )

    console.print("[green]✓ Model loaded successfully![/green]")
    
    # Basic inference test
    console.print("[cyan]Testing inference...[/cyan]")
    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=50,
            do_sample=True,
            temperature=0.7
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    console.print(f"[bold magenta]Response: {response}[/bold magenta]")
    
    # Interactive loop
    console.print("\n[bold green]Starting interactive mode. Type 'exit' to quit.[/bold green]")
    while True:
        # Get user input
        console.print("[bold blue]You:[/bold blue] ", end="")
        user_input = input().strip()
        
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Generate response
        inputs = tokenizer(user_input, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=100,
                do_sample=True,
                temperature=0.7
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up the response (remove the input prompt)
        if user_input in response:
            response = response.replace(user_input, "").strip()
            
        console.print(f"[bold green]TinyLlama:[/bold green] {response}")
        
        # Clear memory after each response
        gc.collect()
        
except Exception as e:
    console.print(f"[bold red]Error: {str(e)}[/bold red]")
    console.print("\n[bold yellow]Debug information:[/bold yellow]")
    console.print(f"[yellow]Model path: {MODEL_PATH}[/yellow]")
    
    # Check if files exist
    tokenizer_files = ["tokenizer.model", "tokenizer.json", "special_tokens_map.json"]
    for file in tokenizer_files:
        file_path = os.path.join(MODEL_PATH, file)
        if os.path.exists(file_path):
            console.print(f"[green]File exists: {file}[/green]")
        else:
            console.print(f"[red]File missing: {file}[/red]") 