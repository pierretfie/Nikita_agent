#!/usr/bin/env python3

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import json
from datetime import datetime
from pathlib import Path
import re
import shlex
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import signal
import gc
import readline  # Add readline for command editing

console = Console()

# ===============================
# === CONFIG ====================
# ===============================

# Base directory
BASE_DIR = os.path.join(os.path.expanduser("~"), "TinyLlama_Assistant")

# Config paths
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Model name
MODEL_PATH = "/home/eclipse/tinyllama/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"  # Local model path
CHAT_HISTORY_FILE = Path(os.path.join(BASE_DIR, "chat_history.json"))
HISTORY_FILE = Path(os.path.join(BASE_DIR, "command_history"))

# Create necessary directories
os.makedirs(BASE_DIR, exist_ok=True)

# Set up readline
readline.set_completer_delims(' \t\n;')
if os.path.exists(HISTORY_FILE):
    readline.read_history_file(HISTORY_FILE)
readline.set_history_length(1000)  # Remember 1000 commands

# Model parameters
MAX_TOKENS = 512
TEMPERATURE = 0.7
MEMORY_LIMIT = 10  # Number of messages to keep in memory

# System prompt for model responses
SYSTEM_PROMPT = """You are Nikita Mini, a helpful assistant that specializes in Python programming and general conversation.
You can provide code snippets, explanations, and help with various tasks.
For Python code, you'll format it with proper indentation and comments.
You're friendly, concise, and focus on being helpful."""

# Use CPU for inference
device = "cpu"
console.print("[cyan]Using CPU mode for inference[/cyan]")

console.print(" [cyan]Initializing Nikita Mini Assistant...[/cyan]")

# Get system RAM info
ram = psutil.virtual_memory()
total_ram_gb = ram.total / (1024**3)
available_ram_gb = ram.available / (1024**3)

console.print(f"[cyan]System RAM: {total_ram_gb:.2f} GB (Available: {available_ram_gb:.2f} GB)[/cyan]")

# Free up memory before loading
gc.collect()

# Limit number of threads to save memory
if hasattr(torch, 'set_num_threads'):
    num_cpus = psutil.cpu_count(logical=False) or 4
    # Use 3 cores out of 4 available
    thread_count = 3  # Fixed to 3 cores
    torch.set_num_threads(thread_count)
    console.print(f"[cyan]Using {thread_count} PyTorch threads (3 out of 4 CPU cores)[/cyan]")

# Set PyTorch and other compute libraries to use more resources
os.environ["OMP_NUM_THREADS"] = str(thread_count)
os.environ["MKL_NUM_THREADS"] = str(thread_count)
os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread_count)
os.environ["NUMEXPR_NUM_THREADS"] = str(thread_count)

# Try to reserve less memory for the model - conservative approach
memory_to_reserve = max(0.5, min(available_ram_gb - 1.5, 1.5))  # Use up to 1.5GB if available
console.print(f"[cyan]Reserving {memory_to_reserve:.2f}GB for model operations[/cyan]")

# ===============================
# === MODEL LOADING =============
# ===============================

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
    except Exception as e:
        console.print(f"[red]Error accessing model directory: {e}[/red]")
        exit(1)

    # Load tokenizer 
    console.print("[cyan]Loading tokenizer...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )
    console.print("✅ [green]Tokenizer loaded successfully![/green]")

    # Load model with memory efficient settings
    console.print("[cyan]Loading model with ultra-low memory settings...[/cyan]")

    # Force garbage collection before loading
    gc.collect()
    
    # Memory map config - very conservative
    memory_map = {0: f"{int(memory_to_reserve * 1024)}MB"}
    
    # Use offloading folder for temporary files
    offload_dir = os.path.join(BASE_DIR, "offload")
    os.makedirs(offload_dir, exist_ok=True)
    
    # Load model with ultra-low memory footprint
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,      # Use half precision
        low_cpu_mem_usage=True,         # Low memory usage
        device_map="cpu",               # Force CPU
        local_files_only=True,          # Don't try to download
        offload_folder=offload_dir,     # Use disk offloading
        offload_state_dict=True,        # Offload weights to disk
        max_memory=memory_map,          # Strict memory limit
    )
    
    console.print("🔥 [green]TinyLlama model loaded successfully![/green]")
except Exception as e:
    console.print(f"[red]Error loading TinyLlama model: {e}[/red]")
    exit(1)

# ===============================
# === UTILITY FUNCTIONS ========
# ===============================

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

# Function to handle timeouts
def timeout_handler(signum, frame):
    raise TimeoutError("Generation took too long")

def generate_response(prompt, max_length=MAX_TOKENS, max_retries=3):
    """Generate response using TinyLlama model with timeout protection"""
    for attempt in range(max_retries + 1):
        try:
            # Free memory before processing
            gc.collect()
            
            if attempt > 0:
                console.print(f"[yellow]Retry attempt {attempt}/{max_retries}...[/yellow]")
            
            # Set up timeout (60 seconds - increased for better completion)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)
            
            # Extract user query
            user_query = ""
            if "User:" in prompt and "Nikita:" in prompt:
                user_part = prompt.split("User:")[-1].split("Nikita:")[0].strip()
                user_query = user_part
            elif "User:" in prompt and "Assistant:" in prompt:
                user_part = prompt.split("User:")[-1].split("Assistant:")[0].strip()
                user_query = user_part
            
            # Only use canned responses for basic greeting queries
            if user_query.lower() in ["hello", "hi", "hey"]:
                signal.alarm(0)
                return "Hello! How can I help you today?"
            
            # Use larger context for better reasoning
            max_context = 128 if attempt == 0 else 192
            
            # Add system context for better reasoning
            reasoning_prompt = f"""You are Nikita Mini, a helpful assistant specializing in Python programming and general conversation.
            
When responding to programming requests, provide complete working code with full implementations.
For network scanning scripts, demonstrate good security practices. 
Always think step by step about the problem before answering.

User query: {user_query}

Let's think through this carefully to provide a complete response:
"""
            
            # Create enhanced prompt for deeper reasoning
            enhanced_prompt = f"{reasoning_prompt}\n\nNikita:"
            
            inputs = tokenizer(enhanced_prompt, return_tensors="pt", padding=True, 
                             truncation=True, max_length=max_context)
            
            with torch.no_grad():
                # Configure for complete, well-reasoned responses
                generation_config = {
                    "max_length": 350,  # Much longer responses to prevent truncation
                    "temperature": 0.7,  # Standardized temperature for balanced creativity
                    "do_sample": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "num_return_sequences": 1,
                    "min_length": 50,  # Longer minimum for substantial responses
                    
                    # Settings for more complete responses
                    "use_cache": True,
                    "repetition_penalty": 1.2,  # Standardized to reduce repetition
                    "top_p": 0.85,  # Standardized for consistent sampling
                    "top_k": 40,  # Standardized for better diversity
                    "max_time": 45.0,  # Extended timeout for better reasoning
                }
                
                # Generate without progress indicator
                outputs = model.generate(**inputs, **generation_config)
                
            # Reset the alarm
            signal.alarm(0)
                
            # Process response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response
            if enhanced_prompt in response:
                response = response.replace(enhanced_prompt, "").strip()
            elif "User query:" in response and "Nikita:" in response:
                response = response.split("Nikita:")[1].strip()
            elif "User:" in response and "Nikita:" in response:
                response = response.split("Nikita:")[1].strip()
            elif "User:" in response and "Assistant:" in response:
                response = response.split("Assistant:")[1].strip()
            
            # Remove any thinking process that made it through
            if "Let's think through this" in response:
                response = response.split("Let's think through this")[0].strip()
            
            # Clean up memory
            gc.collect()
            
            # Ensure we have a non-empty response
            if not response or len(response.strip()) < 10:
                response = "I understand your request. Let me work on that for you."
                
            return response
            
        except TimeoutError:
            signal.alarm(0)
            if attempt == max_retries:
                console.print("[red]Generation timed out after multiple attempts[/red]")
                return "I need more time to process this complex request. Could you break it down into smaller parts or clarify what aspect you'd like me to focus on?"
            gc.collect()
            console.print("[yellow]Generation timed out, retrying with simpler settings...[/yellow]")
            
        except Exception as e:
            signal.alarm(0)
            console.print(f"[red]Error generating response: {e}[/red]")
            return "I encountered an error while processing your request. Let's try something different."

# ===============================
# === MAIN LOOP ===============
# ===============================

def main():
    chat_memory = load_chat_history()
    
    # Print version banner
    console.print("\n[bold red]╔══════════════════════════════════════╗[/bold red]")
    console.print("[bold red]║[/bold red]   [bold white]NIKITA MINI ASSISTANT[/bold white]    [bold red]║[/bold red]")
    console.print("[bold red]╚══════════════════════════════════════╝[/bold red]\n")
    
    console.print("✅ [bold green]Nikita Mini Loaded[/bold green]")
    console.print("[cyan]💡 Use arrow keys to navigate history, and left/right to edit commands[/cyan]")
    
    if chat_memory:
        console.print(f"📚 [cyan]Loaded {len(chat_memory)} previous messages[/cyan]")
        
    # Only keep basic greeting responses
    common_responses = {
        "hello": "Hello! How can I help you today?",
        "hi": "Hi there! What would you like assistance with?",
    }
        
    while True:
        try:
            # Print the command prompt on a new line
            console.print("\n[bold cyan]┌──(SUDO)[/bold cyan]")
            console.print(f"[bold cyan]└─>[/bold cyan] ", end="")
            
            # Get user input with readline features
            try:
                user_input = input().strip()
                # Save to readline history file
                readline.write_history_file(HISTORY_FILE)
            except EOFError:
                # Handle Ctrl+D gracefully
                console.print("\n[bold red]Nikita:[/bold red] Detected end of input. Exiting gracefully.\n")
                save_chat_history(chat_memory)
                break
            
            # Exit commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                save_chat_history(chat_memory)
                console.print("\n[bold red]Nikita:[/bold red] Goodbye! Stay frosty.\n")
                break
                
            # Skip empty inputs
            if not user_input:
                continue
            
            # Use common responses for basic queries to avoid generation
            if user_input.lower() in common_responses:
                response = common_responses[user_input.lower()]
                console.print(f"[bold magenta]┌──(NIKITA 🐺)[/bold magenta]")
                console.print(f"[bold magenta]└─>[/bold magenta] {response}")
                
                # Save to chat history
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                chat_memory.append({"role": "user", "content": user_input, "timestamp": timestamp})
                chat_memory.append({"role": "assistant", "content": response, "timestamp": timestamp})
                save_chat_history(chat_memory)
                continue
                
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

            # Show spinner with text
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]Thinking...[/bold cyan]"),
                transient=True,
            ) as progress:
                task_id = progress.add_task("", total=None)
                
                # Use ultra-minimal prompt for faster generation
                prompt = f"User: {user_input}\nNikita:"
                response = generate_response(prompt)
                
                # Add fallback for extremely short responses
                if len(response.split()) < 3:
                    if "who are you" in user_input.lower():
                        response = "I'm Nikita Mini, an AI assistant that specializes in Python programming and general conversation. I can help with code, explanations, and everyday questions."
                    elif "what can you do" in user_input.lower() or "capabilities" in user_input.lower():
                        response = "I can help with Python programming tasks, provide explanations, assist with general questions, and engage in friendly conversation."
                    elif "hello" in user_input.lower() or "hi" in user_input.lower():
                        response = "Hello! How can I help you today with Python programming or any other questions you might have?"
                
                # Check if response contains Python code and format it nicely
                if "```python" in response and "```" in response.split("```python", 1)[1]:
                    parts = response.split("```python", 1)
                    code_and_rest = parts[1].split("```", 1)
                    code = code_and_rest[0].strip()
                    rest = code_and_rest[1] if len(code_and_rest) > 1 else ""
                    
                    response = parts[0] + "\n```python\n" + code + "\n```\n" + rest

                # Stop the progress
                progress.update(task_id, completed=True)
            
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
