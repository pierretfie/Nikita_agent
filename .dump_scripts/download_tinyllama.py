#!/usr/bin/env python3

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def download_tinyllama():
    """Download TinyLlama model and tokenizer with progress indication"""
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    console.print(f"[cyan]Downloading {model_name}...[/cyan]")
    console.print("[yellow]This will download approximately 1.1GB of data[/yellow]")
    
    # Set download directory to standard cache location
    os.makedirs(os.path.expanduser("~/.cache/huggingface"), exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        # Download tokenizer
        task1 = progress.add_task("[cyan]Downloading tokenizer...", total=None)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        progress.remove_task(task1)
        
        # Download model
        task2 = progress.add_task("[cyan]Downloading model weights...", total=None)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        progress.remove_task(task2)
    
    console.print("[green]✓ TinyLlama model downloaded successfully![/green]")
    console.print(f"[cyan]Model saved to: {os.path.expanduser('~/.cache/huggingface/hub')}[/cyan]")
    
    # Verify the model is loadable
    console.print("[cyan]Verifying model can be loaded...[/cyan]")
    test_input = tokenizer("Hello, my name is", return_tensors="pt")
    with torch.no_grad():
        test_output = model.generate(
            **test_input, 
            max_length=20,
            num_return_sequences=1
        )
    test_response = tokenizer.decode(test_output[0], skip_special_tokens=True)
    console.print(f"[green]Model verification successful: '{test_response}'[/green]")
    
    return True

if __name__ == "__main__":
    try:
        success = download_tinyllama()
        if success:
            console.print("[bold green]✅ Model download complete and ready to use with Nikita![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error downloading model: {e}[/bold red]")
        exit(1) 