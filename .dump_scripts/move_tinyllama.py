#!/usr/bin/env python3

import os
import shutil
import glob
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def move_tinyllama_to_home():
    """Move or copy TinyLlama model from HuggingFace cache to ~/tinyllama folder"""
    
    # Define paths
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    target_dir = os.path.expanduser("~/tinyllama")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    console.print(f"[cyan]Moving TinyLlama model to {target_dir}...[/cyan]")
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Find TinyLlama files in cache
    files_found = False
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        task = progress.add_task("[cyan]Searching for TinyLlama files in cache...", total=None)
        
        # Look in models directory
        model_dirs = []
        for root, dirs, files in os.walk(cache_dir):
            if any("tinyllama" in d.lower() for d in dirs):
                model_dirs = [os.path.join(root, d) for d in dirs if "tinyllama" in d.lower()]
                break
        
        if not model_dirs:
            # Try another approach - look for safetensors files
            safetensor_files = []
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    if file.endswith(".safetensors") and "tinyllama" in root.lower():
                        safetensor_files.append(os.path.join(root, file))
                        model_dirs.append(root)
            
            if safetensor_files:
                files_found = True
        else:
            files_found = True
        
        progress.remove_task(task)
    
    if not files_found:
        console.print("[red]Could not find TinyLlama files in cache. Please ensure the model is downloaded.[/red]")
        return False
    
    # Remove duplicates
    model_dirs = list(set(model_dirs))
    console.print(f"[green]Found TinyLlama model files in: {model_dirs}[/green]")
    
    # Copy files
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        task = progress.add_task("[cyan]Copying model files to target directory...", total=len(model_dirs))
        
        for source_dir in model_dirs:
            dir_name = os.path.basename(source_dir)
            target_subdir = os.path.join(target_dir, dir_name)
            os.makedirs(target_subdir, exist_ok=True)
            
            # Copy all files
            for item in os.listdir(source_dir):
                source_item = os.path.join(source_dir, item)
                target_item = os.path.join(target_subdir, item)
                
                if os.path.isfile(source_item):
                    shutil.copy2(source_item, target_item)
                    console.print(f"[green]Copied: {item}[/green]")
                elif os.path.isdir(source_item):
                    shutil.copytree(source_item, target_item, dirs_exist_ok=True)
                    console.print(f"[green]Copied directory: {item}[/green]")
            
            progress.update(task, advance=1)
    
    # Copy tokenizer and config files if they exist in a different location
    tokenizer_files = []
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json"] and "tinyllama" in root.lower():
                tokenizer_files.append(os.path.join(root, file))
    
    for source_file in tokenizer_files:
        file_name = os.path.basename(source_file)
        target_file = os.path.join(target_dir, file_name)
        if not os.path.exists(target_file):
            shutil.copy2(source_file, target_file)
            console.print(f"[green]Copied tokenizer file: {file_name}[/green]")
    
    console.print(f"[bold green]✅ TinyLlama model files have been copied to {target_dir}[/bold green]")
    console.print("[yellow]Note: You'll need to update the Nikita_phi2.py MODEL_PATH to use this location.[/yellow]")
    
    # Update model path instructions
    console.print("\n[cyan]To use this model in Nikita, update your MODEL_PATH:[/cyan]")
    console.print("[yellow]Edit Nikita_phi2.py and change:[/yellow]")
    console.print(f'[red]MODEL_PATH = "/home/eclipse/phi-2/phi-2"[/red]')
    console.print("[yellow]to:[/yellow]")
    console.print(f'[green]MODEL_PATH = "{target_dir}"[/green]')
    
    return True

if __name__ == "__main__":
    try:
        success = move_tinyllama_to_home()
        if not success:
            console.print("[red]Failed to move model files. Make sure TinyLlama is downloaded.[/red]")
            exit(1)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        exit(1) 