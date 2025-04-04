#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Displays detailed GPU, system, and derived workload information using PyTorch and Rich.
"""

import platform
import torch
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Initialize Rich Console
console = Console()

def format_bytes(size_bytes: int, suffix: str = "B") -> str:
    """Converts bytes to a human-readable format (KB, MB, GB, TB)."""
    if size_bytes == 0:
        return "0 B"
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if size_bytes < factor:
            return f"{size_bytes:.2f} {unit}{suffix}"
        size_bytes /= factor
    return f"{size_bytes:.2f} P{suffix}" # Fallback for very large sizes

def get_system_info():
    """Gathers basic system and library version information."""
    info = {
        "System Platform": platform.platform(),
        "System Memory": format_bytes(psutil.virtual_memory().total),
        "Python Version": platform.python_version(),
        "PyTorch Version": torch.__version__,
        "CUDA Available": Text("Yes", style="bold green") if torch.cuda.is_available() else Text("No", style="bold red"),
        "CUDA Version (PyTorch)": getattr(torch.version, 'cuda', "N/A"), # Handles cases where PyTorch is CPU-only build
    }
    return info

def get_gpu_details(device_id: int = 0):
    """Gathers detailed information for a specific CUDA device."""
    if not torch.cuda.is_available():
        return None, "CUDA is not available on this system."
    
    device_count = torch.cuda.device_count()
    if device_count == 0:
        return None, "No CUDA-enabled GPUs found."
    if device_id >= device_count:
         return None, f"Device ID {device_id} is invalid. Found {device_count} devices (IDs 0 to {device_count-1})."

    try:
        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory
        
        # Get current memory usage (may require GPU context)
        try:
            # Briefly allocate something small to ensure context if needed, though get_device_properties usually does this
            # _ = torch.tensor([1.0], device=device_id) 
            mem_allocated = torch.cuda.memory_allocated(device_id)
            mem_reserved = torch.cuda.memory_reserved(device_id) # Often called 'cached' memory by nvidia-smi
            free_mem, total_mem_driver = torch.cuda.mem_get_info(device_id) # Alternative way via driver
        except Exception as mem_err:
            console.print(f"[yellow]Warning:[/yellow] Could not get detailed memory usage for device {device_id}. Error: {mem_err}", style="dim")
            mem_allocated = 0
            mem_reserved = 0
            free_mem = total_memory # Best guess if allocation fails
            total_mem_driver = total_memory

        utilization = (mem_allocated / total_memory * 100) if total_memory > 0 else 0.0
        free_mem_gb = free_mem / (1024**3)
        total_mem_gb = total_memory / (1024**3)

        gpu_info = {
            "Device ID": str(device_id),
            "Device Name": props.name,
            "Compute Capability": f"{props.major}.{props.minor}",
            "Multi-Processor Count": str(props.multi_processor_count),
            "Total Memory (PyTorch)": format_bytes(total_memory, "B"),
            "Total Memory (Driver)": format_bytes(total_mem_driver, "B"),
            "Free Memory (Driver)": format_bytes(free_mem, "B"),
            "Allocated Memory (PyTorch)": format_bytes(mem_allocated, "B"),
            "Reserved Memory (PyTorch)": format_bytes(mem_reserved, "B"),
            "Memory Utilization (Allocated)": f"{utilization:.1f}%",
            "Shared Memory per Block": format_bytes(props.shared_mem_per_block, "B"),
            "Max Threads per Multi-Processor": str(props.max_threads_per_multi_processor),
            "Warp Size": str(props.warp_size),
        }

        # --- Derived Workload Heuristics ---
        compute_units = props.multi_processor_count
        max_work_group_size = 1024 # Common default, could refine based on compute capability if needed

        # Llama layers heuristic based on *available* memory seems more practical
        if free_mem_gb >= 14: # Generous buffer for OS/other processes
             llama_layers = -1 # Suggests using all layers possible
        elif free_mem_gb >= 10:
             llama_layers = 32 # Example tier
        elif free_mem_gb >= 6:
             llama_layers = 24 # Example tier
        elif free_mem_gb >= 3:
             llama_layers = 16 # Example tier
        else:
             llama_layers = 8  # Example for very low memory

        workload_info = {
            "Compute Units (MPs)": str(compute_units),
            "Approx. Available Memory": f"{free_mem_gb:.1f} GB",
            "Suggested Max Work Group Size": str(max_work_group_size),
            "Heuristic Llama Layers": str(llama_layers),
        }
        
        return gpu_info, workload_info

    except Exception as e:
        return None, f"An error occurred while getting properties for device {device_id}: {e}"


def display_info():
    """Gets and displays system and GPU information."""
    
    console.print(Panel("System & Environment Information", style="bold blue", expand=False))
    system_info = get_system_info()
    sys_table = Table(show_header=False, box=None, padding=(0, 2))
    sys_table.add_column("Property", style="cyan")
    sys_table.add_column("Value", style="green")
    for key, value in system_info.items():
        sys_table.add_row(key, value)
    console.print(sys_table)

    console.print(Panel("GPU Information (Device 0)", style="bold blue", expand=False, margin=(1, 0, 0, 0)))
    gpu_info, workload_or_error = get_gpu_details(device_id=0)

    if gpu_info:
        # GPU Details Table
        gpu_table = Table(show_header=False, box=None, padding=(0, 2))
        gpu_table.add_column("Property", style="cyan")
        gpu_table.add_column("Value", style="green")
        for key, value in gpu_info.items():
            gpu_table.add_row(key, value)
        console.print(gpu_table)

        # Workload Heuristics Table
        console.print("\n[bold cyan]Derived Workload Heuristics:[/bold cyan]")
        workload_table = Table(show_header=False, box=None, padding=(0, 2))
        workload_table.add_column("Metric", style="cyan")
        workload_table.add_column("Value", style="green")
        for key, value in workload_or_error.items(): # Here it's the workload dict
            workload_table.add_row(key, value)
        console.print(workload_table)
        console.print("[dim i]Note: Llama layers heuristic is based on currently free memory and is just a rough guideline.[/dim i]")

    else:
        # Display the error message if GPU info retrieval failed
        console.print(f"[bold red]Error:[/bold red] {workload_or_error}")


if __name__ == "__main__":
    display_info()