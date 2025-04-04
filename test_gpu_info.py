import torch
import psutil
import platform
from rich.console import Console
from rich.table import Table

console = Console()

def get_gpu_info():
    """Get detailed GPU information using PyTorch"""
    if not torch.cuda.is_available():
        console.print("[red]CUDA is not available. No GPU detected.[/red]")
        return

    device_count = torch.cuda.device_count()
    if device_count == 0:
        console.print("[red]No CUDA devices found.[/red]")
        return

    # Get properties of the first CUDA device
    device_props = torch.cuda.get_device_properties(0)
    
    # Get memory usage
    try:
        mem_allocated = torch.cuda.memory_allocated(0)
        mem_reserved = torch.cuda.memory_reserved(0)
        total_memory = device_props.total_memory
        utilization = 100.0 * mem_allocated / total_memory
    except:
        mem_allocated = 0
        mem_reserved = 0
        total_memory = 0
        utilization = 0.0

    # Create a rich table for display
    table = Table(title="GPU Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    # Add GPU information
    table.add_row("Device Name", device_props.name)
    table.add_row("CUDA Version", torch.version.cuda)
    table.add_row("Compute Capability", f"{device_props.major}.{device_props.minor}")
    table.add_row("Multi-Processor Count", str(device_props.multi_processor_count))
    table.add_row("Max Threads per Block", str(device_props.max_threads_per_block))
    table.add_row("Max Shared Memory per Block", f"{device_props.max_shared_memory_per_block / 1024:.1f} KB")
    table.add_row("Total Memory", f"{total_memory / (1024**3):.1f} GB")
    table.add_row("Allocated Memory", f"{mem_allocated / (1024**3):.1f} GB")
    table.add_row("Reserved Memory", f"{mem_reserved / (1024**3):.1f} GB")
    table.add_row("Memory Utilization", f"{utilization:.1f}%")

    # Add system information
    table.add_row("System Memory", f"{psutil.virtual_memory().total / (1024**3):.1f} GB")
    table.add_row("System Platform", platform.platform())
    table.add_row("Python Version", platform.python_version())
    table.add_row("PyTorch Version", torch.__version__)

    # Print the table
    console.print(table)

    # Print additional workload information
    console.print("\n[bold cyan]Workload Information:[/bold cyan]")
    workload_table = Table()
    workload_table.add_column("Metric", style="cyan")
    workload_table.add_column("Value", style="green")

    # Calculate workload metrics
    gpu_memory_gb = total_memory / (1024**3)
    compute_units = device_props.multi_processor_count
    
    # Determine max work group size
    max_work_group_size = 1024  # Default for most CUDA devices
    if gpu_memory_gb >= 15 and compute_units >= 40:
        max_work_group_size = 1024  # Tesla T4 specific
    
    # Determine Llama layers
    if gpu_memory_gb >= 15 and compute_units >= 40:
        llama_layers = -1  # Use all layers
    elif gpu_memory_gb >= 12:
        llama_layers = -1
    elif gpu_memory_gb >= 8:
        llama_layers = 32
    elif gpu_memory_gb >= 4:
        llama_layers = 24
    else:
        llama_layers = 16

    workload_table.add_row("Max Work Group Size", str(max_work_group_size))
    workload_table.add_row("Assigned Llama Layers", str(llama_layers))
    workload_table.add_row("Compute Units", str(compute_units))
    workload_table.add_row("Memory Available for Workload", f"{gpu_memory_gb:.1f} GB")

    console.print(workload_table)

if __name__ == "__main__":
    get_gpu_info() 