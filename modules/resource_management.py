import os
import psutil
from rich.console import Console
import time

console = Console()

def get_system_info():
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu_count = os.cpu_count() or 4
    ram_gb = ram.total / (1024 * 1024 * 1024)
    return ram, swap, cpu_count, ram_gb

def get_dynamic_params():
    ram, swap, cpu_count, ram_gb = get_system_info()

    # More conservative RAM allocation for stability
    available_ram = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    used_ram = ram.used / (1024 * 1024 * 1024)

    # RAM allocation tiers with conservative settings
    if ram_gb >= 32:  # High-end systems
        ram_target_utilization = 0.7  # Reduced from 0.65 for stability
        context_limit = 2048  # Reduced from 4096
        n_batch = 512  # Restored original value
    elif ram_gb >= 16:  # Mid-range systems
        ram_target_utilization = 0.7  # Reduced from 0.65
        context_limit = 1024  # Reduced from 2048
        n_batch = 512  # Restored original value
    elif ram_gb >= 8:  # Common systems
        ram_target_utilization = 0.7  # Reduced from 0.65
        context_limit = 1024
        n_batch = 384  # Restored original value
    else:  # Low-memory systems
        ram_target_utilization = 0.6  # Reduced from 0.55
        context_limit = 1024  # Restored original value
        n_batch = 256  # Restored original value

    # Ensure we don't exceed available RAM
    memory_limit = int(min(ram_gb * ram_target_utilization, available_ram * 0.7))
    memory_target_gb = memory_limit

    # Core configuration with conservative inference parameters
    base_config = {
        'n_threads': max(1, min(int(cpu_count * 0.5), 4)),  # Reduced from 0.85, max 4 threads
        'n_batch': n_batch,
        'max_tokens': 2048,  # Restored original value
        'context_limit': context_limit,
        'memory_limit': memory_limit,
        'memory_target_gb': memory_target_gb,
        'memory_target_pct': ram_target_utilization * 100,
        'temperature': 0.7,  # Increased for more diverse responses
        'top_k': 40,  # Increased for more diverse responses
        'top_p': 0.9,  # Increased for more diverse responses
        'repeat_penalty': 1.2,
        'n_gpu_layers': 32,  # Optimized for AMD 7570M (1GB VRAM)
        'use_mmap': True,
        'f16_kv': True,  # Use half-precision for GPU memory efficiency
        'rope_scaling': {"type": "linear", "factor": 0.25},
        'low_vram': True,  # Enable low VRAM mode for 1GB GPU
        'gpu_device': 0,  # Use first GPU device
        'gpu_memory_utilization': 0.85,  # Use 85% of available VRAM
        'tensor_split': [0.5, 0.5],  # Split tensors between CPU and GPU
        'use_cuda': True,  # Enable CUDA support
        'use_rocm': True,  # Enable ROCm support for AMD GPU
        'gpu_layers': 32,  # Number of layers to offload to GPU
        'main_gpu': 0,  # Main GPU device index
        'tensor_parallel': True,  # Enable tensor parallelism
        'thread_batch': True,  # Enable thread batching
        'thread_batch_size': 8,  # Thread batch size
        'thread_batch_parallel': True,  # Enable parallel thread batching
        'thread_batch_parallel_size': 4  # Parallel thread batch size
    }

    return base_config

def optimize_memory_resources():
    """Optimize memory usage with conservative settings"""
    try:
        # Run garbage collection
        import gc
        gc.collect()

        # Get available RAM
        available_ram = psutil.virtual_memory().available / (1024 * 1024 * 1024)

        # Configure environment for stable memory usage
        aggressive_mode = False  # Disable aggressive mode for stability
        
        # Set conservative memory limits - Commented out for stability on low-RAM systems
        # Setting RLIMIT_AS can cause crashes if the process temporarily needs more virtual memory
        # It might be safer to let the OS handle swapping if necessary.
        # try:
        #     import resource
        #     soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        #     new_limit = min(int(available_ram * 0.7 * 1024 * 1024 * 1024), hard)  # Reduced from 0.95
        #     resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
        # except:
        #     pass

        return True, aggressive_mode
    except Exception as e:
        return False, False

def optimize_cpu_usage():
    """Optimize CPU usage with conservative core allocation"""
    process = psutil.Process()
    cpu_count = os.cpu_count()

    if cpu_count > 1:
        # Calculate target cores based on system load
        current_load = psutil.getloadavg()[0] / cpu_count
        if current_load < 0.9:  # Reduced threshold
            # Use 50% of available cores when load is low
            target_cores = max(1, min(int(cpu_count * 0.75), 4))  # Max 4 cores
        else:
            # Use 30% of available cores when load is high
            target_cores = max(1, min(int(cpu_count * 0.5), 2))  # Max 2 cores
        
        # Ensure we don't use all cores
        target_cores = min(target_cores, cpu_count - 1)

        # Create affinity list with correct number of cores
        affinity = list(range(target_cores))
        try:
            process.cpu_affinity(affinity)
            
            # Try to set moderate process priority
            try:
                if os.name == 'nt':  # Windows
                    process.nice(psutil.NORMAL_PRIORITY_CLASS)
                else:  # Linux/Unix
                    process.nice(0)  # Normal priority
            except:
                pass
                
            return True, target_cores, current_load
        except Exception as e:
            return False, cpu_count, current_load
    else:
        return False, 1, 0

def prewarm_model(llm, base_prompt="You are an AI assistant", spinner_style="dots"):
    """Prewarm the model to reduce initial latency"""
    try:
        start_time = time.time()
        
        # Create continuous timer thread
        timer_running = True
        import threading
        
        # Use Rich's built-in spinner with live updating clock
        with console.status("[bold red]🔥[/bold red] Prewarming model...", spinner=spinner_style) as status:
            def continuous_timer():
                while timer_running:
                    elapsed = time.time() - start_time
                    status.update(f"[bold red]🔥[/bold red] Prewarming model... ⏱️ {elapsed:.1f}s")
                    time.sleep(0.05)
            
            timer_thread = threading.Thread(target=continuous_timer)
            timer_thread.daemon = True
            timer_thread.start()
            
            try:
                # Run inference with minimal tokens
                _ = llm(base_prompt, max_tokens=1)
            except Exception as e:
                console.print(f"[yellow]Model prewarming failed: {e}[/yellow]")
                return 0
            finally:
                timer_running = False
                timer_thread.join()  # Wait for timer thread to finish
        
        duration = time.time() - start_time
        return duration
    except Exception as e:
        if 'timer_running' in locals():
            timer_running = False
        console.print(f"[yellow]Model prewarming failed: {e}[/yellow]")
        return 0