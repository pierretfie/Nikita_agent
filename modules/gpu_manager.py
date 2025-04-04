#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU Manager: Detects, initializes, and provides information about system GPUs
using PyOpenCL and PyTorch (CUDA).
"""

import platform
import time
import sys
import os
import subprocess
import gc
import contextlib
import numpy as np
from rich.console import Console
import psutil
from rich.panel import Panel
# Optional dependencies - handle import errors gracefully
try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    cl = None # Define cl as None if import fails

try:
    import torch
    PYTORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available() if PYTORCH_AVAILABLE else False
except ImportError:
    PYTORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    torch = None # Define torch as None if import fails

console = Console()

# --- Helper Functions ---

def format_bytes(size_bytes: int, suffix: str = "B") -> str:
    """Converts bytes to a human-readable format (KB, MB, GB, TB)."""
    if not isinstance(size_bytes, (int, float)) or size_bytes < 0:
        return "N/A"
    if size_bytes == 0:
        return f"0 {suffix}"
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(size_bytes) < factor:
            return f"{size_bytes:.2f} {unit}{suffix}"
        size_bytes /= factor
    return f"{size_bytes:.2f} P{suffix}" # Fallback for very large sizes

def get_system_info():
    """Gathers basic system and library version information."""
    info = {
        "System Platform": platform.platform(),
        "System Memory": format_bytes(psutil.virtual_memory().total),
        "Python Version": platform.python_version(),
        "PyTorch Available": "Yes" if PYTORCH_AVAILABLE else "No",
        "PyOpenCL Available": "Yes" if PYOPENCL_AVAILABLE else "No",
    }
    if PYTORCH_AVAILABLE:
        info["PyTorch Version"] = torch.__version__
        info["CUDA Available (PyTorch)"] = "Yes" if CUDA_AVAILABLE else "No"
        if CUDA_AVAILABLE:
             info["CUDA Version (PyTorch Build)"] = getattr(torch.version, 'cuda', "N/A")
    return info

# --- GPU Manager Class ---

class GPUManager:
    def __init__(self, suppress_output=False):
        self.available_devices = [] # List of discovered devices (dict per device)
        self.active_device_info = None # Info dict of the initialized device
        self.active_interface = None # 'opencl' or 'cuda'
        self.initialized = False
        self._suppress_output = suppress_output

        # OpenCL specific attributes
        self.cl_context = None
        self.cl_queue = None
        self.cl_program = None # Store compiled program

        # Kernel source for GPU-only operations (OpenCL)
        self.opencl_kernel_source = """
        __kernel void matrix_mul(__global const float *A,
                                __global const float *B,
                                __global float *C,
                                const int M, // Rows of A / C
                                const int N, // Cols of B / C
                                const int K) // Cols of A / Rows of B
        {
            int row = get_global_id(0); // Row index of C
            int col = get_global_id(1); // Col index of C

            // Check boundary conditions
            if (row >= M || col >= N) {
                return;
            }

            float sum = 0.0f;
            // Compute dot product for C[row, col]
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }

        __kernel void simple_add(__global const float *a,
                                 __global const float *b,
                                 __global float *c)
        {
            int gid = get_global_id(0);
            c[gid] = a[gid] + b[gid];
        }
        """
        # Discover devices on instantiation
        self._discover_devices()

    def _log(self, message, level="info", style=None):
        """Controlled logging method."""
        if self._suppress_output:
            return

        styles = {
            "info": "green",
            "warning": "yellow",
            "error": "bold red",
            "debug": "dim",
        }
        final_style = style if style else styles.get(level, "white")
        console.print(f"[{final_style}]{message}[/{final_style}]")

    def set_suppress_output(self, suppress=True):
        """Enable/disable console output."""
        self._suppress_output = suppress

    def _discover_devices(self):
        """Scan for available GPU devices using PyOpenCL and PyTorch."""
        self.available_devices = []
        device_id_counter = 0

        # 1. Discover OpenCL Devices
        if PYOPENCL_AVAILABLE:
            try:
                platforms = cl.get_platforms()
                for plat_idx, platform in enumerate(platforms):
                    try:
                        devices = platform.get_devices(device_type=cl.device_info.TYPE_GPU)
                        for dev_idx, device in enumerate(devices):
                            mem_size = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
                            info = {
                                'id': device_id_counter,
                                'interface': 'opencl',
                                'platform_name': platform.get_info(cl.platform_info.NAME),
                                'device_name': device.get_info(cl.device_info.NAME).strip(),
                                'vendor': device.get_info(cl.device_info.VENDOR).strip(),
                                'driver_version': device.get_info(cl.device_info.DRIVER_VERSION),
                                'compute_units': device.get_info(cl.device_info.MAX_COMPUTE_UNITS),
                                'global_memory': mem_size,
                                'max_work_group_size': device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE),
                                'opencl_version': device.get_info(cl.device_info.VERSION),
                                'raw_device': device, # Keep the pyopencl device object
                                'details': f"OpenCL {device.get_info(cl.device_info.VERSION)}",
                            }
                            self.available_devices.append(info)
                            device_id_counter += 1
                    except cl.LogicError as cle:
                         # Sometimes querying devices fails for a specific platform
                         self._log(f"Could not query OpenCL devices for platform '{platform.get_info(cl.platform_info.NAME)}': {cle}", "warning")
                    except Exception as e:
                        self._log(f"Unexpected error querying OpenCL devices for platform '{platform.get_info(cl.platform_info.NAME)}': {e}", "warning")

            except cl.Error as e:
                self._log(f"PyOpenCL error during discovery: {e}. OpenCL detection skipped.", "warning")
            except Exception as e:
                 self._log(f"General error during OpenCL discovery: {e}. OpenCL detection skipped.", "warning")
        else:
            self._log("PyOpenCL not found. Skipping OpenCL device discovery.", "info")

        # 2. Discover CUDA Devices (PyTorch)
        if PYTORCH_AVAILABLE and CUDA_AVAILABLE:
            try:
                count = torch.cuda.device_count()
                for i in range(count):
                    props = torch.cuda.get_device_properties(i)
                    mem_size = props.total_memory
                    # Avoid adding duplicates if OpenCL already found an NVIDIA card
                    is_duplicate = False
                    for dev in self.available_devices:
                        # Simple name check (might need refinement)
                        if dev['interface'] == 'opencl' and props.name in dev['device_name'] and abs(dev['global_memory'] - mem_size) < 1024*1024: # Allow 1MB tolerance
                             # Mark the OpenCL device as also CUDA capable if not already marked
                             if 'cuda_capable' not in dev:
                                 dev['cuda_capable'] = True
                                 dev['details'] += f" / CUDA {torch.version.cuda}"
                             is_duplicate = True
                             break

                    if not is_duplicate:
                         info = {
                            'id': device_id_counter,
                            'interface': 'cuda',
                            'platform_name': 'CUDA (PyTorch)',
                            'device_name': props.name,
                            'vendor': 'NVIDIA',
                            'driver_version': getattr(torch.version, 'cuda', 'N/A'), # PyTorch CUDA build version
                            'compute_units': props.multi_processor_count,
                            'global_memory': mem_size,
                            'max_work_group_size': 1024, # Common CUDA default
                            'opencl_version': 'N/A',
                            'raw_device': i, # Store the torch device index
                            'cuda_compute_capability': f"{props.major}.{props.minor}",
                            'details': f"CUDA {getattr(torch.version, 'cuda', 'N/A')} (Compute Capability {props.major}.{props.minor})",
                            'cuda_capable': True,
                         }
                         self.available_devices.append(info)
                         device_id_counter += 1
            except Exception as e:
                self._log(f"Error during PyTorch CUDA discovery: {e}", "error")
        elif PYTORCH_AVAILABLE and not CUDA_AVAILABLE:
            self._log("PyTorch found, but CUDA is not available.", "info")
        else:
            self._log("PyTorch not found. Skipping CUDA device discovery.", "info")


        # Add heuristic Llama compatibility after discovery
        for dev_info in self.available_devices:
            self._assign_llama_heuristics(dev_info)

    def list_available_devices(self):
        """Prints a list of discovered GPU devices."""
        if not self.available_devices:
            self._log("No compatible GPU devices found.", "warning")
            return

        console.print("\n[bold cyan]Available GPU Devices:[/bold cyan]")
        for dev in self.available_devices:
            mem_gb = dev['global_memory'] / (1024**3)
            # Choose compute metric based on interface
            compute_metric = dev.get('cuda_compute_capability', dev.get('compute_units', 'N/A'))
            compute_label = "Cap." if 'cuda_compute_capability' in dev else "CUs"

            console.print(
                f"  ID: {dev['id']} | Interface: [yellow]{dev['interface'].upper()}[/yellow] | Name: [green]{dev['device_name']}[/green] "
                f"| Mem: {mem_gb:.1f} GB | {compute_label}: {compute_metric} "
                f"| Llama Layers (heuristic): {dev.get('llama_layers_assigned', 'N/A')}"
            )
        print("-" * 70)


    def _assign_llama_heuristics(self, device_info):
        """Assign Llama compatibility heuristics based on device info."""
        # Base compatibility primarily on memory and interface type
        mem_gb = device_info['global_memory'] / (1024**3)
        is_cuda = device_info['interface'] == 'cuda' or device_info.get('cuda_capable', False)

        # Llama.cpp and similar often require CUDA or have better CUDA support
        compatible = False
        layers = 0

        if is_cuda and mem_gb >= 4: # Minimum threshold for CUDA devices
             compatible = True
             if mem_gb >= 22: # e.g., A100, RTX 4090
                 layers = -1 # Use all layers (or let the model decide)
             elif mem_gb >= 14: # e.g., T4, RTX 3080/3090
                 layers = -1 # Use all layers (or let the model decide)
             elif mem_gb >= 10: # e.g., RTX 3070
                 layers = 32
             elif mem_gb >= 6: # e.g., RTX 3060, older cards
                 layers = 24
             else: # 4-6 GB
                 layers = 16
        elif device_info['interface'] == 'opencl' and mem_gb >= 6:
            # OpenCL compatibility is less certain for Llama, require more memory
            # Could add vendor checks (e.g., prefer AMD over Intel integrated)
            if "AMD" in device_info['vendor']:
                compatible = True # Tentative compatibility for AMD via OpenCL
                if mem_gb >= 14:
                    layers = 32
                elif mem_gb >= 8:
                    layers = 24
                else:
                    layers = 16
            # else: Intel OpenCL support for large models is often poor

        device_info['llama_compatible'] = compatible
        device_info['llama_layers_assigned'] = layers


    def initialize(self, device_id=None, preferred_interface=None):
        """
        Initializes a specific GPU device.

        Args:
            device_id (int, optional): The ID of the device to initialize from list_available_devices().
                                       Defaults to the first available CUDA device, then the first OpenCL device.
            preferred_interface (str, optional): 'cuda' or 'opencl'. If specified, tries to find a device
                                                 with this interface matching device_id (or the first one).
        """
        if self.initialized:
            self._log("GPUManager already initialized.", "warning")
            return True
        if not self.available_devices:
            self._log("No devices available to initialize.", "error")
            return False

        selected_device = None

        # --- Device Selection Logic ---
        if device_id is not None:
            # Find by ID
            candidates = [dev for dev in self.available_devices if dev['id'] == device_id]
            if not candidates:
                 self._log(f"Device ID {device_id} not found.", "error")
                 return False
            # If preference is given, filter further
            if preferred_interface:
                 interface_candidates = [dev for dev in candidates if dev['interface'] == preferred_interface.lower()]
                 if not interface_candidates:
                      self._log(f"Device ID {device_id} found, but not with preferred interface '{preferred_interface}'.", "warning")
                      selected_device = candidates[0] # Fallback to the found device
                 else:
                      selected_device = interface_candidates[0]
            else:
                 selected_device = candidates[0]
        else:
             # Auto-select: Prioritize CUDA, then OpenCL
             cuda_devices = [dev for dev in self.available_devices if dev['interface'] == 'cuda']
             opencl_devices = [dev for dev in self.available_devices if dev['interface'] == 'opencl']

             if preferred_interface == 'cuda' and cuda_devices:
                  selected_device = cuda_devices[0]
             elif preferred_interface == 'opencl' and opencl_devices:
                  selected_device = opencl_devices[0]
             elif cuda_devices: # Default priority: CUDA first
                  selected_device = cuda_devices[0]
             elif opencl_devices:
                  selected_device = opencl_devices[0]

        if not selected_device:
            self._log("Could not select a device based on criteria.", "error")
            return False

        self.active_device_info = selected_device
        self.active_interface = selected_device['interface']
        self._log(f"Selected Device ID: {selected_device['id']}, Name: {selected_device['device_name']}, Interface: {self.active_interface.upper()}")

        # --- Interface-Specific Initialization ---
        if self.active_interface == 'opencl':
            if not PYOPENCL_AVAILABLE:
                 self._log("Cannot initialize OpenCL device: PyOpenCL is not installed.", "error")
                 return False
            try:
                self._log("Initializing OpenCL context and queue...")
                self.cl_context = cl.Context(devices=[selected_device['raw_device']])
                self.cl_queue = cl.CommandQueue(self.cl_context, properties=cl.command_queue_properties.PROFILING_ENABLE)

                # Build the OpenCL program
                self._log("Building OpenCL kernels...")
                self.cl_program = cl.Program(self.cl_context, self.opencl_kernel_source).build()
                self._log("OpenCL Kernels built successfully.")

                # Run a quick test
                if not self._test_opencl_gpu():
                     self._log("Initial OpenCL GPU test failed.", "error")
                     self.release() # Clean up partially initialized state
                     return False

                self.initialized = True
                self._log("OpenCL GPU initialized successfully.", "info")
                return True

            except cl.Error as e:
                self._log(f"OpenCL Initialization Error: {e}", "error")
                self.active_device_info = None
                self.active_interface = None
                return False
            except Exception as e:
                 self._log(f"Unexpected error during OpenCL initialization: {e}", "error")
                 self.active_device_info = None
                 self.active_interface = None
                 return False

        elif self.active_interface == 'cuda':
            if not PYTORCH_AVAILABLE or not CUDA_AVAILABLE:
                 self._log("Cannot initialize CUDA device: PyTorch or CUDA support is not available.", "error")
                 return False
            try:
                self._log("Initializing CUDA device via PyTorch...")
                # Test by allocating a small tensor on the selected device
                device_index = selected_device['raw_device']
                with torch.cuda.device(device_index):
                    test_tensor = torch.tensor([1.0], device=f'cuda:{device_index}')
                    self._log(f"Successfully allocated tensor on cuda:{device_index}.")
                    del test_tensor
                    gc.collect()
                    torch.cuda.empty_cache() # Clean up test allocation

                # Run a quick PyTorch test
                if not self._test_pytorch_cuda_gpu(device_index):
                     self._log("Initial PyTorch CUDA GPU test failed.", "error")
                     return False

                self.initialized = True
                self._log(f"CUDA GPU (cuda:{device_index}) initialized successfully via PyTorch.", "info")
                return True

            except Exception as e:
                self._log(f"PyTorch CUDA Initialization Error: {e}", "error")
                self.active_device_info = None
                self.active_interface = None
                return False
        else:
             self._log(f"Unknown interface type '{self.active_interface}' for selected device.", "error")
             return False


    def _test_opencl_gpu(self):
        """Test initialized OpenCL GPU with a simple operation."""
        if not self.initialized or self.active_interface != 'opencl':
             self._log("OpenCL interface not initialized for testing.", "warning")
             return False
        self._log("Running OpenCL test (simple vector addition)...", "debug")
        try:
            vector_size = 1024
            a = np.random.rand(vector_size).astype(np.float32)
            b = np.random.rand(vector_size).astype(np.float32)
            c = np.empty_like(a)

            mf = cl.mem_flags
            a_buf = cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
            b_buf = cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
            c_buf = cl.Buffer(self.cl_context, mf.WRITE_ONLY, c.nbytes)

            kernel = self.cl_program.simple_add
            kernel.set_args(a_buf, b_buf, c_buf)

            global_size = (vector_size,)
            local_size = None # Let OpenCL decide

            event = cl.enqueue_nd_range_kernel(self.cl_queue, kernel, global_size, local_size)
            event.wait() # Wait for kernel execution

            cl.enqueue_copy(self.cl_queue, c, c_buf).wait() # Wait for copy

            # Verify
            expected = a + b
            if np.allclose(c, expected):
                 self._log("OpenCL test successful.", "debug")
                 return True
            else:
                 self._log("OpenCL test verification failed.", "error")
                 return False
        except cl.Error as e:
            self._log(f"Error during OpenCL GPU test: {e}", "error")
            return False
        except Exception as e:
             self._log(f"Unexpected error during OpenCL GPU test: {e}", "error")
             return False


    def _test_pytorch_cuda_gpu(self, device_index):
        """Test initialized PyTorch CUDA GPU with a simple operation."""
        if not self.initialized or self.active_interface != 'cuda':
            self._log("CUDA interface not initialized for testing.", "warning")
            return False
        self._log("Running PyTorch CUDA test (simple tensor addition)...", "debug")
        try:
             with torch.cuda.device(device_index):
                  a = torch.randn(1024, device=f'cuda:{device_index}')
                  b = torch.randn(1024, device=f'cuda:{device_index}')
                  c = a + b
                  # Simple check to ensure computation happened on GPU
                  if c.device.type == 'cuda' and c.shape == a.shape:
                       self._log("PyTorch CUDA test successful.", "debug")
                       # Clean up memory
                       del a, b, c
                       gc.collect()
                       torch.cuda.empty_cache()
                       return True
                  else:
                       self._log("PyTorch CUDA test verification failed (result not on CUDA or wrong shape).", "error")
                       return False
        except Exception as e:
             self._log(f"Error during PyTorch CUDA GPU test: {e}", "error")
             return False

    def get_active_device_info(self):
        """Get information about the currently initialized GPU device."""
        if not self.initialized or not self.active_device_info:
            self._log("No device initialized.", "warning")
            return None

        # Add current utilization if possible
        info = self.active_device_info.copy() # Return a copy
        info['current_utilization'] = self.get_current_utilization()
        return info


    def get_current_utilization(self):
        """Get current utilization percentage (heuristic)."""
        if not self.initialized:
            return "N/A"

        utilization = 0.0
        try:
            if self.active_interface == 'opencl':
                # OpenCL doesn't provide direct utilization metrics easily.
                # This is a placeholder; real utilization requires vendor tools (nvidia-smi, rocm-smi)
                # or more complex performance counter monitoring.
                utilization = "N/A (OpenCL)"
            elif self.active_interface == 'cuda':
                device_index = self.active_device_info['raw_device']
                # Use PyTorch memory stats as a proxy for activity
                mem_info = torch.cuda.mem_get_info(device_index)
                total_mem = mem_info[1]
                used_mem = total_mem - mem_info[0]
                if total_mem > 0:
                     # This is memory utilization, not compute utilization
                     utilization = f"{(used_mem / total_mem * 100):.1f}% (Memory)"
                else:
                     utilization = "0.0% (Memory)"
                # Note: For actual compute utilization on NVIDIA, nvidia-smi is the standard tool.
        except Exception as e:
            self._log(f"Could not get utilization stats: {e}", "warning")
            utilization = "Error"

        return utilization

    def run_matrix_multiply(self, A, B):
        """
        Runs matrix multiplication C = A * B on the active GPU.

        Args:
            A (numpy.ndarray): Left matrix (float32).
            B (numpy.ndarray): Right matrix (float32).

        Returns:
            numpy.ndarray: Result matrix C, or None if failed.
        """
        if not self.initialized:
            self._log("GPU not initialized.", "error")
            return None

        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)

        if A.ndim != 2 or B.ndim != 2:
            self._log("Input arrays must be 2D matrices.", "error")
            return None
        if A.shape[1] != B.shape[0]:
             self._log(f"Matrix dimensions mismatch for multiplication: A({A.shape}) B({B.shape})", "error")
             return None

        M, K = A.shape
        K_B, N = B.shape # K_B should be equal to K

        # --- Execute based on active interface ---
        if self.active_interface == 'opencl':
            self._log("Running matrix multiply using OpenCL...", "debug")
            try:
                C = np.zeros((M, N), dtype=np.float32)
                mf = cl.mem_flags
                a_buf = cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
                b_buf = cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
                c_buf = cl.Buffer(self.cl_context, mf.WRITE_ONLY, C.nbytes)

                kernel = self.cl_program.matrix_mul
                # Args: A, B, C, M, N, K
                kernel.set_args(a_buf, b_buf, c_buf, np.int32(M), np.int32(N), np.int32(K))

                # Use 2D work groups
                max_wg_size = self.active_device_info['max_work_group_size']
                # Determine local size (e.g., 16x16, clamped by max_wg_size)
                local_sz_x = min(16, int(np.sqrt(max_wg_size)))
                local_sz_y = min(16, int(np.sqrt(max_wg_size)))
                local_size = (local_sz_x, local_sz_y)

                # Ensure global size is multiple of local size
                global_size_x = ((M + local_sz_x - 1) // local_sz_x) * local_sz_x
                global_size_y = ((N + local_sz_y - 1) // local_sz_y) * local_sz_y
                global_size = (global_size_x, global_size_y)

                event = cl.enqueue_nd_range_kernel(self.cl_queue, kernel, global_size, local_size)
                event.wait()

                cl.enqueue_copy(self.cl_queue, C, c_buf).wait()
                self._log("OpenCL matrix multiply completed.", "debug")
                return C

            except cl.Error as e:
                 self._log(f"OpenCL matrix multiply error: {e}", "error")
                 return None
            except Exception as e:
                  self._log(f"Unexpected error during OpenCL matrix multiply: {e}", "error")
                  return None

        elif self.active_interface == 'cuda':
            self._log("Running matrix multiply using PyTorch CUDA...", "debug")
            try:
                device_index = self.active_device_info['raw_device']
                with torch.cuda.device(device_index):
                    a_torch = torch.from_numpy(A).to(f'cuda:{device_index}')
                    b_torch = torch.from_numpy(B).to(f'cuda:{device_index}')
                    c_torch = torch.matmul(a_torch, b_torch)
                    C = c_torch.cpu().numpy() # Copy result back to host
                    # Clean up GPU memory
                    del a_torch, b_torch, c_torch
                    gc.collect()
                    torch.cuda.empty_cache()
                self._log("PyTorch CUDA matrix multiply completed.", "debug")
                return C
            except Exception as e:
                self._log(f"PyTorch CUDA matrix multiply error: {e}", "error")
                return None
        else:
             self._log("No active interface to run matrix multiply.", "error")
             return None

    def release(self):
        """Clean up GPU resources."""
        self._log("Releasing GPUManager resources...", "info")
        if self.active_interface == 'opencl':
            try:
                if self.cl_queue:
                    self.cl_queue.finish()
                # Context and program are usually managed by pyopencl's GC
                self.cl_queue = None
                self.cl_context = None
                self.cl_program = None
                self._log("OpenCL resources released.", "debug")
            except Exception as e:
                 # Use print as console might be unavailable during shutdown
                 print(f"Error releasing OpenCL resources: {e}")
        elif self.active_interface == 'cuda':
             try:
                  # Clean PyTorch cache
                  if PYTORCH_AVAILABLE and CUDA_AVAILABLE:
                       gc.collect()
                       torch.cuda.empty_cache()
                  self._log("PyTorch CUDA cache cleared.", "debug")
             except Exception as e:
                  print(f"Error clearing PyTorch CUDA cache: {e}")

        self.active_device_info = None
        self.active_interface = None
        self.initialized = False
        self._log("GPUManager released.", "info")

    def is_initialized(self):
        """Check if GPU is initialized."""
        return self.initialized

    def __enter__(self):
        """Context manager enter."""
        # Initialization should happen separately
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


# --- Example Usage ---
if __name__ == "__main__":

    console.print(Panel("GPU Manager Test", style="bold magenta", expand=False))

    # Print System Info
    console.print("\n[bold]System Information:[/bold]")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        console.print(f"  {key}: [cyan]{value}[/cyan]")

    # Initialize Manager
    # manager = GPUManager(suppress_output=False) # To see all logs
    manager = GPUManager()

    # List detected devices
    manager.list_available_devices()

    if not manager.available_devices:
         console.print("\n[bold red]Exiting: No suitable GPUs found.[/bold red]")
         sys.exit(1)

    # Initialize the default device (prioritizes CUDA)
    console.print("\n[bold]Initializing Default GPU...[/bold]")
    if manager.initialize():
        console.print("[bold green]Initialization Successful![/bold green]")

        # Get info about the active device
        info = manager.get_active_device_info()
        if info:
             console.print("\n[bold]Active Device Information:[/bold]")
             # Sort keys for consistent output
             sorted_keys = sorted([k for k in info if k not in ['raw_device']])
             for key in sorted_keys:
                  value = info[key]
                  if key == 'global_memory':
                       value = format_bytes(value)
                  console.print(f"  {key.replace('_', ' ').title()}: [cyan]{value}[/cyan]")

        # Perform a matrix multiplication test
        console.print("\n[bold]Running Matrix Multiplication Test (128x128)...[/bold]")
        size = 128
        mat_a = np.random.rand(size, size).astype(np.float32)
        mat_b = np.random.rand(size, size).astype(np.float32)

        start_time = time.time()
        result_gpu = manager.run_matrix_multiply(mat_a, mat_b)
        end_time = time.time()

        if result_gpu is not None:
             console.print(f"GPU computation time: [yellow]{(end_time - start_time)*1000:.2f} ms[/yellow]")

             # Optional: Verify against CPU result (can be slow for large matrices)
             # console.print("Verifying against CPU result...")
             # start_time_cpu = time.time()
             # result_cpu = np.matmul(mat_a, mat_b)
             # end_time_cpu = time.time()
             # console.print(f"CPU computation time: {(end_time_cpu - start_time_cpu)*1000:.2f} ms")
             # if np.allclose(result_gpu, result_cpu, atol=1e-3):
             #      console.print("[green]Verification successful![/green]")
             # else:
             #      console.print("[bold red]Verification FAILED![/bold red]")
        else:
            console.print("[bold red]Matrix multiplication test failed.[/bold red]")

        # Clean up
        console.print("\n[bold]Releasing GPU resources...[/bold]")
        manager.release()
        console.print("[bold green]Resources Released.[/bold green]")

    else:
        console.print("[bold red]GPU Initialization Failed.[/bold red]")

    console.print(Panel("GPU Manager Test Complete", style="bold magenta", expand=False))