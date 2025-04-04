# --- START OF FILE gpu_manager.py ---

import numpy as np
import time
import sys
import platform
import subprocess
import os
import threading
import queue
import gc
import contextlib

# Optional PyOpenCL import
try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    # Define dummy cl.Error if pyopencl is not installed
    class CLError(Exception): pass
    class cl:
        class Error(CLError): pass # Define Error within cl for consistent except blocks
        device_type = type('obj', (object,), {'GPU': None})() # Dummy type
        device_info = type('obj', (object,), {})() # Dummy type
        command_queue_properties = type('obj', (object,), {'PROFILING_ENABLE': None})() # Dummy type
        mem_flags = type('obj', (object,), {'READ_ONLY': None, 'COPY_HOST_PTR': None, 'WRITE_ONLY': None})() # Dummy type
        @staticmethod
        def get_platforms(): return []
        @staticmethod
        def Context(*args, **kwargs): raise ImportError("pyopencl not installed")
        @staticmethod
        def CommandQueue(*args, **kwargs): raise ImportError("pyopencl not installed")
        @staticmethod
        def Buffer(*args, **kwargs): raise ImportError("pyopencl not installed")
        @staticmethod
        def Program(*args, **kwargs): raise ImportError("pyopencl not installed")
        @staticmethod
        def enqueue_nd_range_kernel(*args, **kwargs): raise ImportError("pyopencl not installed")
        @staticmethod
        def enqueue_copy(*args, **kwargs): raise ImportError("pyopencl not installed")


# Required imports
import torch
import psutil
from rich.console import Console

console = Console()

# Helper function for formatting bytes
def format_bytes(size_bytes: int, suffix: str = "B") -> str:
    """Converts bytes to a human-readable format (KB, MB, GB, TB)."""
    if not isinstance(size_bytes, (int, float)) or size_bytes < 0:
        return "N/A"
    if size_bytes == 0:
        return "0 B"
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if size_bytes < factor:
            return f"{size_bytes:.2f} {unit}{suffix}"
        size_bytes /= factor
    return f"{size_bytes:.2f} P{suffix}" # Fallback for very large sizes

class GPUManager:
    def __init__(self):
        # Core state
        self.initialized = False
        self.selected_device_info = None # Stores info of the chosen device
        self.selected_device_source = None # 'cuda' or 'opencl'

        # CUDA specific state (PyTorch)
        self.torch_device = None # Will be torch.device object if CUDA is used

        # OpenCL specific state
        self.cl_context = None
        self.cl_device = None # pyopencl device object
        self.cl_queue = None
        self.cl_platform = None

        # Detected devices (raw info before selection)
        self._detected_cuda_devices = []
        self._detected_opencl_devices = []

        # Llama related state
        self.llama_compatible = False
        self.llama_layers_assigned = 0

        # Workload distribution state (only for non-Llama compatible OpenCL devices)
        self.work_queue = queue.Queue()
        self.worker_thread = None
        self.running = False # Worker thread control flag

        # Internal state
        self._suppress_output = False
        self._opencl_available_runtime = PYOPENCL_AVAILABLE # Check if import succeeded

        # Kernel source for GPU-only operations (OpenCL)
        self.kernel_source = """
        __kernel void matrix_mul(__global const float *A,
                                __global const float *B,
                                __global float *C,
                                const int M, // Rows of A / C
                                const int N, // Cols of B / C
                                const int K) // Cols of A / Rows of B
        {
            int row = get_global_id(0); // Row index of C
            int col = get_global_id(1); // Col index of C

            // Check bounds
            if (row >= M || col >= N) {
                return;
            }

            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
        """

    def _log(self, message, level="info"):
        """Controlled logging method."""
        if self._suppress_output:
            return
        styles = {"info": "green", "warning": "yellow", "error": "red", "debug": "dim"}
        console.print(f"[{styles.get(level, 'white')}]{message}[/]")

    def set_suppress_output(self, suppress=True):
        """Enable or disable logging output."""
        self._suppress_output = suppress

    def _get_cuda_device_info(self, device_id):
        """Get detailed info for a specific CUDA device using PyTorch."""
        try:
            props = torch.cuda.get_device_properties(device_id)
            free_mem, total_mem = torch.cuda.mem_get_info(device_id)
            gpu_memory_gb = total_mem / (1024**3)
            compute_capability = f"{props.major}.{props.minor}"

            # Llama layers heuristic (adjust thresholds as needed)
            # Based on TOTAL memory, assuming Llama needs most of it
            if gpu_memory_gb >= 22:  # e.g., A100 40GB/80GB, H100
                llama_layers = -1 # Use all layers
            elif gpu_memory_gb >= 14: # e.g., T4, RTX 3090/4090, A10
                llama_layers = -1 # Use all layers
            elif gpu_memory_gb >= 10: # e.g., RTX 3080, RTX 2080Ti
                 llama_layers = 40 # High number, adjust based on model
            elif gpu_memory_gb >= 7:  # e.g., RTX 2070/3070
                llama_layers = 32
            elif gpu_memory_gb >= 5:  # e.g., GTX 1070/1080, RTX 2060/3060
                llama_layers = 24
            else: # Lower end
                llama_layers = 16

            # Determine Llama compatibility based on memory and compute capability
            # Example thresholds: require >= 6GB and CC >= 6.0
            is_compatible = gpu_memory_gb >= 6 and props.major >= 6
            max_compute_units = props.multi_processor_count

            return {
                'id': device_id,
                'source': 'cuda',
                'name': props.name,
                'vendor': 'NVIDIA',
                'version': f"CUDA {torch.version.cuda}",
                'compute_capability': compute_capability,
                'multi_processor_count': props.multi_processor_count,
                'global_mem_size': total_mem,
                'max_compute_units': max_compute_units,
                'free_mem_size': free_mem,
                'max_work_group_size': getattr(props, 'maxThreadsPerBlock', 1024), # Common CUDA limit
                'llama_compatible': is_compatible,
                'llama_layers_assigned': llama_layers if is_compatible else 0,
            }
        except Exception as e:
            self._log(f"Warning: Could not get CUDA device info for ID {device_id}: {e}", "warning")
            return None

    def _get_opencl_device_info(self, device, platform):
        """Get detailed info for a specific OpenCL device."""
        if not self._opencl_available_runtime:
            return None
        try:
            mem_size = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
            # OpenCL doesn't have a standard way to get free memory easily
            return {
                'id': device.int_ptr, # Unique identifier
                'source': 'opencl',
                'name': device.get_info(cl.device_info.NAME).strip(),
                'vendor': device.get_info(cl.device_info.VENDOR).strip(),
                'version': device.get_info(cl.device_info.VERSION).strip(),
                'driver_version': device.get_info(cl.device_info.DRIVER_VERSION).strip(),
                'max_compute_units': device.get_info(cl.device_info.MAX_COMPUTE_UNITS),
                'global_mem_size': mem_size,
                'free_mem_size': None, # Not reliably available in OpenCL
                'max_work_group_size': device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE),
                'cl_device': device, # Keep reference to object
                'cl_platform': platform, # Keep reference to object
                'llama_compatible': False, # Assume OpenCL is not primary Llama path
                'llama_layers_assigned': 0,
            }
        except cl.Error as e:
            self._log(f"Warning: Could not get OpenCL device info for {device.name}: {e}", "warning")
            return None
        except Exception as e:
            self._log(f"Warning: Unexpected error getting OpenCL info for {device.name}: {e}", "warning")
            return None


    def _discover_devices(self):
        """Discover available CUDA and OpenCL devices."""
        self._detected_cuda_devices = []
        self._detected_opencl_devices = []

        # 1. Discover CUDA devices via PyTorch
        if torch.cuda.is_available():
            try:
                count = torch.cuda.device_count()
                self._log(f"Found {count} CUDA device(s) via PyTorch.")
                for i in range(count):
                    info = self._get_cuda_device_info(i)
                    if info:
                        self._detected_cuda_devices.append(info)
            except Exception as e:
                self._log(f"Error discovering CUDA devices: {e}", "error")
        else:
            self._log("PyTorch CUDA not available.", "info")

        # 2. Discover OpenCL devices (if available)
        if self._opencl_available_runtime:
            self._log("Attempting to discover OpenCL devices...")
            try:
                platforms = cl.get_platforms()
                if not platforms:
                    self._log("No OpenCL platforms found.", "warning")

                for platform in platforms:
                    try:
                        gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                        self._log(f"Platform '{platform.name}' has {len(gpu_devices)} GPU device(s).")
                        for device in gpu_devices:
                            # Avoid adding duplicates if a CUDA device also appears here
                            is_duplicate_cuda = False
                            if self._detected_cuda_devices:
                                # Simple name check (might need refinement)
                                cl_name = device.get_info(cl.device_info.NAME).strip()
                                if any(cuda_dev['name'] == cl_name for cuda_dev in self._detected_cuda_devices):
                                     self._log(f"Skipping OpenCL device '{cl_name}' as it seems to be a duplicate of a detected CUDA device.", "debug")
                                     is_duplicate_cuda = True

                            if not is_duplicate_cuda:
                                info = self._get_opencl_device_info(device, platform)
                                if info:
                                    self._detected_opencl_devices.append(info)
                    except cl.Error as e:
                        # Often happens if drivers are mismatched or no devices of type GPU exist
                         self._log(f"Warning: OpenCL error querying devices on platform '{platform.name}': {e}", "warning")
                    except Exception as e:
                         self._log(f"Warning: Unexpected error querying devices on platform '{platform.name}': {e}", "warning")

            except cl.Error as e:
                 self._log(f"Error getting OpenCL platforms: {e}", "warning")
                 self._opencl_available_runtime = False # Mark as unavailable if platform check fails
            except Exception as e:
                 self._log(f"Unexpected error during OpenCL discovery: {e}", "error")
                 self._opencl_available_runtime = False
        else:
            self._log("PyOpenCL not installed or available.", "info")

    def _select_device(self, device_index=None, preferred_gpu=None):
        """Selects the best available device based on criteria."""
        all_devices = self._detected_cuda_devices + self._detected_opencl_devices

        if not all_devices:
            self._log("No compatible GPU devices found.", "error")
            return None

        selected_device = None

        # 1. By Index (if specified and valid)
        if device_index is not None:
            if 0 <= device_index < len(all_devices):
                selected_device = all_devices[device_index]
                self._log(f"Selected device by index {device_index}: {selected_device['name']} ({selected_device['source']})")
            else:
                self._log(f"Warning: Invalid device_index {device_index}. Max index is {len(all_devices)-1}.", "warning")

        # 2. By Preferred Name/Vendor (if index not used or invalid)
        if not selected_device and preferred_gpu:
            preference = preferred_gpu.lower()
            # Prioritize exact name match, then vendor match
            for device in all_devices:
                if preference == device['name'].lower():
                    selected_device = device
                    break
            if not selected_device:
                 for device in all_devices:
                    if preference in device['vendor'].lower():
                        selected_device = device
                        break # Take first vendor match
            if selected_device:
                 self._log(f"Selected device by preference '{preferred_gpu}': {selected_device['name']} ({selected_device['source']})")
            else:
                 self._log(f"Warning: Preferred GPU '{preferred_gpu}' not found.", "warning")

        # 3. Default: Select first CUDA device if available, otherwise first OpenCL
        if not selected_device:
            if self._detected_cuda_devices:
                selected_device = self._detected_cuda_devices[0]
            elif self._detected_opencl_devices:
                selected_device = self._detected_opencl_devices[0]

        if selected_device:
             self._log(f"Final selected device: {selected_device['name']} ({selected_device['source']})")
             return selected_device
        else:
             # This case should ideally not be reached if all_devices was not empty
             self._log("Could not select a suitable device.", "error")
             return None

    def _test_cuda_device(self):
        """Perform a simple test on the selected CUDA device."""
        self._log(f"Testing CUDA device {self.torch_device}...", "debug")
        try:
            # Simple tensor creation and operation
            a = torch.randn(100, 100, device=self.torch_device)
            b = torch.randn(100, 100, device=self.torch_device)
            c = torch.matmul(a, b)
            torch.cuda.synchronize(self.torch_device) # Wait for completion
            _ = c.cpu() # Bring back a result to ensure it worked
            self._log("CUDA device test successful.", "debug")
            return True
        except Exception as e:
            self._log(f"CUDA device test failed: {e}", "error")
            return False

    def _test_opencl_device(self):
        """Perform matrix multiplication test on the selected OpenCL device."""
        self._log(f"Testing OpenCL device {self.selected_device_info['name']}...", "debug")
        if not self.cl_context or not self.cl_queue or not self.cl_device:
             self._log("OpenCL context/queue/device not initialized for testing.", "error")
             return False
        try:
            # Simple matrix multiplication test (adjust size if needed)
            matrix_size = 64
            mem_size = self.selected_device_info['global_mem_size']
            # Limit test size to avoid excessive memory usage
            max_matrix_elems = mem_size * 0.05 # Use max 5% of memory
            max_matrix_size = int(np.sqrt(max_matrix_elems / (3 * 4))) # 3 matrices, float32
            matrix_size = min(matrix_size, max_matrix_size, 256) # Absolute cap
            if matrix_size < 16: # Minimum reasonable size
                 self._log(f"Skipping OpenCL test due to very low available memory (estimated test size: {matrix_size})", "warning")
                 return True # Skip test but don't fail init

            self._log(f"Using matrix size {matrix_size}x{matrix_size} for OpenCL test.", "debug")

            A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
            B = np.random.rand(matrix_size, matrix_size).astype(np.float32)
            C = np.zeros((matrix_size, matrix_size), dtype=np.float32)

            # Create buffers
            mf = cl.mem_flags
            A_buf = cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
            B_buf = cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
            C_buf = cl.Buffer(self.cl_context, mf.WRITE_ONLY, C.nbytes)

            # Build program
            program = cl.Program(self.cl_context, self.kernel_source).build()
            kernel = program.matrix_mul

            # Set kernel arguments (M, N, K for the C = A * B kernel)
            M, K = A.shape
            K_B, N = B.shape # K should match K_B
            kernel.set_args(A_buf, B_buf, C_buf, np.int32(M), np.int32(N), np.int32(K))

            # Determine optimal work group size (heuristic)
            max_wg_size = self.selected_device_info['max_work_group_size']
            # Use smaller workgroups, often safer across different CL implementations
            work_group_dim = int(np.sqrt(max_wg_size)) // 2
            work_group_dim = max(min(work_group_dim, 8), 4) # Clamp between 4 and 8
            local_size = (work_group_dim, work_group_dim)

            # Global size needs to be multiple of local size
            global_size_x = ((M + local_size[0] - 1) // local_size[0]) * local_size[0]
            global_size_y = ((N + local_size[1] - 1) // local_size[1]) * local_size[1]
            global_size = (global_size_x, global_size_y)

            self._log(f"Executing OpenCL kernel with global_size={global_size}, local_size={local_size}", "debug")

            # Execute kernel
            cl.enqueue_nd_range_kernel(self.cl_queue, kernel, global_size, local_size)
            self.cl_queue.finish() # Wait for completion

            # Read back results
            cl.enqueue_copy(self.cl_queue, C, C_buf).wait()

            # Verify results
            expected = np.matmul(A, B)
            if not np.allclose(C, expected, atol=1e-2):
                self._log("OpenCL matrix multiplication test FAILED verification.", "error")
                # Optionally log C and expected for debugging if needed
                return False
            else:
                self._log("OpenCL matrix multiplication test successful.", "debug")
                return True

        except cl.Error as e:
            self._log(f"OpenCL Error during GPU test: {e}", "error")
            return False
        except Exception as e:
            self._log(f"Unexpected Error during OpenCL GPU test: {e}", "error")
            return False

    def initialize(self, device_index=None, preferred_gpu=None):
        """
        Initialize the GPU Manager.

        Detects available GPUs (CUDA via PyTorch first, then OpenCL),
        selects one based on criteria, initializes it, and performs a basic test.

        Args:
            device_index (int, optional): Preferred index of the device to use from the combined list.
            preferred_gpu (str, optional): Substring to match against device name or vendor.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        self._log("Initializing GPU Manager...")
        self.cleanup() # Ensure clean state before starting

        self._discover_devices()
        self.selected_device_info = self._select_device(device_index, preferred_gpu)

        if not self.selected_device_info:
            self._log("Initialization failed: No suitable GPU device selected.", "error")
            return False

        self.selected_device_source = self.selected_device_info['source']
        self.llama_compatible = self.selected_device_info.get('llama_compatible', False)
        self.llama_layers_assigned = self.selected_device_info.get('llama_layers_assigned', 0)

        init_success = False
        test_success = False

        # Initialize based on the source of the selected device
        if self.selected_device_source == 'cuda':
            try:
                cuda_id = self.selected_device_info['id']
                self.torch_device = torch.device(f'cuda:{cuda_id}')
                self._log(f"Initializing CUDA device: {self.selected_device_info['name']} (torch.device='{self.torch_device}')")
                # PyTorch handles context management implicitly
                init_success = True
                test_success = self._test_cuda_device()
            except Exception as e:
                self._log(f"Failed to initialize selected CUDA device: {e}", "error")
                init_success = False

        elif self.selected_device_source == 'opencl':
            if not self._opencl_available_runtime:
                 self._log("Initialization failed: OpenCL selected but PyOpenCL is not available.", "error")
                 return False
            try:
                self.cl_device = self.selected_device_info['cl_device']
                self.cl_platform = self.selected_device_info['cl_platform']
                self._log(f"Initializing OpenCL device: {self.selected_device_info['name']}")

                # Create context and queue
                self.cl_context = cl.Context(devices=[self.cl_device])
                self.cl_queue = cl.CommandQueue(
                    self.cl_context,
                    properties=cl.command_queue_properties.PROFILING_ENABLE
                )
                init_success = True
                test_success = self._test_opencl_device()

                # Start worker thread only if OpenCL is used AND not Llama compatible
                if init_success and test_success and not self.llama_compatible:
                     self._start_worker_thread()

            except cl.Error as e:
                self._log(f"Failed to initialize selected OpenCL device: {e}", "error")
                init_success = False
            except Exception as e:
                self._log(f"Unexpected error initializing OpenCL device: {e}", "error")
                init_success = False
        else:
             self._log(f"Initialization failed: Unknown device source '{self.selected_device_source}'.", "error")
             return False


        if init_success and test_success:
            self.initialized = True
            self._log("GPU Manager initialized successfully.", "info")
            # Display summary after successful init
            self.display_selected_device_summary()
            return True
        else:
            self._log(f"Initialization failed (Init OK: {init_success}, Test OK: {test_success}). Cleaning up.", "error")
            self.cleanup() # Clean up partial initialization
            return False

    def display_selected_device_summary(self):
        """Prints a summary of the initialized device."""
        if not self.initialized or not self.selected_device_info:
            self._log("Cannot display summary: GPU Manager not initialized.", "warning")
            return

        info = self.selected_device_info
        console.print("\n--- Initialized GPU Summary ---", style="bold cyan")
        print(f"  Name:       {info.get('name', 'N/A')}")
        print(f"  Source:     {info.get('source', 'N/A').upper()}")
        print(f"  Vendor:     {info.get('vendor', 'N/A')}")
        mem_gb = info.get('global_mem_size', 0) / (1024**3)
        print(f"  Memory:     {mem_gb:.2f} GB")
        if info['source'] == 'cuda':
             free_mem_gb = info.get('free_mem_size', 0) / (1024**3)
             print(f"  Free Mem:   {free_mem_gb:.2f} GB (approx)")
             print(f"  Capability: {info.get('compute_capability', 'N/A')}")
        print(f"  Llama Comp: {'Yes' if self.llama_compatible else 'No'}")
        if self.llama_compatible:
             layers = self.llama_layers_assigned if self.llama_layers_assigned > 0 else "All"
             print(f"  Llama Layers: {layers} (heuristic)")
        if self.selected_device_source == 'opencl' and not self.llama_compatible:
             print(f"  Work Queue: Active for OpenCL tasks")
        console.print("-----------------------------", style="bold cyan")


    # --- Worker Thread and Task Processing (for OpenCL non-Llama path) ---

    def _start_worker_thread(self):
        """Starts the background worker thread for OpenCL tasks."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self._log("Worker thread already running.", "debug")
            return

        if not self.selected_device_source == 'opencl' or self.llama_compatible:
             self._log("Worker thread not needed for this configuration.", "debug")
             return

        if not self.cl_context or not self.cl_queue:
            self._log("Cannot start worker thread: OpenCL context/queue not ready.", "error")
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self._log("Started OpenCL worker thread.", "info")

    def _worker_loop(self):
        """The main loop for the background worker thread."""
        self._log("OpenCL worker thread started.", "debug")
        while self.running:
            try:
                task = self.work_queue.get(timeout=1.0) # Wait for 1 second
                if task is None: # Allow graceful exit signal?
                    continue

                self._process_cl_task(task)
                self.work_queue.task_done()

            except queue.Empty:
                continue # No task, loop again
            except Exception as e:
                self._log(f"Error in OpenCL worker thread: {e}", "error")
        self._log("OpenCL worker thread finished.", "debug")


    def _process_cl_task(self, task):
        """Processes a task using OpenCL."""
        task_type = task.get('type')
        data = task.get('data')
        callback = task.get('callback')
        result = None
        error = None

        try:
            if task_type == 'matrix_mul':
                A = data.get('A')
                B = data.get('B')
                if A is not None and B is not None:
                    result = self._matrix_mul_cl(A, B)
                else:
                    error = "Missing matrix data 'A' or 'B'"
            # Add more OpenCL task types here if needed
            # elif task_type == 'some_other_cl_op':
            #     result = self._some_other_cl_op(data)
            else:
                error = f"Unknown OpenCL task type: {task_type}"

        except Exception as e:
             self._log(f"Error processing OpenCL task '{task_type}': {e}", "error")
             error = str(e)

        if callback and callable(callback):
            try:
                callback(result, error) # Pass result and potential error to callback
            except Exception as cb_e:
                 self._log(f"Error executing task callback: {cb_e}", "error")


    def _matrix_mul_cl(self, A, B):
        """Performs matrix multiplication using the initialized OpenCL device."""
        if not self.initialized or self.selected_device_source != 'opencl' or not self.cl_queue:
            raise RuntimeError("OpenCL environment not ready for matrix multiplication.")

        try:
            # Validate and prepare inputs
            A = np.asarray(A, dtype=np.float32)
            B = np.asarray(B, dtype=np.float32)
            if A.ndim != 2 or B.ndim != 2:
                 raise ValueError("Inputs must be 2D matrices.")
            if A.shape[1] != B.shape[0]:
                raise ValueError(f"Matrix dimensions mismatch for multiplication: A({A.shape}) and B({B.shape})")

            M, K = A.shape
            K_B, N = B.shape
            C = np.zeros((M, N), dtype=np.float32)

            # Create buffers (consider buffer reuse strategies for performance)
            mf = cl.mem_flags
            with contextlib.ExitStack() as stack:
                A_buf = stack.enter_context(cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A))
                B_buf = stack.enter_context(cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B))
                C_buf = stack.enter_context(cl.Buffer(self.cl_context, mf.WRITE_ONLY, C.nbytes))

                # Build program (consider caching compiled programs)
                program = cl.Program(self.cl_context, self.kernel_source).build()
                kernel = program.matrix_mul

                kernel.set_args(A_buf, B_buf, C_buf, np.int32(M), np.int32(N), np.int32(K))

                # Determine execution parameters (same logic as test)
                max_wg_size = self.selected_device_info['max_work_group_size']
                work_group_dim = int(np.sqrt(max_wg_size)) // 2
                work_group_dim = max(min(work_group_dim, 8), 4)
                local_size = (work_group_dim, work_group_dim)
                global_size_x = ((M + local_size[0] - 1) // local_size[0]) * local_size[0]
                global_size_y = ((N + local_size[1] - 1) // local_size[1]) * local_size[1]
                global_size = (global_size_x, global_size_y)

                # Execute and wait
                cl.enqueue_nd_range_kernel(self.cl_queue, kernel, global_size, local_size)
                self.cl_queue.finish()

                # Read result
                cl.enqueue_copy(self.cl_queue, C, C_buf).wait()

            return C

        except cl.Error as e:
            self._log(f"OpenCL error during matrix multiplication task: {e}", "error")
            raise # Re-raise cl.Error
        except Exception as e:
            self._log(f"Unexpected error during matrix multiplication task: {e}", "error")
            raise # Re-raise other exceptions


    def add_task(self, task_type, data, callback=None):
        """
        Add a task to the workload queue (only if using OpenCL and not Llama compatible).

        Args:
            task_type (str): The type of task (e.g., 'matrix_mul').
            data (dict): Data required for the task.
            callback (callable, optional): Function to call with (result, error) upon completion.

        Returns:
            bool: True if the task was successfully added to the queue, False otherwise.
        """
        if not self.initialized:
            self._log("Cannot add task: GPU manager not initialized.", "warning")
            return False

        if self.selected_device_source == 'opencl' and not self.llama_compatible:
            if self.running and self.worker_thread and self.worker_thread.is_alive():
                task = {'type': task_type, 'data': data, 'callback': callback}
                self.work_queue.put(task)
                self._log(f"Added OpenCL task '{task_type}' to queue.", "debug")
                return True
            else:
                 self._log("Cannot add task: OpenCL worker thread is not running.", "warning")
                 return False
        else:
            # Log why task is not added (either CUDA path or Llama compatible)
            reason = "using CUDA" if self.selected_device_source == 'cuda' else "GPU is Llama-compatible"
            self._log(f"Task '{task_type}' not added to queue because {reason}. Process directly if needed.", "info")
            return False

    # --- Public Accessors and Control ---

    def get_device_info(self):
        """
        Get information about the currently initialized GPU device.

        Returns:
            dict: A dictionary containing device details, or None if not initialized.
                  Keys might include: 'name', 'vendor', 'version', 'global_mem_size', etc.
                  Includes 'llama_compatible' and 'llama_layers_assigned'.
        """
        if not self.initialized:
            self._log("Cannot get device info: GPU Manager not initialized.", "warning")
            return None
        # Return a copy to prevent external modification
        return self.selected_device_info.copy() if self.selected_device_info else None

    def is_initialized(self):
        """Check if the GPU Manager has been successfully initialized."""
        return self.initialized

    def cleanup(self):
        """Clean up resources: stop worker thread, finish queues."""
        self._log("Cleaning up GPU Manager resources...", "debug")
        # Stop worker thread first
        if self.running:
            self.running = False
            if self.worker_thread and self.worker_thread.is_alive():
                self._log("Waiting for worker thread to finish...", "debug")
                # Put a None task to potentially wake up the thread from queue.get
                # try: self.work_queue.put(None, timeout=0.1)
                # except queue.Full: pass
                self.worker_thread.join(timeout=2.0)
                if self.worker_thread.is_alive():
                    self._log("Warning: Worker thread did not terminate gracefully.", "warning")
        self.worker_thread = None

        # Clear the queue (after stopping the worker)
        while not self.work_queue.empty():
            try:
                self.work_queue.get_nowait()
                self.work_queue.task_done()
            except queue.Empty:
                break
            except Exception: pass # Ignore errors during cleanup

        # Release OpenCL resources (primarily finish queue)
        # Context/Device release is tricky, often GC is safer. Finishing queue is important.
        if self.cl_queue:
            try:
                self.cl_queue.finish()
                self._log("Finished OpenCL command queue.", "debug")
            except Exception as e:
                # Use print here as console might be problematic during shutdown
                print(f"GPUManager Cleanup Warning: Error finishing OpenCL queue: {e}")
        # Reset state
        self.cl_queue = None
        self.cl_context = None
        self.cl_device = None
        self.cl_platform = None
        self.torch_device = None
        self.selected_device_info = None
        self.selected_device_source = None
        self.initialized = False
        self.llama_compatible = False
        self.llama_layers_assigned = 0
        # Don't reset _suppress_output or detected devices list here

        # Explicitly run garbage collection (optional, might help release CUDA memory)
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        self._log("GPU Manager cleanup complete.", "debug")
        return True # Maintain signature

    def __del__(self):
        """Destructor attempts to clean up resources."""
        # Be careful in __del__, avoid complex logging or operations that might fail
        try:
            if self.initialized: # Only cleanup if it was initialized
                 # Use print as logging might be unstable during interpreter shutdown
                 print("GPUManager: __del__ triggering cleanup...")
                 self.cleanup()
        except Exception as e:
            # Keep __del__ silent on errors during interpreter shutdown
            print(f"GPUManager: Error in __del__ cleanup: {e}")
            pass

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    console.print("[bold blue]GPU Manager Example Usage[/bold blue]")

    manager = GPUManager()
    # manager.set_suppress_output(False) # Ensure output is visible

    # Initialize (try default first)
    if manager.initialize():
        console.print("\n[bold green]Initialization Successful![/bold green]")
        info = manager.get_device_info()
        # console.print("Device Info:", info) # Raw info

        # Example: Add a task if OpenCL worker is active
        if manager.selected_device_source == 'opencl' and not manager.llama_compatible:
            console.print("\nAdding Matrix Multiplication task to OpenCL queue...")
            mat_a = np.random.rand(128, 64).astype(np.float32)
            mat_b = np.random.rand(64, 128).astype(np.float32)

            # Define a callback function
            task_results = {}
            task_event = threading.Event()
            def mat_mul_callback(result, error):
                if error:
                    print(f"Callback received error: {error}")
                    task_results['error'] = error
                else:
                    print(f"Callback received result matrix of shape: {result.shape}")
                    task_results['result'] = result
                task_event.set() # Signal completion

            if manager.add_task('matrix_mul', {'A': mat_a, 'B': mat_b}, callback=mat_mul_callback):
                print("Waiting for task completion via callback...")
                task_event.wait(timeout=10) # Wait for callback signal
                if 'result' in task_results:
                     print("Task completed successfully.")
                     # print("Result (first few elements):\n", task_results['result'][:3,:3])
                elif 'error' in task_results:
                     print(f"Task failed with error: {task_results['error']}")
                else:
                     print("Task timed out or callback failed.")
            else:
                print("Failed to add task to queue.")

        elif manager.selected_device_source == 'cuda':
             console.print("\nRunning on CUDA. Tasks should be processed directly using PyTorch on device:", manager.torch_device)
             # Example direct CUDA operation
             try:
                 t1 = torch.randn(5,5, device=manager.torch_device)
                 t2 = torch.randn(5,5, device=manager.torch_device)
                 t3 = t1 @ t2
                 print("Performed simple matrix multiplication directly on CUDA device.")
             except Exception as e:
                 print(f"Error during direct CUDA operation: {e}")

        # Cleanup
        console.print("\nCleaning up manager...")
        manager.cleanup()
        console.print("Cleanup finished.")

    else:
        console.print("\n[bold red]Initialization Failed.[/bold red]")

    console.print("\n[bold blue]Example finished.[/bold blue]")

# --- END OF FILE gpu_manager.py ---