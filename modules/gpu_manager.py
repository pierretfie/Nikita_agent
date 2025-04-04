import pyopencl as cl
import numpy as np
import time
import sys
from rich.console import Console
import psutil
import platform
import subprocess
import os
import threading
import queue
import torch
import gc
import contextlib

console = Console()

class GPUManager:
    def __init__(self):
        self.context = None
        self.device = None
        self.queue = None
        self.platform = None
        self.initialized = False
        self.available_devices = []
        self.selected_device_index = 0
        self.gpu_type = None  # 'cuda', 'amd', 'intel', etc.
        self.llama_compatible = False  # Whether the GPU is compatible with Llama
        self.llama_layers_assigned = 0  # Number of layers assigned to this GPU
        self.work_queue = queue.Queue()  # Queue for workload distribution
        self.worker_thread = None  # Thread for processing workload
        self.running = False  # Flag to control worker thread
        self.opencl_available = True  # Flag to indicate if OpenCL is available
        self.cuda_device_info = None  # Store CUDA device info if OpenCL fails but CUDA is available
        self._suppress_output = False
        
        # Kernel source for GPU-only operations
        self.kernel_source = """
        __kernel void matrix_mul(__global const float *A,
                                __global const float *B,
                                __global float *C,
                                const int matrix_dim)
        {
            int row = get_global_id(0);
            int col = get_global_id(1);

            if (row >= matrix_dim || col >= matrix_dim) {
                return;
            }

            float sum = 0.0f;
            for (int k = 0; k < matrix_dim; ++k) {
                sum += A[row * matrix_dim + k] * B[k * matrix_dim + col];
            }
            C[row * matrix_dim + col] = sum;
        }
        """

    def _log(self, message, level="info"):
        """Controlled logging method that respects output suppression"""
        if self._suppress_output:
            return
            
        if level == "info":
            console.print(f"[green]{message}[/green]")
        elif level == "warning":
            console.print(f"[yellow]{message}[/yellow]")
        elif level == "error":
            console.print(f"[red]{message}[/red]")

    def set_suppress_output(self, suppress=True):
        """Set whether to suppress output"""
        self._suppress_output = suppress

    def _detect_gpu_type(self):
        """Detect the type of GPU available in the system"""
        try:
            # First check if CUDA is available through PyTorch
            if torch.cuda.is_available():
                return 'cuda'
                
            # Check for NVIDIA GPU
            if os.path.exists('/proc/driver/nvidia'):
                return 'cuda'
            
            # Check for AMD GPU
            if os.path.exists('/sys/class/drm/card0/device/vendor') and os.path.exists('/sys/class/drm/card0/device/uevent'):
                with open('/sys/class/drm/card0/device/vendor', 'r') as f:
                    vendor = f.read().strip()
                if vendor == '0x1002':  # AMD vendor ID
                    return 'amd'
            
            # Check for Intel GPU
            if os.path.exists('/sys/class/drm/card0/device/vendor') and os.path.exists('/sys/class/drm/card0/device/uevent'):
                with open('/sys/class/drm/card0/device/vendor', 'r') as f:
                    vendor = f.read().strip()
                if vendor == '0x8086':  # Intel vendor ID
                    return 'intel'
            
            # Try to detect using lspci
            try:
                result = subprocess.run(['lspci', '-v'], capture_output=True, text=True)
                if 'NVIDIA' in result.stdout:
                    return 'cuda'
                elif 'AMD' in result.stdout and 'VGA' in result.stdout:
                    return 'amd'
                elif 'Intel' in result.stdout and 'VGA' in result.stdout:
                    return 'intel'
            except:
                pass
                
            return 'unknown'
        except Exception as e:
            self._log(f"Warning: Could not detect GPU type: {e}", "warning")
            return 'unknown'

    def _check_llama_compatibility(self):
        """Check if the GPU is compatible with Llama"""
        try:
            # Check if PyTorch is available and CUDA is supported
            if not torch.cuda.is_available():
                self._log("PyTorch CUDA is not available. Llama may not use this GPU.", "warning")
                return False
                
            # Check GPU memory
            if self.gpu_type == 'cuda':
                # For NVIDIA GPUs, check CUDA memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
                if gpu_memory < 4:  # Llama typically needs at least 4GB
                    self._log(f"GPU memory ({gpu_memory:.1f}GB) may be insufficient for Llama.", "warning")
                    return False
                    
            # Check OpenCL capabilities
            if self.device:
                # Check if the device supports double precision
                double_support = self.device.get_info(cl.device_info.DOUBLE_FP_CONFIG)
                if double_support == 0:
                    self._log("GPU does not support double precision. Llama may not use this GPU.", "warning")
                    return False
                    
                # Check compute capability
                compute_units = self.device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
                if compute_units < 8:  # Arbitrary threshold
                    self._log(f"GPU compute capability ({compute_units} units) may be insufficient for Llama.", "warning")
                    return False
            
            return True
        except Exception as e:
            self._log(f"Warning: Could not check Llama compatibility: {e}", "warning")
            return False

    def _get_gpu_utilization(self):
        """Get GPU utilization percentage"""
        try:
            if self.device:
                mem_info = self.device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
                mem_used = self.device.get_info(cl.device_info.GLOBAL_MEM_SIZE) - self.device.get_info(cl.device_info.GLOBAL_MEM_CACHE_SIZE)
                return (mem_used / mem_info) * 100
            return 0
        except Exception as e:
            self._log(f"Warning: Could not get GPU utilization: {e}", "warning")
            return 0

    def _get_cuda_device_info(self):
        """Get CUDA device info using PyTorch when OpenCL is not available"""
        try:
            if not torch.cuda.is_available():
                return None
                
            device_count = torch.cuda.device_count()
            if device_count == 0:
                return None
                
            # Get properties of the first CUDA device
            device_props = torch.cuda.get_device_properties(0)
            
            # Get memory usage for utilization calculation
            try:
                # Try to get memory utilization
                mem_allocated = torch.cuda.memory_allocated(0)
                mem_reserved = torch.cuda.memory_reserved(0)
                utilization = 100.0 * mem_allocated / device_props.total_memory
            except:
                # Fallback if memory stats fail
                utilization = 0.0
            
            # Calculate appropriate layer assignment based on GPU memory
            # T4 has 16GB VRAM, determine the proper number of layers
            gpu_memory_gb = device_props.total_memory / (1024**3)
            
            # Determine Llama layers assignment based on GPU memory
            
            if gpu_memory_gb >= 24:
                # For high-end GPUs with lots of VRAM (A100, etc)
                llama_layers = -1  # Use all layers
            elif gpu_memory_gb >= 12:
                # For GPUs with good VRAM (RTX 3080, etc)
                llama_layers = -1  # Use all layers
            elif gpu_memory_gb >= 8:
                # For GPUs with decent VRAM
                llama_layers = 32
            elif gpu_memory_gb >= 4:
                # For GPUs with limited VRAM
                llama_layers = 24
            else:
                # For GPUs with very limited VRAM
                llama_layers = 16
            
            # Format the device info similar to OpenCL format
            device_info = {
                'name': device_props.name,
                'vendor': 'NVIDIA',
                'version': f"CUDA {torch.version.cuda}",
                'driver_version': torch.version.cuda,
                'max_compute_units': device_props.multi_processor_count,
                'global_mem_size': device_props.total_memory,
                'max_work_group_size': 1024,  # Default for most CUDA devices
                'gpu_type': 'cuda',
                'llama_compatible': True,
                'llama_layers_assigned': llama_layers,
                'utilization': utilization
            }
            
            return device_info
        except Exception as e:
            self._log(f"Warning: Could not get CUDA device info: {e}", "warning")
            return None

    def _get_available_devices(self):
        """Scan for available GPU devices"""
        self.available_devices = []
        try:
            platforms = cl.get_platforms()
            for platform in platforms:
                # Get all GPU devices
                devices = platform.get_devices(device_type=cl.device_info.GPU)
                for device in devices:
                    device_info = {
                        'device': device,
                        'platform': platform,
                        'name': device.get_info(cl.device_info.NAME),
                        'vendor': device.get_info(cl.device_info.VENDOR),
                        'utilization': self._get_gpu_utilization(),
                        'available': device.get_info(cl.device_info.AVAILABLE),
                        'max_work_group_size': device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE),
                        'max_compute_units': device.get_info(cl.device_info.MAX_COMPUTE_UNITS),
                        'global_mem_size': device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
                    }
                    self.available_devices.append(device_info)
        except Exception:
            # Silently handle OpenCL errors
            self.opencl_available = False
            
            # Try to get CUDA device info if OpenCL fails
            cuda_info = self._get_cuda_device_info()
            if cuda_info:
                self.cuda_device_info = cuda_info

    def initialize(self, device_index=None, preferred_gpu=None):
        """Initialize GPU with support for multiple GPU types"""
        try:
            # Detect GPU type
            self.gpu_type = self._detect_gpu_type()
            
            # Get available devices
            self._get_available_devices()
            
            # Special handling for CUDA when OpenCL is not available
            if not self.opencl_available and self.gpu_type == 'cuda' and self.cuda_device_info:
                self.llama_compatible = True
                # Use the layers assigned in _get_cuda_device_info
                self.llama_layers_assigned = self.cuda_device_info['llama_layers_assigned']
                self.initialized = True
                return True
            
            if not self.available_devices:
                # Check if we have CUDA but OpenCL failed
                if self.gpu_type == 'cuda' and torch.cuda.is_available():
                    self.llama_compatible = True
                    
                    # Get the T4 info if available
                    cuda_info = self._get_cuda_device_info()
                    if cuda_info:
                        self.cuda_device_info = cuda_info
                        self.llama_layers_assigned = cuda_info['llama_layers_assigned']
                    else:
                        # Default if no detailed info available
                        self.llama_layers_assigned = -1  # Use all layers as default
                        
                    self.initialized = True
                    return True
                    
                return False

            # Select device based on preference or use the first available
            selected_device = None
            
            if preferred_gpu:
                # Try to find the preferred GPU type
                for device_info in self.available_devices:
                    if preferred_gpu.lower() in device_info['name'].lower() or preferred_gpu.lower() in device_info['vendor'].lower():
                        selected_device = device_info
                        break
            
            # If no preferred device found or no preference specified, use the first available
            if not selected_device and self.available_devices:
                selected_device = self.available_devices[0]
            
            if not selected_device:
                return False

            self.device = selected_device['device']
            self.platform = selected_device['platform']

            # Create context with appropriate optimizations based on GPU type
            self.context = cl.Context([self.device])
            
            # Create command queue with profiling enabled
            self.queue = cl.CommandQueue(
                self.context,
                properties=cl.command_queue_properties.PROFILING_ENABLE
            )

            # Test GPU with matrix multiplication
            if not self._test_gpu():
                return False
                
            # Check Llama compatibility
            self.llama_compatible = self._check_llama_compatibility()
            if not self.llama_compatible:
                # Start the worker thread for workload splitting
                self._start_worker_thread()

            self.initialized = True
            return True

        except Exception:
            # Silently handle initialization errors
            return False

    def _test_gpu(self):
        """Test GPU with matrix multiplication optimized for the detected GPU type"""
        try:
            # Adjust matrix size based on GPU type and available memory
            if self.gpu_type == 'cuda':
                console.print('CUDAAAAAAA')
                # NVIDIA GPUs typically have more VRAM
                matrix_size = 128
            elif self.gpu_type == 'amd':
                # AMD GPUs vary in VRAM, use a moderate size
                matrix_size = 64
            else:
                # Default to a smaller size for other GPUs
                matrix_size = 32
                
            # Adjust based on available memory
            if self.device:
                mem_size = self.device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
                # Use at most 10% of available memory for the test
                max_matrix_size = int(np.sqrt(mem_size * 0.1 / (3 * 4)))  # 3 matrices, 4 bytes per float
                matrix_size = min(matrix_size, max_matrix_size)
            
            A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
            B = np.random.rand(matrix_size, matrix_size).astype(np.float32)
            C = np.zeros((matrix_size, matrix_size), dtype=np.float32)

            # Create buffers
            mf = cl.mem_flags
            A_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
            B_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
            C_buf = cl.Buffer(self.context, mf.WRITE_ONLY, C.nbytes)

            # Build program
            program = cl.Program(self.context, self.kernel_source).build()
            kernel = program.matrix_mul

            # Set kernel arguments
            kernel.set_args(A_buf, B_buf, C_buf, np.int32(matrix_size))

            # Get device's maximum work group size
            max_work_group_size = self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            # Optimize work group size based on GPU type
            if self.gpu_type == 'cuda':
                # NVIDIA GPUs often work well with larger work groups
                work_group_size = min(16, max_work_group_size)
            elif self.gpu_type == 'amd':
                # AMD GPUs often work well with smaller work groups
                work_group_size = min(8, max_work_group_size)
            else:
                # Default to a conservative size
                work_group_size = min(4, max_work_group_size)
                
            global_size = (matrix_size, matrix_size)
            local_size = (work_group_size, work_group_size)

            # Execute kernel
            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
            self.queue.finish()

            # Read back results
            cl.enqueue_copy(self.queue, C, C_buf)
            self.queue.finish()

            # Verify results
            expected = np.matmul(A, B)
            if not np.allclose(C, expected, atol=1e-2):
                self._log("Matrix multiplication test failed!")
                return False

            return True

        except Exception as e:
            self._log(f"Error during GPU test: {e}", "error")
            return False

    def _start_worker_thread(self):
        """Start the worker thread for processing workload"""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self._log("Started worker thread for workload processing")
        
    def _worker_loop(self):
        """Worker thread loop for processing workload"""
        while self.running:
            try:
                # Get a task from the queue with a timeout
                task = self.work_queue.get(timeout=1.0)
                if task is None:
                    continue
                    
                # Process the task
                self._process_task(task)
                
                # Mark the task as done
                self.work_queue.task_done()
            except queue.Empty:
                # No tasks available, continue waiting
                continue
            except Exception as e:
                self._log(f"Error in worker thread: {e}", "error")
                
    def _process_task(self, task):
        """Process a workload task"""
        try:
            task_type = task.get('type')
            data = task.get('data')
            
            if task_type == 'matrix_mul':
                # Process matrix multiplication
                A = data.get('A')
                B = data.get('B')
                if A is not None and B is not None:
                    result = self._matrix_mul(A, B)
                    # Store the result or send it back
                    if 'callback' in task and callable(task['callback']):
                        task['callback'](result)
            elif task_type == 'tensor_ops':
                # Process tensor operations
                tensor = data.get('tensor')
                operation = data.get('operation')
                if tensor is not None and operation is not None:
                    result = self._process_tensor(tensor, operation)
                    if 'callback' in task and callable(task['callback']):
                        task['callback'](result)
            else:
                self._log(f"Unknown task type: {task_type}", "warning")
        except Exception as e:
            self._log(f"Error processing task: {e}", "error")
            
    def _matrix_mul(self, A, B):
        """Perform matrix multiplication on the GPU"""
        try:
            # Ensure matrices are float32
            A = np.asarray(A, dtype=np.float32)
            B = np.asarray(B, dtype=np.float32)
            
            # Check dimensions
            if A.shape[1] != B.shape[0]:
                raise ValueError(f"Matrix dimensions mismatch: A({A.shape}) and B({B.shape})")
                
            # Create result matrix
            C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
            
            # Create buffers
            mf = cl.mem_flags
            A_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
            B_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
            C_buf = cl.Buffer(self.context, mf.WRITE_ONLY, C.nbytes)
            
            # Build program
            program = cl.Program(self.context, self.kernel_source).build()
            kernel = program.matrix_mul
            
            # Set kernel arguments
            kernel.set_args(A_buf, B_buf, C_buf, np.int32(A.shape[1]))
            
            # Get device's maximum work group size
            max_work_group_size = self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            
            # Optimize work group size based on GPU type
            if self.gpu_type == 'cuda':
                work_group_size = min(16, max_work_group_size)
            elif self.gpu_type == 'amd':
                work_group_size = min(8, max_work_group_size)
            else:
                work_group_size = min(4, max_work_group_size)
                
            global_size = (A.shape[0], B.shape[1])
            local_size = (min(work_group_size, A.shape[0]), min(work_group_size, B.shape[1]))
            
            # Execute kernel
            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
            self.queue.finish()
            
            # Read back results
            cl.enqueue_copy(self.queue, C, C_buf)
            self.queue.finish()
            
            return C
        except Exception as e:
            self._log(f"Error in matrix multiplication: {e}", "error")
            return None
            
    def _process_tensor(self, tensor, operation):
        """Process tensor operations on the GPU"""
        try:
            # Convert tensor to numpy array if it's a PyTorch tensor
            if torch.is_tensor(tensor):
                tensor = tensor.detach().cpu().numpy()
                
            # Process based on operation
            if operation == 'transpose':
                result = np.transpose(tensor)
            elif operation == 'sum':
                result = np.sum(tensor)
            elif operation == 'mean':
                result = np.mean(tensor)
            else:
                self._log(f"Unsupported tensor operation: {operation}", "warning")
                return None
                
            return result
        except Exception as e:
            self._log(f"Error in tensor operation: {e}", "error")
            return None
            
    def add_task(self, task_type, data, callback=None):
        """Add a task to the workload queue"""
        if not self.initialized:
            self._log("GPU manager not initialized!", "warning")
            return False
            
        if not self.llama_compatible:
            # Add task to the queue
            task = {
                'type': task_type,
                'data': data,
                'callback': callback
            }
            self.work_queue.put(task)
            return True
        else:
            self._log("GPU is Llama-compatible, tasks will be processed by Llama directly.")
            return False
            
    def get_device_info(self):
        """Get information about the current GPU device"""
        if not self.device and not self.cuda_device_info:
            return None
            
        try:
            # If using CUDA directly (OpenCL failed or not available)
            if self.cuda_device_info:
                return self.cuda_device_info
                
            # If using OpenCL
            return {
                'name': self.device.get_info(cl.device_info.NAME),
                'vendor': self.device.get_info(cl.device_info.VENDOR),
                'version': self.device.get_info(cl.device_info.VERSION),
                'driver_version': self.device.get_info(cl.device_info.DRIVER_VERSION),
                'max_compute_units': self.device.get_info(cl.device_info.MAX_COMPUTE_UNITS),
                'global_mem_size': self.device.get_info(cl.device_info.GLOBAL_MEM_SIZE),
                'max_work_group_size': self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE),
                'gpu_type': self.gpu_type,
                'llama_compatible': self.llama_compatible,
                'llama_layers_assigned': self.llama_layers_assigned,
                'utilization': self._get_gpu_utilization()
            }
        except Exception as e:
            self._log(f"Error getting device info: {e}", "error")
            return None

    def cleanup(self):
        """Clean up GPU resources"""
        try:
            # Stop the worker thread
            self.running = False
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2.0)
                
            # Clear the queue
            while not self.work_queue.empty():
                try:
                    self.work_queue.get_nowait()
                    self.work_queue.task_done()
                except queue.Empty:
                    break
                    
            # Clean up GPU resources - only finish the queue
            if self.queue:
                self.queue.finish()
            # if self.context:  # Context release is likely handled by GC
            #     self.context.release()
            self.initialized = False
            return True
        except Exception as e:
            # Use print instead of console.print to avoid issues during shutdown
            print(f"Error during cleanup: {e}")
            return False

    def is_initialized(self):
        """Check if GPU is initialized"""
        return self.initialized

    def __del__(self):
        """Cleanup on deletion"""
        # Use a safer cleanup approach during shutdown
        try:
            # Only explicitly finish the queue if it exists
            if hasattr(self, 'queue') and self.queue:
                self.queue.finish()
            # Context release is likely handled by GC
            # if hasattr(self, 'context') and self.context:
            #     self.context.release()
        except:
            # Silently fail during shutdown
            pass 