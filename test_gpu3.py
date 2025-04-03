import pyopencl as cl
import numpy as np
import time
import sys

# --- Helper function to select platform/device (same as before) ---
def get_clover_turks_context():
    target_platform_name = "Clover"
    target_device_name = "AMD TURKS"
    platform = None
    device = None

    print("Looking for OpenCL platforms and devices...")
    try:
        platforms = cl.get_platforms()
        if not platforms:
            print("ERROR: No OpenCL platforms found!")
            return None

        for p in platforms:
            p_name = p.get_info(cl.platform_info.NAME).strip()
            # print(f"Found platform: '{p_name}'") # Debug
            if target_platform_name in p_name:
                platform = p
                try:
                    gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                    if not gpu_devices: continue

                    for d in gpu_devices:
                        d_name = d.get_info(cl.device_info.NAME).strip()
                        # print(f"  Found device: '{d_name}'") # Debug
                        if target_device_name in d_name:
                            device = d
                            print(f"\nUsing Platform: {p_name}")
                            print(f"Using Device: {d_name}\n")
                            break
                    if device: break
                except cl.Error as e:
                     print(f"  -> CL Error getting devices for '{p_name}': {e}")
                except Exception as e:
                     print(f"  -> Error getting devices for '{p_name}': {e}")


        if not device:
            print(f"ERROR: Target device '{target_device_name}' on platform '{target_platform_name}' not found.")
            print("Check 'clinfo' output again for the exact device name.")
            return None

        # --- Add check for required CL version if needed ---
        # cl_version_str = device.get_info(cl.device_info.VERSION) # e.g., "OpenCL 1.1 Mesa 25.0.2-1"
        # print(f"Device OpenCL Version: {cl_version_str}")
        # if "OpenCL 1.0" in cl_version_str:
        #    print("Warning: Only OpenCL 1.0 detected, might lack features.")
        # ----

        return cl.Context([device])

    except cl.Error as e:
        print(f"OpenCL setup error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during setup: {e}")
        return None

# --- Main Test Logic ---
MATRIX_SIZE = 512 # Keep it smaller initially to increase chance of success
                  # Increase later (e.g., 1024) if 512 works

# Kernel source for standard matrix multiplication C = A * B
# Each work item computes one element of C
kernel_source_template = """
__kernel void matrix_mul(__global const float *A,
                         __global const float *B,
                         __global float *C,
                         const int matrix_dim) // Width of A, Height/Width of B/C
{
    int row = get_global_id(0); // Row index of C
    int col = get_global_id(1); // Col index of C

    // Check bounds (optional but good practice)
    if (row >= matrix_dim || col >= matrix_dim) {{
        return;
    }}

    float sum = 0.0f;
    // Calculate dot product of row 'row' from A and col 'col' from B
    for (int k = 0; k < matrix_dim; ++k) {{
        // A[row, k] * B[k, col]
        sum += A[row * matrix_dim + k] * B[k * matrix_dim + col];
    }}

    // Write result C[row, col]
    C[row * matrix_dim + col] = sum;
}
"""
# Use the template (no string replacement needed here as matrix_dim is passed)
kernel_source = kernel_source_template

# Get OpenCL context
context = get_clover_turks_context()
if not context:
    sys.exit(1)

device = context.devices[0]
queue = cl.CommandQueue(context)

# --- Prepare Data ---
print(f"Preparing data (matrix size: {MATRIX_SIZE}x{MATRIX_SIZE})...")
# Use float32
A_host = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
B_host = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
C_host = np.empty((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)

# --- Create Device Buffers ---
mf = cl.mem_flags
try:
    print("Creating device buffers...")
    A_device = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_host)
    B_device = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_host)
    C_device = cl.Buffer(context, mf.WRITE_ONLY, C_host.nbytes)
except cl.Error as e:
    print(f"ERROR creating buffers: {e}")
    sys.exit(1)

# --- Compile Kernel ---
print("Compiling kernel...")
try:
    program = cl.Program(context, kernel_source).build() # Options e.g. '-Werror'
    matrix_mul_kernel = program.matrix_mul
    print("Kernel compiled successfully.")
except cl.RuntimeError as e:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("ERROR: Kernel compilation failed!")
    try:
        print("Build Log:")
        print(program.get_build_info(device, cl.program_build_info.LOG))
    except Exception:
         print("(Could not retrieve build log)")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    sys.exit(1)
except cl.Error as e:
    print(f"ERROR during kernel compilation setup: {e}")
    sys.exit(1)

# --- Execute Kernel ---
print("Executing kernel...")
# Define global size (equal to the dimensions of C)
global_size = (MATRIX_SIZE, MATRIX_SIZE)
# Local size can be None (let driver choose) or a tuple e.g., (16, 16)
local_size = None # Start with None for compatibility

# Set kernel arguments (match kernel signature)
# Pass matrix_dim as int32
matrix_dim_arg = np.int32(MATRIX_SIZE)
kernel_args = (A_device, B_device, C_device, matrix_dim_arg)
matrix_mul_kernel.set_args(*kernel_args)

try:
    # Warm-up runs (optional, but good for timing)
    print("Warm-up runs...")
    for _ in range(5):
         warmup_event = cl.enqueue_nd_range_kernel(queue, matrix_mul_kernel, global_size, local_size)
         warmup_event.wait()

    # Timed execution
    print("Timed execution...")
    start_time = time.time()
    num_runs = 20 # Average over a few runs
    exec_events = []
    for _ in range(num_runs):
        exec_events.append(cl.enqueue_nd_range_kernel(queue, matrix_mul_kernel, global_size, local_size))

    # Wait for the last event to ensure all are finished
    if exec_events:
        exec_events[-1].wait()

    end_time = time.time()
    gpu_time = (end_time - start_time) / num_runs # Average time per run

    print(f"Kernel execution finished.")
    print(f"Average GPU Kernel Time: {gpu_time:.6f} seconds.")


    # --- Read Result Back ---
    print("Reading result back from device...")
    read_event = cl.enqueue_copy(queue, C_host, C_device)
    read_event.wait() # Wait for copy to finish
    print("Result copied back.")

    # --- Verify Result ---
    print("Verifying result (this might take a moment)...")
    start_verify_time = time.time()
    # Calculate expected result using NumPy on the CPU
    expected_result = np.matmul(A_host, B_host)
    verify_time = time.time() - start_verify_time
    print(f"(CPU verification took {verify_time:.4f} seconds)")

    # Compare using np.allclose for floating point tolerance
    tolerance = 1e-4 # May need adjustment for float32 matmul
    if np.allclose(C_host, expected_result, atol=tolerance, rtol=tolerance):
        print("\nResult verification PASSED!")
    else:
        # More detailed failure info
        abs_diff = np.abs(C_host - expected_result)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        num_diff = np.sum(~np.isclose(C_host, expected_result, atol=tolerance, rtol=tolerance))
        print(f"\nResult verification FAILED! ({num_diff} elements differ)")
        print(f"  Max absolute difference: {max_abs_diff:.4E}")
        print(f"  Mean absolute difference: {mean_abs_diff:.4E}")

except cl.Error as e:
    print(f"\nAn OpenCL error occurred during execution/copy: {e.what()} (Code: {e.code})")
    # Common errors: CL_INVALID_WORK_GROUP_SIZE, CL_MEM_OBJECT_ALLOCATION_FAILURE, CL_OUT_OF_RESOURCES, or crashes
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
finally:
    print("Script finished.")