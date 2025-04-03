import os
import psutil
cpu_count = os.cpu_count()
if cpu_count > 1:
    # Use 50-75% of available cores based on load
    current_load = psutil.getloadavg()[0] / cpu_count
    print(f"Current load: {current_load}")
    if current_load < 0.8:
        target_cores = min(cpu_count - 1, max(2, int(cpu_count * 0.75)))
    else:
        target_cores = min(cpu_count - 1, max(1, int(cpu_count * 0.5)))
    print(f"Using {target_cores} cores")