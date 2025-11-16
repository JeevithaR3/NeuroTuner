# monitor.py
import psutil
def get_hardware_state():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    # Optional: try to get GPU util via pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
    except Exception:
        util = 0
    return {"cpu_percent": cpu, "mem_percent": mem, "gpu_percent": util}
