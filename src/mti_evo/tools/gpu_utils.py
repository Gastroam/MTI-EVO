
import subprocess
import shutil
import xml.etree.ElementTree as ET

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

def get_gpu_stats() -> dict | None:
    """
    Returns a dictionary of GPU stats (first GPU only for now) or None if not available.
    Tries pynvml first, then falls back to nvidia-smi parsing.
    Keys: 
      - name: str
      - vram_used_mb: float
      - vram_total_mb: float
      - gpu_util_percent: int
    """
    # 1. Try pynvml (fastest, in-process)
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8', errors='ignore')
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                except Exception:
                    gpu_util = 0
                
                return {
                    "name": name,
                    "vram_used_mb": mem_info.used / 1024 / 1024,
                    "vram_total_mb": mem_info.total / 1024 / 1024,
                    "gpu_util_percent": gpu_util
                }
        except Exception:
            pass # Fallthrough to nvidia-smi
        finally:
            try:
                if HAS_PYNVML: pynvml.nvmlShutdown()
            except: pass

    # 2. Fallback to nvidia-smi (slower, subprocess)
    if shutil.which("nvidia-smi"):
        try:
            # Run nvidia-smi -q -x for XML output
            cmd = ["nvidia-smi", "-q", "-x"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                root = ET.fromstring(result.stdout)
                gpu = root.find("gpu")
                if gpu is not None:
                    prod_name = gpu.find("product_name").text
                    
                    fb_memory = gpu.find("fb_memory_usage")
                    used_str = fb_memory.find("used").text # e.g. "100 MiB"
                    total_str = fb_memory.find("total").text
                    
                    util_node = gpu.find("utilization")
                    gpu_util_str = util_node.find("gpu_util").text # e.g. "10 %"

                    def parse_val(s):
                        return float(s.split()[0])

                    return {
                        "name": prod_name,
                        "vram_used_mb": parse_val(used_str),
                        "vram_total_mb": parse_val(total_str),
                        "gpu_util_percent": int(parse_val(gpu_util_str))
                    }
        except Exception:
            pass

    return None

# Alias for backward compatibility
get_vram_info = get_gpu_stats

def check_cuda():
    """Check if CUDA is available (Basic check via nvidia-smi)."""
    return get_gpu_stats() is not None

def optimize_gpu_settings():
    """Apply GPU optimizations (Placeholder)."""
    pass



