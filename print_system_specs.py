#!/usr/bin/env python3
"""
print_system_specs.py
A single-file script that prints hardware & software system specs.

Usage:
    python print_system_specs.py
"""

import os
import sys
import platform
import shutil
import subprocess
from typing import Any, Dict, Optional


# ------------------------- helpers ------------------------- #
def _safe_run(cmd, timeout: float = 5.0):
    """Run a command safely with a timeout; return (code, stdout, stderr)."""
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
        return p.returncode, (out or "").strip(), (err or "").strip()
    except Exception as e:
        return -1, "", str(e)


def _import_or_none(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def _bytes_to_gb(n: Optional[int]) -> Optional[float]:
    if n is None:
        return None
    return round(n / (1024**3), 2)


# ------------------------- collectors ------------------------- #
def get_os_info() -> Dict[str, Any]:
    u = platform.uname()
    return {
        "system": u.system,
        "node": u.node,
        "release": u.release,
        "version": u.version,
        "machine": u.machine,
        "platform": platform.platform(),
    }


def get_cpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["architecture"] = platform.machine()
    info["processor"] = platform.processor() or platform.uname().processor
    try:
        import multiprocessing as mp
        info["logical_cores"] = mp.cpu_count()
    except Exception:
        info["logical_cores"] = None

    # Optional: py-cpuinfo for richer details
    cpuinfo = _import_or_none("cpuinfo")
    if cpuinfo:
        try:
            raw = cpuinfo.get_cpu_info()
            info["brand_raw"] = raw.get("brand_raw")
            info["hz_advertised"] = raw.get("hz_advertised_friendly")
            info["l2_cache_size"] = raw.get("l2_cache_size")
            info["l3_cache_size"] = raw.get("l3_cache_size")
        except Exception:
            pass
    return info


def get_ram_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    # Preferred: psutil (if available)
    psutil = _import_or_none("psutil")
    if psutil:
        try:
            v = psutil.virtual_memory()
            info["total_bytes"] = int(v.total)
            info["available_bytes"] = int(v.available)
            return info
        except Exception:
            pass

    # Fallback (Linux): parse /proc/meminfo
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.lower().startswith("memtotal"):
                        kb = int(line.split(":")[1].strip().split()[0])
                        info["total_bytes"] = kb * 1024
                    if line.lower().startswith("memavailable"):
                        kb = int(line.split(":")[1].strip().split()[0])
                        info["available_bytes"] = kb * 1024
        except Exception:
            pass

    return info


def get_disk_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        usage = shutil.disk_usage(os.getcwd())
        info["total_bytes"] = int(usage.total)
        info["used_bytes"] = int(usage.used)
        info["free_bytes"] = int(usage.free)
    except Exception:
        pass
    return info


def get_python_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["version"] = platform.python_version()
    info["executable"] = sys.executable
    info["venv"] = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")

    # label -> import name
    pkg_map = {
        "pip": "pip",
        "setuptools": "setuptools",
        "wheel": "wheel",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "transformers": "transformers",
        "sentence_transformers": "sentence_transformers",
        "torch": "torch",
        "faiss": "faiss",  # faiss-cpu/faiss-gpu usually import as 'faiss'
        "tensorflow": "tensorflow",
    }
    pkgs: Dict[str, Optional[str]] = {}
    for label, mod_name in pkg_map.items():
        mod = _import_or_none(mod_name)
        if mod:
            pkgs[label] = getattr(mod, "__version__", "unknown")
    info["packages"] = pkgs
    return info


def get_gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"has_cuda": False}

    # Torch (if present)
    torch = _import_or_none("torch")
    if torch:
        try:
            info["torch_cuda_available"] = bool(torch.cuda.is_available())
            info["has_cuda"] = info["torch_cuda_available"]
            if info["torch_cuda_available"]:
                info["cuda_device_count"] = torch.cuda.device_count()
                devices = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    devices.append({
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_memory_bytes": getattr(props, "total_memory", None),
                        "capability": f"{getattr(props, 'major', '?')}.{getattr(props, 'minor', '?')}",
                    })
                info["devices"] = devices
        except Exception:
            pass

    # nvidia-smi (if present)
    code, out, _ = _safe_run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,cuda_version",
        "--format=csv,noheader,nounits"
    ])
    if code == 0 and out:
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        gpus = []
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            g = {"index": i}
            if len(parts) > 0: g["name"] = parts[0]
            if len(parts) > 1:
                try:
                    g["total_memory_mb"] = int(parts[1])
                except Exception:
                    g["total_memory_mb"] = parts[1]
            if len(parts) > 2: g["driver_version"] = parts[2]
            if len(parts) > 3: g["cuda_version"] = parts[3]
            gpus.append(g)
        info["nvidia_smi"] = gpus
        info["has_cuda"] = True if gpus else info.get("has_cuda", False)

    return info


def collect_specs() -> Dict[str, Any]:
    return {
        "os": get_os_info(),
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "disk": get_disk_info(),
        "python": get_python_info(),
        "gpu": get_gpu_info(),
    }


def print_report(specs: Dict[str, Any]) -> None:
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    os_info = specs.get("os", {})
    cpu = specs.get("cpu", {})
    ram = specs.get("ram", {})
    disk = specs.get("disk", {})
    py = specs.get("python", {})
    gpu = specs.get("gpu", {})

    def fmt(val, suffix=""):
        return "N/A" if val is None else f"{val}{suffix}"

    print("=== System Specs Report ===")
    print(f"Timestamp (UTC): {ts}")
    print(f"OS: {os_info.get('platform')} (machine={os_info.get('machine')})")
    print(f"CPU: {cpu.get('brand_raw') or cpu.get('processor') or 'unknown'}")
    print(f"  Arch: {cpu.get('architecture')} | Logical cores: {cpu.get('logical_cores')}")
    print(f"RAM: total={fmt(_bytes_to_gb(ram.get('total_bytes')), ' GB')}, "
          f"available={fmt(_bytes_to_gb(ram.get('available_bytes')), ' GB')}")
    print(f"Disk: total={fmt(_bytes_to_gb(disk.get('total_bytes')), ' GB')}, "
          f"used={fmt(_bytes_to_gb(disk.get('used_bytes')), ' GB')}, "
          f"free={fmt(_bytes_to_gb(disk.get('free_bytes')), ' GB')}")
    print(f"Python: {py.get('version')} @ {py.get('executable')}")
    print(f"Virtual env: {py.get('venv') or 'None'}")
    if py.get("packages"):
        print("Key packages:")
        for k, v in py["packages"].items():
            print(f"  - {k}: {v}")
    print("GPU:")
    if gpu.get("has_cuda"):
        if gpu.get("devices"):
            for d in gpu["devices"]:
                print(f"  - Torch device {d['index']}: {d.get('name')} | "
                      f"VRAM={fmt(_bytes_to_gb(d.get('total_memory_bytes')), ' GB')} | "
                      f"CC={d.get('capability')}")
        if gpu.get("nvidia_smi"):
            for gdev in gpu["nvidia_smi"]:
                print(f"  - nvidia-smi {gdev.get('index')}: {gdev.get('name')} | "
                      f"VRAM={gdev.get('total_memory_mb')} MB | "
                      f"driver={gdev.get('driver_version')} | CUDA={gdev.get('cuda_version')}")
    else:
        print("  - No CUDA-capable GPU detected (or drivers not available).")


# ------------------------- main ------------------------- #
if __name__ == "__main__":
    specs = collect_specs()
    print_report(specs)
