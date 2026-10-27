"""What the hardware is doing, for the Queue's hardware cards and the sidebar: each NVIDIA GPU's use, memory and
temperature (nvidia-smi, on Linux and Windows), the CPU's use and temperature, and system RAM.

Linux reports the CPU's temperature through its sensor drivers (coretemp for Intel, k10temp or zenpower for
AMD). Windows doesn't without admin rights or a tool like LibreHardwareMonitor, so there it's None."""

from __future__ import annotations

import functools
import re
import subprocess
import threading
import time

import psutil

from .procs import WINDOWS

GPU_TTL = 1.0  # seconds a GPU reading is reused: the sidebar and the Queue both ask
_gpu_lock = threading.Lock()
_gpu_cache: tuple[float, list[dict]] = (0.0, [])

psutil.cpu_percent(None)  # the first reading starts the count: the next one is the use since then


def gpus() -> list[dict]:
    """[{index, name, mem_used, mem_total (MiB), util (%), temp (°C)}] for each NVIDIA GPU; [] without any. A value
    nvidia-smi doesn't report ("[N/A]": a GPU without a sensor, or one sharing system memory) is None."""
    global _gpu_cache
    with _gpu_lock:
        at, cached = _gpu_cache
        if time.monotonic() - at < GPU_TTL:
            return cached
        found = []
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
                                  "temperature.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL,
                                 creationflags=subprocess.CREATE_NO_WINDOW if WINDOWS else 0).stdout
            for line in out.strip().splitlines():
                fields = [x.strip() for x in line.split(",")]
                if len(fields) != 6 or not fields[0].isdigit():
                    continue  # not a GPU's line (a warning nvidia-smi printed)
                i, name, used, total, util, temp = fields
                found.append({"index": int(i), "name": name, "mem_used": _number(used), "mem_total": _number(total),
                              "util": _number(util), "temp": _number(temp)})
        except (OSError, subprocess.SubprocessError):
            found = []
        _gpu_cache = (time.monotonic(), found)
        return found


def _number(v: str) -> int | None:
    try:
        return int(float(v))
    except ValueError:
        return None


def short_cpu_name(name: str) -> str:
    """The CPU's name without the trademarks, generation and speed: "12th Gen Intel(R) Core(TM) i5-12600KF" is
    "Intel Core i5-12600KF", "AMD Ryzen 9 7950X 16-Core Processor" is "AMD Ryzen 9 7950X"."""
    name = re.sub(r"\((R|TM|tm|r)\)|\b\d+(st|nd|rd|th) Gen\b|\bCPU\b|@.*$|\b\d+-Cores?\b|\bProcessor\b", " ", name)
    return " ".join(name.split())


@functools.cache
def cpu_name() -> str:
    try:
        if WINDOWS:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return short_cpu_name(winreg.QueryValueEx(k, "ProcessorNameString")[0])
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith(("model name", "Model")):  # x86; ARM boards name the board
                    return short_cpu_name(line.split(":", 1)[1])
    except (OSError, ImportError, IndexError):
        pass
    import platform

    return short_cpu_name(platform.processor()) or "CPU"


def cpu_temp(sensors: dict | None = None) -> float | None:
    """The CPU package's temperature (°C) from Linux's sensor drivers (``sensors`` as psutil reports them), or
    None when there's none: on Windows, or a machine whose drivers don't report it. A motherboard's thermal zone
    (acpitz) isn't the CPU, so it isn't used."""
    if sensors is None:
        if not hasattr(psutil, "sensors_temperatures"):
            return None
        try:
            sensors = psutil.sensors_temperatures()
        except (OSError, RuntimeError):
            return None

    def first(driver: str, *labels: str) -> float | None:
        for label in labels:
            values = [s.current for s in sensors.get(driver, []) if s.label.startswith(label) and s.current]
            if values:
                return max(values)  # a dual-socket machine: the hotter package
        return None

    for driver, labels in (("coretemp", ("Package id",)), ("k10temp", ("Tdie", "Tctl")), ("zenpower", ("Tdie", "Tctl")),
                           ("cpu_thermal", ("",)), ("soc_thermal", ("",))):
        t = first(driver, *labels)
        if t is not None:
            return round(t, 1)
    return None


def cpu() -> dict:
    """{name, percent (use since the last reading), temp (°C or None), cores, threads}."""
    return {"name": cpu_name(), "percent": psutil.cpu_percent(None), "temp": cpu_temp(),
            "cores": psutil.cpu_count(logical=False) or 0, "threads": psutil.cpu_count() or 0}


def ram() -> dict:
    """{total_gb, available_gb, used_gb}: in use is what isn't available (the page cache can be freed)."""
    vm = psutil.virtual_memory()
    gb = 1024**3
    return {"total_gb": vm.total / gb, "available_gb": vm.available / gb, "used_gb": (vm.total - vm.available) / gb}
