"""The Queue's hardware cards (dainamo/server/hardware.py): GPUs from nvidia-smi, the CPU's name and temperature
on Linux and Windows, and the /api/hardware answer.

    .venv/bin/python -m pytest tests/test_hardware.py -v
"""

import subprocess
import sys
import types
from collections import namedtuple

import pytest

from dainamo.server import hardware

Sensor = namedtuple("Sensor", "label current high critical")


@pytest.mark.parametrize("raw, short", [
    ("12th Gen Intel(R) Core(TM) i5-12600KF", "Intel Core i5-12600KF"),
    ("Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz", "Intel Core i9-9900K"),
    ("Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz", "Intel Xeon E5-2680 v4"),
    ("AMD Ryzen 9 7950X 16-Core Processor", "AMD Ryzen 9 7950X"),
    ("AMD Ryzen Threadripper PRO 5995WX 64-Cores", "AMD Ryzen Threadripper PRO 5995WX"),
    ("  Raspberry Pi 4 Model B Rev 1.4 ", "Raspberry Pi 4 Model B Rev 1.4"),
])
def test_cpu_names_lose_trademarks_and_speeds(raw, short):
    assert hardware.short_cpu_name(raw) == short


def test_cpu_temperature_is_the_package_one():
    s = lambda *pairs: [Sensor(label, t, None, None) for label, t in pairs]  # noqa: E731
    # Intel: the package (the hotter of two sockets), not a core or the motherboard
    assert hardware.cpu_temp({"acpitz": s(("", 27.8)), "coretemp": s(("Package id 0", 61.0), ("Core 0", 70.0),
                                                                       ("Package id 1", 64.5))}) == 64.5
    # AMD: Tdie where there is one (Tctl can be offset on older Ryzens), else Tctl
    assert hardware.cpu_temp({"k10temp": s(("Tctl", 75.0), ("Tdie", 65.0), ("Tccd1", 60.0))}) == 65.0
    assert hardware.cpu_temp({"k10temp": s(("Tctl", 48.25), ("Tccd1", 44.0))}) == 48.2
    assert hardware.cpu_temp({"zenpower": s(("Tdie", 55.0))}) == 55.0
    assert hardware.cpu_temp({"cpu_thermal": s(("", 51.6))}) == 51.6  # a Raspberry Pi
    # a motherboard zone or a drive isn't the CPU
    assert hardware.cpu_temp({"acpitz": s(("", 27.8)), "nvme": s(("Composite", 30.0))}) is None
    assert hardware.cpu_temp({}) is None


def test_windows_reports_no_cpu_temperature_and_names_the_cpu_from_the_registry(monkeypatch):
    monkeypatch.delattr(hardware.psutil, "sensors_temperatures", raising=False)  # as psutil is on Windows
    assert hardware.cpu_temp() is None
    class Key:  # the registry key winreg.OpenKey opens
        def __init__(self, root, path):
            assert (root, path) == ("HKLM", r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    winreg = types.SimpleNamespace(HKEY_LOCAL_MACHINE="HKLM", OpenKey=Key,
                                   QueryValueEx=lambda k, name: ("AMD Ryzen 7 5800X3D 8-Core Processor           ", 1))
    monkeypatch.setitem(sys.modules, "winreg", winreg)
    monkeypatch.setattr(hardware, "WINDOWS", True)
    hardware.cpu_name.cache_clear()
    try:
        assert hardware.cpu_name() == "AMD Ryzen 7 5800X3D"
    finally:
        hardware.cpu_name.cache_clear()


def test_gpus_come_from_nvidia_smi_once_a_second(monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="0, NVIDIA GeForce RTX 3090, 596, 24576, 37, 61\n"
                                                         "1, NVIDIA GeForce RTX 3090, 18, 24576, 0, 30\n")
    monkeypatch.setattr(hardware.subprocess, "run", run)
    monkeypatch.setattr(hardware, "_gpu_cache", (0.0, []))
    first = hardware.gpus()
    assert first == [{"index": 0, "name": "NVIDIA GeForce RTX 3090", "mem_used": 596, "mem_total": 24576, "util": 37, "temp": 61},
                     {"index": 1, "name": "NVIDIA GeForce RTX 3090", "mem_used": 18, "mem_total": 24576, "util": 0, "temp": 30}]
    assert hardware.gpus() == first and len(calls) == 1  # the sidebar and the Queue share a reading

    def missing(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(hardware.subprocess, "run", missing)
    monkeypatch.setattr(hardware, "_gpu_cache", (0.0, []))
    assert hardware.gpus() == []  # no NVIDIA GPU or driver


def test_the_api_reports_gpus_cpu_and_ram():
    from fastapi.testclient import TestClient

    from dainamo.server.app import app

    with TestClient(app) as client:
        hw = client.get("/api/hardware").json()
    assert isinstance(hw["gpus"], list)
    cpu, ram = hw["cpu"], hw["ram"]
    assert cpu["name"] and 0 <= cpu["percent"] <= 100 and cpu["threads"] >= cpu["cores"] >= 1
    assert cpu["temp"] is None or 0 < cpu["temp"] < 130
    assert 0 < ram["used_gb"] < ram["total_gb"] and abs(ram["used_gb"] + ram["available_gb"] - ram["total_gb"]) < 1e-6
