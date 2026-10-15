"""Low-frequency, read-only resource samples; observations are not diagnoses."""

import time


class ResourceSampler:
    def __init__(self):
        self.process = None
        self.previous_io = None
        self.cpu_ready = False
        try:
            import psutil
        except ImportError:
            psutil = None
        self.psutil = psutil

    def attach(self, pid):
        self.process = None
        if self.psutil is not None:
            try:
                # Retain the Process identity, not just a potentially reused PID.
                self.process = self.psutil.Process(pid)
            except self.psutil.Error:
                pass

    def sample(self, now=None):
        if self.psutil is None:
            return {"unavailable": "RAM/CPU monitoring unavailable; run Setup.cmd after closing the app."}
        now = time.monotonic() if now is None else now
        ps = self.psutil
        values = {}
        try:
            memory = ps.virtual_memory()
            values.update(ram_percent=memory.percent, available_ram_gib=memory.available / 1024**3)
            cpu = ps.cpu_percent(interval=None)
            values["cpu_percent"] = cpu if self.cpu_ready else None
            self.cpu_ready = True
            values["app_ram_mib"] = ps.Process().memory_info().rss / 1024**2
            disk = ps.disk_io_counters()
            if disk is not None:
                previous = self.previous_io
                self.previous_io = (now, disk.read_bytes, disk.write_bytes)
                if previous is not None and now > previous[0]:
                    values["disk_read_mib_s"] = max(0, disk.read_bytes - previous[1]) / (now - previous[0]) / 1024**2
                    values["disk_write_mib_s"] = max(0, disk.write_bytes - previous[2]) / (now - previous[0]) / 1024**2
        except (ps.Error, OSError, NotImplementedError):
            values["unavailable"] = "Some system readings are unavailable."
        if self.process is not None:
            try:
                if self.process.is_running():
                    processes = [self.process, *self.process.children(recursive=True)]
                    rss = 0
                    partial = False
                    for process in processes:
                        try:
                            rss += process.memory_info().rss
                        except (ps.Error, OSError):
                            partial = True
                    values.update(trainer_ram_mib=rss / 1024**2, trainer_sample_partial=partial)
            except (ps.Error, OSError):
                values["trainer_sample_partial"] = True
        return values


def resource_summary(resources, phase):
    if resources is None:
        return "Resource monitoring off. Enable Measure performance in Environment before training."
    if not resources:
        return "Waiting for resource samples (every 5 seconds)..."
    parts, hints = [], []
    if resources.get("ram_percent") is not None:
        parts.append(f"System RAM {resources['ram_percent']:.0f}% | Available {resources['available_ram_gib']:.1f} GiB")
        if resources["ram_percent"] >= 90:
            hints.append("High system RAM pressure; paging may slow training. Check other apps and memory settings.")
    if resources.get("trainer_ram_mib") is not None:
        qualifier = " (partial)" if resources.get("trainer_sample_partial") else ""
        parts.append(f"Trainer tree RAM ~{resources['trainer_ram_mib']:.0f} MiB{qualifier}")
    if resources.get("app_ram_mib") is not None:
        parts.append(f"App RAM {resources['app_ram_mib']:.0f} MiB")
    if resources.get("cpu_percent") is not None:
        parts.append(f"System CPU {resources['cpu_percent']:.0f}%")
    if resources.get("disk_read_mib_s") is not None:
        parts.append(f"System disk R/W {resources['disk_read_mib_s']:.1f}/{resources['disk_write_mib_s']:.1f} MiB/s")
    else:
        parts.append("System disk rate pending/unavailable")
    gpu = resources.get("gpu")
    if gpu:
        parts.append(f"GPU 0 {gpu['utilization']:.0f}% | VRAM {gpu['memory_mib']:.0f} MiB")
        if phase == "Training" and gpu["utilization"] < 30:
            hints.append("Low sampled GPU activity: compare CPU, RAM and disk readings. A single sample does not identify a bottleneck.")
    else:
        parts.append("GPU reading unavailable")
    if resources.get("unavailable"):
        hints.append(resources["unavailable"])
    return "Last resource sample: " + " | ".join(parts) + ("\n" + " ".join(hints) if hints else "")
