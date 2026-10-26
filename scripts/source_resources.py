"""Container-aware resource accounting. Pure stdlib; no model/data access."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def effective_limits(host_cpus, host_ram_bytes, affinity_count=None, cpu_max=None, memory_max=None):
    cpu = float(host_cpus)
    if affinity_count:
        cpu = min(cpu, affinity_count)
    if cpu_max:
        quota, period = cpu_max.split()
        if quota != 'max' and int(quota) > 0 and int(period) > 0:
            cpu = min(cpu, int(quota)/int(period))
    memory = host_ram_bytes
    if memory_max and memory_max != 'max' and int(memory_max) > 0:
        memory = min(memory, int(memory_max))
    return {'cpu_count': cpu, 'ram_bytes': memory}


def snapshot(host_cpus, host_ram_bytes):
    def read(path):
        p = Path(path)
        return p.read_text(encoding='utf-8').strip() if p.is_file() else None
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    cpu_max = read('/sys/fs/cgroup/cpu.max') if sys.platform.startswith('linux') else None
    memory_max = read('/sys/fs/cgroup/memory.max') if sys.platform.startswith('linux') else None
    # Current instance uses namespaced cgroup v2. On v1 translate the visible limits.
    if sys.platform.startswith('linux') and cpu_max is None:
        quota = read('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
        period = read('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
        if quota and period:
            cpu_max = f'{quota if int(quota) > 0 else "max"} {period}'
    if sys.platform.startswith('linux') and memory_max is None:
        memory_max = read('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    limits = effective_limits(host_cpus, host_ram_bytes, len(affinity) if affinity else None,
                              cpu_max, memory_max)
    return {**limits, 'host_cpu_count': host_cpus, 'host_ram_bytes': host_ram_bytes,
            'affinity_cpus': affinity, 'cgroup_cpu_max': cpu_max,
            'cgroup_memory_max': memory_max,
            'shared_memory_total_bytes': shutil.disk_usage('/dev/shm').total
                if sys.platform.startswith('linux') and Path('/dev/shm').is_dir() else None}


def shared_memory_risk(resources, minimum_bytes=1024**3):
    total = resources.get('shared_memory_total_bytes')
    return total is not None and total < minimum_bytes
