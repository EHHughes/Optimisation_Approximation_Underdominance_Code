"""Detect how many CPU cores this process may actually use.

Under SLURM the job is confined to the cores allocated to it (often fewer than
the node's physical core count), so sizing a worker pool from ``os.cpu_count()``
would oversubscribe a shared node.  Outside SLURM we fall back to the full
machine.  Use :func:`available_cpu_count` wherever a ``multiprocess.Pool`` is
sized so the same code runs sensibly on a laptop and on the cluster.
"""

import os


def available_cpu_count() -> int:
    """Return the number of CPU cores this process should use for parallel work.

    Resolution order:

    1. Inside a SLURM allocation (``SLURM_JOB_ID`` set), use the cores granted to
       the job: ``SLURM_CPUS_PER_TASK`` if present, otherwise
       ``SLURM_CPUS_ON_NODE``.
    2. Otherwise fall back to the cores the OS scheduler allows this process to
       run on (``os.sched_getaffinity`` respects cgroup/cpuset limits where
       available), then to ``os.cpu_count()``.

    Always returns at least 1.
    """
    if os.environ.get("SLURM_JOB_ID"):
        for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
            value = os.environ.get(var)
            if value:
                try:
                    return max(1, int(value))
                except ValueError:
                    pass

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        # sched_getaffinity is Linux-only; fall back to the full machine.
        return max(1, os.cpu_count() or 1)
