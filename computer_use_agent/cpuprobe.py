# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Who is eating the CPU, in one line.

The dashboard stutters and the preview pump's own numbers say the event loop
is running four to seven times slower than it asks to. That is as far as
`preview.py` can see: it knows it was late, not what made it late. The two
answers need completely different fixes -- more cores, or a blocking call
moved off the loop -- so guessing between them is exactly the mistake this
file exists to stop being made a third time.

Everything here reads `/proc`, which is cheap (a few dozen small reads) and
needs no dependency. Deliberately *not* `/proc/stat`: under a sandboxed
runtime its idle/busy columns describe something other than our container, and
a number that is wrong in a way nobody can see is worse than no number. Summing
per-process CPU time is safe -- those are our own processes either way -- and
answers the question that matters: of the cores we are allowed, how many are we
actually burning, and in whose name.
"""

from __future__ import annotations

import logging
import os
import resource
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("cctv_audit.cpu")

_CLOCK_TICKS = float(os.sysconf("SC_CLK_TCK") or 100)


def cpu_quota() -> float:
    """Cores this container may use, per cgroup -- not the host's core count.

    `os.cpu_count()` reports the machine, so on a 4-core allowance of a bigger
    host it overstates the ceiling and every percentage computed from it comes
    out flatteringly low.
    """
    for path, parse in (
        ("/sys/fs/cgroup/cpu.max", _parse_cpu_max),          # cgroup v2
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", None),        # v1, with period
    ):
        try:
            text = Path(path).read_text().strip()
        except OSError:
            continue
        if parse is not None:
            value = parse(text)
        else:
            try:
                period = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
                quota = float(text)
                value = quota / period if quota > 0 and period > 0 else None
            except (OSError, ValueError):
                value = None
        if value:
            return value
    return float(os.cpu_count() or 1)


def _parse_cpu_max(text: str) -> Optional[float]:
    parts = text.split()
    if len(parts) != 2 or parts[0] == "max":
        return None
    try:
        quota, period = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return quota / period if quota > 0 and period > 0 else None


def _snapshot() -> Dict[int, Tuple[str, float]]:
    """CPU seconds burned so far, per live process, by pid."""
    out: Dict[int, Tuple[str, float]] = {}
    try:
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return out
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            # Processes come and go between the listing and the read; a clip's
            # ffmpeg is the most likely one to vanish here. Nothing to report.
            continue
        # comm can contain spaces and parentheses, so split on the LAST ')'.
        close = stat.rfind(")")
        open_ = stat.find("(")
        if close < 0 or open_ < 0:
            continue
        name = stat[open_ + 1:close]
        fields = stat[close + 2:].split()
        try:
            # utime and stime are fields 14 and 15 of stat, i.e. 12 and 13
            # counting from the field after state.
            used = (float(fields[11]) + float(fields[12])) / _CLOCK_TICKS
        except (IndexError, ValueError):
            continue
        out[pid] = (name, used)
    return out


def runqueue_wait_seconds() -> Optional[float]:
    """Time this process spent ready to run but not given a CPU, since boot.

    Field 2 of `/proc/self/schedstat`, in nanoseconds. It is the one number
    that separates the two explanations outright, because the kernel keeps it
    and neither the GIL nor our own code can influence it: a task that is
    runnable and waiting is a task somebody else decided not to schedule.

    Returns None where the file is absent -- sandboxed runtimes often omit it,
    and a fabricated zero would read as "definitely not throttled".
    """
    try:
        fields = Path("/proc/self/schedstat").read_text().split()
    except OSError:
        return None
    try:
        return float(fields[1]) / 1e9
    except (IndexError, ValueError):
        return None


class ThreadTicker:
    """The same 83ms tick as the preview pump, on a plain thread.

    The pump measures its own lateness, but it is an asyncio task, so lateness
    there has two possible authors: the platform not scheduling the process, or
    something inside the event loop refusing to yield. This thread only ever
    calls `time.sleep`, which releases the GIL, so it is late only for reasons
    outside our event loop. Run the two side by side and the answer is
    whichever one lags:

        both late   -> the process is not being scheduled; look at the platform
        pump alone  -> something is blocking the loop; look at our own code
    """

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._late_ms: List[float] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="tickprobe", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last = time.monotonic()
        while not self._stop.wait(self.interval):
            now = time.monotonic()
            with self._lock:
                self._late_ms.append((now - last) * 1000.0)
            last = now

    def stop(self) -> None:
        self._stop.set()

    def drain(self) -> List[float]:
        with self._lock:
            out, self._late_ms = self._late_ms, []
        return out


class CpuProbe:
    """Deltas between two calls to `report`, formatted for a log line."""

    def __init__(self, tick_interval: float = 0.083) -> None:
        self.quota = cpu_quota()
        self._previous = _snapshot()
        self._waited = runqueue_wait_seconds()
        self._usage = resource.getrusage(resource.RUSAGE_SELF)
        self.ticker = ThreadTicker(tick_interval)
        self.ticker.start()

    def report(self, window_seconds: float, top: int = 3) -> str:
        current = _snapshot()
        if window_seconds <= 0:
            return ""
        deltas: List[Tuple[float, str]] = []
        for pid, (name, used) in current.items():
            was = self._previous.get(pid)
            # A pid absent from the previous snapshot started inside this
            # window; all of its CPU time belongs to the window.
            before = was[1] if was else 0.0
            spent = used - before
            if spent > 0:
                deltas.append((spent, name))
        self._previous = current

        total = sum(spent for spent, _ in deltas)
        deltas.sort(reverse=True)
        # Group by name: Chromium is a dozen processes and reporting the
        # busiest single renderer would understate it against a lone ffmpeg.
        by_name: Dict[str, float] = {}
        for spent, name in deltas:
            by_name[name] = by_name.get(name, 0.0) + spent
        ranked = sorted(by_name.items(), key=lambda kv: kv[1], reverse=True)[:top]
        named = ", ".join(
            f"{name} {spent / window_seconds * 100:.0f}%" for name, spent in ranked
        )
        try:
            load = os.getloadavg()[0]
        except OSError:
            load = -1.0
        cores_used = total / window_seconds
        return (
            f"cpu: {cores_used:.1f} of {self.quota:.0f} cores "
            f"({cores_used / self.quota * 100:.0f}% of the allowance), "
            f"load {load:.1f}, {len(current)} procs; top: {named}"
        )

    def scheduling_report(self, window_seconds: float) -> str:
        """Who is late, and whether the kernel says we were kept waiting."""
        if window_seconds <= 0:
            return ""
        ticks = sorted(self.ticker.drain())
        if ticks:
            asked = self.ticker.interval * 1000.0
            tick_part = (
                f"thread tick median {ticks[len(ticks) // 2]:.0f}ms "
                f"p90 {ticks[min(len(ticks) - 1, int(len(ticks) * 0.9))]:.0f}ms "
                f"worst {ticks[-1]:.0f}ms (asked {asked:.0f}ms, "
                f"{len(ticks)} in {window_seconds:.0f}s)"
            )
        else:
            tick_part = "thread tick: no samples"

        waited = runqueue_wait_seconds()
        if waited is None or self._waited is None:
            wait_part = "runqueue wait unavailable"
        else:
            delta = waited - self._waited
            self._waited = waited
            wait_part = (f"runqueue wait {delta * 1000:.0f}ms "
                         f"({delta / window_seconds * 100:.1f}% of the window)")

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Involuntary switches are the process being taken off a CPU it still
        # wanted. A big number next to near-zero CPU is the signature of an
        # allowance being enforced rather than of work being done.
        involuntary = usage.ru_nivcsw - self._usage.ru_nivcsw
        voluntary = usage.ru_nvcsw - self._usage.ru_nvcsw
        own = ((usage.ru_utime + usage.ru_stime)
               - (self._usage.ru_utime + self._usage.ru_stime))
        self._usage = usage

        return (f"sched: {tick_part}; {wait_part}; "
                f"this process {own / window_seconds * 100:.0f}% of one core, "
                f"switches {involuntary} forced / {voluntary} voluntary")

    def stop(self) -> None:
        self.ticker.stop()
