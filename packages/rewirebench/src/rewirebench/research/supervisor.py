"""Single process-tree watchdog with durable logs and no shell interpretation."""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import psutil


class ResourceStop(RuntimeError):
    pass


def workspace_size(root):
    size = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(directory)/d).is_symlink()]
        for name in files:
            path = Path(directory)/name
            if not path.is_symlink():
                try:
                    size += path.stat().st_size
                except FileNotFoundError:
                    pass
    return size


def terminate_tree(process, observed=None):
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        children = []
    children.extend(observed or [])
    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    # The leader may already have exited while an unobserved child ignores TERM.
    # Its process group still exists; always finish cleanup with a group KILL.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for child in children:
        try:
            if child.is_running():
                child.kill()
        except psutil.Error:
            pass


def run_guarded(argv, *, cwd, workspace, stdout, stderr, seconds, memory_bytes,
                workspace_bytes, deadline, stopped=lambda: False, env=None,
                input_text=None, on_start=lambda pid: None, event_check=lambda: None):
    """Memory enforcement is sampled RSS, not a hard kernel allocation ceiling."""
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) for a in argv):
        raise ValueError("A fixed argv list is required")
    start = time.monotonic()
    peak, process, observed = 0, None, {}
    if time.time() >= deadline:
        raise ResourceStop("campaign_deadline")
    if workspace_size(workspace) > workspace_bytes:
        raise ResourceStop("workspace_limit")
    if stopped():
        raise ResourceStop("stop_requested")
    with Path(stdout).open("w") as out, Path(stderr).open("w") as err, \
            tempfile.TemporaryFile(mode="w+", dir=workspace) as prompt:
        try:
            if input_text:
                prompt.write(input_text)
            prompt.seek(0)
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=prompt,
                                       stdout=out, stderr=err, text=True, start_new_session=True)
            on_start(process.pid)
            while process.poll() is None:
                if stopped():
                    raise ResourceStop("stop_requested")
                if time.time() >= deadline:
                    raise ResourceStop("campaign_deadline")
                if time.monotonic()-start >= seconds:
                    raise ResourceStop("job_timeout")
                try:
                    tree = psutil.Process(process.pid)
                    descendants = tree.children(recursive=True)
                    for child in descendants:
                        observed[child.pid] = child
                    rss = sum(p.memory_info().rss for p in [tree, *descendants]
                              if p.is_running())
                    peak = max(peak, rss)
                    if rss > memory_bytes:
                        raise ResourceStop("memory_limit")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if workspace_size(workspace) > workspace_bytes:
                    raise ResourceStop("workspace_limit")
                event_check()
                time.sleep(.1)
            event_check()
            if workspace_size(workspace) > workspace_bytes:
                raise ResourceStop("workspace_limit")
            return {"returncode": process.returncode, "wall_seconds": time.monotonic()-start,
                    "peak_rss_bytes": peak}
        finally:
            if process is not None:
                # Clean the group even after a successful leader exit. A child
                # cannot survive by letting its parent finish early.
                terminate_tree(process, list(observed.values()))


def local_environment():
    # Scientific operations receive no account credentials. Numerical libraries
    # run one CPU thread; remote model/data download routes are disabled.
    keep = {k: v for k, v in os.environ.items()
            if k in {"PATH", "HOME", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR", "PYTHONPATH"}}
    return dict(keep, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                HF_DATASETS_OFFLINE="1", CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
