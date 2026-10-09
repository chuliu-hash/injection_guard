import atexit
import json
import os
import tempfile
import time
from contextlib import contextmanager
from typing import Iterable, List, Optional


_ASSIGNED = {}


def parse_gpu_ids(value: Optional[str], device_count: int) -> List[int]:
    if device_count <= 0:
        return []
    if not value:
        return list(range(device_count))

    ids = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            gpu_id = int(part)
        except ValueError:
            continue
        if 0 <= gpu_id < device_count:
            ids.append(gpu_id)
    return sorted(set(ids)) or list(range(device_count))


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@contextmanager
def _spin_file_lock(path: str, timeout: float = 10.0):
    start = time.time()
    lock_fd = None
    while True:
        try:
            lock_fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(lock_fd, str(os.getpid()).encode("ascii"))
            break
        except FileExistsError:
            if time.time() - start > timeout:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            time.sleep(0.05)
    try:
        yield
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _state_paths(namespace: str):
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in namespace)
    base = os.path.join(tempfile.gettempdir(), f"model_shield_gpu_{safe_name}")
    return base + ".json", base + ".lock"


def _load_state(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
            if isinstance(state, dict):
                return state
    except Exception:
        pass
    return {"assignments": {}, "cursor": 0}


def _save_state(path: str, state: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def release_process_gpu(namespace: str) -> None:
    key = (namespace, os.getpid())
    if key not in _ASSIGNED:
        return

    state_path, lock_path = _state_paths(namespace)
    with _spin_file_lock(lock_path):
        state = _load_state(state_path)
        assignments = state.get("assignments", {})
        assignments.pop(str(os.getpid()), None)
        state["assignments"] = assignments
        _save_state(state_path, state)
    _ASSIGNED.pop(key, None)


def assign_process_gpu(
    namespace: str,
    device_count: int,
    gpu_ids: Optional[Iterable[int]] = None,
) -> Optional[int]:
    """Return a stable visible CUDA device index for the current process."""
    if device_count <= 0:
        return None

    pid = os.getpid()
    key = (namespace, pid)
    if key in _ASSIGNED:
        return _ASSIGNED[key]

    ids = list(gpu_ids) if gpu_ids is not None else list(range(device_count))
    ids = [gpu_id for gpu_id in ids if 0 <= int(gpu_id) < device_count]
    if not ids:
        ids = list(range(device_count))

    state_path, lock_path = _state_paths(namespace)
    with _spin_file_lock(lock_path):
        state = _load_state(state_path)
        assignments = state.get("assignments", {})
        assignments = {
            str(old_pid): int(gpu_id)
            for old_pid, gpu_id in assignments.items()
            if pid_alive(int(old_pid)) and int(gpu_id) in ids
        }

        existing = assignments.get(str(pid))
        if existing is not None:
            selected = int(existing)
        else:
            counts = {gpu_id: 0 for gpu_id in ids}
            for gpu_id in assignments.values():
                counts[int(gpu_id)] = counts.get(int(gpu_id), 0) + 1

            min_count = min(counts.values())
            candidates = [gpu_id for gpu_id in ids if counts[gpu_id] == min_count]
            cursor = int(state.get("cursor", 0))
            selected = candidates[cursor % len(candidates)]
            state["cursor"] = cursor + 1
            assignments[str(pid)] = selected

        state["assignments"] = assignments
        _save_state(state_path, state)

    _ASSIGNED[key] = selected
    atexit.register(release_process_gpu, namespace)
    return selected
