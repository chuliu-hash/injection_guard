"""跨进程 GPU 分配。

通过 ``tempfile`` 下的 JSON + 自旋文件锁记录「进程 -> GPU」映射，
按负载均衡挑选 GPU，进程退出时释放。多个服务共用同一命名空间即可协调。
"""

from __future__ import annotations

import atexit
import json
import os
import tempfile
import time
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Optional

_ASSIGNED: Dict[tuple, int] = {}


def parse_gpu_ids(value: Optional[str], device_count: int) -> List[int]:
    """把 ``"0,1,2"`` 这类字符串解析为合法的 GPU id 列表。"""
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


def free_memory_mb(device_ids: Iterable[int]) -> Dict[int, int]:
    """查询各 GPU 的当前空闲显存（MiB）；不可用或查询失败时该卡记 0。

    用于在加载模型前避开已被其他任务占满的卡（见 ``DefenseService`` 的 BERT
    分类器选卡逻辑），避免直接 CUDA OOM。torch 为惰性导入，未安装/无 CUDA 时返回空表。
    """
    ids = [int(device_id) for device_id in device_ids]
    if not ids:
        return {}
    try:
        import torch
    except Exception:  # noqa: BLE001 - 无 torch 时不做显存检查
        return {}

    free: Dict[int, int] = {}
    for device_id in ids:
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device_id)
            free[device_id] = int(free_bytes // (1024 * 1024))
        except Exception:  # noqa: BLE001 - 单卡查询失败不阻断其余卡
            free[device_id] = 0
    return free


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@contextmanager
def _spin_file_lock(path: str, timeout: float = 10.0) -> Iterator[None]:
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


def _state_paths(namespace: str) -> tuple:
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in namespace)
    base = os.path.join(tempfile.gettempdir(), f"model_shield_gpu_{safe_name}")
    return base + ".json", base + ".lock"


def _load_state(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
            if isinstance(state, dict):
                return state
    except Exception:  # noqa: BLE001
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
    """返回当前进程稳定绑定的可见 CUDA 设备索引。"""
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
