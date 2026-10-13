"""跨进程 GPU 分配：统一命名空间与占用查询（用临时命名空间，不碰真实 GPU）。"""

from __future__ import annotations

import os

from injection_guard.services import gpu


def test_resolve_namespace_prefers_explicit_then_env_then_default(monkeypatch):
    monkeypatch.delenv("GPU_ALLOC_NAMESPACE", raising=False)
    assert gpu.resolve_namespace() == gpu.DEFAULT_NAMESPACE
    assert gpu.resolve_namespace("custom") == "custom"

    monkeypatch.setenv("GPU_ALLOC_NAMESPACE", "shared-ns")
    assert gpu.resolve_namespace() == "shared-ns"
    assert gpu.resolve_namespace("custom") == "custom"


def test_allocated_gpus_are_visible_in_same_namespace(monkeypatch, tmp_path):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    namespace = f"test-ns-{os.getpid()}"
    gpu._ASSIGNED.clear()

    selected = gpu.assign_process_gpu(namespace, device_count=4)
    assert selected is not None
    assert gpu.gpu_assignment_load(namespace) == {selected: 1}

    # 另一个命名空间互不影响
    assert gpu.gpu_assignment_load(f"{namespace}-other") == {}

    with gpu._spin_file_lock(gpu._state_paths(namespace)[1]):
        pass
    gpu.release_process_gpu(namespace)
    assert gpu.gpu_assignment_load(namespace) == {}
