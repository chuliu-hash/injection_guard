"""输入检测器选卡：按空闲显存跳过被占满的 GPU（不加载真实模型）。"""

from __future__ import annotations

from types import SimpleNamespace

from injection_guard.api import dependencies as deps


def _service(
    monkeypatch,
    *,
    free_mb=None,
    min_free=4096,
    gpu_ids=(0, 1, 2, 3),
    enable_bert=True,
    busy=None,
):
    """只装配 ``_select_input_devices`` 需要的属性，避开 GPU / 模型初始化。"""
    service = deps.DefenseService.__new__(deps.DefenseService)
    service.settings = SimpleNamespace(
        enable_bert=enable_bert,
        input_detector_min_free_mb=min_free,
    )
    service.input_gpu_ids = list(gpu_ids)
    if free_mb is not None:
        monkeypatch.setattr(deps, "free_memory_mb", lambda ids: {i: free_mb.get(i, 0) for i in ids})
    # 不让用例读到本机真实的 GPU 分配状态文件
    monkeypatch.setattr(deps, "gpu_assignment_load", lambda namespace: dict(busy or {}))
    return service


def test_bert_disabled_keeps_one_lightweight_detector_per_gpu(monkeypatch):
    service = _service(monkeypatch, enable_bert=False)
    assert service._select_input_devices() == (
        ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        False,
    )


def test_gpu_without_enough_free_memory_is_skipped(monkeypatch):
    service = _service(monkeypatch, free_mb={0: 528, 1: 18680, 2: 8843, 3: 18680})
    devices, load_bert = service._select_input_devices()
    assert devices == ["cuda:1", "cuda:2", "cuda:3"]
    assert load_bert is True


def test_all_gpus_too_full_degrades_to_llm_only(monkeypatch):
    service = _service(monkeypatch, free_mb={0: 100, 1: 200, 2: 300, 3: 400})
    assert service._select_input_devices() == ([None], False)


def test_no_visible_gpu_lets_detector_pick_device(monkeypatch):
    service = _service(monkeypatch, gpu_ids=())
    assert service._select_input_devices() == ([None], True)


def test_zero_threshold_disables_memory_check(monkeypatch):
    service = _service(monkeypatch, free_mb={0: 0, 1: 0}, gpu_ids=(0, 1), min_free=0)
    assert service._select_input_devices() == (["cuda:0", "cuda:1"], True)


def test_gpus_used_by_other_services_are_avoided(monkeypatch):
    """别的服务（LLM / Embedding）已占用的卡优先避开，即使显存也够。"""
    service = _service(
        monkeypatch,
        free_mb={0: 18764, 1: 20669, 2: 21688, 3: 21688},
        busy={0: 2, 2: 1},
    )
    devices, load_bert = service._select_input_devices()
    assert devices == ["cuda:1", "cuda:3"]
    assert load_bert is True


def test_falls_back_to_busy_gpus_when_all_are_taken(monkeypatch):
    """所有显存足够的卡都被占用时仍加载（并告警），只有显存不足才退化。"""
    service = _service(
        monkeypatch,
        free_mb={0: 18764, 1: 20669, 2: 21688, 3: 21688},
        busy={0: 1, 1: 1, 2: 1, 3: 1},
    )
    devices, load_bert = service._select_input_devices()
    assert devices == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert load_bert is True
