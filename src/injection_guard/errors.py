"""流水线异常类型。

独立成模块，避免 ``api`` 层与 ``agent`` 编排层互相 import 造成循环依赖。
``injection_guard.api.dependencies`` 会原样再导出这两个类型，历史调用方
（``api.routes`` / 测试）无需改动。
"""

from __future__ import annotations


class ServiceBusyError(RuntimeError):
    """推理服务繁忙，无法在指定时间内获取并发名额（-> HTTP 503）。"""


class DetectionError(RuntimeError):
    """检测链路内部异常（-> HTTP 500）。"""
