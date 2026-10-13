"""确保 `src/` 在导入路径上（无需安装即可运行 pytest）。"""

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 测试里不预热真实检测器（BERT 加载依赖 GPU / 权重）；由各用例自行构造桩。
os.environ.setdefault("WARMUP_ON_STARTUP", "false")
