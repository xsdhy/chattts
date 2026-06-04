"""pytest 全局夹具配置。

把仓库根目录插入 ``sys.path``，让测试可以用 ``backend.app.xxx`` 的方式导入应用
代码（与 ``scripts/`` 下脚本的导入方式保持一致）。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
