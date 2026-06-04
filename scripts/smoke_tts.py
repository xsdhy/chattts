#!/usr/bin/env python3
"""最小合成冒烟测试。

执行完整链路：text -> ChatTTS 推理 -> WAV 文件，用于验证模型与引擎可用。

运行示例：
    python scripts/smoke_tts.py --text "你好，欢迎使用 ChatTTS 语音合成服务。"
    python scripts/smoke_tts.py --speaker 42 --speed 1.1 --output out.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 允许直接执行本脚本，而不必手动设置 PYTHONPATH。
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.tts_engine import engine, write_wav


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成一段中文 WAV，用于验证 ChatTTS 引擎可用。"
    )
    parser.add_argument(
        "--text",
        default="你好，欢迎使用 ChatTTS 语音合成服务。",
        help="待合成文本",
    )
    parser.add_argument("--speaker", default=None, help="音色 id（seed），例如 42")
    parser.add_argument("--speed", type=float, default=1.0, help="语速，范围 0.5-2.0")
    parser.add_argument("--refine", action="store_true", help="启用文本 refine")
    parser.add_argument("--output", default="audio.wav", help="输出 WAV 文件路径")
    args = parser.parse_args()

    result = engine.synthesize(
        args.text,
        speaker=args.speaker,
        speed=args.speed,
        refine=args.refine,
    )
    output = Path(args.output)
    write_wav(output, result.samples, result.sample_rate)
    print(
        f"已生成 {output.resolve()}，采样率 {result.sample_rate} Hz，"
        f"样本数 {len(result.samples)}。"
    )


if __name__ == "__main__":
    main()
