"""``segment_text`` 分片器单元测试（覆盖需求 10.2）。

验证：不切开 atomic_spans（token/日期/金额/单位/缩写/URL/长编号）、双阈值行为、
段落/列表边界优先成片、普通与流式同一预处理结果仅分片长度不同。
"""

from __future__ import annotations

from backend.app.text_preprocess import PreprocessOptions, preprocess_text
from backend.app.tts_engine import segment_text


def _segments(text: str, *, first: int, rest: int, cap: int | None = None, **opts):
    """预处理 + 分片，返回 (result, segments)。"""

    result = preprocess_text(text, PreprocessOptions(**opts))
    segs = segment_text(
        result.text,
        result.atomic_spans,
        first_chars=first,
        rest_chars=rest,
        first_hard_cap=cap,
    )
    return result, segs


def _assert_atoms_intact(result, segments):
    """每个原子片段必须完整落在某一个分片内，不被切开。"""

    text = result.text
    for span in result.atomic_spans:
        fragment = text[span.start:span.end]
        assert any(fragment in seg for seg in segments), (fragment, segments)


# ---------------------------------------------------------------------- #
# 原子片段不被切开
# ---------------------------------------------------------------------- #
def test_atomic_spans_never_split():
    text = (
        "今天是 2026-06-04，气温 28.5°C，下载速度 12MB/s，"
        "订单号 202606041234，详见 https://example.com 谢谢。"
    )
    result, segs = _segments(text, first=12, rest=20)
    _assert_atoms_intact(result, segs)
    # 关键语义单元不被切碎。
    joined = "".join(segs)
    assert "二零二六年六月四日" in joined
    assert "二零二六零六零四一二三四" in joined


def test_token_not_split_even_with_tiny_threshold():
    # 极小阈值也不得把 [uv_break] 切成 [uv_ 与 break]。
    result, segs = _segments("你好。世界。再见。", first=3, rest=4, prosody="dialogue")
    for seg in segs:
        assert "[uv_" not in seg or "[uv_break]" in seg
    _assert_atoms_intact(result, segs)


# ---------------------------------------------------------------------- #
# 双阈值
# ---------------------------------------------------------------------- #
def test_dual_threshold_first_small_rest_large():
    text = "你好，欢迎使用流式语音合成服务，" * 6 + "希望第一声尽快出来。"
    _, segs = _segments(text, first=50, rest=100, cap=90)
    assert len(segs) >= 2
    # 首片受首片硬上限约束。
    assert len(segs[0]) <= 90
    # 首片明显小于后续片的目标（求快）。
    assert len(segs[0]) <= 60


def test_normal_mode_single_threshold():
    text = "你好，欢迎使用流式语音合成服务，" * 6 + "希望第一声尽快出来。"
    _, stream = _segments(text, first=50, rest=100, cap=90)
    _, normal = _segments(text, first=120, rest=120)
    # 普通（大阈值）分片数不多于流式（小首片）。
    assert len(normal) <= len(stream)


# ---------------------------------------------------------------------- #
# 段落 / 列表边界优先成片
# ---------------------------------------------------------------------- #
def test_list_boundaries_form_independent_segments():
    text = "# 标题\n- 项目一\n- 项目二\n- 项目三"
    result, segs = _segments(text, first=50, rest=100, profile="balanced")
    # 不再注入 [lbreak]：列表项边界改用自然句末标点显式化。
    assert "[lbreak]" not in result.text
    assert "标题。项目一。项目二。项目三" in result.text
    # 用极小阈值时，应在句末标点处拆成多段，且不留下控制 token。
    _, small = _segments(text, first=4, rest=4, profile="balanced")
    assert len(small) >= 3
    assert all("[lbreak]" not in seg for seg in small)


# ---------------------------------------------------------------------- #
# 普通 / 流式一致性（10.3.6）
# ---------------------------------------------------------------------- #
def test_normalized_text_identical_across_thresholds():
    text = "2026-06-04 转化率 12.5%，下载 12MB/s。请联系 API 团队。"
    r1, normal = _segments(text, first=120, rest=120)
    r2, stream = _segments(text, first=50, rest=100, cap=90)
    # 同一文本：归一化结果完全一致，只有分片长度不同（D3）。
    assert r1.text == r2.text
    # 两种切分都不切开原子片段。
    _assert_atoms_intact(r1, normal)
    _assert_atoms_intact(r2, stream)


def test_empty_text_returns_empty():
    assert segment_text("", [], first_chars=50, rest_chars=100) == []
    assert segment_text("   ", [], first_chars=50, rest_chars=100) == []
