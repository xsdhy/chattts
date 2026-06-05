"""``text_preprocess`` 预处理管线单元测试（覆盖需求 10.1）。

测试纯函数管线，不依赖模型与网络。覆盖：数字阈值、日期/时间/金额/百分比/温度/单位、
token 白名单、英文缩写、Markdown/结构化、幂等性与回退策略。
"""

from __future__ import annotations

from backend.app.text_preprocess import (
    PreprocessOptions,
    preprocess_text,
)


def _run(text: str, **kwargs) -> str:
    """跑预处理并返回最终可朗读文本。"""

    return preprocess_text(text, PreprocessOptions(**kwargs)).text


# ---------------------------------------------------------------------- #
# 数字读法阈值（6.4）
# ---------------------------------------------------------------------- #
def test_small_integer_reads_as_cardinal():
    assert _run("123") == "一百二十三"


def test_long_number_reads_digit_by_digit():
    # 位数 ≥ 7 → 逐位读。
    assert _run("202606041234") == "二零二六零六零四一二三四"


def test_leading_zero_reads_digit_by_digit():
    assert _run("007") == "零零七"


def test_grouped_number_reads_digit_by_digit():
    assert _run("400-800-1234") == "四零零八零零一二三四"


def test_context_keyword_forces_digit_by_digit():
    # 上下文关键词「订单号」→ 即便只有 4 位也逐位读。
    assert _run("订单号 1234") == "订单号 一二三四"


def test_decimal_reads_with_dian():
    assert _run("3.14") == "三点一四"


def test_version_keeps_understandable_form():
    assert _run("v1.2.3") == "v一点二点三"


# ---------------------------------------------------------------------- #
# 日期 / 时间 / 金额 / 百分比 / 温度 / 单位（6.4）
# ---------------------------------------------------------------------- #
def test_date_reading():
    assert _run("2026-06-04") == "二零二六年六月四日"
    assert _run("2026/06/04") == "二零二六年六月四日"


def test_ambiguous_date_not_misread_as_date():
    # 越界月份 → 日期规则不强行转换成「...年...月...日」（歧义格式不当真，见 6.4）。
    out = _run("2026-13-40")
    assert "年" not in out and "月" not in out and "日" not in out


def test_time_reading():
    assert _run("14:30") == "十四点三十分"
    assert _run("09:05") == "九点零五分"


def test_money_percent_temperature():
    assert _run("¥99.9") == "九十九点九元"
    assert _run("￥99.9") == "九十九点九元"
    assert _run("12.5%") == "百分之十二点五"
    assert _run("28.5°C") == "二十八点五摄氏度"
    assert _run("100°F") == "一百华氏度"


def test_simple_and_compound_units():
    assert _run("10kg") == "十千克"
    assert _run("5km") == "五公里"
    assert _run("24kHz") == "二十四千赫兹"
    assert _run("12MB/s") == "十二兆字节每秒"
    assert _run("60km/h") == "六十公里每小时"


# ---------------------------------------------------------------------- #
# 控制 token 白名单（6.3）
# ---------------------------------------------------------------------- #
def test_tokens_stripped_by_default():
    # 默认剥离用户原文 token，避免提示注入式控制。
    assert _run("请读这句话 [laugh_2][break_7]").strip() == "请读这句话"


def test_whitelisted_tokens_kept_when_allowed():
    out = _run("保留 [uv_break] 这里", allow_control_tokens=True, profile="plain")
    assert "[uv_break]" in out


def test_non_whitelisted_tokens_not_passed_even_when_allowed():
    # [break_9] 不在白名单（break_0~7）→ 即使 allow=True 也不透传。
    out = _run("文本 [break_9]", allow_control_tokens=True, profile="plain")
    assert "[break_9]" not in out


# ---------------------------------------------------------------------- #
# 英文缩写 / 混读（6.5）
# ---------------------------------------------------------------------- #
def test_acronyms_and_word_map():
    out = _run("ChatTTS 支持 API 调用")
    assert "Chat T T S" in out
    assert "A P I" in out


def test_acronym_consistent_between_profiles():
    # 普通 / 流式共用同一预处理：不同 profile 下缩写结果一致。
    a = _run("GPU 与 CPU", profile="balanced")
    b = _run("GPU 与 CPU", profile="expressive")
    assert "G P U" in a and "C P U" in a
    assert a == b  # 缩写处理与 profile 无关


# ---------------------------------------------------------------------- #
# 安全清洗与结构化（6.2）
# ---------------------------------------------------------------------- #
def test_markdown_symbols_not_spoken():
    out = _run("# 今日更新\n- 支持 API 调用\n- 下载速度 12MB/s")
    assert "#" not in out
    assert "*" not in out
    # 不再注入控制 token；列表项之间用自然句末标点形成边界。
    assert "[lbreak]" not in out
    assert "[uv_break]" not in out
    assert "今日更新。支持" in out


def test_control_chars_removed():
    # 含零宽空格、BOM、控制符 \x00 → 应被清除且不崩溃。
    raw = "你好" + "\u200b" + "\ufeff" + "\x00" + "世界"
    out = _run(raw)
    assert "\u200b" not in out and "\ufeff" not in out and "\x00" not in out
    assert "你好世界" in out.replace(" ", "")

def test_bare_url_compressed_to_placeholder():
    out = _run("详见 https://example.com/path 谢谢")
    assert "http" not in out
    assert "链接" in out


# ---------------------------------------------------------------------- #
# 幂等性（10.1 / D4）
# ---------------------------------------------------------------------- #
def test_idempotent_under_repeat():
    opt = PreprocessOptions(allow_control_tokens=True, profile="balanced", prosody="dialogue")
    for text in ["你好。世界！再见。", "# 标题\n- 项目一，很好\n- 项目二", "金额 ¥9.9。"]:
        once = preprocess_text(text, opt).text
        twice = preprocess_text(once, opt).text
        assert once == twice, (text, once, twice)


# ---------------------------------------------------------------------- #
# 回退策略（D4 / 7.4）
# ---------------------------------------------------------------------- #
def test_single_rule_failure_skips_only_that_rule(monkeypatch):
    # 让数字读法内部抛异常：数字规则被跳过，但其它规则（token 剥离）仍生效，不阻断整体。
    import backend.app.text_preprocess as tp

    def _boom(value):
        raise RuntimeError("boom")

    # _int_to_chinese 被数字规则内部调用；令其抛错 → 整条 _normalize_numbers 被跳过。
    monkeypatch.setattr(tp, "_int_to_chinese", _boom)
    out = _run("订单 123 [laugh_2]")
    # 数字未转换（规则被跳过），但 token 仍被剥离、文本未崩溃。
    assert "123" in out
    assert "[laugh_2]" not in out


def test_disabled_preprocess_only_safe_cleans():
    # preprocess=false：仅安全清洗 + 空白合并，不做读法规范化。
    out = _run("订单 123\n下一行", enabled=False)
    assert "123" in out  # 数字未转换
    assert "\n" not in out  # 空白已合并


def test_refine_path_mutual_exclusion():
    # refine=false → token_injection 模式，但不再自动注入任何手工停顿 token。
    r_false = preprocess_text("你好。世界。", PreprocessOptions(prosody="dialogue"))
    assert r_false.refine_mode == "token_injection"
    assert r_false.refine_prompt is None
    assert "[uv_break]" not in r_false.text
    assert "[lbreak]" not in r_false.text

    # refine=true → refine_prompt 模式，同样不注入手工 token。
    r_true = preprocess_text("你好。世界。", PreprocessOptions(prosody="dialogue", refine=True))
    assert r_true.refine_mode == "refine_prompt"
    assert r_true.refine_prompt == "[oral_4][laugh_0][break_5]"
    assert "[uv_break]" not in r_true.text


def test_empty_and_symbol_only_inputs():
    assert preprocess_text("", PreprocessOptions()).text == ""
    assert preprocess_text("   ", PreprocessOptions()).text == ""
    # 纯符号不应崩溃。
    preprocess_text("！！！？？？……", PreprocessOptions())
