"""TTS 文本预处理管线（纯函数）。

本模块是需求《TTS 文本处理与流式分片一体化优化》的核心落地点之一，负责把用户原始
输入转换成「更适合 ChatTTS 朗读」的可朗读文本，并产出供分片器使用的「原子片段标注」、
按韵律策略决定的 ``refine_prompt``、以及可观测的变更记录。

设计要点（对应需求 D1~D5、第 5/6/7 节）：

- **一条有序、可解释、可回退的规则管线**：每条规则都是独立的纯函数，单条失败只跳过
  该条、不影响其它规则；整条管线异常则回退到「安全清洗后的原文」，绝不阻断 TTS（D4）。
- **原子片段保护**：数字 / 日期 / 时间 / 金额 / 百分比 / 单位 / 英文缩写 / URL 占位符 /
  长编号 / 用户原文保留的控制 token，在转换时统一登记为「原子片段」。实现手段是把它们
  替换成私有区占位符（``\\ue000…\\ue001``），后续规则的正则不会再误伤它们；管线末尾再把
  占位符展开成最终文本并计算精确字符区间（``AtomicSpan``），交给分片器，保证任何层级
  （含硬切）都不会把一个语义单元切成两半（D3 / 6.8）。
- **不自动注入控制 token**：本模块不会主动插入 ``[uv_break]``/``[lbreak]`` 等停顿 / 结构
  token，停顿完全交由文本自身标点与 ChatTTS 决定。``refine=false``（默认）仅做文本规范化；
  ``refine=true`` 则把 ``prosody`` 映射成 ``refine_prompt`` 交给 ChatTTS 的 refine_text 负责。
  用户原文里自带且在白名单内的控制 token 在 ``allow_control_tokens`` 打开时仍会被保留。
- **段落边界以标点显式化（D2）**：必须在「把换行压成空格」之前，把 Markdown 结构、段落、
  列表项边界转成自然句末标点，否则换行信息会在归一化时被抹掉。

本模块不依赖任何重型运行时库，只用标准库 ``re``，满足「预处理 P95 < 20ms、不新增大型
依赖」的性能与质量约束（第 9 节）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Literal

logger = logging.getLogger("chattts.preprocess")


# ====================================================================== #
# 数据结构（对应需求 7.2）
# ====================================================================== #
@dataclass(frozen=True)
class PreprocessOptions:
    """预处理选项。

    Attributes:
        enabled: 是否启用完整预处理。``False`` 时退化为仅「安全清洗 + 空白合并」，
            等价旧 ``_normalize_text``，但仍保留原子保护框架（见 7.5）。
        profile: 增强强度。``plain`` 不主动注入停顿 token；``balanced`` 适度；
            ``expressive`` 更明显地利用停顿。
        prosody: 朗读风格，决定停顿倾向 / refine prompt 映射。
        allow_control_tokens: 是否允许用户原文中的 ChatTTS 控制 token 生效。
        refine: 是否启用 ChatTTS refine_text。它决定 D1 走哪条互斥路径。
    """

    enabled: bool = True
    profile: Literal["plain", "balanced", "expressive"] = "balanced"
    prosody: Literal["flat", "natural", "dialogue", "narration"] = "natural"
    allow_control_tokens: bool = False
    refine: bool = False


@dataclass(frozen=True)
class PreprocessChange:
    """一次具体的文本变更记录（可观测，用于预览与调试）。

    ``source``/``target`` 在序列化为 API 响应时分别映射为 ``from``/``to``（见 6.9）。
    """

    type: str
    source: str
    target: str


@dataclass(frozen=True)
class AtomicSpan:
    """最终文本中的一个原子片段区间 ``[start, end)``。

    分片器据此保护语义单元：任何层级（含硬切）都不得切开该区间。
    """

    start: int
    end: int
    # token / date / time / money / percent / unit / acronym / url / longnum / number / version
    kind: str


@dataclass(frozen=True)
class PreprocessResult:
    """预处理产出。

    Attributes:
        text: 可朗读文本（不含自动注入的停顿 / 结构 token，仅含规范化标点）。
        atomic_spans: 原子片段区间列表（按 ``start`` 升序），供分片器保护。
        refine_prompt: D1 决定的最终 refine prompt；``token_injection`` 路径下为 ``None``。
        refine_mode: ``"token_injection"`` 或 ``"refine_prompt"``，对应 D1 两条路径。
        changes: 变更记录列表。
    """

    text: str
    atomic_spans: list[AtomicSpan] = field(default_factory=list)
    refine_prompt: str | None = None
    refine_mode: Literal["token_injection", "refine_prompt"] = "token_injection"
    changes: list[PreprocessChange] = field(default_factory=list)


# ====================================================================== #
# 常量表
# ====================================================================== #
# ChatTTS 控制 token 白名单（见 6.3）。固定 token + 带参数 token 两类。
_WHITELIST_FIXED = ("[uv_break]", "[laugh]", "[lbreak]")
_WHITELIST_PARAM_RE = re.compile(r"^\[(oral_[0-9]|laugh_[0-2]|break_[0-7])\]$")
# 任何「形如 token」的方括号片段，用于剥离 / 校验用户原文中的 token。
_TOKEN_LIKE_RE = re.compile(r"\[[a-zA-Z][a-zA-Z0-9_]*\]")

# prosody → refine prompt 映射（仅 refine=true 时使用，见 6.7）。
_PROSODY_REFINE_PROMPT = {
    "flat": "[oral_0][laugh_0][break_2]",
    "natural": "[oral_2][laugh_0][break_4]",
    "dialogue": "[oral_4][laugh_0][break_5]",
    "narration": "[oral_2][laugh_0][break_5]",
}

# 阿拉伯数字 → 中文数字。
_DIGITS = "零一二三四五六七八九"

# 整数「逐位读」的上下文关键词（见 6.4）。出现在数字紧邻的前文时强制逐位读。
_DIGIT_BY_DIGIT_KEYWORDS = (
    "订单号", "单号", "编号", "电话", "手机", "卡号",
    "验证码", "快递", "QQ", "微信号", "邮编", "房间",
)

# 单位表（见 6.4）。按「先长后短」匹配，避免 ``km`` 被 ``m`` 抢先匹配。
_UNIT_MAP = {
    "kHz": "千赫兹", "MHz": "兆赫兹", "Hz": "赫兹",
    "GB": "吉字节", "MB": "兆字节", "KB": "千字节",
    "min": "分钟", "ms": "毫秒",
    "kg": "千克", "km": "公里", "cm": "厘米", "mm": "毫米",
    "g": "克", "m": "米", "s": "秒", "h": "小时",
}
# 复合速率分母 → 读法。
_RATE_MAP = {"s": "每秒", "h": "每小时", "min": "每分钟"}

# 技术缩写：拆成单字母朗读，降低连读失败概率（见 6.5）。
_ACRONYMS = (
    "HTTPS", "HTTP", "JSON", "HTML", "URL", "API",
    "GPU", "CPU", "SDK", "TTS", "SQL", "AI",
)
# 工程词表：整词映射（优先于缩写规则）。
_WORD_MAP = {
    "ChatTTS": "Chat T T S",
}

# ---- 占位符编码 ----
# 用私有区字符包裹「原子片段 id」。id 的每一位十进制数字映射到 ``+digit``，
# 这样占位符内部不含 ASCII 数字 / 字母，后续的数字 / 单位 / 缩写正则都不会误伤它。
_PH_START = ""
_PH_END = ""
_PH_DIGIT_BASE = 0xE010
_PH_RE = re.compile(_PH_START + r"([-]+)" + _PH_END)


def _encode_ph_index(index: int) -> str:
    """把原子片段 id 编码成私有区数字串。"""

    return "".join(chr(_PH_DIGIT_BASE + int(ch)) for ch in str(index))


def _decode_ph_index(encoded: str) -> int:
    """把私有区数字串还原成原子片段 id。"""

    return int("".join(str(ord(ch) - _PH_DIGIT_BASE) for ch in encoded))


# ====================================================================== #
# 中文数字读法工具
# ====================================================================== #
def _int_below_10000_to_chinese(value: int) -> str:
    """0~9999 的中文基数读法（内部零按中文习惯压缩）。"""

    if value == 0:
        return "零"
    units = ("", "十", "百", "千")
    text = str(value)
    length = len(text)
    result = ""
    pending_zero = False
    for i, ch in enumerate(text):
        digit = int(ch)
        position = length - i - 1
        if digit == 0:
            pending_zero = True
            continue
        if pending_zero:
            result += "零"
            pending_zero = False
        result += _DIGITS[digit] + units[position]
    return result


def _int_to_chinese(value: int) -> str:
    """任意非负整数 → 中文基数读法（支持 万 / 亿 分组）。

    例：``123`` → ``一百二十三``；``20000`` → ``二万``；``10`` → ``十``。
    """

    if value == 0:
        return "零"
    big_units = ("", "万", "亿", "万亿")
    groups: list[int] = []
    while value > 0:
        groups.append(value % 10000)
        value //= 10000
    result = ""
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if group == 0:
            # 组为 0 时补一个「零」以表达数位间断（避免重复零，末尾零最后再清理）。
            if result and not result.endswith("零"):
                result += "零"
            continue
        # 非最高组且高位不足 1000 时，需要补零（如 100005 的「零五」）。
        if result and group < 1000:
            result += "零"
        result += _int_below_10000_to_chinese(group) + big_units[index]
    result = result.rstrip("零")
    # 中文习惯：10~19 读「十X」而非「一十X」。
    if result.startswith("一十"):
        result = result[1:]
    return result


def _digits_to_chinese(digits: str) -> str:
    """逐位读：``2026`` → ``二零二六``。"""

    return "".join(_DIGITS[int(ch)] for ch in digits)


def _read_decimal(num_str: str) -> str:
    """读一个十进制数：整数部分基数读，小数部分逐位读。

    例：``3.14`` → ``三点一四``；``99.9`` → ``九十九点九``。
    """

    if "." in num_str:
        int_part, frac_part = num_str.split(".", 1)
        int_spoken = _int_to_chinese(int(int_part)) if int_part else "零"
        frac_spoken = _digits_to_chinese(frac_part)
        return f"{int_spoken}点{frac_spoken}"
    return _int_to_chinese(int(num_str))


# ====================================================================== #
# 原子片段登记表
# ====================================================================== #
class _AtomicRegistry:
    """登记原子片段并发放占位符；管线末尾统一展开成最终文本 + 区间。"""

    def __init__(self) -> None:
        self._items: list[tuple[str, str]] = []  # (spoken_text, kind)

    def add(self, spoken: str, kind: str) -> str:
        """登记一个原子片段，返回其占位符字符串。"""

        index = len(self._items)
        self._items.append((spoken, kind))
        return f"{_PH_START}{_encode_ph_index(index)}{_PH_END}"

    def expand(self, text: str) -> tuple[str, list[AtomicSpan]]:
        """把文本中的全部占位符展开成最终文本，并计算原子片段区间。"""

        result: list[str] = []
        spans: list[AtomicSpan] = []
        cursor = 0  # 已写入 result 的字符数（即最终文本中的偏移）
        last = 0
        for match in _PH_RE.finditer(text):
            head = text[last:match.start()]
            result.append(head)
            cursor += len(head)
            spoken, kind = self._items[_decode_ph_index(match.group(1))]
            start = cursor
            result.append(spoken)
            cursor += len(spoken)
            spans.append(AtomicSpan(start=start, end=cursor, kind=kind))
            last = match.end()
        result.append(text[last:])
        merged = "".join(result)
        spans.sort(key=lambda span: span.start)
        return merged, spans


# ====================================================================== #
# 管线规则（每条纯函数；签名统一为 (text, ctx) -> text）
# ====================================================================== #
@dataclass
class _Context:
    """规则间共享的可变状态：选项、登记表、变更记录。"""

    options: PreprocessOptions
    registry: _AtomicRegistry
    changes: list[PreprocessChange]

    def record(self, type_: str, source: str, target: str) -> None:
        self.changes.append(PreprocessChange(type=type_, source=source, target=target))


# 不可见 / 零宽 / 控制字符（保留 \n \t \r，结构化阶段仍需换行）。
_INVISIBLE_RE = re.compile(
    "["
    "\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"   # C0/C1 控制符（保留 \t\n\r）
    "​-‏‪-‮⁠﻿"  # 零宽 / 方向控制 / BOM
    "-"                          # 私有区（含本模块占位符区）
    "]"
)
# 行 / 段分隔符统一成 \n（  行分隔、  段分隔、\r 回车）。
_LINE_SEP_RE = re.compile("[  \r]")


def _clean_invisible_chars(text: str, ctx: _Context) -> str:
    """去除不可见控制字符、零宽字符、私有区字符；统一换行符。

    必须最先执行：既清掉用户可能注入的私有区字符（避免与本模块占位符冲突），
    也为后续结构化保留 ``\\n``。
    """

    text = _LINE_SEP_RE.sub("\n", text)
    return _INVISIBLE_RE.sub("", text)


def _normalize_width_and_punctuation(text: str, ctx: _Context) -> str:
    """全角 → 半角（仅数字 / 字母 / 空格），保留中文标点语义。

    数字 / 字母转半角是为了让后续数字、单位、缩写规则能稳定匹配；标点不做激进转换，
    避免改变中文朗读语气（句末标点的硬断点判定在分片器侧同时认全角与半角）。
    """

    chars: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0xFF10 <= code <= 0xFF19 or 0xFF21 <= code <= 0xFF3A or 0xFF41 <= code <= 0xFF5A:
            # 全角数字 / 大写字母 / 小写字母 → 半角。
            chars.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            chars.append(" ")  # 全角空格 → 普通空格
        else:
            chars.append(ch)
    return "".join(chars)


# Markdown 行内元素
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_BARE_URL_RE = re.compile(r"https?://[^\s，。！？;；]+|www\.[^\s，。！？;；]+")
# 列表符号：``- `` / ``* `` / ``1. `` / ``1、``。有序前缀限定 1~2 位数字且要求其后有
# 空白，避免把小数（如 ``3.14``）误判成有序列表项。
_MD_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]\s+|\d{1,2}[.、]\s+)")
_HARD_PUNCT_TAIL_RE = re.compile(r"[。！？!?；;：:]$")


def _structurize_markdown_and_paragraphs(text: str, ctx: _Context) -> str:
    """Markdown → 可朗读文本，并在抹掉换行前把段落 / 列表边界 token 化（D2 / 6.2）。

    步骤：
    1. 代码块 / 行内代码、链接、裸 URL 先做整体替换（URL 登记为原子占位符）。
    2. 移除标题 ``#`` 前缀。
    3. 按行切块（每个非空行视为一个块），剥离列表符号。
    4. 块间边界一律用自然句末标点（非末块缺句末标点时补「。」），靠标点形成停顿与
       分片边界——不再注入 ``[lbreak]`` 控制 token。
    5. 合并行内多余空白（段落 / 列表边界此时已显式化，换行可安全压缩）。
    """

    # 代码块 → 「代码片段」（不逐字符朗读）。
    if _MD_FENCE_RE.search(text):
        text = _MD_FENCE_RE.sub("代码片段。", text)
        ctx.record("code", "```...```", "代码片段")
    # 行内代码：去掉反引号，保留内容（朗读标识符通常是用户期望）。
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    # 链接：保留「文本」。
    text = _MD_LINK_RE.sub(r"\1", text)
    # 裸 URL → 「链接」占位符（原子片段）。
    def _url_repl(match: re.Match[str]) -> str:
        ctx.record("url", match.group(0), "链接")
        return ctx.registry.add("链接", "url")

    text = _BARE_URL_RE.sub(_url_repl, text)

    # 移除标题前缀。
    text = _MD_HEADING_RE.sub("", text)

    # 按行成块。
    raw_lines = text.split("\n")
    blocks: list[str] = []
    for line in raw_lines:
        stripped = _MD_LIST_MARKER_RE.sub("", line).strip()
        # 合并块内多余空白。
        stripped = re.sub(r"[ \t]+", " ", stripped).strip()
        if stripped:
            blocks.append(stripped)

    if not blocks:
        return ""

    # 块间一律用自然标点：确保非末块以句末标点收尾，靠标点形成停顿与分片边界。
    normalized_blocks: list[str] = []
    for index, block in enumerate(blocks):
        if index != len(blocks) - 1 and not _HARD_PUNCT_TAIL_RE.search(block):
            block = block + "。"
        normalized_blocks.append(block)

    return "".join(normalized_blocks)


def _protect_or_strip_control_tokens(text: str, ctx: _Context) -> str:
    """按白名单保护 / 剥离用户原文中的 ChatTTS 控制 token（6.3）。

    注意：本模块自己注入的 token 已是私有区占位符，不会被本步误伤。
    """

    def _is_whitelisted(token: str) -> bool:
        return token in _WHITELIST_FIXED or bool(_WHITELIST_PARAM_RE.match(token))

    def _repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if ctx.options.allow_control_tokens and _is_whitelisted(token):
            # 保留白名单 token，并登记为原子片段（分片器不得切开）。
            return ctx.registry.add(token, "token")
        # 默认：剥离（防提示注入式控制），或非白名单 token 不透传。
        ctx.record("strip_token", token, "")
        return ""

    return _TOKEN_LIKE_RE.sub(_repl, text)


_DATE_RE = re.compile(r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)")
_TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")


def _read_clock_minute(minute: str) -> str:
    """时钟分钟读法：``30`` → ``三十``；``05`` → ``零五``。"""

    value = int(minute)
    if value == 0:
        return "零"
    if minute.startswith("0") and value < 10:
        return "零" + _DIGITS[value]
    return _int_to_chinese(value)


def _normalize_dates_times(text: str, ctx: _Context) -> str:
    """日期 / 时间读法规范化（仅明确格式，歧义格式保留原样，见 6.4）。"""

    def _date_repl(match: re.Match[str]) -> str:
        year, month, day = match.group(1), int(match.group(2)), int(match.group(3))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return match.group(0)  # 越界 → 保留原样
        spoken = (
            f"{_digits_to_chinese(year)}年"
            f"{_int_to_chinese(month)}月{_int_to_chinese(day)}日"
        )
        ctx.record("date", match.group(0), spoken)
        return ctx.registry.add(spoken, "date")

    def _time_repl(match: re.Match[str]) -> str:
        hour, minute = int(match.group(1)), match.group(2)
        spoken = f"{_int_to_chinese(hour)}点{_read_clock_minute(minute)}分"
        ctx.record("time", match.group(0), spoken)
        return ctx.registry.add(spoken, "time")

    text = _DATE_RE.sub(_date_repl, text)
    text = _TIME_RE.sub(_time_repl, text)
    return text


_NUM = r"\d+(?:\.\d+)?"
_PERCENT_RE = re.compile(rf"({_NUM})\s*%")
_MONEY_RE = re.compile(rf"[¥￥]\s*({_NUM})")
_TEMP_RE = re.compile(rf"({_NUM})\s*°\s*([CF])")
_COMPOUND_UNIT_RE = re.compile(
    rf"(?<![A-Za-z])({_NUM})\s*([A-Za-z]+)/(min|s|h)(?![A-Za-z])"
)
# 单位按「先长后短」排列，避免 km 被 m 抢匹配。
_UNIT_RE = re.compile(
    rf"(?<![A-Za-z])({_NUM})\s*({'|'.join(_UNIT_MAP)})(?![A-Za-z])"
)


def _normalize_money_percent_units(text: str, ctx: _Context) -> str:
    """百分比 / 金额 / 温度 / 复合速率单位 / 常见单位读法规范化（6.4）。"""

    def _percent_repl(match: re.Match[str]) -> str:
        spoken = f"百分之{_read_decimal(match.group(1))}"
        ctx.record("percent", match.group(0), spoken)
        return ctx.registry.add(spoken, "percent")

    def _money_repl(match: re.Match[str]) -> str:
        spoken = f"{_read_decimal(match.group(1))}元"
        ctx.record("money", match.group(0), spoken)
        return ctx.registry.add(spoken, "money")

    def _temp_repl(match: re.Match[str]) -> str:
        unit = "摄氏度" if match.group(2) == "C" else "华氏度"
        spoken = f"{_read_decimal(match.group(1))}{unit}"
        ctx.record("temperature", match.group(0), spoken)
        return ctx.registry.add(spoken, "unit")

    def _compound_repl(match: re.Match[str]) -> str:
        base = _UNIT_MAP.get(match.group(2))
        if base is None:
            return match.group(0)  # 未知基础单位 → 保守保留
        spoken = f"{_read_decimal(match.group(1))}{base}{_RATE_MAP[match.group(3)]}"
        ctx.record("unit", match.group(0), spoken)
        return ctx.registry.add(spoken, "unit")

    def _unit_repl(match: re.Match[str]) -> str:
        spoken = f"{_read_decimal(match.group(1))}{_UNIT_MAP[match.group(2)]}"
        ctx.record("unit", match.group(0), spoken)
        return ctx.registry.add(spoken, "unit")

    text = _PERCENT_RE.sub(_percent_repl, text)
    text = _MONEY_RE.sub(_money_repl, text)
    text = _TEMP_RE.sub(_temp_repl, text)
    text = _COMPOUND_UNIT_RE.sub(_compound_repl, text)  # 复合必须先于简单单位
    text = _UNIT_RE.sub(_unit_repl, text)
    return text


_VERSION_RE = re.compile(r"(?<![A-Za-z])[vV](\d+(?:\.\d+)+)")
_GROUPED_NUM_RE = re.compile(r"(?<!\d)\d{2,}(?:-\d{2,})+(?!\d)")
_PLAIN_NUM_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?")


def _normalize_numbers(text: str, ctx: _Context) -> str:
    """数字读法规范化，含 6.4 的「基数 vs 逐位」阈值判定。"""

    # 版本号：v1.2.3 → v 一点二点三（各段基数，点连接，保留可懂形式）。
    def _version_repl(match: re.Match[str]) -> str:
        parts = match.group(1).split(".")
        spoken = "v" + "点".join(_int_to_chinese(int(p)) for p in parts)
        ctx.record("version", match.group(0), spoken)
        return ctx.registry.add(spoken, "version")

    text = _VERSION_RE.sub(_version_repl, text)

    # 分组编号（含分隔符）：400-800-1234 → 逐位读。
    def _grouped_repl(match: re.Match[str]) -> str:
        digits = match.group(0).replace("-", "")
        spoken = _digits_to_chinese(digits)
        ctx.record("number", match.group(0), spoken)
        return ctx.registry.add(spoken, "longnum")

    text = _GROUPED_NUM_RE.sub(_grouped_repl, text)

    # 普通数字：按阈值判定基数 / 逐位。
    def _plain_repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        start = match.start()
        preceding = text[max(0, start - 8):start]
        has_keyword = any(kw in preceding for kw in _DIGIT_BY_DIGIT_KEYWORDS)
        if "." in raw:
            # 小数一律按十进制读（小数极少作为编号 / 电话）。
            spoken = _read_decimal(raw)
            kind = "number"
        else:
            digit_by_digit = (
                len(raw) >= 7            # 条件 1：位数 ≥ 7
                or (len(raw) > 1 and raw[0] == "0")  # 条件 2：前导零
                or has_keyword           # 条件 4：上下文关键词
            )
            if digit_by_digit:
                spoken = _digits_to_chinese(raw)
                kind = "longnum"
            else:
                spoken = _int_to_chinese(int(raw))
                kind = "number"
        ctx.record("number", raw, spoken)
        return ctx.registry.add(spoken, kind)

    text = _PLAIN_NUM_RE.sub(_plain_repl, text)
    return text


def _normalize_english_acronyms(text: str, ctx: _Context) -> str:
    """英文缩写 / 工程词表混读处理（6.5）。"""

    # 工程整词映射优先（如 ChatTTS → Chat T T S）。
    for word, spoken in _WORD_MAP.items():
        if word in text:
            pattern = re.compile(re.escape(word))

            def _word_repl(match: re.Match[str], spoken=spoken, word=word) -> str:
                ctx.record("acronym", word, spoken)
                return ctx.registry.add(spoken, "acronym")

            text = pattern.sub(_word_repl, text)

    # 技术缩写：拆成单字母（A P I），降低连读失败。
    acronym_re = re.compile(rf"(?<![A-Za-z])({'|'.join(_ACRONYMS)})(?![A-Za-z])")

    def _acronym_repl(match: re.Match[str]) -> str:
        word = match.group(1)
        spoken = " ".join(word)
        ctx.record("acronym", word, spoken)
        return ctx.registry.add(spoken, "acronym")

    return acronym_re.sub(_acronym_repl, text)


# 连续标点压缩（保留省略号单独处理）。
_REPEAT_BANG_RE = re.compile(r"[!！]{2,}")
_REPEAT_QUESTION_RE = re.compile(r"[?？]{2,}")
_REPEAT_PERIOD_RE = re.compile(r"。{2,}")
_ELLIPSIS_RE = re.compile(r"\.{3,}|。{3,}|…+")


def _enhance_punctuation_breaks(text: str, ctx: _Context) -> str:
    """标点规范化（不再注入任何文本内停顿 token）。

    标点清理对所有 profile 生效；停顿改为完全依赖文本自身的句末 / 句中标点，由
    ChatTTS 自行决定韵律，本步不再主动插入 ``[uv_break]``/``[lbreak]``。
    """

    # 省略号统一为「……」（先于其它重复标点，避免被错误压缩）。
    text = _ELLIPSIS_RE.sub("……", text)
    # 连续标点压缩：！！！ → ！
    text = _REPEAT_BANG_RE.sub("！", text)
    text = _REPEAT_QUESTION_RE.sub("？", text)
    text = _REPEAT_PERIOD_RE.sub("。", text)
    return text


# 管线规则顺序（对应 7.3）。
_PIPELINE: tuple[Callable[[str, _Context], str], ...] = (
    _clean_invisible_chars,
    _normalize_width_and_punctuation,
    _structurize_markdown_and_paragraphs,
    _protect_or_strip_control_tokens,
    _normalize_dates_times,
    _normalize_money_percent_units,
    _normalize_numbers,
    _normalize_english_acronyms,
    _enhance_punctuation_breaks,
)

# 仅「安全清洗」用到的规则子集（preprocess=false 或整体回退时使用）。
_SAFE_PIPELINE: tuple[Callable[[str, _Context], str], ...] = (
    _clean_invisible_chars,
    _normalize_width_and_punctuation,
)


def _collapse_whitespace(text: str) -> str:
    """合并连续空白并去首尾空白（全流程只在预处理阶段做一次，见 6.8）。"""

    return re.sub(r"\s+", " ", text or "").strip()


def _resolve_refine(options: PreprocessOptions) -> tuple[str | None, str]:
    """决定 D1 路径：返回 ``(refine_prompt, refine_mode)``。"""

    if options.refine:
        prompt = _PROSODY_REFINE_PROMPT.get(
            options.prosody, _PROSODY_REFINE_PROMPT["natural"]
        )
        return prompt, "refine_prompt"
    return None, "token_injection"


def _run_safe_only(text: str, ctx: _Context) -> str:
    """仅跑安全清洗子集（用于 preprocess=false 与整体回退）。"""

    for rule in _SAFE_PIPELINE:
        text = rule(text, ctx)
    return text


def preprocess_text(text: str, options: PreprocessOptions) -> PreprocessResult:
    """文本预处理管线主入口（纯函数）。

    Args:
        text: 用户原始文本。
        options: 预处理选项。

    Returns:
        ``PreprocessResult``：可朗读文本 + 原子片段 + refine_prompt + 变更记录。

    回退策略（D4 / 7.4）：
    - 单条规则抛异常 → 记录规则名（不记全文）、跳过该条、用其输入继续后续规则。
    - 整条管线异常 → 回退到「安全清洗后的原文」，``refine_mode="token_injection"``、
      ``refine_prompt=None``、``changes=[]``，绝不阻断 TTS。
    """

    refine_prompt, refine_mode = _resolve_refine(options)
    registry = _AtomicRegistry()
    ctx = _Context(options=options, registry=registry, changes=[])

    try:
        working = text or ""
        rules = _PIPELINE if options.enabled else _SAFE_PIPELINE
        for rule in rules:
            try:
                working = rule(working, ctx)
            except Exception:  # noqa: BLE001 - 单条规则失败不影响整体
                logger.warning("预处理规则 %s 失败，已跳过该规则。", rule.__name__)
        # 全流程仅在此处合并一次空白（结构标记此时已 token 化，安全）。
        working = _collapse_whitespace(working)
        final_text, atomic_spans = registry.expand(working)
        return PreprocessResult(
            text=final_text,
            atomic_spans=atomic_spans,
            refine_prompt=refine_prompt,
            refine_mode=refine_mode,
            changes=ctx.changes,
        )
    except Exception:  # noqa: BLE001 - 整条管线异常 → 回退安全清洗
        logger.warning("预处理管线整体异常，已回退到安全清洗后的原文。", exc_info=True)
        fallback_ctx = _Context(
            options=options, registry=_AtomicRegistry(), changes=[]
        )
        try:
            safe = _collapse_whitespace(_run_safe_only(text or "", fallback_ctx))
        except Exception:  # noqa: BLE001 - 安全清洗也失败时退回最朴素归一化
            safe = _collapse_whitespace(text or "")
        return PreprocessResult(
            text=safe,
            atomic_spans=[],
            refine_prompt=None,
            refine_mode="token_injection",
            changes=[],
        )
