# TTS 文本处理与流式分片一体化优化需求

## 0. 文档说明

本文档将「文本预处理」与「流式分片优化」作为**一条有依赖关系的处理管线**统一设计。原因是：分片必须发生在预处理之后，且分片要能识别预处理插入的 ChatTTS token、展开后的数字/单位、以及由段落/列表转换出的停顿标记——把两者分开会导致 token 被切断、段落边界丢失、普通接口与流式接口结果不一致等问题。因此本期把它们作为**同一个工作项**一起落地。

---

## 1. 背景与目标

### 1.1 背景

当前工程已把 ChatTTS 封装成可用的中文语音合成服务，具备文本归一化、长文本切分、speaker seed、采样参数、refine 开关、语速调节、段间静音和 SSE 流式输出等基础能力。但用户输入进入 ChatTTS 前基本保持原样，仅做空白合并和按标点切分（见 `ChatTTSEngine._normalize_text` 与 `split_text`）。

这会让数字、金额、日期、英文缩写、URL、过密标点、长句、列表、LLM 输出残留 Markdown 等输入在听感上不稳定。ChatTTS 又是面向对话的生成式模型，输入文本若缺少适合朗读的结构，模型能力很难稳定释放。

本需求希望通过工程手段，在文本进入 ChatTTS 前做**轻量、可解释、可回退**的预处理，并**同步**重构分片策略，使两件事一起达成：让用户更快听到第一段声音，且整体效果更清楚、自然、稳定。

### 1.2 用户目标

用户提交一段文本后，即使包含数字、符号、列表、换行、英文缩写、长句或轻微格式噪声，也能得到更自然、清楚、稳定的语音；并且可以**在生成前一键预览**文本将被如何朗读。

### 1.3 工程目标

- 新增可组合的纯函数文本预处理管线，输出"可朗读文本 + 原子片段标注 + refine_prompt + 变更记录"。
- 把分片器重构为 **token/原子-aware** 且支持流式**双阈值**，分片依据是预处理后的文本。
- 普通接口与流式接口共用同一套预处理与分片语义，仅长度阈值不同。
- 默认策略保守、不改变语义、不制造夸张效果、不破坏用户合法 token。
- 预处理结果可观测、可预览、可回退；预处理失败不阻断 TTS。
- 不显著增加首包延迟与整体合成耗时。

### 1.4 效果目标

- 提升数字、日期、金额、百分比、单位的朗读准确率。
- 降低长句卡顿、吞字、异常停顿、读出 Markdown 符号的概率。
- 提升列表、公告、客服、旁白、短对话等文本的停顿自然度。
- 通过可控的 token 注入释放 ChatTTS 的口语感与停顿能力，而非依赖用户手写 token。

---

## 2. 现状与约束

### 2.1 现有代码关键事实（实现时必须基于这些事实）

| 事实 | 位置 | 对本需求的影响 |
|---|---|---|
| `_normalize_text` 把所有空白（含 `\n`）压成单空格 | `tts_engine.py` `_normalize_text` | 段落/换行信息会被抹掉，**必须在预处理阶段先把段落/列表边界转成显式 token 或标点**，不能依赖换行存活 |
| `split_text` 入口**再次** `re.sub(r"\s+", " ")` | `tts_engine.py` `split_text` 开头 | 即使预处理保留了换行，进入 split_text 仍会被二次抹掉，**必须改造该入口** |
| `_hard_wrap` 按固定字符数硬切 | `tts_engine.py` `_hard_wrap` | 这是切断 `[uv_break]`、日期、单位的真正风险点，**token/原子-aware 的修复重点在这里**，而非分句正则 |
| `split_text` 全程只用单一 `max_chars`，`_merge_short_segments` 贪婪等长合并 | `tts_engine.py` | 流式"首片小、后续片大"的双阈值是**结构性改造**，需贯穿 `plan_segments → split_text → merge`，不是加两个常量 |
| refine prompt 仅在 `refine=true`（`skip_refine_text=False`）时生效 | `tts_engine.py` `synthesize_segment` / `_infer_segment` | `refine` 默认 `false`，因此 prosody 若只映射 refine prompt，**默认配置下完全失效**，详见 [6.7](#67-韵律与-refine互斥策略) |
| `refine_text` 是会**重写文本**的 GPT | ChatTTS | 它会自行插入/改写 token，与预处理手工注入的 `[uv_break]` 冲突，二者**不能叠加** |
| 普通接口与流式接口共用 `plan_segments` | `tts_engine.py` / `main.py` | 这是"普通/流式预处理结果一致"几乎零成本的红利，应保持 |
| 文本内 token `[uv_break]/[lbreak]/[laugh]` 在 `skip_refine_text=True` 下也生效 | ChatTTS | 这是本期**可靠、可控、可观测**的韵律主力杠杆 |
| `TTSRequest` 现有字段：text/speaker/speed/refine/temperature/top_p/top_k/format/max_segment_chars | `schemas.py` | 新增字段需在此扩展 |
| `inference_gate` 并发门控保护显存 | `concurrency.py` / `main.py` | 纯文本预处理接口**不应**占用推理槽位 |

### 2.2 ChatTTS 可利用能力

- `InferCodeParams`：`spk_emb`、`temperature`、`top_P`、`top_K`。
- `RefineTextParams(prompt=...)`：`[oral_0~9]`、`[laugh_0~2]`、`[break_0~7]` 做句子级控制（仅 refine 开启时生效，且会重写文本）。
- 文本内 token：`[uv_break]`、`[laugh]`、`[lbreak]` 做局部停顿/笑声（refine 开关无关均可生效）。
- 当前公开模型可靠的 token 仅限 `[laugh]`、`[uv_break]`、`[lbreak]`；更丰富的情绪控制不作为本期能力。

参考：
- ChatTTS 官方仓库：https://github.com/2noise/ChatTTS
- ChatTTS README 高级用法：https://github.com/2noise/ChatTTS/blob/main/README.md
- NVIDIA NeMo Text Normalization：https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/nlp/text_normalization/nn_text_normalization.html

### 2.3 取向

不建设大型通用 TN 系统、不接入 LLM 改写、不做情感识别。优先规则驱动、低延迟、可测试的中文场景增强，保留人工 token 的受控入口。

---

## 3. 核心设计决策

这些决策贯穿全文。

### D1 韵律实现以「文本内 token 注入」为主力，refine prompt 与之互斥

> **修订（最新决策）**：取消「自动注入 `[uv_break]`/`[lbreak]` 等控制 token」的逻辑。
> `refine=false` 路径**不再注入任何手工停顿/结构 token**，停顿完全交由文本自身的标点与
> ChatTTS 决定；段落/列表边界改为只用自然句末标点显式化（见 D2）。`refine=true` 路径维持
> 不变（映射 `refine_prompt`）。用户原文中自带、且在 `allow_control_tokens` 打开时命中白名单
> 的 token 仍会被保留为原子片段。下文中关于「按 prosody/profile 注入 token」的描述均作废，
> `prosody`/`profile` 现仅用于分片长度调优与 `refine_prompt` 映射。`refine_mode` 字段仍保留
> `token_injection`/`refine_prompt` 两值以兼容接口，其中 `token_injection` 现表示「仅规范化、
> 不注入」。

（以下为历史设计，已被上面的修订覆盖：）

因为文本内 token 在 `refine=false` 下也生效、可预览、可解释，而 refine prompt 只在 `refine=true` 下生效且会重写文本吃掉手工 token，两者**永不叠加**：

- **`refine=false`（默认）**：~~由预处理按 `prosody`/`profile` 在文本中注入受控的 `[uv_break]`/`[lbreak]`~~（已取消）；`refine_prompt` 为 `None`，不走 ChatTTS refine。
- **`refine=true`（用户显式开启）**：预处理**不注入**文本内停顿 token（避免被 refine 重写打架），改为把 `prosody` 映射成 `refine_prompt`（如 `[oral_2][laugh_0][break_4]`）交给 ChatTTS，由 refine 全权负责停顿。

无论哪条路径，`changes` 记录采用了哪种策略，预览接口回显最终 `refine_prompt`。

### D2 预处理与分片是一条管线，段落边界必须先 token 化

预处理在 split_text 抹掉换行**之前**，把段落/列表/对话轮次边界转成显式 `[lbreak]` 或自然标点；分片器再基于这些显式标记切分。两者必须一起改造。

### D3 分片器是 token/原子-aware 的，普通/流式只差长度阈值

预处理输出"原子片段列表"（token、日期、时间、金额、百分比、单位、英文缩写、URL 占位符、长编号），分片器在任何层级（含硬切）都不得切开原子片段。普通接口与流式接口跑**完全相同**的预处理与原子保护，只有目标长度阈值不同。

### D4 预处理幂等且可回退

预处理应尽量幂等（对已处理文本再次处理不应二次破坏）。单条规则失败 → 跳过该规则继续；整条管线异常 → 回退到"安全清洗后的原文"，绝不阻断 TTS。

### D5 预览接口开放给前端，但 TTS 仍服务端重处理

前端"一键整理"调用 `/api/tts/preprocess` 仅用于**预览展示**，不改变请求真值。提交 TTS 时仍发送用户原始文本，由服务端再次完整预处理，保证"所见即所得"且行为单一可靠。

---

## 4. 非目标

- 不做训练、微调、音色克隆或情绪模型。
- 不引入 LLM 作为默认预处理步骤。
- 不追求覆盖所有语言、所有单位、所有行业符号。
- 不做 SSML 完整兼容。
- 不保证所有 ChatTTS token 稳定生效；只支持官方公开、当前模型明确可用的 token。
- 不改变当前音频编码、speaker seed、SSE 事件协议的基本形态。

---

## 5. 总体架构与数据流

```text
原始输入
  │
  ▼  preprocess_text(text, options)        ← backend/app/text_preprocess.py（新增，纯函数管线）
  ├─ 安全清洗（不可见字符、空白、全半角标点）
  ├─ 结构化（Markdown → 可朗读文本；段落/列表边界 → 显式 [lbreak]/标点）   ★ 必须在抹掉换行前完成
  ├─ 控制 token 保护（白名单校验 / 转义 / 还原）
  ├─ 数字 / 日期 / 时间 / 金额 / 百分比 / 单位 读法规范化（标注为原子片段）
  ├─ 英文缩写 / 混读处理（标注为原子片段）
  ├─ 标点与停顿增强（按 D1 注入文本内 token，或留给 refine）
  └─ 产出 PreprocessResult{ text, atomic_spans, refine_prompt, changes }
  │
  ▼  segment_text(result, first_chars, rest_chars)   ← 由现有 split_text 重构而来，token/原子-aware
  ├─ 句末硬断点 → 软断点 → 受约束硬切（不切开 atomic_spans / token）
  ├─ 流式：首片用 first_chars（小），后续片用 rest_chars（大）
  └─ 普通：统一用较大阈值
  │
  ▼  ChatTTS infer（逐段；refine 按 D1 决定是否启用）
```

集成点：现有 `ChatTTSEngine.plan_segments()` 内部由
`_normalize_text(text) → split_text()`
改为
`preprocess_text(text, options) → segment_text(result, ...)`。
`synthesize()` / `synthesize_segment()` 增加接收 `refine_prompt`。

---

## 6. 功能需求

### 6.1 请求参数与默认值

`TTSRequest` 新增字段（普通接口 `/api/tts` 与流式接口 `/api/tts/stream` **都生效**；分片相关字段仅流式生效）：

```jsonc
{
  "preprocess": true,
  "preprocess_profile": "balanced",
  "prosody": "natural",
  "allow_control_tokens": false,
  "first_segment_chars": null,   // 仅流式：首片目标上限，缺省走服务端默认
  "max_segment_chars": null      // 语义变更见下表，仅流式：后续分片目标上限
}
```

| 字段 | 类型 | 默认 | 作用域 | 说明 |
|---|---|---|---|---|
| `preprocess` | boolean | `true` | 两者 | 是否启用文本预处理 |
| `preprocess_profile` | enum | `balanced` | 两者 | `plain` / `balanced` / `expressive` |
| `prosody` | enum | `natural` | 两者 | `flat` / `natural` / `dialogue` / `narration` |
| `allow_control_tokens` | boolean | `false` | 两者 | 是否允许用户原文 ChatTTS token 生效 |
| `first_segment_chars` | int? | `null` | 流式 | 流式首片目标上限；缺省用 `STREAM_FIRST_SEGMENT_CHARS` |
| `max_segment_chars` | int? | `null` | 流式 | **语义变更**：由"硬切上限"改为"后续分片目标上限"；缺省用 `STREAM_SEGMENT_CHARS` |

> `max_segment_chars` 语义变更属于行为变更。老客户端仍可传该字段，仅含义从"硬切点"调整为"目标上限参考"，分片器允许为保护语义单元略微超出。

profile 行为：

| profile | 行为 |
|---|---|
| `plain` | 只做安全清洗、结构化、空白/标点规范化、读法规范化、基础分段；**不主动注入** ChatTTS 停顿 token |
| `balanced` | 默认。常见读法规范化 + 适度停顿增强（列表/段落/冒号解释处有限注入） |
| `expressive` | 更明显地利用停顿（首片可略短、保留更多 `[uv_break]`），但**仍不主动注入笑声**，并限制 token 密度 |

### 6.2 安全清洗与结构化

必须处理：

- 去除不可见控制字符、零宽字符、无法朗读的私有区字符。
- 合并连续空白；**段落/列表/对话轮次边界在抹掉换行前转成显式 `[lbreak]` 或自然标点**（落实 D2）。
- 标准化全角/半角标点（`，。！？；：` 等）。
- Markdown 转可朗读文本：
  - 标题前缀 `#` 移除。
  - 列表符号 `-`、`*`、`1.` 转为自然断句 + 项间停顿。
  - 代码块（``` ``` 及行内 `code`）默认压缩为"代码片段"，不逐字符朗读。
  - 链接 `[文本](URL)` 保留"文本"；裸 URL 压缩为"链接"占位符（标注为原子片段）。
- Emoji 默认移除；少量常见情绪 emoji 可选映射为"微笑/大笑"等，本期默认不开启。

验收：
- 输入 Markdown 文案不读出"井号""星号""反引号"。
- 含控制字符文本不导致 ChatTTS 异常或读出奇怪音节。
- 多段落/列表文本，段落与列表项之间有自然停顿（依赖 D2 的显式标记）。

### 6.3 控制 token 保护与白名单

白名单：

```text
[uv_break]  [laugh]  [lbreak]
[oral_0]..[oral_9]   [laugh_0]..[laugh_2]   [break_0]..[break_7]
```

策略：
- `allow_control_tokens=false`（默认）：用户原文中形如 token 的内容一律转义/移除，防提示注入式控制。
- `allow_control_tokens=true`：仅保留白名单内 token，非白名单 token 不透传。
- 系统按 `prosody`/`profile` 注入的 token 必须来自白名单。
- 注入的 token 全部标注为原子片段，分片器不得切开。

验收：
- 用户输入 `[laugh_2]`，默认不改变全局笑声倾向。
- `allow_control_tokens=true` 且 token 合法时，原文 token 保留。
- 非白名单 token 一律不透传。

### 6.4 数字 / 日期 / 金额 / 单位读法规范化

覆盖高频、低歧义场景，转换结果统一标注为原子片段：

| 类型 | 示例 | 期望 spoken form |
|---|---|---|
| 整数（基数） | `123` | `一百二十三` |
| 小数 | `3.14` | `三点一四` |
| 百分比 | `12.5%` | `百分之十二点五` |
| 金额 | `¥99.9`、`￥99.9` | `九十九点九元` |
| 时间 | `14:30` | `十四点三十分` |
| 日期 | `2026-06-04`、`2026/06/04` | `二零二六年六月四日`（年份逐位、月日基数） |
| 温度 | `28.5°C` | `二十八点五摄氏度`（`°F` → 华氏度） |
| 常见单位 | `10kg`、`5km`、`24kHz` | `十千克`、`五公里`、`二十四千赫兹` |
| 复合速率单位 | `12MB/s`、`60km/h` | `十二兆字节每秒`、`六十公里每小时` |
| 电话/编号 | `400-800-1234`、`13800138000` | 逐位读 |
| 版本号 | `v1.2.3` | `v 一点二点三`（保留可懂形式） |

**整数读法判定阈值（消除歧义，必须明确实现）**：

- 满足任一条件 → **逐位读**：
  1. 数字位数 ≥ 7；
  2. 含前导零（如 `007`）；
  3. 含分组分隔符（`-`、空格分组，如 `400-800-1234`）；
  4. 紧邻上下文关键词：`订单号`、`单号`、`编号`、`电话`、`手机`、`卡号`、`验证码`、`快递`、`QQ`、`微信号` 等（词表可扩展）。
- 否则（≤ 6 位且无上述特征）→ **基数读**（`一百二十三`）。

设计原则：
- 无法判断语义时宁可保守、不强行转换。
- 日期/时间只转换明确格式（`YYYY-MM-DD`、`YYYY/MM/DD`、`HH:MM`），歧义格式保留原样。
- 单位表先覆盖：`kg`、`g`、`km`、`m`、`cm`、`mm`、`GB`、`MB`、`KB`、`Hz`、`kHz`、`MHz`、`ms`、`s`、`min`、`h`、`°C`、`°F`、`%`，以及 `X/s`、`X/h` 复合速率。

验收：
- `今天是 2026-06-04，气温 28.5°C。` → 自然中文日期 + 温度。
- `下载速度 12MB/s` → 清晰单位读法，不逐字符含混。
- `订单号 202606041234` → 逐位读，不读成巨大数值。

### 6.5 英文、缩写与混读

- 常见技术缩写保留大写并适当空格降低连读失败：`API`、`HTTP`、`URL`、`GPU`、`CPU`、`TTS`、`AI`（可拆为 `A P I`）。
- 连续英文短词保持原样，不强制翻译。
- 混合词如 `ChatTTS` 按工程词表映射（默认 `Chat T T S`，可配置 `恰特 T T S`）。
- 处理结果标注为原子片段，普通/流式一致。

验收：
- `ChatTTS 支持 API 调用` 不被读成难懂的连续英文串。
- `GPU`、`CPU`、`HTTP` 在普通与流式接口结果一致。

### 6.6 标点与停顿增强

- 句末标点 `。！？` 保留为硬断点。
- 逗号、顿号、分号为软断点。
- 冒号后若是解释/列表，适度插入停顿。
- 连续标点压缩：`！！！` → `！`。
- 省略号统一为 `……`，必要时转较长停顿。
- 列表项、段落之间插入明确停顿（由 D2 的显式标记驱动）。

停顿 token 注入（仅在 D1 的 `refine=false` 路径，且受 profile 限量）：

| 场景 | 建议 |
|---|---|
| 自然短暂停顿 | `[uv_break]` |
| 段落/列表项切换 | `[lbreak]` 或依赖分段静音 |
| 笑声 | 默认不注入 |

验收：
- 长列表不像一整句连续读完。
- 逗号密集文本不因注入过多 token 而碎裂（profile 限量生效）。

### 6.7 韵律与 refine（互斥策略）

落实 [D1](#d1-韵律实现以文本内-token-注入为主力refine-prompt-与之互斥)。`prosody` 与 `refine` 的组合行为：

| `refine` | 行为 | `refine_prompt` 产出 | 文本内 token 注入 |
|---|---|---|---|
| `false`（默认） | 由预处理按 prosody 注入文本内 `[uv_break]`/`[lbreak]`，不走 ChatTTS refine | `None` | 是（受 profile 限量） |
| `true` | 把 prosody 映射成 refine prompt 交给 ChatTTS，refine 全权负责停顿 | 见下表 | **否**（避免被 refine 重写打架） |

`refine=true` 时 prosody → refine prompt 映射：

| prosody | refine prompt | 适用场景 |
|---|---|---|
| `flat` | `[oral_0][laugh_0][break_2]` | 报数、公告、严肃 |
| `natural` | `[oral_2][laugh_0][break_4]` | 默认通用 |
| `dialogue` | `[oral_4][laugh_0][break_5]` | 对话、客服、助手 |
| `narration` | `[oral_2][laugh_0][break_5]` | 长文、旁白、解说 |

`refine=false` 时 prosody → 文本内 token 强度映射（密度随 profile 与 prosody 调节，`plain` profile 不注入）：

| prosody | 停顿倾向 |
|---|---|
| `flat` | 仅在硬断点与段落处少量 `[lbreak]` |
| `natural` | 句间适度 `[uv_break]`，段落 `[lbreak]` |
| `dialogue` | 句间略多 `[uv_break]`，重标点处停顿更明显 |
| `narration` | 长句中部 `[uv_break]`，段落 `[lbreak]` 略长 |

说明：
- 本期不使用 `[laugh_1]`/`[laugh_2]`，避免不合时宜笑声。
- `refine` 字段语义明确为"是否启用 ChatTTS refine_text"。默认 `[oral_2][laugh_0][break_4]` 行为保留，但来源改为 prosody 配置化。
- `changes` 必须记录本次走了哪条路径（token 注入 / refine prompt）及最终 `refine_prompt`。

### 6.8 统一分片策略（token/原子-aware + 双阈值）

落实 [D3](#d3-分片器是-token原子-aware-的普通流式只差长度阈值)。由现有 `split_text` 重构为 `segment_text`，输入为预处理结果（含 `atomic_spans`）。

硬性规则：
- 分片必须发生在预处理之后，依据是预处理后的可朗读文本。
- **任何层级（含硬切）都不得切开 `atomic_spans`**：ChatTTS token、日期、时间、金额、百分比、单位、英文缩写、URL 占位符、长编号。修复重点是现有 `_hard_wrap`。
- 切分优先级：句末硬断点（`。！？；：`）→ 软断点（`，、` 空格）→ 受约束硬切（在不破坏原子片段前提下尽量靠近上限）。
- 列表项、段落、对话轮次边界优先形成独立分片。
- 每个分片尽量包含完整语义，不追求机械等长。
- 修正现有双重空白压缩：`segment_text` 入口**不得**再 `re.sub(r"\s+", " ")` 抹掉由 D2 转换出的结构标记；空白合并只在预处理阶段做一次。

双阈值（流式）：

```text
STREAM_FIRST_SEGMENT_CHARS = 50    # 首片目标，求快
STREAM_SEGMENT_CHARS       = 100   # 后续片目标，求连贯
STREAM_FIRST_HARD_CAP      = 90    # 首片无法在自然断点结束时的硬上限
```

执行逻辑：
1. 首片用 `first_chars`（默认 50），优先在第一个完整短句或自然停顿处结束，让用户尽快出声。
2. 第二片起用 `rest_chars`（默认 100），减少推理次数与段间割裂。
3. 首片无法在自然断点结束时，延长到硬上限（默认 90）。
4. 某自然句超过上限时按软断点切；无软断点才受约束硬切（仍不切原子片段）。
5. `expressive` profile 首片可略短、保留更多停顿；`flat` 可略长、减少段间静音。

普通接口：统一用较大目标阈值（沿用 `MAX_SEGMENT_CHARS`，默认 120），**不使用**双阈值；预处理与原子保护与流式完全一致。

接口兼容：
- `max_segment_chars` 语义改为"后续分片目标上限参考"（见 6.1）。
- 新增 `first_segment_chars` 仅影响流式首片。
- 不传字段时使用服务端默认双阈值。
- 每个分片输出携带原文片段索引，便于日志与调试。

验收：
- 流式首片仍明显快于普通接口返回完整 WAV。
- 同一文本的分片不切断 token、日期、金额、单位、缩写、URL 占位符。
- 分片数不因 token 注入异常膨胀。
- 开启增强后流式段间停顿不明显比普通接口更碎。
- 前端时间线与播放缓冲仍可按分片顺序稳定拼接。
- `[uv_break]` 不被切成 `[uv_` 与 `break]`；`2026-06-04` 不被切开。

### 6.9 预处理预览接口

```http
POST /api/tts/preprocess
```

- 请求体复用 `TTSRequest` 的文本与预处理字段（`text`/`preprocess`/`preprocess_profile`/`prosody`/`allow_control_tokens`，及可选 `first_segment_chars`/`max_segment_chars` 以预览分片）。
- **不经过 `inference_gate` 推理门控**（纯文本处理，不占显存槽位）。
- 响应：

```json
{
  "original_text": "...",
  "normalized_text": "...",
  "refine_prompt": "[oral_2][laugh_0][break_4]",
  "prosody": "natural",
  "profile": "balanced",
  "refine_mode": "token_injection",
  "segments": [
    { "index": 0, "text": "..." },
    { "index": 1, "text": "..." }
  ],
  "changes": [
    { "type": "number", "from": "12.5%", "to": "百分之十二点五" }
  ]
}
```

字段约定：
- `refine_prompt`：D1 决定的最终 prompt，`token_injection` 路径下为 `null`。
- `refine_mode`：`"token_injection"` 或 `"refine_prompt"`，对应 D1 两条路径。
- `changes[].type`/`from`/`to`：序列化字段名统一为 `type`/`from`/`to`（内部 dataclass 用 `type`/`source`/`target`，序列化时映射）。
- `segments`：预览分片（流式默认双阈值；可被请求字段覆盖）。

### 6.10 可观测性与隐私

- 普通合成接口**不**默认返回预处理详情，避免影响音频响应。
- 日志只记录长度、分段数、命中的规则类型、`refine_mode`，**不记录完整文本**。
- 测试环境可开启完整 diff。

---

## 7. 后端实现设计

### 7.1 模块结构

```text
backend/app/
  text_preprocess.py   # 新增：纯函数预处理管线
  tts_engine.py        # 改造：plan_segments 集成、segment_text 重构、refine_prompt 透传
  schemas.py           # 改造：TTSRequest 扩字段、PreprocessResponse 新增
  config.py            # 改造：新增双阈值配置项
  main.py              # 改造：/api/tts/preprocess 路由、两接口透传新字段
```

### 7.2 数据结构

```python
@dataclass(frozen=True)
class PreprocessOptions:
    enabled: bool
    profile: Literal["plain", "balanced", "expressive"]
    prosody: Literal["flat", "natural", "dialogue", "narration"]
    allow_control_tokens: bool
    refine: bool                      # 决定 D1 走哪条路径

@dataclass(frozen=True)
class PreprocessChange:
    type: str
    source: str                       # 序列化为 "from"
    target: str                       # 序列化为 "to"

@dataclass(frozen=True)
class AtomicSpan:
    start: int                        # 在 text 中的字符区间，分片器据此保护
    end: int
    kind: str                         # token / date / time / money / percent / unit / acronym / url / longnum

@dataclass(frozen=True)
class PreprocessResult:
    text: str                         # 可朗读文本（含显式 [lbreak] 等结构标记）
    atomic_spans: list[AtomicSpan]
    refine_prompt: str | None
    refine_mode: Literal["token_injection", "refine_prompt"]
    changes: list[PreprocessChange]

def preprocess_text(text: str, options: PreprocessOptions) -> PreprocessResult:
    ...
```

### 7.3 管线规则（每条纯函数，可单独禁用排查）

```text
clean_invisible_chars
normalize_width_and_punctuation
structurize_markdown_and_paragraphs   # ★ 在抹掉换行前把段落/列表 → [lbreak]/标点
protect_or_strip_control_tokens
normalize_dates_times
normalize_money_percent_units
normalize_numbers                     # 含 6.4 整数/逐位阈值
normalize_english_acronyms
enhance_punctuation_breaks            # 按 D1 注入 token 或留给 refine
collect_atomic_spans                  # 汇总原子片段，供分片器
```

### 7.4 回退策略（落实 D4）

- 单条规则抛异常 → 记录规则名（不记全文）、跳过该规则、用其输入继续后续规则。
- 整条管线异常 → 回退到仅"安全清洗后的原文"，`refine_mode="token_injection"`、`refine_prompt=None`、`changes=[]`，不阻断 TTS。
- 预处理应尽量幂等：对已规范化文本再次处理不应二次破坏（用于"应用预览结果后再提交"的极端场景）。

### 7.5 与引擎集成

`plan_segments()`：
```text
旧：_normalize_text(text) → split_text(max_chars)
新：preprocess_text(text, options) → segment_text(result, first_chars, rest_chars)
```
- `preprocess=false` 时，`preprocess_text` 内部退化为仅安全清洗（等价旧 `_normalize_text` 但保留原子保护框架）。
- 普通接口：`first_chars=rest_chars=MAX_SEGMENT_CHARS`（即不启用双阈值）。
- 流式接口：`first_chars=first_segment_chars or STREAM_FIRST_SEGMENT_CHARS`，`rest_chars=max_segment_chars or STREAM_SEGMENT_CHARS`。

`synthesize()` / `synthesize_segment()`：
- 接收 `refine_prompt`；`refine=true` 时用它构造 `RefineTextParams`，否则 `skip_refine_text=True`。
- `_build_refine_params` 改为接收 prompt 参数，默认值保留 `[oral_2][laugh_0][break_4]`。

### 7.6 配置项（config.py）

```text
STREAM_FIRST_SEGMENT_CHARS = 50
STREAM_SEGMENT_CHARS       = 100   # 取代/重定义原 STREAM_MAX_SEGMENT_CHARS 语义
STREAM_FIRST_HARD_CAP      = 90
```
保留向后兼容：若仅设置旧 `STREAM_MAX_SEGMENT_CHARS`，将其作为 `STREAM_SEGMENT_CHARS` 读取并告警。

### 7.7 API 兼容

现有请求不传新字段时默认：`preprocess=true`、`preprocess_profile=balanced`、`prosody=natural`、`allow_control_tokens=false`、`refine=false`。规则保守，因此默认开启 `balanced` 对存量请求是安全的。如担心上线风险，可在前端提供"文本增强"开关由用户控制（默认开）。

---

## 8. 前端需求

### 8.1 高级参数区（折叠区，现有 `advanced` 区域扩展）

- "文本增强"开关：默认开启，绑定 `preprocess`。
- "增强强度"分段控件：简洁 / 均衡 / 表现力 → `plain` / `balanced` / `expressive`。
- "朗读风格"下拉/分段控件：平实 / 自然 / 对话 / 旁白 → `flat` / `natural` / `dialogue` / `narration`。
- "允许控制 token"开关：默认关闭，附 tooltip 提醒仅适合高级用户。
- 现有"文本 refine"开关保留，tooltip 说明其与朗读风格的关系（D1）。

### 8.2 「一键整理」预览按钮（落实 D5）

- 文本输入区附近放置"一键整理 / 预览处理结果"按钮。
- 点击调用 `POST /api/tts/preprocess`，把当前文本与高级参数发过去。
- 在只读预览面板展示 `normalized_text`、分片列表（按 `index` 顺序）、`changes`（`type: from → to` 列表）、以及 `refine_mode`/`refine_prompt`。
- **默认不改写输入框**：输入框保留用户原文，提交 TTS 时仍发送原文，由服务端再次完整预处理（D5）。
- 可选提供"应用到输入框"次级操作；若用户应用，需提示"再次提交将基于已处理文本"，依赖 7.4 的幂等性保证不二次破坏。
- 预览请求与 TTS 请求互不阻塞；生成进行中禁用该按钮，避免状态错乱（与现有"生成中禁用交互"一致）。

### 8.3 请求构造

`requestBody` 中：普通与流式都带上 `preprocess`/`preprocess_profile`/`prosody`/`allow_control_tokens`；流式额外带 `first_segment_chars`/`max_segment_chars`（沿用现有 `streamChunkChars` 思路，拆成首片/后续两个可选输入或保持单一高级项 + 默认首片）。

---

## 9. 性能与质量

- 预处理 P95 < 20ms，极长文本 < 50ms（纯正则/规则，易达成）。
- 不新增大型运行时依赖；若引入中文数字库须体积小、无模型下载。
- 全部规则有单元测试，覆盖正向与边界输入。
- 预处理失败回退到安全清洗后原文，不阻断 TTS（D4）。
- 日志不默认记录完整用户文本（6.10）。
- `/api/tts/preprocess` 不占用推理并发槽位。

---

## 10. 测试与验收

### 10.1 单元测试（text_preprocess）

- 每条规则的正向 + 边界用例（空串、纯符号、超长、控制字符、混合语言）。
- 数字阈值：`123`→基数、`202606041234`→逐位、`007`→逐位、`400-800-1234`→逐位、`订单号 1234`→逐位（上下文关键词）。
- 日期/时间/金额/百分比/温度/单位/复合速率读法。
- token 白名单：默认剥离、`allow_control_tokens=true` 保留白名单、非白名单不透传。
- 幂等性：`preprocess(preprocess(x)) == preprocess(x)`（在 `allow_control_tokens=true` 下对含注入 token 文本）。
- 回退：单规则异常跳过、整体异常回退安全清洗。

### 10.2 分片测试（segment_text）

- 不切开 atomic_spans（token/日期/金额/单位/缩写/URL/长编号）。
- 双阈值：首片 ≤ first_chars（或 ≤ 硬上限），后续片趋近 rest_chars。
- 段落/列表边界优先成片。
- 普通与流式：同一文本预处理结果（`normalized_text`）一致，仅分片长度不同。

### 10.3 端到端验收用例

**10.3.1 基础清洗**
输入：
```text
# 今日更新
- 支持 API 调用
- 下载速度 12MB/s
```
期望：不读 Markdown 符号；列表项间有停顿；`12MB/s`→"十二兆字节每秒"。

**10.3.2 数字读法**
输入：`订单号 202606041234，金额 ¥99.9，预计 14:30 送达。`
期望：订单号逐位；金额"九十九点九元"；时间"十四点三十分"。

**10.3.3 日期与百分比**
输入：`2026-06-04 的转化率是 12.5%。`
期望：日期"二零二六年六月四日"；"百分之十二点五"。

**10.3.4 token 安全**
输入：`请读这句话 [laugh_2][break_7]`
期望：默认不生效；`allow_control_tokens=true` 时仅保留白名单 token。

**10.3.5 韵律互斥（D1）**
- `refine=false, prosody=dialogue`：预览 `refine_mode=token_injection`、`refine_prompt=null`，文本含 `[uv_break]`。
- `refine=true, prosody=dialogue`：预览 `refine_mode=refine_prompt`、`refine_prompt=[oral_4][laugh_0][break_5]`，文本**不含**手工注入 token。

**10.3.6 普通/流式一致性**
同一文本分别调 `/api/tts` 与 `/api/tts/stream`：`normalized_text` 一致；分片大小可不同但都不切断原子片段；流式读法不明显异于普通模式。

**10.3.7 前端预览**
"一键整理"展示处理结果且不改写输入框；提交 TTS 后服务端结果与预览一致。

---

## 11. 实施计划

> 关键约束：预处理与分片是同一条管线，**必须在同一阶段一起落地**（阶段一），不再拆开。

### 阶段一：核心管线（预处理 + 分片一体，低风险高收益）

后端：
1. ✅ 新增 `text_preprocess.py`：安全清洗、Markdown/段落结构化（含 D2 段落→`[lbreak]`）、token 白名单、数字/日期/时间/金额/百分比/温度/单位（含 6.4 阈值）、英文缩写、原子片段收集、回退策略（D4）。本阶段停顿增强仅做"基础注入 + plain 不注入"，复杂 prosody 强度放阶段二。
2. ✅ 重构 `split_text → segment_text`：token/原子-aware（修 `_hard_wrap`），修正双重空白压缩（D2/6.8），支持首片/后续双阈值。
3. ✅ `plan_segments` 集成 `preprocess_text → segment_text`；`config.py` 加双阈值配置（含旧变量兼容）。
4. ✅ `schemas.py` 扩 `TTSRequest` 字段（两接口生效），新增 `PreprocessResponse`。
5. ✅ `main.py` 新增 `POST /api/tts/preprocess`（不走推理门控），两接口透传新字段。
6. ✅ 单测覆盖 10.1 / 10.2，CI 跑通。

验收：10.3.1 / 10.3.2 / 10.3.3 / 10.3.4 / 10.3.6 通过；普通/流式一致；无 token/原子被切断。

### 阶段二：韵律实现与前端接入

后端：
1. ✅ 实现 D1 互斥策略：`refine=false` 注入文本内 token（按 profile/prosody 限量密度）、`refine=true` 映射 refine prompt；`synthesize*` 透传 `refine_prompt`，`_build_refine_params` 参数化。
2. ✅ 列表/段落/冒号解释句的有限 `[uv_break]`/`[lbreak]` 注入与 expressive profile 调优。
3. ✅ 预览接口补 `refine_mode`/`refine_prompt`。

前端：
4. ✅ 高级参数区接入"文本增强 / 增强强度 / 朗读风格 / 允许控制 token"（8.1）。
5. ✅ "一键整理"预览按钮 + 只读预览面板（8.2）；请求构造带新字段（8.3）。

验收：10.3.5 / 10.3.7 通过。

### 阶段三：细化与评测

1. ✅ 扩充单位表、技术词表、上下文关键词表（`text_preprocess.py` 中 `_UNIT_MAP` /
   `_RATE_MAP` / `_ACRONYMS` / `_WORD_MAP` / `_DIGIT_BY_DIGIT_KEYWORDS`，已结构化、可持续扩充）。
2. ⬜ 建人工试听用例集，按真实失败样本迭代规则（评测/运营类，待真实样本到位后迭代）。
3. ✅ 记录规则命中统计与 `refine_mode` 分布（不记全文）：`plan_segments` 已按 6.10 输出
   `chars_in/chars_out/segments/refine_mode/rules` 计数日志。
4. ⬜ 评估是否引入小型中文数字库替换自研规则（当前自研规则已覆盖高频场景，按需评估）。

---

## 12. 风险与取舍

| 风险 | 说明 | 应对 |
|---|---|---|
| refine 重写吃掉手工 token | refine_text 会改写文本 | D1 两条路径互斥，永不叠加；预览回显 `refine_mode` |
| 段落边界丢失 | 现有代码两处抹掉换行 | D2 在抹掉前 token 化；segment_text 入口去掉二次空白压缩 |
| 双阈值改造范围被低估 | split_text/merge 是单阈值贪婪合并 | 作为结构性改造在阶段一一次性完成并加分片测试 |
| 数字读法歧义 | 基数 vs 逐位 | 6.4 明确阈值规则，无法判断时保守保留 |
| 过度规范化 | 改变用户本意 | 默认保守，提供 `plain` 模式 + 预览接口 |
| token 过多导致碎裂 | 停顿过密或读出 token | profile 限量注入，按 prosody 控制密度 |
| 预览与实际不一致 | 前端展示与服务端结果偏差 | D5：TTS 服务端重处理 + 幂等性保证 |
| 长文本风格漂移 | 分段多导致轻微漂移 | 固定 speaker seed、统一 refine 来源、合并短段 |

---

## 13. 成功标准

- 用户默认输入普通中文、LLM 回复、列表文案、含数字文案时，语音可懂度明显提升。
- ChatTTS 的口语感与停顿以可控方式释放，不依赖用户手写 token。
- 预处理与分片是一条可测试、可逐步扩展的统一管线。
- 现有 API 不传新字段仍正常工作。
- 普通生成与流式生成保持同一套预处理与分片语义。
- 用户可在生成前一键预览处理结果，所见即所得。
