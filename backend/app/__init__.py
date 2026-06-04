"""ChatTTS 中文 TTS 后端应用包。

本包是对 `2noise/ChatTTS` 模型的服务化封装，工程范式对标姊妹项目
``kokoro-zh-tts``：单进程、单端口、关注点分离、Docker 优先。
不再 vendoring 模型源码，而是直接依赖官方 ``ChatTTS`` PyPI 包作为推理引擎。
"""
