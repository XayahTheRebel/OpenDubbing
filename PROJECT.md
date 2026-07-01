# OpenDubbing V1 开发任务

## 项目目标

从零开始开发 OpenDubbing。

OpenDubbing 必须采用全新的模块化架构。

目标：

建立一个长期维护的 AI 视频配音平台。

当前支持：

- 离线视频翻译
- AI 视频配音
- 多语言配音
- 声音克隆
- AI 对口型

未来支持：

- 实时直播翻译
- Hallo-Live
- WebRTC
- OBS
- 数字人
- 多语言直播
- 插件市场

整个架构必须从第一天开始考虑扩展性。

---

# 架构原则

遵循：

```
Application
  ↓
Pipeline
  ↓
Engine
  ↓
Provider
  ↓
Model
```

禁止：

- GUI 调模型
- Engine 调 Engine
- Provider 调 Provider
- 业务代码依赖模型

整个项目必须：

- 高内聚
- 低耦合
- 插件化
- 依赖注入
- 接口编程

---

# 开发原则

OpenDubbing 不允许按照模型开发。

例如：

禁止：

- CosyVoice2Engine
- Hallo3Engine
- QwenEngine

正确：

- TTSEngine
- FaceAnimationEngine
- ASREngine

模型只是 Provider。

以后任何模型都可以替换。

---

# Engine

需要实现：

- Input Engine
- ASR Engine
- Timeline Engine
- Translation Engine
- Length Optimizer
- Prosody Engine
- TTS Engine
- Audio Post Processor
- Face Animation Engine
- Video Post Processor
- Output Engine
- Pipeline Orchestrator
- Workspace Manager
- Provider Registry

---

# 第一阶段默认模型

| 能力 | 默认模型 |
|---|---|
| Audio Separation | Demucs |
| Noise Reduction | DeepFilterNet3 |
| VAD | Silero VAD |
| ASR | Qwen3-ASR |
| Forced Alignment | Qwen3-ForcedAligner |
| Translation | NLLB / GPT / Claude / Gemini |
| TTS | CosyVoice2 |
| Face Animation | Hallo3 |

---

# Timeline

Timeline 是整个项目唯一时间轴。

所有 Engine：

- 读取 Timeline
- 更新 Timeline

Timeline 包含：

- Sentence
- Word
- Phoneme
- Start
- End
- Duration
- Pause
- Speech Rate
- Emotion
- Confidence

---

# Workspace

所有中间结果必须写入 Workspace。

```
workspace/
  input/
  cache/
  timeline/
  translation/
  tts/
  face/
  output/
```

禁止：

- Engine 之间传递对象。

---

# Provider

所有模型必须 Provider 化。

例如：

```
tts/
  cosyvoice2.py
  f5tts.py
translation/
  gpt.py
  claude.py
  gemini.py
  nllb.py
face/
  hallo3.py
  hallolive.py
```

Provider 必须实现统一接口。

---

# Engine Interface

所有 Engine：

- `initialize()`
- `process()`
- `save()`
- `release()`

---

# Provider Interface

所有 Provider：

- `initialize()`
- `load_model()`
- `infer()`
- `release()`

---

# Pipeline

Pipeline 是整个系统唯一调度中心。

负责：

- 执行流程
- 错误恢复
- 缓存
- 断点续跑
- 日志

禁止：

- Pipeline 内部出现模型推理代码。

---

# 后续开发路线

```
V1  离线视频翻译
  ↓
V2  批量翻译
  ↓
V3  直播翻译
  ↓
V4  数字人
  ↓
V5  插件生态
```

整个架构必须保证：

**未来升级无需重构。**
