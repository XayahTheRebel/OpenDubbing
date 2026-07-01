# OpenDubbing V1 详细实现执行文档

> **本文件用途**：本文档是写给负责写代码的执行模型的"施工图纸"。骨架与接口已全部定义好，本文档只负责把每个 Engine / Provider 从"占位"变成"能真正跑出一条配音"。
>
> **执行模型阅读须知**：
> - **不要新建文件、不要改接口签名**。所有文件路径都已存在，逐一替换内容即可。
> - **严格遵守分层**：Engine 只调 Provider 接口，不 import 任何具体模型库；Provider 才 import torch/transformers/sdk。Pipeline 不出现任何模型推理。
> - **每个任务都有验收命令**，做完一个跑一个，全绿再进下一个。
> - 禁止跨任务跳读。文档自包含一切你需要的签名、路径、库版本、输入输出契约。

---

## 0. 项目现状（执行前必读）

骨架已完成并全绿：
- `pytest` 34 passed / `ruff` passed / `pyright` 0 errors
- 11 个 Engine 全部是占位（`process()` 原样返回 Timeline，带 `# TODO`）
- 11 个 Provider 全部是占位（`load_model()`/`infer()` 抛 `NotImplementedError`）
- `utils/media.py` 是空文件
- 前端骨架已就位，本批次不动前端

**本批次目标是：打通一条真实可跑的离线配音链路。**

### 0.1 已存在且禁止修改的契约

#### Timeline 结构（`src/opendubbing/core/timeline.py`，已实现，禁改）

```python
@dataclass
class Phoneme:
    symbol: str = ""
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0

@dataclass
class Word:
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    pause: float = 0.0
    speech_rate: float = 1.0
    confidence: float = 1.0
    phonemes: list[Phoneme] = field(default_factory=list)

@dataclass
class Sentence:
    id: str = ""
    text: str = ""
    translation: str = ""
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    pause: float = 0.0
    speech_rate: float = 1.0
    emotion: str = "neutral"
    confidence: float = 1.0
    words: list[Word] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Timeline:
    version: str = "1.0"
    source_language: str = ""
    target_language: str = ""
    sentences: list[Sentence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # 方法: to_dict() / from_dict(d) / save(path) / load(path) / append(s) / update_sentence(id, **kw)
```

序列化格式：**JSONL**，第一行是元信息 `{"version","source_language","target_language","metadata"}`，后续每行一个 `Sentence`（`asdict` 全量）。

#### Workspace（`src/opendubbing/core/workspace.py`，已实现，禁改）

```python
Workspace.SUBDIRECTORIES = ("input","cache","timeline","translation","tts","face","output")
workspace.path_for(step, name) -> Path      # step 必须在 SUBDIRECTORIES 内
workspace.cache_path   -> Path   # = root/cache
workspace.timeline_path -> Path  # = root/timeline/timeline.jsonl
workspace.exists(step, name) -> bool
```

#### 接口（`src/opendubbing/core/interfaces.py`，已实现，禁改）

```python
class Engine(ABC):
    name: str
    def initialize(self, config: dict) -> None: ...
    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline: ...
    def save(self, timeline: Timeline, workspace: Workspace) -> None: ...
    def release(self) -> None: ...

class Provider(ABC):
    name: str; kind: str
    def initialize(self, config: dict) -> None: ...
    def load_model(self) -> None: ...
    def infer(self, inputs: Any) -> Any: ...
    def release(self) -> None: ...
```

#### Registry（已实现）

`registry.build(kind, name, config)` 会 `provider_class()` + `provider.initialize(config)`，**不会**调用 `load_model()`——`load_model()` 由 Engine 在 `process()` 首次使用前显式调用。

#### Orchestrator 流程（已实现，禁改）

每个 step：检查取消 → `cache.should_skip` → `cache.hit` → 否则 `engine = engine_registry.build(step, config)` → `engine.process` → `engine.save` → `cache.commit` → `engine.release()`。
Engine 收到的 `config` dict 已被 orchestrator 注入 `"step"` 和 `"registry"`（`ProviderRegistry` 实例）两个键。

### 0.2 流程总览（本批次实现的端到端链路）

```
input ─→ asr ─→ timeline(对齐) ─→ translation ─→ length_optimizer ─→ prosody
   ─→ tts ─→ audio_post ─→ face ─→ video_post ─→ output
```

每个 Engine 的 `process()` 职责本批全部填实。Provider 的 `load_model()`/`infer()` 填实。

---

## 1. 环境与依赖准备

### 1.1 pyproject.toml 依赖修改（唯一允许的"骨架文件"改动）

在 `pyproject.toml` 的 `[project] dependencies` 中**新增**以下条目（保留原有 pydantic/pyyaml/structlog）：

```toml
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "structlog>=24.0",
    # --- 新增：媒体处理 ---
    "ffmpeg-python>=0.2.0",
    "numpy>=1.26,<2.0",
    "soundfile>=0.12.0",
    "librosa>=0.10.0",
    # --- 新增：ML（provider 延迟加载，但装上以便本地验证） ---
    "torch>=2.2.0",
    "torchaudio>=2.2.0",
    "transformers>=4.40.0",
    "torchvision>=0.17.0",
    "accelerate>=0.30.0",
    "sentencepiece>=0.2.0",
    # --- 新增：API 客户端（LLM 翻译 provider 用） ---
    "openai>=1.30.0",
    "anthropic>=0.30.0",
    "google-generativeai>=0.7.0",
]
```

新增 optional 分组（heavy / 可选 provider 不强制安装）：

```toml
[project.optional-dependencies]
heavy = [
    "demucs>=0.0.3",
    "deepfilternet>=0.5.0",
    "silero-vad>=5.1",
    "funasr>=1.0.27",
]
```

> **字段约束**：不要动 `[tool.hatch.build.targets.wheel]`、`[project.scripts]`、ruff/pyright/pytest 配置块。

### 1.2 FFmpeg 系统要求

文档与 README 里声明 OpenDubbing 依赖系统 **FFmpeg ≥ 4.4**（`ffmpeg` 在 PATH 中）。`utils/media.py` 用 `ffmpeg-python` 调用。不写 Python 侧 ffmpeg 安装逻辑。

### 验收 1

```bash
python -m pip install -e ".[all]"
python -c "import ffmpeg, librosa, soundfile, torch; print('ok')"
ffmpeg -version | head -1
```
全绿通过。

---

## 2. utils/media.py — FFmpeg 音视频工具层

**文件**：`src/opendubbing/utils/media.py`（当前为空）

本模块是 Engine 与 Provider 共用的音视频原语。**只依赖 `ffmpeg-python`、`numpy`、`soundfile`、`subprocess`**，不依赖任何模型库。

实现以下函数（全部带 docstring，遵循 `ruff` 的 google 风格）：

```python
"""FFmpeg-based media I/O utilities shared by engines and providers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ffmpeg
import numpy as np
import soundfile as sf


def probe_duration(path: Path | str) -> float:
    """Return media duration in seconds via ffprobe."""

def extract_audio(
    source: Path | str,
    out_path: Path | str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    """Extract audio track to 16-bit PCM wav at sample_rate.

    Returns the resolved out_path. Uses ffmpeg-python to run ffmpeg.
    Raises ffmpeg.Error with stderr on failure.
    """

def read_audio(
    path: Path | str,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    """Read audio as float32 numpy array. Returns (samples, sr).
    If sample_rate given, resample via librosa."""

def write_audio(
    samples: np.ndarray,
    sample_rate: int,
    path: Path | str,
) -> Path:
    """Write float32 samples to wav."""

def concat_audio(segments: list[Path], out_path: Path | str) -> Path:
    """Concatenate wav segments in order using ffmpeg concat demuxer."""

def mux_audio_video(
    video: Path | str,
    audio: Path | str,
    out_path: Path | str,
    codec: str = "libx264",
) -> Path:
    """Mux audio into video, replacing original audio track."""

def segment_audio(
    audio: Path | str,
    segments: list[tuple[float, float]],
    out_dir: Path | str,
    prefix: str = "seg",
) -> list[Path]:
    """Cut audio into [start,end) second segments. Returns ordered list of paths."""
```

实现约束：
- `probe_duration` 用 `ffmpeg.probe(str(path))` 取 `format.duration`，找不到时遍历 `streams`。
- `extract_audio`/`mux_audio_video`/`concat_audio` 用 `ffmpeg.run()`，`overwrite_output=True`，捕获 stderr 到 `RuntimeError`。
- `segment_audio` 对每段用 `ffmpeg.input(..., ss=start, t=duration)` 切到 `out_dir/{prefix}_{i:04d}.wav`。
- 所有路径返回前 `.resolve()`。
- 不要 import `librosa` 在模块顶层之外的执行路径里只 import 一次。

### 验收 2

新增测试 `tests/test_media.py`（用 `pytest` + `tmp_path`）：
- 生成 2 秒静音 wav（用 `numpy` + `write_audio` 写 16kHz float32 zeros）。
- 断言 `read_audio` 返回 sample 数量符合 `2 * sr`。
- 断言 `concat_audio` 两段后长度翻倍。
- 断言 `probe_duration` 返回约 2.0（容差 0.1）。
- `mux_audio_video`/`segment_audio` 这类需要真实视频的测试用 `pytest.skip("no ffmpeg-media fixture")` 跳过即可，但函数必须实现且能 import。

```bash
python -m pytest tests/test_media.py -q
python -m ruff check src tests
```
全绿通过。

---

## 3. Engine 实现任务

> **通用规则**（每个 Engine 都遵守）：
> 1. `initialize` 把 `self.provider_name` 设为 `self.provider_config.get("name")`，`self.provider_options = self.provider_config.get("options", {})`，`self.registry` 已在 config 注入。
> 2. `process()` 开头 `provider = self.registry.build(KIND, self.provider_name, self.provider_config)`；首次使用前 `provider.load_model()`；用完 `provider.release()`。**注意**：Engine 的 `release()` 是 Engine 维度的资源释放，Provider 用完即释。
> 3. 读写 Workspace 一律用 `workspace.path_for(...)` / `workspace.exists(...)`，**禁止跨 Engine 直接传对象**，Engine 间只通过 Timeline（内存中传递，由 orchestrator 持有）+ Workspace 文件（落盘产物）通信。
> 4. 既有的 `save()` 实现保留即可（已写好），本批只在必要时微调输出路径。
> 5. 任何"可选重"能力（demucs/deepfilter/silero/funasr）走 try-import，import 失败时抛 `ProviderModelLoadError("install opendubbing[heavy]")`，**不要在 Engine 层 try**，让错误冒泡由 Pipeline 记录。

### 3.1 InputEngine — `engines/input_engine.py`

职责：从 `config["input_path"]` 读视频/音频，提取干净音频，写入 `workspace/input/`。

`initialize` 后已有 `self.input_path = config.get("input_path")`。补充 `self.target_sample_rate = config.get("sample_rate", 16000)`。

`process` 逻辑：
```python
if not self.input_path:
    raise ValueError("input_path is required")
input_path = Path(self.input_path)
audio_out = workspace.path_for("input", "audio.wav")
video_out = workspace.path_for("input", "video.mp4")
if not audio_out.exists():
    media.extract_audio(input_path, audio_out, sample_rate=self.target_sample_rate)
# 如果输入是视频，复制一份到 video_out 供后续 face/video_post 使用
if media.probe_duration(input_path) and str(input_path).lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
    if not video_out.exists():
        shutil.copyfile(input_path, video_out)
timeline.metadata["input_audio"] = str(workspace.relative(audio_out))
timeline.metadata["input_video"] = str(workspace.relative(video_out)) if video_out.exists() else None
duration = media.probe_duration(audio_out)
timeline.metadata["duration"] = duration
return timeline
```
import 顶部加 `import shutil` 和 `from opendubbing.utils import media`。

### 3.2 ASREngine — `engines/asr_engine.py`

职责：调用 ASR Provider 对整段音频转录，产出 Sentence 列表填进 Timeline。
KIND = `"asr"`。

`process`：
```python
provider = self.registry.build("asr", self.provider_name, self.provider_config)
provider.load_model()
audio_path = Path(timeline.metadata["input_audio"])  # 相对 workspace.root
audio_abs = workspace.root / audio_path if not audio_path.is_absolute() else audio_path
result = provider.infer({"audio": str(audio_abs), "language": self.provider_config.get("source_language", "auto")})
provider.release()
# result 约定: {"segments":[ {"text","start","end","words":[{"text","start","end"}], "confidence"} ]}
timeline.source_language = result.get("language", timeline.source_language)
for i, seg in enumerate(result["segments"]):
    s = Sentence(
        id=f"s{i:04d}", text=seg["text"].strip(),
        start=seg["start"], end=seg["end"], duration=seg["end"]-seg["start"],
        confidence=seg.get("confidence", 1.0),
        words=[Word(text=w["text"], start=w["start"], end=w["end"], duration=w["end"]-w["start"]) for w in seg.get("words", [])],
    )
    timeline.append(s)
return timeline
```
**注意**：`metadata["input_audio"]` 是相对路径（InputEngine 存的是 `workspace.relative(...)`），需要拼回绝对路径。

### 3.3 TimelineEngine — `engines/timeline_engine.py`

职责：用 Forced Alignment Provider 精修每个 sentence 的 word/phoneme 时间。
KIND = `"forced_alignment"`。

`process` 遍历 `timeline.sentences`，对每个有 words 的 sentence 调 `provider.infer({"audio":..., "text":sentence.text, "words":[w.text...], "start":sentence.start, "end":sentence.end})`。Provider 返回 `{"words":[{"text","start","end","phonemes":[{symbol,start,end}]}], "confidence"}`。把返回覆盖到 `sentence.words`，并填 phonemes；更新 `sentence.confidence = min(old, returned)`。无 words 的 sentence 跳过。

### 3.4 TranslationEngine — `engines/translation_engine.py`

职责：翻译每个 sentence.text 到目标语言，填 `sentence.translation`。
KIND = `"translation"`，`self.provider_name` 来自 `providers.translation.name`，`self.target_language = config.get("target_language", "zh")`（从 `provider_config.get("options",{}).get("target_language")` 兜底）。

`process`：批量更高效，但占位实现先逐句调 `provider.infer({"text":s.text, "source_language":timeline.source_language,"target_language":self.target_language})` → `{"translation": str}`。把结果写入 `s.translation`。`timeline.target_language = self.target_language`。

### 3.5 LengthOptimizer — `engines/length_optimizer.py`

本批**不接 Provider**（规则式即可，保持轻量）。逻辑：
- 对每个 sentence，`target_duration = sentence.duration`（原语时长）。
- 若 `s.translation` 为空，跳过。
- 估算译文时长：`chars = len(s.translation)`，按目标语言系数（中文 `0.28s/字`，英文 `0.18s/字`，无语言默认 0.25）算 `est = chars * rate`。
- 若 `est > target_duration * 1.15`：在 `metadata` 标 `{"length_warning":"too_long","est":est,"target":target_duration}`，并把 `s.speech_rate = clamp(est/target_duration, 0.8, 1.3)` 作为给 TTS 的提示。
- 否则 `s.speech_rate = 1.0`。
- 不接 Provider，不调 `registry`。

### 3.6 ProsodyEngine — `engines/prosody_engine.py`

本批**规则式**，不接 Provider。逻辑：
- 对每个 sentence，`s.emotion = "neutral"`（默认）。
- 简单标点启发式：文本以 `!`/`？`结尾 → `surprise`/`question`；含省略号 → `calm`；其余 `neutral`。
- `s.pause`：句末停顿 = `max(0.0, next_sentence.start - s.end)`（找一个非空起始）。
- 对每个 word，`w.pause = max(0.0, next_word.start - w.end)`。
- 写 `s.metadata["prosody_done"] = True`。

### 3.7 TTSEngine — `engines/tts_engine.py`

职责：对每个有 translation 的 sentence 合成配音音频，写入 `workspace/tts/{sentence_id}.wav`。
KIND = `"tts"`。

`process`：
```python
provider = self.registry.build("tts", self.provider_name, self.provider_config)
provider.load_model()
for s in timeline.sentences:
    if not s.translation:
        continue
    out = workspace.path_for("tts", f"{s.id}.wav")
    provider.infer({
        "text": s.translation,
        "language": timeline.target_language,
        "speech_rate": s.speech_rate,
        "emotion": s.emotion,
        "out_path": str(out),
    })
    s.metadata["tts_audio"] = str(workspace.relative(out))
provider.release()
return timeline
```
**Provider 契约**：`infer` 接收 dict 并把音频直接写到 `out_path`，返回 `{"duration": float}`。Engine 把 duration 存进 `s.metadata["tts_duration"]`。

### 3.8 AudioPostProcessor — `engines/audio_post_processor.py`

职责：把各 sentence 的 TTS 片段按原时间轴插入成一条与原视频等长的完整配音轨，背景空白用静音填充。不接 Provider。

`process`：
- 读 `timeline.metadata["duration"]`（InputEngine 写入）。
- 用 `soundfile` 读每个 `tts/*.wav`，按 `sentence.start` 在总轨定位，片段超过 `sentence.end` 则截断。
- 句间用静音填（保持与原画面时长一致）。
- 用 `media.write_audio` 写到 `workspace/tts/dub_full.wav`（虽然放 tts 目录，命名加前缀避免与 sentence 片段冲突；或写到 `workspace/input/dub_full.wav`——**统一用 `workspace.path_for("tts","dub_full.wav")`**）。
- `timeline.metadata["dub_audio"] = str(workspace.relative(<path>))`。

### 3.9 FaceAnimationEngine — `engines/face_animation_engine.py`

职责：调 Face Provider 对原视频做人像对口型，输出 `workspace/face/face_video.mp4`。
KIND = `"face"`。

`process`：
```python
video = timeline.metadata.get("input_video")
if not video:
    return timeline  # 纯音频输入，跳过
dub = timeline.metadata["dub_audio"]
out = workspace.path_for("face", "face_video.mp4")
provider = self.registry.build("face", self.provider_name, self.provider_config)
provider.load_model()
provider.infer({
    "video": str((workspace.root / video) if not Path(video).is_absolute() else video),
    "audio": str((workspace.root / dub) if not Path(dub).is_absolute() else dub),
    "out_path": str(out),
})
provider.release()
timeline.metadata["face_video"] = str(workspace.relative(out))
return timeline
```
**Provider 契约**：`infer` 把对口型视频写到 `out_path`，返回 `{"path": str}`。

### 3.10 VideoPostProcessor — `engines/video_post_processor.py`

职责：把对口型视频与配音轨合成最终视频（若 face 步有输出则用 face 视频，否则用原视频；音频统一用 dub 音轨）。不接 Provider。

`process`：
```python
face_video = timeline.metadata.get("face_video")
video = (workspace.root / face_video) if face_video else (timeline.metadata.get("input_video") and (workspace.root / timeline.metadata["input_video"]))
dub = workspace.root / timeline.metadata["dub_audio"]
out = workspace.path_for("output", "final.mp4")
media.mux_audio_video(video, dub, out, codec=self.config.get("output",{}).get("codec","libx264"))
timeline.metadata["final_video"] = str(workspace.relative(out))
return timeline
```

### 3.11 OutputEngine — `engines/output_engine.py`

职责：把最终视频命名为配置里的 `output.filename`，放在 `workspace/output/`。
`process`：`final = workspace.path_for("output","final.mp4")`；`dst = workspace.path_for("output", self.config.get("output",{}).get("filename","output.mp4"))`；若两者不同则 `shutil.move`/`copyfile`；`timeline.metadata["output"] = str(workspace.relative(dst))`。

### 验收 3

新增 `tests/test_engines/test_integration.py`：用一个 fake provider 注册表跑完整 pipeline。
```python
@pytest.fixture
def fake_registries():
    # 注册一组 fake Provider 类到 ProviderRegistry，每个 infer 返回契约化数据
    ...
    # 注册真实 Engine 到 EngineRegistry（create_default_engine_registry）
    ...
```
断言走完 11 步后 `workspace/output/output.mp4` 存在、`timeline.metadata["output"]` 有值。用 fake provider 模拟 ASR/Translation/TTS/Face 返回值，其余用规则式 engine 自然产出。

```bash
python -m pytest tests/test_engines/test_integration.py -q
python -m ruff check src tests
```
全绿。

---

## 4. Provider 实现任务

> **通用规则**：
> 1. `initialize(self, config)`：保存 `self.config`；解析 `self.model = config.get("model")`；解析 `self.options = config.get("options", {})`；解析必要 key（如 `api_key`）。
> 2. `load_model()`：**延迟加载**。import heavy 库在这一刻才发生（函数内 import），失败抛 `ProviderModelLoadError(thename)`。
> 3. `infer(self, inputs)`：严格按 Engine 契约返回 dict/写文件。
> 4. `release()`：`del` 模型引用，`gc.collect()`，GPU 侧 `torch.cuda.empty_cache()` 当存在。
> 5. 顶层不 import heavy 库；只 import 标准库 + `opendubbing.core.interfaces` + `opendubbing.utils.media`（若需要）。

### 4.1 DemucsProvider — `providers/audio_separation/demucs.py`

- `load_model`：`from demucs.pretrained import get_model; self.model = get_model("htdemucs")`；`import torch`；`self.model.to(device)`；`self.model.eval()`。device 取 `self.options.get("device","cuda" if torch.cuda.is_available() else "cpu")`。
- `infer(inputs)`：`inputs = {"audio": path}`（16k mono wav）。用 `torchaudio.load` 读 → `self.model.forward(wav)` → 取 `vocals` stem → `media.write_audio` 到 `inputs.get("out_path")` 或返回 `{"stems": {"vocals": ndarray, "bg": ndarray}}`。本批定契约：**返回 `{"vocals": <wav_path>, "background": <wav_path>}`，两个都落盘**。
- `release`：`self.model=None`。

### 4.2 DeepFilterNet3Provider — `providers/noise_reduction/deep_filter_net3.py`

- `load_model`：`from df import init_df, enhance`；`self.df_model, self.df_state, _ = init_df()`。
- `infer(inputs)`：`{"audio": path, "out": path}` → `enhance(self.df_model, self.df_state, audio_path, output_path=...)` → 返回 `{"enhanced": out_path}`。若 import 失败抛 `ProviderModelLoadError`。

### 4.3 SileroVADProvider — `providers/vad/silero_vad.py`

- `load_model`：`from silero_vad import load_silero_vad, get_speech_timestamps; self.model = load_silero_vad()`；`import torch`。
- `infer(inputs)`：`{"audio": path, "threshold":0.5,"min_silence_ms":500}` → 读 wav 16k → `get_speech_timestamps(wav, self.model, threshold=..., min_silence_duration_ms=...)` → 返回 `{"segments":[{"start":s["start"]/sr,"end":s["end"]/sr} for s in ts]}`。**注意把 sample 转秒**。

### 4.4 Qwen3ForcedAlignerProvider — `providers/forced_alignment/qwen3_forced_aligner.py`

本批用 **funasr** 作为可跑的 Forced Aligner 占位（文档明确写：生产可替换为 Qwen3-ForcedAligner）。
- `load_model`：`from funasr import AutoModel; self.model = AutoModel(model="fa-zh")`（或合适的对齐模型）；import 失败抛 `ProviderModelLoadError("funasr")`。
- `infer(inputs)`：`{"audio","text","words","start","end"}` → 调 `self.model.generate(...)` → 解析为 `{"words":[{"text","start","end","phonemes":[{"symbol","start","end"}]}],"confidence":float}`。若拿不到 phonemes 就空 list，但 words 时间必须填。

### 4.5 翻译 Providers（四个，模式一致）

共通契约：`infer({"text","source_language","target_language"})` → `{"translation": str}`。

#### NLLBProvider — `providers/translation/nllb.py`
- `load_model`：`from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; self.tokenizer = AutoTokenizer.from_pretrained(model); self.model = AutoModelForSeq2SeqLM.from_pretrained(model)`。
- `infer`：tokenizer 设置 `src_lang`/`forced_BCP47` 之类 → 生成 → 解码 → 返回。
- 语种代码映射：维护一个 `_LANG_MAP = {"zh":"zho_Hans","en":"eng_Latn",...}`，缺失时原样给。

#### GPTProvider — `providers/translation/gpt.py`
- `load_model`：读 `self.config.get("api_key")` 或 `os.getenv("OPENAI_API_KEY")`；`from openai import OpenAI; self.client = OpenAI(api_key=...)`；`self.model_name = self.model or "gpt-4o-mini"`。
- `infer`：构造 system prompt "You are a translator. Translate to {target_language}. Return only the translation." → `client.chat.completions.create(...)` → 取 `choices[0].message.content`。

#### ClaudeProvider — `providers/translation/claude.py`
- `load_model`：`os.getenv("ANTHROPIC_API_KEY")`；`from anthropic import Anthropic`；`self.client = Anthropic()`；model 默认 `"claude-3-5-sonnet-latest"`（用 `self.model` 覆盖）。
- `infer`：`client.messages.create(model=..., max_tokens=1024, messages=[{"role":"user","content": prompt}])` → 取 `content[0].text`。

#### GeminiProvider — `providers/translation/gemini.py`
- `load_model`：`os.getenv("GOOGLE_API_KEY")`；`import google.generativeai as genai; genai.configure(api_key=...)`；`self.model = genai.GenerativeModel(self.model or "gemini-1.5-flash")`。
- `infer`：`self.model.generate_content(prompt)` → `.text`。

### 4.6 CosyVoice2Provider — `providers/tts/cosyvoice2.py`

本批 CosyVoice2 是较重的模型，实现一个"可落库替换"的适配层：
- `load_model`：try import CosyVoice2 repo（`from cosyvoice.cli.cosyvoice import CosyVoice2` 默认走 `[heavy]` 分组说明）→ 失败时**抛 `ProviderModelLoadError`，不要静默**。`self.model = CosyVoice2(model_path)`。
- `infer(inputs)`：`{"text","language","speech_rate","emotion","out_path"}` → 用 provider 提供的 inference 接口生成 → `soundfile.write(out_path, audio, sr)` → 返回 `{"duration": len(audio)/sr, "path": out_path}`。
- 若无 reference voice，本批用零样本模式，并在 docstring 标注声音克隆需要 `inputs["reference_audio"]` 字段（V1 保留接口,详细留 TODO）。

### 4.7 Hallo3Provider — `providers/face/hallo3.py`

- `load_model`：try import `hallo3` 包（heavy 分组）→ 失败抛 `ProviderModelLoadError`。
- `infer(inputs)`：`{"video","audio","out_path"}` → 调模型对口型生成到 `out_path` → 返回 `{"path": out_path}`。
- 本批若 hallo3 不易跑通，允许 `load_model` 成功但 `infer` 走"回退"：用 ffmpeg 把 audio 直接 mux 到原 video（不含真对口型），并在返回 dict 加 `"fallback": True` 以便测试区分。**文档要求最终必须真接入 hallo3**，回退仅用于无 GPU 环境冒烟。

### 验收 4

每个 Provider 至少 1 个单测（`tests/test_providers/test_<name>.py`），用 mock：
- mock 掉 heavy 库 import（`monkeypatch.setattr` 或 `sys.modules["demucs"]=MagicMock()`），断言 `infer` 满足契约（返回 dict / 写文件）。
- `load_model` 真缺库时断言抛 `ProviderModelLoadError`。
- 翻译类用 `httpx` mock / 录制响应测 `infer` 解析逻辑。

```bash
python -m pytest tests/test_providers/ -q
python -m ruff check src tests
```
全绿。

---

## 5. 配置与示例文件

### 5.1 `examples/sample_config.yaml` 更新

补全所有 provider 块与 LLM 环境说明：
```yaml
target_language: "zh"
sample_rate: 16000

workspace:
  root: "./workspace"

providers:
  audio_separation:
    name: demucs
    model: "htdemucs"
    options: { device: "auto" }
  noise_reduction:
    name: deep_filter_net3
  vad:
    name: silero_vad
    options: { threshold: 0.5, min_silence_ms: 500 }
  asr:
    name: qwen3_asr   # 本批暂用 funasr 后端，见 provider 注释
    options: { source_language: "auto" }
  forced_alignment:
    name: qwen3_forced_aligner
  translation:
    name: nllb          # 切换 LLM: 改成 gpt/claude/gemini 并在下设 api_key via env
    model: "facebook/nllb-200-3.3B"
  tts:
    name: cosyvoice2
    options: { device: "auto" }
  face:
    name: hallo3

pipeline:
  steps: [input, asr, timeline, translation, length_optimizer, prosody, tts, audio_post, face, video_post, output]

output:
  filename: "output.mp4"
  codec: "libx264"

api:
  host: "127.0.0.1"
  port: 8000
```

### 5.2 `AppConfig`（`src/opendubbing/config.py`）小补丁

新增顶层字段 `target_language: str = "zh"` 与 `sample_rate: int = 16000`（在 `AppConfig` dataclass 里）。`PipelineConfig.steps` 已存在无需改。`to_dict()` 会自动带上。

### 验收 5

```bash
python -c "from opendubbing.config import load_config; c=load_config('examples/sample_config.yaml'); print(c.target_language, c.pipeline.steps)"
python -m pyright src -p .
```
target_language 正确、steps 正确、pyright 0 errors。

---

## 6. orchestrator 衔接（无需改动，仅说明）

`orchestrator` 已把 `self.provider_registry` 透传进 engine config 的 `"registry"` 键，并把每个 step name 注入 `"step"`。`_build_engine` 调 `engine_registry.build(step, engine_config)`，对应 `create_default_engine_registry` 里每个 step name 已注册。**本任务无需改 orchestrator/cache**。

唯一要注意：`config["providers"]` 是 `AppConfig.to_dict()` 里的 `providers: {kind: {name, model, options}}`。`engine_config["providers"]["asr"]["name"]` 即 Provider name。Engine 现有代码用 `config["providers"]["asr"]["name"]`——**本批把 `self.provider_name` 在 `initialize` 里显式取出来**，避免 None。

### 验收 6（端到端冒烟）

```bash
# 准备一段 10 秒测试视频（任一素材；若无可用下载 sample）
opendubbing process --input <path/to/test.mp4> --config examples/sample_config.yaml
ls workspace/output/
```
应出现 `output.mp4`。若模型未装全，至少在 `audio_post` 之前各步落盘正常（用 `--config` 里把 `steps` 缩为 `[input, asr]` 单独验）。

---

## 7. 前端验收（本批不动，但留接口契约对齐备忘）

前端 `src/api/tasks.ts` 已封装 `/tasks` `/providers` 与 WebSocket。后端 `routes.py` 的 `create_task` 接 `{"input_path","config_path","resume"}` 返回 `TaskResponse`。本批保持该契约不变即可。无需改动。

---

## 8. 总验收清单（全部完成后再跑一次）

```bash
python -m pytest tests/ -q                    # 全绿
python -m ruff check src tests                # All checks passed
python -m pyright src -p .                     # 0 errors
opendubbing providers                          # 列出全部 provider
opendubbing process --input <test.mp4> --config examples/sample_config.yaml
test -f workspace/output/output.mp4 && echo OK # 真实链路产出
```

---

## 9. 任务对照表（执行模型按序打勾）

| # | 任务 | 涉及文件 | 验收命令 | 状态 |
|---|---|---|---|---|
| 1 | 环境依赖 | `pyproject.toml` | 见 §1 | ☐ |
| 2 | 媒体工具 | `utils/media.py` + `tests/test_media.py` | `pytest tests/test_media.py` | ☐ |
| 3 | 11 个 Engine | `engines/*.py` + `tests/test_engines/test_integration.py` | `pytest tests/test_engines` | ☐ |
| 4 | 11 个 Provider | `providers/*/*.py` + `tests/test_providers/*` | `pytest tests/test_providers` | ☐ |
| 5 | 配置与示例 | `config.py` + `examples/sample_config.yaml` | §5 | ☐ |
| 6 | 端到端冒烟 | （无新文件） | §6 | ☐ |
| 7 | 总验收 | （全量） | §8 | ☐ |

---

## 10. 守则（违反即返工）

1. 不新建文件结构（除 §2/§3/§4 明确列出的测试文件）；不改接口签名。
2. Engine 不 import 任何 `torch`/`transformers`/SDK；只有 Provider import。
3. Provider 不调 Provider；Engine 不调 Engine。
4. 所有中间结果落 `workspace/<step>/`，Engine 间只交换 Timeline（内存）+ 文件路径（写入 metadata 后落盘）。
5. heavy 库 import 失败必须抛 `ProviderModelLoadError`，不得静默返回空。
6. 每个 `infer`/`process` 有 docstring；遵循 `ruff google` 风格；行宽 100。
7. 任何时间相关数值统一**秒**（float）；采样率默认 16000；音频 wav float32 / 16k。
8. 不动 orchestrator/cache/core/timeline/workspace；如发现需改，停下并报告。
```

---

**给执行模型的一句启动指令**：从任务 1 开始，每完成一节先跑该节"验收命令"全部绿后再推进下一节；若任何验收失败，先修到绿再继续，禁止跳过。