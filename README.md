# OpenDubbing

OpenDubbing 是一个模块化的 AI 视频配音平台，目标是从离线视频翻译配音逐步扩展到实时直播翻译、数字人、插件生态等场景。

## 架构

```
Application → Pipeline → Engine → Provider → Model
```

核心原则：

- 业务代码只依赖接口，不依赖具体模型。
- Engine 按功能命名（如 `TTSEngine`），不按模型命名（禁止 `CosyVoice2Engine`）。
- 所有 Engine 通过统一的 Timeline 与 Workspace 通信，不直接传递对象。
- 前端通过 API 与后端交互，后端 Pipeline 负责调度 Engine 与 Provider。

## 快速开始

```bash
# 安装依赖
pip install -e ".[all]"

# 运行 CLI
opendubbing process --input video.mp4 --config examples/sample_config.yaml

# 启动 API 服务
opendubbing api --config examples/sample_config.yaml

# 列出 Provider
opendubbing providers
```

## 开发

```bash
# 静态检查
ruff check src tests
pyright src

# 测试
pytest
```

## 许可证

MIT
