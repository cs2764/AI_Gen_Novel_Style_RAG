# Style-RAG: Independent RAG System / 独立 RAG 系统

**Independent RAG System for Style Learning and Creative Enhancement**
**用于风格学习和创作优化的独立 RAG 系统**

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## 🚀 Introduction

**Style-RAG** is an independent Retrieval-Augmented Generation (RAG) system designed specifically for Chinese writing style learning and creative enhancement. It supports both local and cloud-based embeddings and can be integrated into multiple applications.

> [!NOTE]
> This project is designed as a companion to the [AI_Gen_Novel](https://github.com/cs2764/AI_Gen_Novel) project but can be used independently for any RAG-based task.

### ✨ Features

- 🏠 **Local First**: Defaults to local embedding models, no API costs required.
- ☁️ **Cloud Compatible**: Supports OpenAI, Zhipu, Aliyun, SiliconFlow, and more.
- 🔄 **Auto Fallback**: Automatically downgrades to local models if cloud services fail.
- 📚 **Smart Chunking**: Intelligently chunks text by dialogue, description, and action.
- 🔍 **Multi-dimensional Retrieval**: Supports semantic search, scene search, and emotion filtering.
- 🌐 **Multiple Interfaces**: Python API, CLI tools, and HTTP service.
- 🔒 **Secure Config**: Model configuration is separated; API keys are not hardcoded.
- 📊 **Progress Tracking**: Detailed file-level progress display during indexing.
- 📜 **Full Results**: Retrieval tests display full, untruncated content.

### 🆕 Update Log (v0.1.1 - 2026-03-25)

- **Cross-Platform Compatibility**: Standardized line endings (CRLF to LF) across the entire codebase to improve compatibility across Windows, Linux, and macOS environments.
- **Improved Ignored Files**: Updated `.gitignore` to better handle massive RAG index backups and archives, preventing accidental commits of redundant user data.

### 📜 Previous Updates (v0.1.0)

- **Configuration Separation**: Sensitive configurations (like API Keys) are now stored in `model_config.py`.
- **Retrieval Optimization**: `run_test` no longer truncates long text in search results.
- **Indexing Progress**: Enhanced progress display during indexing with per-file timing.
- **Documentation**: Comprehensive bilingual developer documentation.

### 📦 Installation

#### Method 1: Using `uv` (Recommended)

```bash
# Activate virtual environment
.\.venv\Scripts\activate  # Windows PowerShell
# or
source .venv/bin/activate  # Linux/macOS

# Install dependencies
uv pip install -r requirements.txt
```

#### Method 2: Using `pip`

```bash
pip install -r requirements.txt
```

### ⚙️ Configuration

Before usage, you need to configure models and API keys:

1. Copy the configuration template:
   ```bash
   cp model_config.py.example model_config.py
   # Windows PowerShell:
   # Copy-Item model_config.py.example model_config.py
   ```

2. Edit `model_config.py` to add your API keys and select the desired model.

### 🚀 Quick Start

#### 1. Prepare Data

Place your novel text files (.txt or .md) into the `input/` directory.

#### 2. Python API

```python
from style_rag import StyleRAGClient

# Initialize client (uses local embedding by default)
client = StyleRAGClient(
    db_path="./my_rag_db",
    embedding_model="Qwen/Qwen3-Embedding-4B"
)

# Index a directory
result = client.index_directory("./my_articles")
print(f"Indexing complete: {result['total_chunks']} chunks")

# Semantic Search
results = client.search("Intense battle scene", top_k=5)
for r in results:
    print(f"Similarity: {r['similarity']:.2f}")
    print(r['content'][:200])

# Scene Search
results = client.search_by_scene(
    scene_description="Meeting under the moon",
    emotion="Romantic",
    writing_type="description",
    top_k=3
)
```

#### CLI Tools

```bash
# Index articles
python -m style_rag.cli.index_articles \
    --input ./articles \
    --output ./rag_db \
    --embedding-model "Qwen/Qwen3-Embedding-4B"

# Query index
python -m style_rag.cli.query \
    --db ./rag_db \
    --query "Intense battle scene" \
    --top-k 5
```

---

<a name="chinese"></a>
## 🚀 简介 (Introduction)

**Style-RAG** 是一个独立的检索增强生成（RAG）系统，专为中文写作风格学习和创作优化设计。支持本地和云端 Embedding，可被多个应用集成使用。

> [!NOTE]
> 本项目是 [AI_Gen_Novel](https://github.com/cs2764/AI_Gen_Novel) 项目的配套组件，但也完全可以独立使用。

### ✨ 特性 (Features)

- 🏠 **本地优先** - 默认使用本地 Embedding 模型，无需 API 费用
- ☁️ **云端兼容** - 支持 OpenAI、智谱、阿里云、SiliconFlow 等云端服务
- 🔄 **自动降级** - 云端失败时自动降级到本地模型
- 📚 **智能分块** - 按对话、描写、动作等类型智能分块
- 🔍 **多维检索** - 支持语义检索、场景检索、情感过滤
- 🌐 **多种接口** - Python API、CLI 工具、HTTP 服务
- 🔒 **安全配置** - 模型配置分离，API 密钥不直接从代码读取
- 📊 **进度追踪** - 索引构建时显示详细的文件级进度
- 📜 **完整结果** - 检索测试显示无截断的完整内容

### 🆕 更新日志 (v0.1.1 - 2026-03-25)

- **跨平台兼容性优化**: 统一了整个代码库的换行符格式（从 CRLF 转换为 LF），提升了在 Windows、Linux 和 macOS 环境下运行的兼容性。
- **Git 忽略规则完善**: 更新了 `.gitignore`，更好地排除庞大的 RAG 索引备份和归档文件，防止冗余的用户个人数据被意外提交。

### 📜 历史更新 (v0.1.0)

- **配置分离**: 敏感配置（如 API Key）现在存储在 `model_config.py` 中，不再硬编码在 `run.py`。
- **检索优化**: `run_test` 检索测试不再截断长文本，便于完整查看检索效果。
- **索引进度**: 增强了索引构建时的进度显示，包含每个文件的处理时间。
- **文档更新**: 完善了中英双语开发文档。

### 📦 安装 (Installation)

#### 方式 1: 使用 `uv` 安装依赖

```bash
# 激活虚拟环境
.\.venv\Scripts\activate  # Windows PowerShell
# 或
source .venv/bin/activate  # Linux/macOS

# 安装依赖
uv pip install -r requirements.txt
```

#### 方式 2: 使用 `pip` 安装

```bash
pip install -r requirements.txt
```

### ⚙️ 配置 (Configuration)

在使用之前，需要配置模型和 API 密钥：

1. 复制配置文件模板：
   ```bash
   cp model_config.py.example model_config.py
   # Windows PowerShell:
   # Copy-Item model_config.py.example model_config.py
   ```

2. 编辑 `model_config.py`，填入你的 API 密钥并选择使用的模型。

### 🚀 快速开始 (Quick Start)

#### 1. 准备数据

将你的小说文本文件（.txt 或 .md）放入 `input/` 目录中。

#### 2. Python API

```python
from style_rag import StyleRAGClient

# 初始化客户端（使用本地 Embedding）
client = StyleRAGClient(
    db_path="./my_rag_db",
    embedding_model="Qwen/Qwen3-Embedding-4B"
)

# 索引文章目录
result = client.index_directory("./my_articles")
print(f"索引完成: {result['total_chunks']} 个片段")

# 语义检索
results = client.search("激烈的战斗场面", top_k=5)
for r in results:
    print(f"相似度: {r['similarity']:.2f}")
    print(r['content'][:200])

# 场景检索
results = client.search_by_scene(
    scene_description="月下相遇",
    emotion="浪漫",
    writing_type="description",
    top_k=3
)
```

#### 命令行工具 (CLI)

```bash
# 索引文章目录
python -m style_rag.cli.index_articles \
    --input ./articles \
    --output ./rag_db \
    --embedding-model "Qwen/Qwen3-Embedding-4B"

# 检索相关片段
python -m style_rag.cli.query \
    --db ./rag_db \
    --query "激烈的战斗场面" \
    --top-k 5
```

## 📄 许可证 (License)

MIT License
