# Style-RAG 系统文档 / System Documentation

## 1. 系统架构 / System Architecture

Style-RAG 是一个模块化的检索增强生成系统。
Style-RAG is a modular Retrieval-Augmented Generation system.

### 核心组件 / Core Components

- **Embedding Manager**: 管理本地和云端 Embedding 模型，处理模型切换和降级。
- **Vector Store (ChromaDB)**: 存储文本片段和向量索引。
- **Article Loader**: 负责加载、清洗和预处理文档。
- **Chunker**: 智能分块器，根据文本类型（对话、描写、动作）进行语义切分。

## 2. 配置说明 / Configuration

### 模型配置 / Model Configuration
配置文件: `model_config.py`

| 变量 / Variable | 说明 / Description | 默认值 / Default |
|----------------|-------------------|-----------------|
| `EMBEDDING_PROVIDER` | Embedding 提供商 (local, openai, etc.) | "local" |
| `API_KEY` | 云端 API 密钥 | "" |
| `EMBEDDING_BATCH_SIZE` | 批处理大小 | 32 |

### 系统配置 / System Configuration
配置文件: `style_rag/core/config.py`

包含数据库路径、默认参数等系统级配置。

## 3. 功能特性 / Features

### 3.1 智能分块 / Smart Chunking
系统会自动识别文本中的不同元素：
- **Dialogue (对话)**: 提取角色对话，保留上下文。
- **Description (描写)**: 提取环境、外貌等描写性段落。
- **Action (动作)**: 提取战斗、动作场景。

### 3.2 混合检索 / Hybrid Retrieval
结合语义相似度和元数据过滤：
- 支持按场景类型 (Scene Type) 过滤
- 支持按情感色彩 (Emotion) 过滤

## 4. 故障排除 / Troubleshooting

### 常见问题 / Common Issues

**Q: 索引速度很慢 / Slow Indexing**
A: 尝试减小 `EMBEDDING_BATCH_SIZE` 或检查 GPU 使用情况。
Try reducing batch size or check GPU usage.

**Q: 检索结果不相关 / Irrelevant Results**
A: 检查使用的 Embedding 模型是否与索引时一致。
Ensure the embedding model matches the one used for indexing.

**Q: 找不到 `model_config.py` / Missing config**
A: 请从 `model_config.py.example` 复制并重命名。
Copy from `model_config.py.example`.
