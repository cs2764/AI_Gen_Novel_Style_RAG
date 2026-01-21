"""
Style-RAG: 独立RAG系统 - 用于风格学习和创作优化
Independent RAG System for Style Learning and Creative Enhancement

主要功能:
- 索引现有文章到向量数据库
- 语义检索相关片段
- 支持本地和云端Embedding
- 提供Python API、CLI工具和HTTP服务
"""

from style_rag.api.client import StyleRAGClient
from style_rag.core.embedding_config import EmbeddingConfig, EmbeddingProvider
from style_rag.core.config import RAGConfig

__version__ = "0.1.0"
__all__ = [
    "StyleRAGClient",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "RAGConfig",
]
