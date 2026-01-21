"""
RAG系统主配置 / Main RAG Configuration
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class RAGConfig:
    """RAG系统配置 / RAG System Configuration"""
    
    # 启用开关 / Enable switch
    enabled: bool = True
    
    # 默认输入目录 / Default input directory
    articles_dir: str = "./input"
    
    # 向量数据库配置 / Vector database configuration
    vector_db_path: str = "./rag_db"
    collection_name: str = "article_styles"
    
    # 检索配置 / Retrieval configuration
    top_k: int = 5
    similarity_threshold: float = 0.5
    enable_reranking: bool = False
    reranker_model: str = "Qwen/Qwen3-Reranker-4B"
    
    # 分块配置 / Chunking configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # 性能配置 / Performance configuration
    batch_size: int = 32
    max_cached_embeddings: int = 10000
    
    # 文件格式支持 / Supported file formats
    supported_extensions: List[str] = field(
        default_factory=lambda: [".txt", ".md"]
    )
    
    def get_db_path(self) -> Path:
        """获取向量数据库路径 / Get vector database path"""
        return Path(self.vector_db_path)
    
    def validate(self) -> bool:
        """验证配置 / Validate configuration"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        return True


# 默认配置 / Default configuration
DEFAULT_RAG_CONFIG = RAGConfig()
