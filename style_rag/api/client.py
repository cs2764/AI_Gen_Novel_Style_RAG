"""
StyleRAG客户端 - 统一接口供其他项目调用
StyleRAG Client - Unified Interface for Integration with Other Projects
"""

import logging
from typing import List, Dict, Optional, Union
from pathlib import Path

from style_rag.core.config import RAGConfig
from style_rag.core.embedding_config import EmbeddingConfig, EmbeddingProvider
from style_rag.core.embeddings import EmbeddingManager
from style_rag.core.chunker import StyleAwareChunker
from style_rag.core.vector_store import StyleVectorStore
from style_rag.indexing.index_manager import IndexManager
from style_rag.retrieval.retriever import StyleRetriever, RetrievalQuery
from style_rag.retrieval.query_builder import QueryBuilder

logger = logging.getLogger(__name__)


class StyleRAGClient:
    """
    风格RAG客户端
    Style RAG Client
    
    提供统一的接口用于索引和检索文章片段
    """
    
    def __init__(
        self,
        db_path: str = "./rag_db",
        embedding_config: Optional[EmbeddingConfig] = None,
        rag_config: Optional[RAGConfig] = None,
        embedding_model: Optional[str] = None,
        collection_name: str = "article_styles"
    ):
        """
        初始化RAG客户端
        
        Args:
            db_path: 向量数据库存储路径
            embedding_config: Embedding配置（优先使用）
            rag_config: RAG配置
            embedding_model: 嵌入模型名称（简化配置）
            collection_name: 集合名称
        """
        self.db_path = Path(db_path)
        
        # 配置
        self.rag_config = rag_config or RAGConfig(
            vector_db_path=str(self.db_path),
            collection_name=collection_name
        )
        
        # Embedding配置
        if embedding_config:
            self.embedding_config = embedding_config
        elif embedding_model:
            self.embedding_config = EmbeddingConfig(
                provider=EmbeddingProvider.LOCAL,
                local_model=embedding_model
            )
        else:
            self.embedding_config = EmbeddingConfig()
        
        # 初始化组件
        self._initialize()
    
    def _initialize(self):
        """初始化所有组件"""
        # Embedding管理器
        self.embedding_manager = EmbeddingManager(self.embedding_config)
        
        # 向量存储
        self.vector_store = StyleVectorStore(
            persist_dir=str(self.db_path),
            collection_name=self.rag_config.collection_name
        )
        
        # 分块器
        self.chunker = StyleAwareChunker(
            chunk_size=self.rag_config.chunk_size,
            chunk_overlap=self.rag_config.chunk_overlap
        )
        
        # 索引管理器
        self.index_manager = IndexManager(
            embedding_manager=self.embedding_manager,
            vector_store=self.vector_store,
            chunker=self.chunker
        )
        
        # 检索器
        self.retriever = StyleRetriever(
            embedding_manager=self.embedding_manager,
            vector_store=self.vector_store,
            default_top_k=self.rag_config.top_k,
            default_min_similarity=self.rag_config.similarity_threshold
        )
        
        # 查询构建器
        self.query_builder = QueryBuilder()
        
        logger.info(
            f"StyleRAGClient initialized: db={self.db_path}, "
            f"model={self.embedding_config.get_effective_model()}"
        )
    
    # ==================== 索引API ====================
    
    def index_directory(
        self,
        articles_dir: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        progress_callback=None,
        delete_after_index: bool = False
    ) -> Dict:
        """
        索引目录下的所有文章
        
        Args:
            articles_dir: 文章目录路径
            file_patterns: 文件匹配模式 (如 ["*.txt", "*.md"])
            recursive: 是否递归子目录
            progress_callback: 进度回调函数 (current, total, message)
            delete_after_index: 索引成功后删除源文件
            
        Returns:
            索引结果统计
        """
        return self.index_manager.index_directory(
            articles_dir=articles_dir,
            recursive=recursive,
            file_patterns=file_patterns,
            progress_callback=progress_callback,
            delete_after_index=delete_after_index
        )
    
    def index_files(self, file_paths: List[str], progress_callback=None) -> Dict:
        """
        索引指定的文件列表
        
        Args:
            file_paths: 文件路径列表
            progress_callback: 进度回调函数
            
        Returns:
            索引结果统计
        """
        return self.index_manager.index_files(
            file_paths=file_paths,
            progress_callback=progress_callback
        )
    
    def index_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> Dict:
        """
        直接索引文本内容
        
        Args:
            texts: 文本列表
            metadatas: 对应的元数据列表
            
        Returns:
            索引结果统计
        """
        return self.index_manager.index_texts(texts, metadatas)
    
    # ==================== 检索API ====================
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[str] = None,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """
        语义检索相关片段
        
        Args:
            query: 检索查询
            top_k: 返回数量
            filter_type: 过滤类型 (dialogue/description/action)
            min_similarity: 最小相似度阈值
            
        Returns:
            相关片段列表，每个包含:
            - content: 片段内容
            - metadata: 元数据
            - similarity: 相似度分数
        """
        return self.retriever.search(
            query=query,
            top_k=top_k,
            filter_type=filter_type,
            min_similarity=min_similarity
        )
    
    def search_by_scene(
        self,
        scene_description: str,
        emotion: Optional[str] = None,
        writing_type: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        按场景检索（专为创作优化）
        
        Args:
            scene_description: 场景描述
            emotion: 情感基调 (激动/温馨/紧张/悲伤/神秘/浪漫)
            writing_type: 写作类型 (dialogue/description/action)
            top_k: 返回数量
            
        Returns:
            相关片段列表
        """
        return self.retriever.search_by_scene(
            scene_description=scene_description,
            emotion=emotion,
            writing_type=writing_type,
            top_k=top_k
        )
    
    def retrieve_for_writing(
        self,
        storyline: str,
        chapter_summary: Optional[str] = None,
        character_states: Optional[Dict] = None,
        writing_phase: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        为写作检索风格参考
        
        Args:
            storyline: 故事线
            chapter_summary: 章节摘要
            character_states: 人物状态
            writing_phase: 写作阶段
            top_k: 返回数量
            
        Returns:
            相关风格参考片段
        """
        from style_rag.retrieval.query_builder import CreativeContext
        
        context = CreativeContext(
            storyline=storyline,
            chapter_summary=chapter_summary,
            character_states=character_states,
            writing_phase=writing_phase
        )
        
        query = self.query_builder.build_query_from_context(context)
        return self.retriever.retrieve(query, top_k=top_k)
    
    # ==================== 管理API ====================
    
    def get_stats(self) -> Dict:
        """
        获取索引统计信息
        
        Returns:
            包含 total_chunks, persist_dir, collection_name 等信息
        """
        stats = self.index_manager.get_stats()
        stats['embedding_model'] = self.embedding_manager.model_name
        stats['embedding_provider'] = self.embedding_manager.provider_name
        return stats
    
    def clear_index(self) -> bool:
        """
        清空索引
        
        Returns:
            是否成功
        """
        return self.index_manager.clear()
    
    def export_index(self, export_path: str) -> bool:
        """
        导出索引到指定路径
        
        Args:
            export_path: 导出路径
            
        Returns:
            是否成功
        """
        return self.index_manager.export_index(export_path)
    
    def import_index(self, import_path: str) -> bool:
        """
        从指定路径导入索引
        
        Args:
            import_path: 导入路径
            
        Returns:
            是否成功
        """
        return self.index_manager.import_index(import_path)
    
    # ==================== 便捷方法 ====================
    
    def format_references(
        self,
        results: List[Dict],
        max_length: int = 500
    ) -> str:
        """
        格式化检索结果为参考文本
        
        Args:
            results: 检索结果列表
            max_length: 每个片段的最大长度
            
        Returns:
            格式化的参考文本
        """
        if not results:
            return ""
        
        lines = ["## 写作风格参考\n"]
        lines.append("以下是与当前场景相似的优秀写作片段，请参考其用词和表达手法：\n")
        
        for i, result in enumerate(results, 1):
            chunk_type = result.get('metadata', {}).get('chunk_type', 'general')
            similarity = result.get('similarity', 0)
            content = result['content']
            
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            lines.append(f"### 参考{i} ({chunk_type}, 相似度: {similarity:.2f})")
            lines.append(f"```\n{content}\n```\n")
        
        lines.append(
            "\n> 请学习上述参考的用词习惯、句式结构和表达手法，但要创作全新的内容。\n"
        )
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"StyleRAGClient(db='{self.db_path}', "
            f"chunks={stats['total_chunks']}, "
            f"model='{self.embedding_manager.model_name}')"
        )
