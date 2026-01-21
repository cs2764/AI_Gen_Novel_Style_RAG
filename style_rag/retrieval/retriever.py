"""
风格检索器 - 多维度检索相关片段
Style Retriever - Multi-dimensional Retrieval of Related Segments
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from style_rag.core.embeddings import EmbeddingManager
from style_rag.core.vector_store import StyleVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalQuery:
    """检索查询 / Retrieval Query"""
    text: str                               # 查询文本
    scene_description: Optional[str] = None # 场景描述
    character_names: Optional[List[str]] = None  # 相关人物
    emotion_tone: Optional[str] = None      # 情感基调
    writing_type: Optional[str] = None      # 写作类型: dialogue/description/action
    style_preference: Optional[str] = None  # 风格偏好


class StyleRetriever:
    """
    风格检索器
    Style Retriever
    
    支持多维度检索相关文章片段，可选重排序
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: StyleVectorStore,
        default_top_k: int = 5,
        default_min_similarity: float = 0.5,
        enable_reranking: bool = False,
        reranker_model: str = "Qwen/Qwen3-Reranker-4B"
    ):
        """
        初始化检索器
        
        Args:
            embedding_manager: Embedding管理器
            vector_store: 向量存储
            default_top_k: 默认返回数量
            default_min_similarity: 默认最小相似度
            enable_reranking: 是否启用重排序
            reranker_model: 重排序模型名称
        """
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.default_min_similarity = default_min_similarity
        self.enable_reranking = enable_reranking
        self.reranker_model = reranker_model
        self._reranker = None
    
    def _get_reranker(self):
        """获取重排序器（延迟初始化）"""
        if self._reranker is None and self.enable_reranking:
            from style_rag.retrieval.reranker import Reranker
            self._reranker = Reranker(model_name=self.reranker_model)
        return self._reranker
    
    def _apply_reranking(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """应用重排序"""
        if not self.enable_reranking or not results:
            return results
        
        reranker = self._get_reranker()
        if reranker:
            return reranker.rerank_results(query, results, top_k=top_k)
        return results
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_type: Optional[str] = None,
        min_similarity: Optional[float] = None
    ) -> List[Dict]:
        """
        语义检索相关片段
        
        Args:
            query: 检索查询
            top_k: 返回数量
            filter_type: 过滤类型 (dialogue/description/action)
            min_similarity: 最小相似度阈值
            
        Returns:
            相关片段列表，每个包含 content, metadata, similarity
        """
        top_k = top_k or self.default_top_k
        min_similarity = min_similarity or self.default_min_similarity
        
        # 生成查询embedding
        query_embedding = self.embedding_manager.embed(query)
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        
        # 构建过滤条件
        filter_dict = None
        if filter_type:
            filter_dict = {"chunk_type": filter_type}
        
        # 检索
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k * 2,  # 多检索一些用于过滤
            filter_dict=filter_dict
        )
        
        # 过滤低相似度结果
        filtered_results = [
            r for r in results 
            if r.get('similarity', 0) >= min_similarity
        ]
        
        return filtered_results[:top_k]
    
    def search_by_scene(
        self,
        scene_description: str,
        emotion: Optional[str] = None,
        writing_type: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        按场景检索（专为创作优化）
        
        Args:
            scene_description: 场景描述
            emotion: 情感基调
            writing_type: 写作类型
            top_k: 返回数量
            
        Returns:
            相关片段列表
        """
        # 构建增强查询
        query_parts = [scene_description]
        
        if emotion:
            query_parts.append(f"情感氛围：{emotion}")
        if writing_type:
            query_parts.append(f"写作类型：{writing_type}")
        
        enhanced_query = " ".join(query_parts)
        
        return self.search(
            query=enhanced_query,
            top_k=top_k,
            filter_type=writing_type
        )
    
    def retrieve(
        self,
        query: RetrievalQuery,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        多维度检索
        
        Args:
            query: 检索查询对象
            top_k: 返回数量
            
        Returns:
            相关片段列表
        """
        # 构建语义查询字符串
        semantic_query = self._build_semantic_query(query)
        
        # 构建过滤条件
        filter_dict = self._build_filter(query)
        
        # 生成查询embedding
        query_embedding = self.embedding_manager.embed(semantic_query)
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        
        # 检索
        top_k = top_k or self.default_top_k
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k * 2,
            filter_dict=filter_dict
        )
        
        # 过滤低相似度
        filtered = [
            r for r in results 
            if r.get('similarity', 0) >= self.default_min_similarity
        ]
        
        return filtered[:top_k]
    
    def _build_semantic_query(self, query: RetrievalQuery) -> str:
        """构建语义查询字符串"""
        parts = []
        
        if query.text:
            parts.append(query.text)
        if query.scene_description:
            parts.append(query.scene_description)
        if query.emotion_tone:
            parts.append(f"情感氛围：{query.emotion_tone}")
        if query.writing_type:
            parts.append(f"写作类型：{query.writing_type}")
        if query.character_names:
            parts.append(f"相关人物：{'、'.join(query.character_names)}")
        
        return " ".join(parts)
    
    def _build_filter(self, query: RetrievalQuery) -> Optional[Dict]:
        """构建过滤条件"""
        filters = {}
        
        if query.writing_type:
            filters["chunk_type"] = query.writing_type
        if query.style_preference:
            filters["detected_style"] = query.style_preference
        
        return filters if filters else None
    
    def get_similar_chunks(
        self,
        content: str,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> List[Dict]:
        """
        获取与给定内容相似的片段
        
        Args:
            content: 给定内容
            top_k: 返回数量
            exclude_self: 是否排除自身（完全匹配）
            
        Returns:
            相似片段列表
        """
        results = self.search(content, top_k=top_k + 1 if exclude_self else top_k)
        
        if exclude_self:
            # 排除完全相同的内容
            results = [r for r in results if r['content'] != content]
            results = results[:top_k]
        
        return results
