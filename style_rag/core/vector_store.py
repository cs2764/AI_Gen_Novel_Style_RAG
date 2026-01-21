"""
向量存储管理 - ChromaDB集成
Vector Store Management - ChromaDB Integration
"""

import logging
import hashlib
from typing import List, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class StyleVectorStore:
    """
    风格向量存储
    Style Vector Store
    
    使用ChromaDB作为向量数据库后端
    """
    
    def __init__(
        self,
        persist_dir: str = "./rag_db",
        collection_name: str = "article_styles"
    ):
        """
        初始化向量存储
        
        Args:
            persist_dir: 持久化目录
            collection_name: 集合名称
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._initialize()
    
    def _initialize(self):
        """初始化ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # 创建目录
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            # 初始化持久化客户端
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 获取或创建集合
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(
                f"Vector store initialized: {self.persist_dir}, "
                f"collection: {self.collection_name}, "
                f"count: {self._collection.count()}"
            )
            
        except ImportError:
            raise ImportError(
                "chromadb is required for vector storage. "
                "Install with: pip install chromadb"
            )
    
    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> int:
        """
        添加文章片段到向量库
        
        Args:
            chunks: 片段列表，每个包含 'content' 和 'metadata'
            embeddings: 对应的嵌入向量列表
            ids: 可选的ID列表
            
        Returns:
            添加的数量
        """
        if not chunks or not embeddings:
            return 0
        
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have same length")
        
        # 生成ID (包含索引确保唯一性)
        if ids is None:
            ids = [self._generate_id(c['content'], i, c.get('metadata', {})) for i, c in enumerate(chunks)]
        
        # 提取文档和元数据
        documents = [c['content'] for c in chunks]
        metadatas = [c.get('metadata', {}) for c in chunks]
        
        # 确保metadata值是基本类型
        metadatas = [self._sanitize_metadata(m) for m in metadatas]
        
        # ChromaDB 批量大小限制 (最大约5461，使用5000作为安全值)
        CHROMA_BATCH_SIZE = 5000
        total_added = 0
        
        try:
            # 分批添加以避免超过ChromaDB限制
            for i in range(0, len(chunks), CHROMA_BATCH_SIZE):
                batch_end = min(i + CHROMA_BATCH_SIZE, len(chunks))
                self._collection.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )
                total_added += (batch_end - i)
            
            logger.info(f"Added {total_added} chunks to vector store")
            return total_added
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
        include_distances: bool = True
    ) -> List[Dict]:
        """
        查询相似片段
        
        Args:
            query_embedding: 查询向量
            n_results: 返回数量
            filter_dict: 过滤条件（ChromaDB where clause）
            include_distances: 是否包含距离
            
        Returns:
            相似片段列表，每个包含 content, metadata, similarity
        """
        include = ["documents", "metadatas"]
        if include_distances:
            include.append("distances")
        
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_dict,
                include=include
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
        
        # 转换结果格式
        output = []
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0] if include_distances else [0] * len(documents)
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            output.append({
                'content': doc,
                'metadata': meta,
                'similarity': 1 - dist if include_distances else None,
                'distance': dist if include_distances else None
            })
        
        return output
    
    def search_by_text(
        self,
        query_text: str,
        embedding_func,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        通过文本查询（自动生成embedding）
        
        Args:
            query_text: 查询文本
            embedding_func: 生成embedding的函数
            n_results: 返回数量
            filter_dict: 过滤条件
            
        Returns:
            相似片段列表
        """
        query_embedding = embedding_func(query_text)
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        return self.query(query_embedding, n_results, filter_dict)
    
    def delete_by_ids(self, ids: List[str]) -> int:
        """删除指定ID的片段"""
        try:
            self._collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks")
            return len(ids)
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return 0
    
    def delete_by_filter(self, filter_dict: Dict) -> int:
        """根据过滤条件删除片段"""
        try:
            self._collection.delete(where=filter_dict)
            logger.info(f"Deleted chunks by filter: {filter_dict}")
            return -1  # ChromaDB不返回删除数量
        except Exception as e:
            logger.error(f"Delete by filter failed: {e}")
            return 0
    
    def clear(self) -> bool:
        """清空集合"""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Vector store cleared")
            return True
        except Exception as e:
            logger.error(f"Clear failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        return {
            'total_chunks': self._collection.count(),
            'persist_dir': str(self.persist_dir),
            'collection_name': self.collection_name
        }
    
    def count(self) -> int:
        """获取片段总数"""
        return self._collection.count()
    
    def _generate_id(self, content: str, index: int = 0, metadata: Dict = None) -> str:
        """
        生成内容ID
        
        Args:
            content: 片段内容
            index: 片段在批次中的索引
            metadata: 元数据（用于增加唯一性）
        """
        # 组合多个因素确保唯一性：内容 + 索引 + 来源文件
        source = metadata.get('source', '') if metadata else ''
        unique_str = f"{content}_{index}_{source}"
        hash_obj = hashlib.md5(unique_str.encode('utf-8'))
        return f"chunk_{hash_obj.hexdigest()[:16]}"
    
    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """清理metadata，确保值是基本类型"""
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif isinstance(v, list):
                # 转换为字符串
                sanitized[k] = str(v)
            elif v is None:
                continue
            else:
                sanitized[k] = str(v)
        return sanitized
