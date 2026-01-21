"""
重排序器 - 使用Qwen3-Reranker对检索结果进行精排
Reranker - Using Qwen3-Reranker for result reranking
"""

import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class Reranker:
    """
    重排序器
    Reranker
    
    使用Qwen3-Reranker或其他重排序模型对检索结果进行精排
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        device: str = "auto",
        use_fp16: bool = True
    ):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称
            device: 设备 (auto, cpu, cuda)
            use_fp16: 是否使用FP16
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self._model = None
        self._tokenizer = None
        self._initialized = False
    
    def _initialize(self):
        """延迟初始化模型"""
        if self._initialized:
            return
        
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            logger.info(f"Loading reranker model: {self.model_name}")
            
            # 确定设备
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # 加载tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 加载模型
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.use_fp16 and device == "cuda" else torch.float32
            )
            self._model.to(device)
            self._model.eval()
            
            self._device = device
            self._initialized = True
            logger.info(f"Reranker model loaded on {device}")
            
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for reranking. "
                f"Install with: pip install transformers torch. Error: {e}"
            )
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前k个结果（None表示全部）
            
        Returns:
            重排序后的 (原始索引, 分数) 列表
        """
        if not documents:
            return []
        
        self._initialize()
        
        import torch
        
        # 构建查询-文档对
        pairs = [[query, doc] for doc in documents]
        
        # 编码
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self._device)
            
            # 获取分数
            outputs = self._model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().tolist()
        
        # 如果只有一个文档，scores可能是float
        if isinstance(scores, float):
            scores = [scores]
        
        # 创建 (索引, 分数) 对并排序
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]
        
        return indexed_scores
    
    def rerank_results(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            results: 检索结果列表，每个包含 'content' 键
            top_k: 返回前k个结果
            
        Returns:
            重排序后的结果列表，添加 'rerank_score' 键
        """
        if not results:
            return []
        
        documents = [r['content'] for r in results]
        ranked = self.rerank(query, documents, top_k)
        
        reranked_results = []
        for orig_idx, score in ranked:
            result = results[orig_idx].copy()
            result['rerank_score'] = score
            reranked_results.append(result)
        
        return reranked_results
    
    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        批量预测查询-文档对的相关性分数
        
        Args:
            pairs: (query, document) 元组列表
            
        Returns:
            分数列表
        """
        if not pairs:
            return []
        
        self._initialize()
        
        import torch
        
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self._device)
            
            outputs = self._model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().tolist()
        
        if isinstance(scores, float):
            scores = [scores]
        
        return scores
    
    @property
    def is_initialized(self) -> bool:
        """检查模型是否已初始化"""
        return self._initialized
