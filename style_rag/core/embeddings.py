"""
统一Embedding管理器 - 支持本地/云端/混合模式
Unified Embedding Manager - Supporting Local/Cloud/Hybrid Modes
"""

import logging
from typing import List, Optional, Union
import numpy as np

from style_rag.core.embedding_config import EmbeddingConfig, EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    统一的Embedding管理器 - 支持本地/云端/混合
    Unified Embedding Manager - Supporting Local/Cloud/Hybrid
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        初始化Embedding管理器
        
        Args:
            config: Embedding配置，如果为None则使用默认本地配置
        """
        self.config = config or EmbeddingConfig()
        self._local_model = None
        self._gguf_model = None  # GGUF模型实例
        self._api_client = None
        self._api_model = None
        self._fallback_model = None
        self._dimension = None
        self._initialize()
    
    def _initialize(self):
        """初始化Embedding后端 / Initialize embedding backend"""
        provider = self.config.provider
        
        if provider == EmbeddingProvider.LOCAL:
            self._init_local_model()
        elif provider == EmbeddingProvider.LOCAL_GGUF:
            self._init_gguf_model()
        elif provider == EmbeddingProvider.LM_STUDIO:
            self._init_openai_compatible(
                base_url=self.config.lm_studio_url,
                api_key="not-needed",
                model=self.config.lm_studio_model
            )
        elif provider == EmbeddingProvider.OLLAMA:
            self._init_openai_compatible(
                base_url=f"{self.config.ollama_url}/v1",
                api_key="ollama",
                model=self.config.ollama_model
            )
        elif provider == EmbeddingProvider.ZENMUX:
            self._init_openai_compatible(
                base_url=self.config.zenmux_base_url,
                api_key=self.config.api_key or "zenmux-key",
                model=self.config.zenmux_model
            )
        elif provider == EmbeddingProvider.OPENROUTER:
            self._init_openai_compatible(
                base_url=self.config.openrouter_url,
                api_key=self.config.api_key,
                model=self.config.api_model or self.config.openrouter_model
            )
        else:
            # 云端API (OpenAI, 智谱, 阿里云, SiliconFlow)
            self._init_cloud_api()
        
        # 初始化降级备选
        if self.config.enable_fallback and self.config.fallback_to_local:
            self._init_fallback_local()
    
    def _init_local_model(self):
        """初始化本地sentence-transformers模型，优先使用GPU"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            import os
            
            # 模型缓存目录（项目本地）
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
            os.makedirs(cache_dir, exist_ok=True)
            
            # 设置环境变量，让transformers也使用本地缓存
            os.environ['HF_HOME'] = cache_dir
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            
            # 设备选择优先级：cuda > mps > cpu
            device = self.config.local_device
            if device == "auto":
                print("🔍 检测可用设备...")
                if torch.cuda.is_available():
                    device = "cuda"
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"   ✅ GPU 检测到: {gpu_name}")
                    logger.info(f"GPU detected: {gpu_name}")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                    print("   ✅ Apple MPS 检测到")
                    logger.info("Apple MPS detected")
                else:
                    device = "cpu"
                    print("   ⚠️  未检测到 GPU，使用 CPU")
                    logger.info("No GPU detected, using CPU")
            
            print(f"\n📥 正在加载嵌入模型: {self.config.local_model}")
            print(f"   缓存目录: {cache_dir}")
            print(f"   运行设备: {device}")
            print("   (首次运行需要下载模型，请耐心等待...)\n")
            
            logger.info(f"Loading local embedding model: {self.config.local_model}")
            logger.info(f"Model cache directory: {cache_dir}")
            logger.info(f"Using device: {device}")
            
            self._local_model = SentenceTransformer(
                self.config.local_model,
                device=device,
                cache_folder=cache_dir
            )
            
            # 获取embedding维度
            self._dimension = self._local_model.get_sentence_embedding_dimension()
            print(f"   ✅ 模型加载完成! 向量维度: {self._dimension}\n")
            logger.info(f"Local model loaded, dimension: {self._dimension}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embedding. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def _init_openai_compatible(self, base_url: str, api_key: str, model: str):
        """初始化OpenAI兼容API客户端"""
        try:
            from openai import OpenAI
            
            logger.info(f"Initializing OpenAI-compatible client: {base_url}")
            self._api_client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=self.config.timeout
            )
            self._api_model = model
        except ImportError:
            raise ImportError(
                "openai package is required for API embedding. "
                "Install with: pip install openai"
            )
    
    def _init_cloud_api(self):
        """初始化云端API客户端"""
        base_url = self.config.get_base_url()
        model = self.config.get_effective_model()
        
        if not self.config.api_key:
            raise ValueError(
                f"API key is required for provider: {self.config.provider.value}"
            )
        
        self._init_openai_compatible(
            base_url=base_url,
            api_key=self.config.api_key,
            model=model
        )
    
    def _init_fallback_local(self):
        """初始化降级用本地模型"""
        if self._local_model is not None:
            # 已经使用本地模型，不需要降级
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(
                f"Initializing fallback local model: {self.config.fallback_local_model}"
            )
            self._fallback_model = SentenceTransformer(
                self.config.fallback_local_model
            )
        except Exception as e:
            logger.warning(f"Failed to initialize fallback model: {e}")
            self._fallback_model = None
    
    def _init_gguf_model(self):
        """初始化GGUF量化模型 / Initialize GGUF quantized model"""
        try:
            from llama_cpp import Llama
            import os
            
            model_path = self.config.gguf_model_path
            if not model_path:
                raise ValueError("GGUF model path is not configured")
            
            # 支持相对路径
            if not os.path.isabs(model_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                model_path = os.path.join(base_dir, model_path)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"GGUF model not found: {model_path}")
            
            n_gpu_layers = self.config.gguf_n_gpu_layers
            device_info = f"GPU (layers: {n_gpu_layers})" if n_gpu_layers != 0 else "CPU"
            
            print(f"\n📥 正在加载GGUF嵌入模型: {os.path.basename(model_path)}")
            print(f"   模型路径: {model_path}")
            print(f"   运行设备: {device_info}")
            print("   (首次加载可能需要几秒钟...)\n")
            
            logger.info(f"Loading GGUF embedding model: {model_path}")
            logger.info(f"GPU layers: {n_gpu_layers}")
            
            self._gguf_model = Llama(
                model_path=model_path,
                embedding=True,  # 关键：启用嵌入模式
                n_gpu_layers=n_gpu_layers,
                n_ctx=self.config.gguf_n_ctx,
                n_batch=self.config.gguf_n_batch,
                verbose=False
            )
            
            self._dimension = self.config.gguf_embedding_dim
            print(f"   ✅ GGUF模型加载完成! 向量维度: {self._dimension}\n")
            logger.info(f"GGUF model loaded, dimension: {self._dimension}")
            
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF embedding. "
                "Install with: pip install llama-cpp-python\n"
                "For GPU support: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python"
            )
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        生成文本嵌入向量
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            嵌入向量数组，shape为 (n_texts, dimension)
        """
        # 确保输入是列表
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            texts = list(texts)
            single_input = False
        
        if not texts:
            return np.array([])
        
        try:
            if self._local_model is not None:
                embeddings = self._embed_local(texts)
            elif self._gguf_model is not None:
                embeddings = self._embed_gguf(texts)
            else:
                embeddings = self._embed_api(texts)
        except Exception as e:
            if self.config.enable_fallback and self._fallback_model is not None:
                logger.warning(f"Primary embedding failed, falling back to local: {e}")
                embeddings = self._embed_fallback(texts)
            else:
                raise
        
        return embeddings[0] if single_input else embeddings
    
    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """使用本地模型嵌入"""
        return self._local_model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
    
    def _embed_gguf(self, texts: List[str]) -> np.ndarray:
        """使用GGUF模型生成嵌入 / Generate embeddings using GGUF model"""
        embeddings = []
        for text in texts:
            try:
                result = self._gguf_model.create_embedding(text)
                # llama-cpp-python 返回格式: {'data': [{'embedding': [...], 'index': 0, 'object': 'embedding'}]}
                embeddings.append(result['data'][0]['embedding'])
            except Exception as e:
                logger.error(f"GGUF embedding failed for text: {e}")
                raise
        return np.array(embeddings)
    
    def _embed_api(self, texts: List[str]) -> np.ndarray:
        """
        使用API嵌入 - 支持并发
        Embed using API - with concurrency support
        """
        # 检查是否启用并发
        if self.config.enable_concurrency and self.config.max_concurrency > 1:
            return self._embed_api_concurrent(texts)
        else:
            return self._embed_api_sequential(texts)
    
    def _embed_api_sequential(self, texts: List[str]) -> np.ndarray:
        """顺序处理API嵌入 / Sequential API embedding"""
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            for attempt in range(self.config.max_retries):
                try:
                    response = self._api_client.embeddings.create(
                        model=self._api_model,
                        input=batch
                    )
                    batch_embeddings = [d.embedding for d in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    logger.warning(
                        f"API embedding attempt {attempt + 1} failed: {e}, retrying..."
                    )
        
        return np.array(all_embeddings)
    
    def _embed_api_concurrent(self, texts: List[str]) -> np.ndarray:
        """
        并发处理API嵌入 / Concurrent API embedding
        
        使用线程池实现并发，提高处理速度
        """
        import concurrent.futures
        
        # 分批
        batches = []
        for i in range(0, len(texts), self.config.batch_size):
            batches.append(texts[i:i + self.config.batch_size])
        
        all_embeddings = [None] * len(batches)
        max_workers = min(self.config.max_concurrency, len(batches))
        
        def embed_batch(batch_idx: int, batch: List[str]) -> tuple:
            """嵌入单个批次"""
            for attempt in range(self.config.max_retries):
                try:
                    response = self._api_client.embeddings.create(
                        model=self._api_model,
                        input=batch
                    )
                    embeddings = [d.embedding for d in response.data]
                    return batch_idx, embeddings
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        logger.error(f"Batch {batch_idx} failed after {self.config.max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Batch {batch_idx} attempt {attempt + 1} failed: {e}, retrying...")
            return batch_idx, []
        
        # 使用线程池并发执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(embed_batch, idx, batch): idx 
                for idx, batch in enumerate(batches)
            }
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_idx, embeddings = future.result()
                    all_embeddings[batch_idx] = embeddings
                except Exception as e:
                    logger.error(f"Concurrent embedding failed: {e}")
                    raise
        
        # 合并结果
        final_embeddings = []
        for batch_embeddings in all_embeddings:
            if batch_embeddings:
                final_embeddings.extend(batch_embeddings)
        
        return np.array(final_embeddings)
    
    def _embed_fallback(self, texts: List[str]) -> np.ndarray:
        """使用降级模型嵌入"""
        return self._fallback_model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
    
    @property
    def dimension(self) -> Optional[int]:
        """获取嵌入向量维度"""
        if self._dimension is not None:
            return self._dimension
        
        # 如果使用API，通过测试获取维度
        if self._api_client is not None:
            try:
                test_embedding = self.embed("test")
                self._dimension = len(test_embedding)
                return self._dimension
            except:
                pass
        
        return None
    
    @property
    def provider_name(self) -> str:
        """获取当前提供商名称"""
        return self.config.provider.value
    
    @property
    def model_name(self) -> str:
        """获取当前模型名称"""
        return self.config.get_effective_model()
