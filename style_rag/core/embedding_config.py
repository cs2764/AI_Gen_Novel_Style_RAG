"""
Embedding配置 - 支持本地/云端/混合模式
Embedding Configuration - Supporting Local/Cloud/Hybrid Modes
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class EmbeddingProvider(Enum):
    """Embedding提供商 / Embedding Providers"""
    LOCAL = "local"                 # 本地模型 (sentence-transformers)
    LOCAL_GGUF = "local_gguf"       # 本地GGUF量化模型 (llama-cpp-python)
    OPENAI = "openai"               # OpenAI API
    OPENROUTER = "openrouter"       # OpenRouter API (多模型聚合)
    ZHIPU = "zhipu"                 # 智谱AI
    ALIYUN = "aliyun"               # 阿里云
    SILICONFLOW = "siliconflow"     # SiliconFlow
    ZENMUX = "zenmux"               # Zenmux聚合网关
    LM_STUDIO = "lm_studio"         # LM Studio本地
    OLLAMA = "ollama"               # Ollama本地


# 云端API基础URL / Cloud API base URLs
PROVIDER_BASE_URLS = {
    EmbeddingProvider.OPENAI: "https://api.openai.com/v1",
    EmbeddingProvider.OPENROUTER: "https://openrouter.ai/api/v1",
    EmbeddingProvider.ZHIPU: "https://open.bigmodel.cn/api/paas/v4",
    EmbeddingProvider.ALIYUN: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    EmbeddingProvider.SILICONFLOW: "https://api.siliconflow.cn/v1",
}

# 默认模型 / Default models
PROVIDER_DEFAULT_MODELS = {
    EmbeddingProvider.LOCAL: "Qwen/Qwen3-Embedding-4B",
    EmbeddingProvider.OPENAI: "text-embedding-3-small",
    EmbeddingProvider.OPENROUTER: "baai/bge-m3",
    EmbeddingProvider.ZHIPU: "embedding-3",
    EmbeddingProvider.ALIYUN: "text-embedding-v3",
    EmbeddingProvider.SILICONFLOW: "Qwen/Qwen3-Embedding-4B",
    EmbeddingProvider.LM_STUDIO: "nomic-embed-text",
    EmbeddingProvider.OLLAMA: "nomic-embed-text",
}


@dataclass
class EmbeddingConfig:
    """Embedding配置 / Embedding Configuration"""
    
    # 主要提供商 / Primary provider
    provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    
    # 本地模型配置 / Local model configuration
    local_model: str = "Qwen/Qwen3-Embedding-4B"
    local_device: str = "auto"  # auto, cpu, cuda, mps
    
    # 云端API配置 / Cloud API configuration
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_model: Optional[str] = None
    
    # Zenmux配置 / Zenmux configuration
    zenmux_base_url: str = "http://localhost:8000/v1"
    zenmux_model: str = "embedding-default"
    
    # LM Studio配置 / LM Studio configuration
    lm_studio_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "nomic-embed-text"
    
    # Ollama配置 / Ollama configuration
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"
    
    # 降级配置 / Fallback configuration
    enable_fallback: bool = True
    fallback_to_local: bool = True
    fallback_local_model: str = "Qwen/Qwen3-Embedding-0.6B"
    
    # GGUF模型配置 / GGUF model configuration
    gguf_model_path: Optional[str] = None      # GGUF文件路径
    gguf_n_gpu_layers: int = -1                # GPU层数，-1=全部，0=纯CPU
    gguf_n_ctx: int = 512                      # 上下文长度
    gguf_n_batch: int = 512                    # 批处理大小
    gguf_embedding_dim: int = 3584             # Qwen3-Embedding-4B维度
    
    # OpenRouter配置 / OpenRouter configuration
    openrouter_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "baai/bge-m3"
    
    # 批处理配置 / Batch processing configuration
    batch_size: int = 8  # 每次API调用的文本数量 (降低以避免超过token限制)
    max_retries: int = 3
    timeout: float = 60.0
    
    # 并发配置 / Concurrency configuration
    max_concurrency: int = 5          # 最大并发请求数
    enable_concurrency: bool = True   # 是否启用并发
    
    def get_effective_model(self) -> str:
        """获取实际使用的模型名称 / Get the effective model name"""
        if self.provider == EmbeddingProvider.LOCAL:
            return self.local_model
        elif self.provider == EmbeddingProvider.LOCAL_GGUF:
            return self.gguf_model_path or "GGUF model"
        elif self.provider == EmbeddingProvider.LM_STUDIO:
            return self.lm_studio_model
        elif self.provider == EmbeddingProvider.OLLAMA:
            return self.ollama_model
        elif self.provider == EmbeddingProvider.ZENMUX:
            return self.zenmux_model
        elif self.provider == EmbeddingProvider.OPENROUTER:
            return self.api_model or self.openrouter_model
        else:
            return self.api_model or PROVIDER_DEFAULT_MODELS.get(
                self.provider, "text-embedding-3-small"
            )
    
    def get_base_url(self) -> Optional[str]:
        """获取API基础URL / Get API base URL"""
        if self.api_base_url:
            return self.api_base_url
        elif self.provider == EmbeddingProvider.LM_STUDIO:
            return self.lm_studio_url
        elif self.provider == EmbeddingProvider.OLLAMA:
            return f"{self.ollama_url}/v1"
        elif self.provider == EmbeddingProvider.ZENMUX:
            return self.zenmux_base_url
        elif self.provider == EmbeddingProvider.OPENROUTER:
            return self.openrouter_url
        else:
            return PROVIDER_BASE_URLS.get(self.provider)
    
    def validate(self) -> bool:
        """验证配置 / Validate configuration"""
        # GGUF模型需要指定路径
        if self.provider == EmbeddingProvider.LOCAL_GGUF:
            if not self.gguf_model_path:
                raise ValueError("GGUF model path is required for LOCAL_GGUF provider")
        # 云端API需要key（除了LM Studio和Ollama）
        elif self.provider not in [
            EmbeddingProvider.LOCAL, 
            EmbeddingProvider.LM_STUDIO, 
            EmbeddingProvider.OLLAMA
        ]:
            if not self.api_key:
                raise ValueError(
                    f"API key is required for provider: {self.provider.value}"
                )
        return True


# 预设配置 / Preset configurations

def local_embedding_config(model: str = "BAAI/bge-large-zh-v1.5") -> EmbeddingConfig:
    """创建本地Embedding配置 / Create local embedding config"""
    return EmbeddingConfig(
        provider=EmbeddingProvider.LOCAL,
        local_model=model
    )


def openai_embedding_config(api_key: str, model: str = "text-embedding-3-small") -> EmbeddingConfig:
    """创建OpenAI Embedding配置 / Create OpenAI embedding config"""
    return EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI,
        api_key=api_key,
        api_model=model
    )


def zhipu_embedding_config(api_key: str, model: str = "embedding-3") -> EmbeddingConfig:
    """创建智谱AI Embedding配置 / Create Zhipu embedding config"""
    return EmbeddingConfig(
        provider=EmbeddingProvider.ZHIPU,
        api_key=api_key,
        api_model=model
    )


def siliconflow_embedding_config(
    api_key: str, 
    model: str = "BAAI/bge-large-zh-v1.5"
) -> EmbeddingConfig:
    """创建SiliconFlow Embedding配置 / Create SiliconFlow embedding config"""
    return EmbeddingConfig(
        provider=EmbeddingProvider.SILICONFLOW,
        api_key=api_key,
        api_model=model
    )


def lm_studio_embedding_config(
    url: str = "http://localhost:1234/v1",
    model: str = "nomic-embed-text"
) -> EmbeddingConfig:
    """创建LM Studio Embedding配置 / Create LM Studio embedding config"""
    return EmbeddingConfig(
        provider=EmbeddingProvider.LM_STUDIO,
        lm_studio_url=url,
        lm_studio_model=model
    )


def ollama_embedding_config(
    url: str = "http://localhost:11434",
    model: str = "nomic-embed-text"
) -> EmbeddingConfig:
    """创建Ollama Embedding配置 / Create Ollama embedding config"""
    return EmbeddingConfig(
        provider=EmbeddingProvider.OLLAMA,
        ollama_url=url,
        ollama_model=model
    )


def openrouter_embedding_config(
    api_key: str,
    model: str = "openai/text-embedding-3-small",
    max_concurrency: int = 5
) -> EmbeddingConfig:
    """
    创建OpenRouter Embedding配置 / Create OpenRouter embedding config
    
    Args:
        api_key: OpenRouter API密钥
        model: 嵌入模型名称 (格式: provider/model-name)
        max_concurrency: 最大并发请求数
    """
    return EmbeddingConfig(
        provider=EmbeddingProvider.OPENROUTER,
        api_key=api_key,
        api_model=model,
        max_concurrency=max_concurrency,
        enable_concurrency=True
    )
