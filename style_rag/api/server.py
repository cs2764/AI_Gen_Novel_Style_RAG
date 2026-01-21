"""
HTTP服务 - FastAPI实现
HTTP Server - FastAPI Implementation
"""

import logging
import os
from typing import List, Dict, Optional, Union

logger = logging.getLogger(__name__)

# 延迟导入FastAPI相关模块
_app = None
_client = None


def get_embedding_config_from_env():
    """
    从环境变量获取嵌入模型配置
    
    环境变量:
        STYLE_RAG_EMBEDDING_PROVIDER: 嵌入模型提供商 (lm_studio, openrouter, siliconflow, local_gguf, local)
        STYLE_RAG_LM_STUDIO_URL: LM Studio API地址
        STYLE_RAG_LM_STUDIO_MODEL: LM Studio模型名称
        STYLE_RAG_OPENROUTER_API_KEY: OpenRouter API密钥
        STYLE_RAG_OPENROUTER_MODEL: OpenRouter模型名称
        STYLE_RAG_SILICONFLOW_API_KEY: SiliconFlow API密钥
        STYLE_RAG_SILICONFLOW_MODEL: SiliconFlow模型名称
        STYLE_RAG_GGUF_MODEL_PATH: GGUF模型路径
        STYLE_RAG_GGUF_N_GPU_LAYERS: GPU层数
        STYLE_RAG_EMBEDDING_BATCH_SIZE: 批处理大小
        STYLE_RAG_MAX_CONCURRENCY: 最大并发数
    """
    from style_rag.core.embedding_config import EmbeddingConfig, EmbeddingProvider
    
    provider_str = os.environ.get("STYLE_RAG_EMBEDDING_PROVIDER", "lm_studio").lower()
    batch_size = int(os.environ.get("STYLE_RAG_EMBEDDING_BATCH_SIZE", "60"))
    max_concurrency = int(os.environ.get("STYLE_RAG_MAX_CONCURRENCY", "20"))
    
    if provider_str == "lm_studio":
        return EmbeddingConfig(
            provider=EmbeddingProvider.LM_STUDIO,
            lm_studio_url=os.environ.get("STYLE_RAG_LM_STUDIO_URL", "http://localhost:1234/v1"),
            lm_studio_model=os.environ.get("STYLE_RAG_LM_STUDIO_MODEL", "text-embedding-qwen3-embedding-4b"),
            batch_size=batch_size
        )
    elif provider_str == "openrouter":
        return EmbeddingConfig(
            provider=EmbeddingProvider.OPENROUTER,
            api_key=os.environ.get("STYLE_RAG_OPENROUTER_API_KEY", ""),
            api_model=os.environ.get("STYLE_RAG_OPENROUTER_MODEL", "baai/bge-m3"),
            max_concurrency=max_concurrency,
            enable_concurrency=True,
            batch_size=batch_size
        )
    elif provider_str == "siliconflow":
        return EmbeddingConfig(
            provider=EmbeddingProvider.SILICONFLOW,
            api_key=os.environ.get("STYLE_RAG_SILICONFLOW_API_KEY", ""),
            api_model=os.environ.get("STYLE_RAG_SILICONFLOW_MODEL", "BAAI/bge-m3"),
            max_concurrency=max_concurrency,
            enable_concurrency=True,
            batch_size=batch_size
        )
    elif provider_str == "local_gguf":
        return EmbeddingConfig(
            provider=EmbeddingProvider.LOCAL_GGUF,
            gguf_model_path=os.environ.get("STYLE_RAG_GGUF_MODEL_PATH", "./models/Qwen3-Embedding-4B-Q8_0.gguf"),
            gguf_n_gpu_layers=int(os.environ.get("STYLE_RAG_GGUF_N_GPU_LAYERS", "-1")),
            batch_size=batch_size
        )
    else:  # "local" 或其他
        return EmbeddingConfig(
            provider=EmbeddingProvider.LOCAL,
            batch_size=batch_size
        )


def create_app(
    db_path: str = "./rag_db",
    embedding_config=None
):
    """
    创建FastAPI应用
    
    Args:
        db_path: 向量数据库路径
        embedding_config: 嵌入模型配置 (EmbeddingConfig对象)
        
    Returns:
        FastAPI应用实例
    """
    try:
        from fastapi import FastAPI, HTTPException, Request, Response
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        from starlette.types import Message
    except ImportError:
        raise ImportError(
            "FastAPI is required for HTTP server. "
            "Install with: pip install fastapi uvicorn"
        )
    
    from style_rag.api.client import StyleRAGClient
    
    # 创建应用
    app = FastAPI(
        title="Style-RAG API",
        description="独立RAG系统API - 用于风格学习和创作优化",
        version="1.0.0"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Logging Middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        import json
        
        # Log Request
        try:
            body_bytes = await request.body()
            body_str = body_bytes.decode("utf-8")
            
            # Re-create the body iterator so it can be read again by the route handler
            async def receive() -> Message:
                return {"type": "http.request", "body": body_bytes}
            request._receive = receive
            
            print(f"\n{'='*20} [REQUEST] {'='*20}")
            print(f"METHOD: {request.method}")
            print(f"URL: {request.url}")
            if body_str:
                print(f"BODY:\n{body_str}")
            print(f"{'='*50}\n")
        except Exception as e:
            print(f"[REQUEST LOG ERROR]: {e}")
            
        # Call the next handler
        response = await call_next(request)
        
        # Log Response
        try:
            # Capture response body
            resp_body = b""
            async for chunk in response.body_iterator:
                resp_body += chunk
            
            resp_str = resp_body.decode("utf-8")
            
            print(f"\n{'='*20} [RESPONSE] {'='*20}")
            print(f"STATUS: {response.status_code}")
            if resp_str:
                print(f"BODY:\n{resp_str}")
            print(f"{'='*50}\n")
            
            # Re-create response to return to client
            return Response(
                content=resp_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        except Exception as e:
            print(f"[RESPONSE LOG ERROR]: {e}")
            return response
    
    # 如果没有提供embedding_config，从环境变量读取
    if embedding_config is None:
        embedding_config = get_embedding_config_from_env()
    
    # 初始化客户端
    logger.info(f"初始化StyleRAGClient: db_path={db_path}, provider={embedding_config.provider}")
    client = StyleRAGClient(
        db_path=db_path,
        embedding_config=embedding_config
    )
    
    # ==================== 请求/响应模型 ====================
    
    class IndexTextsRequest(BaseModel):
        texts: List[str]
        metadatas: Optional[List[Dict]] = None
    
    class IndexDirectoryRequest(BaseModel):
        directory: str
        recursive: bool = True
        patterns: Optional[List[str]] = None
    
    class SearchRequest(BaseModel):
        query: str
        top_k: int = int(os.environ.get("STYLE_RAG_SEARCH_TOP_K", 5))
        filter_type: Optional[str] = None
        min_similarity: float = 0.5
    
    class SceneSearchRequest(BaseModel):
        scene_description: str
        emotion: Optional[str] = None
        writing_type: Optional[str] = None
        top_k: int = 3
    
    class SearchResult(BaseModel):
        content: str
        metadata: Dict
        similarity: Optional[float] = None
    
    class StatsResponse(BaseModel):
        total_chunks: int
        persist_dir: str
        collection_name: str
        embedding_model: str
        embedding_provider: str
    
    class IndexResult(BaseModel):
        total_files: int = 0
        total_texts: int = 0
        total_chunks: int
        success: bool = True
        errors: Optional[List[Dict]] = None
    
    # ==================== 路由 ====================
    
    @app.get("/")
    async def root():
        """API根路径"""
        return {
            "name": "Style-RAG API",
            "version": "1.0.0",
            "description": "独立RAG系统API - 用于风格学习和创作优化"
        }
    
    @app.get("/stats", response_model=StatsResponse)
    async def get_stats():
        """获取索引统计信息"""
        return client.get_stats()
    
    @app.post("/index/texts", response_model=IndexResult)
    async def index_texts(request: IndexTextsRequest):
        """索引文本内容"""
        try:
            result = client.index_texts(request.texts, request.metadatas)
            return IndexResult(
                total_texts=len(request.texts),
                total_chunks=result.get('total_chunks', 0)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/index/directory", response_model=IndexResult)
    async def index_directory(request: IndexDirectoryRequest):
        """索引目录"""
        try:
            result = client.index_directory(
                articles_dir=request.directory,
                recursive=request.recursive,
                file_patterns=request.patterns
            )
            return IndexResult(
                total_files=result.get('total_files', 0),
                total_chunks=result.get('total_chunks', 0),
                success=result.get('success', True),
                errors=result.get('errors')
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/search", response_model=List[SearchResult])
    async def search(request: SearchRequest):
        """语义检索"""
        try:
            results = client.search(
                query=request.query,
                top_k=request.top_k,
                filter_type=request.filter_type,
                min_similarity=request.min_similarity
            )
            return [
                SearchResult(
                    content=r['content'],
                    metadata=r['metadata'],
                    similarity=r.get('similarity')
                )
                for r in results
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/search/scene", response_model=List[SearchResult])
    async def search_by_scene(request: SceneSearchRequest):
        """按场景检索"""
        try:
            results = client.search_by_scene(
                scene_description=request.scene_description,
                emotion=request.emotion,
                writing_type=request.writing_type,
                top_k=request.top_k
            )
            return [
                SearchResult(
                    content=r['content'],
                    metadata=r['metadata'],
                    similarity=r.get('similarity')
                )
                for r in results
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/index")
    async def clear_index():
        """清空索引"""
        try:
            success = client.clear_index()
            return {"success": success}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="启动Style-RAG HTTP服务"
    )
    parser.add_argument(
        "--db", "-d",
        default="./rag_db",
        help="向量数据库路径 (default: ./rag_db)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="监听地址 (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="监听端口 (default: 8080)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="开发模式，自动重载"
    )
    
    args = parser.parse_args()
    
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install uvicorn")
        return
    
    embedding_config = get_embedding_config_from_env()
    
    print(f"Starting Style-RAG server...")
    print(f"  Database: {args.db}")
    print(f"  Embedding Provider: {embedding_config.provider}")
    print(f"  Address: http://{args.host}:{args.port}")
    
    app = create_app(db_path=args.db, embedding_config=embedding_config)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


# 模块级别的app实例，供uvicorn直接加载使用
# 使用环境变量配置
app = create_app(
    db_path=os.environ.get("STYLE_RAG_DB_PATH", "./rag_db"),
    embedding_config=None  # 将从环境变量自动读取
)


if __name__ == "__main__":
    main()
