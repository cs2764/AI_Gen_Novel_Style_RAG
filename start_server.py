#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动HTTP服务 - 提供REST API接口
Start HTTP Server - Provide REST API Interface

使用方法 / Usage:
    python start_server.py                    # 默认端口8086
    python start_server.py --port 8080        # 指定端口
    python start_server.py --host 0.0.0.0     # 允许外部访问
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="启动Style-RAG HTTP服务"
    )
    parser.add_argument(
        "--host", "-H",
        default="0.0.0.0",
        help="服务地址 (默认: 0.0.0.0 允许外部访问)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8086,
        help="服务端口 (默认: 8086)"
    )
    parser.add_argument(
        "--db", "-d",
        default="./rag_index",
        help="数据库目录 (默认: ./rag_index)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="开发模式（自动重载）"
    )
    
    args = parser.parse_args()
    
    # 检查数据库
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"⚠️  数据库不存在: {args.db}")
        print(f"   服务将使用空索引启动，可通过API添加数据")
    
    print(f"\n{'='*50}")
    print(f"Style-RAG HTTP 服务")
    print(f"{'='*50}")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  数据库: {args.db}")
    print(f"  API文档: http://{args.host}:{args.port}/docs")
    print(f"{'='*50}")
    print(f"\n按 Ctrl+C 停止服务\n")
    
    try:
        # 加载配置
        sys.path.append(os.getcwd())
        try:
            from model_config import (
                API_EMBEDDING_PROVIDER,
                API_EMBEDDING_BATCH_SIZE,
                API_SEARCH_TOP_K,
                API_LM_STUDIO_URL, API_LM_STUDIO_MODEL,
                API_OPENROUTER_API_KEY, API_OPENROUTER_MODEL, API_OPENROUTER_MAX_CONCURRENCY,
                API_SILICONFLOW_API_KEY, API_SILICONFLOW_MODEL, API_SILICONFLOW_MAX_CONCURRENCY,
                API_GGUF_MODEL_PATH, API_GGUF_N_GPU_LAYERS
            )
        except ImportError:
            print("❌ 未找到配置文件 model_config.py")
            print("   请从 model_config.py.example 复制并创建 model_config.py")
            sys.exit(1)

        import uvicorn
        
        # 设置基本环境变量
        os.environ['STYLE_RAG_DB_PATH'] = args.db
        os.environ['STYLE_RAG_EMBEDDING_PROVIDER'] = API_EMBEDDING_PROVIDER
        os.environ['STYLE_RAG_EMBEDDING_BATCH_SIZE'] = str(API_EMBEDDING_BATCH_SIZE)
        os.environ['STYLE_RAG_SEARCH_TOP_K'] = str(API_SEARCH_TOP_K)
        
        # 根据Provider设置特定环境变量
        if API_EMBEDDING_PROVIDER == "lm_studio":
            os.environ['STYLE_RAG_LM_STUDIO_URL'] = API_LM_STUDIO_URL
            os.environ['STYLE_RAG_LM_STUDIO_MODEL'] = API_LM_STUDIO_MODEL
        elif API_EMBEDDING_PROVIDER == "openrouter":
            os.environ['STYLE_RAG_OPENROUTER_API_KEY'] = API_OPENROUTER_API_KEY
            os.environ['STYLE_RAG_OPENROUTER_MODEL'] = API_OPENROUTER_MODEL
            os.environ['STYLE_RAG_MAX_CONCURRENCY'] = str(API_OPENROUTER_MAX_CONCURRENCY)
        elif API_EMBEDDING_PROVIDER == "siliconflow":
            os.environ['STYLE_RAG_SILICONFLOW_API_KEY'] = API_SILICONFLOW_API_KEY
            os.environ['STYLE_RAG_SILICONFLOW_MODEL'] = API_SILICONFLOW_MODEL
            os.environ['STYLE_RAG_MAX_CONCURRENCY'] = str(API_SILICONFLOW_MAX_CONCURRENCY)
        elif API_EMBEDDING_PROVIDER == "local_gguf":
            os.environ['STYLE_RAG_GGUF_MODEL_PATH'] = API_GGUF_MODEL_PATH
            os.environ['STYLE_RAG_GGUF_N_GPU_LAYERS'] = str(API_GGUF_N_GPU_LAYERS)
        
        uvicorn.run(
            "style_rag.api.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    except ImportError as e:
        if "model_config" in str(e):
             print("❌ 导入配置失败")
        else:
             print("❌ 需要安装 uvicorn:")
             print("   uv pip install uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n服务已停止")


if __name__ == "__main__":
    main()
