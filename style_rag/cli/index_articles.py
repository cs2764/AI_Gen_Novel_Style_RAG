"""
索引命令行工具 - 构建文章索引
Index CLI Tool - Building Article Index
"""

import argparse
import sys
import logging
from pathlib import Path


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="索引文章到Style-RAG向量数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法 - 索引整个目录
  python -m style_rag.cli.index_articles --input ./articles --output ./rag_db

  # 指定嵌入模型
  python -m style_rag.cli.index_articles --input ./articles --output ./rag_db \\
      --embedding-model "BAAI/bge-large-zh-v1.5"

  # 仅索引特定类型文件
  python -m style_rag.cli.index_articles --input ./articles --output ./rag_db \\
      --patterns "*.txt" "*.md"

  # 使用云端Embedding
  python -m style_rag.cli.index_articles --input ./articles --output ./rag_db \\
      --provider siliconflow --api-key YOUR_API_KEY
        """
    )
    
    # 输入输出
    parser.add_argument(
        "--input", "-i",
        default="./input",
        help="文章目录路径 (default: ./input)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./rag_db",
        help="向量数据库输出路径 (default: ./rag_db)"
    )
    
    # Embedding配置
    parser.add_argument(
        "--embedding-model", "-m",
        default="Qwen/Qwen3-Embedding-4B",
        help="嵌入模型名称 (default: Qwen/Qwen3-Embedding-4B)"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["local", "openai", "zhipu", "aliyun", "siliconflow", "lm_studio", "ollama"],
        default="local",
        help="Embedding提供商 (default: local)"
    )
    parser.add_argument(
        "--api-key",
        help="云端API密钥（使用云端提供商时需要）"
    )
    parser.add_argument(
        "--api-url",
        help="自定义API URL（可选）"
    )
    
    # 索引选项
    parser.add_argument(
        "--patterns",
        nargs="+",
        help="文件匹配模式 (如 *.txt *.md)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="不递归子目录"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="分块大小 (default: 512)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="分块重叠 (default: 50)"
    )
    
    # 其他
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="索引前清空现有数据"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="索引成功后删除源文件（谨慎使用！）"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # 验证输入目录
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入目录不存在: {args.input}")
        sys.exit(1)
    if not input_path.is_dir():
        logger.error(f"输入路径不是目录: {args.input}")
        sys.exit(1)
    
    # 构建Embedding配置
    from style_rag.core.embedding_config import (
        EmbeddingConfig, 
        EmbeddingProvider
    )
    
    provider_map = {
        "local": EmbeddingProvider.LOCAL,
        "openai": EmbeddingProvider.OPENAI,
        "zhipu": EmbeddingProvider.ZHIPU,
        "aliyun": EmbeddingProvider.ALIYUN,
        "siliconflow": EmbeddingProvider.SILICONFLOW,
        "lm_studio": EmbeddingProvider.LM_STUDIO,
        "ollama": EmbeddingProvider.OLLAMA,
    }
    
    embedding_config = EmbeddingConfig(
        provider=provider_map[args.provider],
        local_model=args.embedding_model,
        api_key=args.api_key,
        api_model=args.embedding_model,
        api_base_url=args.api_url
    )
    
    # 构建RAG配置
    from style_rag.core.config import RAGConfig
    
    rag_config = RAGConfig(
        vector_db_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # 创建客户端
    from style_rag.api.client import StyleRAGClient
    
    logger.info("初始化RAG客户端...")
    try:
        client = StyleRAGClient(
            db_path=args.output,
            embedding_config=embedding_config,
            rag_config=rag_config
        )
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        sys.exit(1)
    
    # 清空现有索引
    if args.clear:
        logger.info("清空现有索引...")
        client.clear_index()
    
    # 计算文件数量
    from style_rag.indexing.article_loader import ArticleLoader
    loader = ArticleLoader()
    total_files = loader.count_files(args.input, not args.no_recursive)
    
    if total_files == 0:
        print(f"警告: 目录 {args.input} 中没有找到支持的文件 (.txt, .md)")
        sys.exit(0)
    
    print(f"\n找到 {total_files} 个文件待索引")
    print("-" * 50)
    
    # 进度回调 - 详细显示
    import time
    start_time = time.time()
    
    def progress_callback(current: int, total: int, message: str):
        percent = (current / total * 100) if total > 0 else 0
        remaining = total - current
        elapsed = time.time() - start_time
        
        # 计算预估剩余时间
        if current > 0:
            avg_time = elapsed / current
            eta = avg_time * remaining
            eta_str = f"ETA: {eta:.0f}s" if eta < 60 else f"ETA: {eta/60:.1f}min"
        else:
            eta_str = "ETA: --"
        
        # 显示进度条
        bar_len = 20
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        
        filename = message[:30] + "..." if len(message) > 30 else message
        print(f"\r[{bar}] {percent:5.1f}% | {current}/{total} | 剩余:{remaining} | {eta_str} | {filename:<35}", end="", flush=True)
    
    # 开始索引
    logger.info(f"开始索引: {args.input}")
    logger.info(f"输出位置: {args.output}")
    logger.info(f"嵌入模型: {args.embedding_model}")
    logger.info(f"提供商: {args.provider}")
    
    # 如果启用了删除，显示警告
    if args.delete:
        print("\n⚠️  警告: 已启用 --delete 选项，索引成功的文件将被删除！")
        print("-" * 50)
    
    try:
        result = client.index_directory(
            articles_dir=args.input,
            recursive=not args.no_recursive,
            file_patterns=args.patterns,
            progress_callback=progress_callback,
            delete_after_index=args.delete
        )
        print()  # 换行
    except Exception as e:
        print()
        logger.error(f"索引失败: {e}")
        sys.exit(1)
    
    # 计算总用时
    total_time = time.time() - start_time
    time_str = f"{total_time:.1f}秒" if total_time < 60 else f"{total_time/60:.1f}分钟"
    
    # 输出结果
    print("\n" + "=" * 50)
    print("索引完成!")
    print("=" * 50)
    print(f"  总文件数: {total_files}")
    print(f"  处理文件数: {result.get('total_files', 0)}")
    print(f"  生成片段数: {result.get('total_chunks', 0)}")
    print(f"  失败文件数: {result.get('failed_files', 0)}")
    if args.delete:
        print(f"  已删除文件: {result.get('deleted_files', 0)}")
    print(f"  总用时: {time_str}")
    print(f"  数据库位置: {args.output}")
    
    if result.get('errors'):
        print("\n错误列表:")
        for err in result['errors'][:5]:
            print(f"  - {err['file']}: {err['error']}")
        if len(result['errors']) > 5:
            print(f"  ... 还有 {len(result['errors']) - 5} 个错误")
    
    # 显示统计
    stats = client.get_stats()
    print(f"\n当前索引总片段数: {stats['total_chunks']}")


if __name__ == "__main__":
    main()
