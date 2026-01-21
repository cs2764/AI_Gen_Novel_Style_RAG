"""
查询命令行工具 - 检索相关片段
Query CLI Tool - Searching Related Segments
"""

import argparse
import sys
import logging


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="从Style-RAG向量数据库检索相关片段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本检索
  python -m style_rag.cli.query --db ./rag_db --query "激烈的战斗场面"

  # 指定返回数量和过滤类型
  python -m style_rag.cli.query --db ./rag_db --query "月下相遇" --top-k 5 --type description

  # 场景检索
  python -m style_rag.cli.query --db ./rag_db --query "离别场景" --emotion 悲伤
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--db", "-d",
        required=True,
        help="向量数据库路径"
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="检索查询"
    )
    
    # 检索选项
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="返回数量 (default: 5)"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["dialogue", "description", "action", "inner_monologue"],
        help="过滤类型"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.5,
        help="最小相似度阈值 (default: 0.5)"
    )
    parser.add_argument(
        "--emotion", "-e",
        choices=["激动", "温馨", "紧张", "悲伤", "神秘", "浪漫", "幽默", "平稳"],
        help="情感基调过滤"
    )
    
    # Embedding配置
    parser.add_argument(
        "--embedding-model", "-m",
        default="BAAI/bge-large-zh-v1.5",
        help="嵌入模型 (default: BAAI/bge-large-zh-v1.5)"
    )
    
    # 输出格式
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "markdown"],
        default="text",
        help="输出格式 (default: text)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=500,
        help="每个片段最大显示长度 (default: 500)"
    )
    
    # 其他
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="显示索引统计信息"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(args.verbose)
    
    # 创建客户端
    from style_rag.api.client import StyleRAGClient
    
    try:
        client = StyleRAGClient(
            db_path=args.db,
            embedding_model=args.embedding_model
        )
    except Exception as e:
        print(f"错误: 无法打开数据库 - {e}", file=sys.stderr)
        sys.exit(1)
    
    # 显示统计信息
    if args.stats:
        stats = client.get_stats()
        print("索引统计信息:")
        print(f"  总片段数: {stats['total_chunks']}")
        print(f"  数据库路径: {stats['persist_dir']}")
        print(f"  集合名称: {stats['collection_name']}")
        print(f"  嵌入模型: {stats['embedding_model']}")
        print(f"  提供商: {stats['embedding_provider']}")
        print()
    
    # 执行检索
    if args.emotion:
        results = client.search_by_scene(
            scene_description=args.query,
            emotion=args.emotion,
            writing_type=args.type,
            top_k=args.top_k
        )
    else:
        results = client.search(
            query=args.query,
            top_k=args.top_k,
            filter_type=args.type,
            min_similarity=args.min_similarity
        )
    
    # 输出结果
    if not results:
        print("未找到相关片段")
        sys.exit(0)
    
    if args.format == "json":
        import json
        output = []
        for r in results:
            output.append({
                "content": r['content'],
                "metadata": r['metadata'],
                "similarity": r.get('similarity')
            })
        print(json.dumps(output, ensure_ascii=False, indent=2))
    
    elif args.format == "markdown":
        print(client.format_references(results, max_length=args.max_length))
    
    else:  # text
        print(f"找到 {len(results)} 个相关片段:\n")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity', 0)
            chunk_type = result.get('metadata', {}).get('chunk_type', 'unknown')
            content = result['content']
            
            if len(content) > args.max_length:
                content = content[:args.max_length] + "..."
            
            print(f"\n【结果 {i}】相似度: {similarity:.3f} | 类型: {chunk_type}")
            print("-" * 60)
            print(content)
            print()
        
        print("=" * 60)


if __name__ == "__main__":
    main()
