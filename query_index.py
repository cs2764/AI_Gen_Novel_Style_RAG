#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查询索引 - 从RAG索引中检索相关内容
Query Index - Search related content from RAG index

使用方法 / Usage:
    python query_index.py "战斗场景"          # 基本检索
    python query_index.py "月下相遇" -k 5     # 返回5条结果
    python query_index.py --stats             # 显示索引统计
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="从RAG索引中检索相关内容"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="检索查询"
    )
    parser.add_argument(
        "--db", "-d",
        default="./rag_index",
        help="数据库目录 (默认: ./rag_index)"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="返回结果数量 (默认: 5)"
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.3,
        help="最小相似度 (默认: 0.3)"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["dialogue", "description", "action"],
        help="按类型过滤"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="显示索引统计信息"
    )
    
    args = parser.parse_args()
    
    # 检查数据库
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"错误: 数据库不存在: {args.db}")
        print("请先运行 create_index.py 创建索引")
        sys.exit(1)
    
    # 导入模块
    from style_rag import StyleRAGClient
    
    # 创建客户端
    print("正在加载索引...")
    client = StyleRAGClient(db_path=args.db)
    
    # 显示统计
    if args.stats or not args.query:
        stats = client.get_stats()
        print(f"\n{'='*50}")
        print(f"索引统计信息")
        print(f"{'='*50}")
        print(f"  总片段数: {stats['total_chunks']}")
        print(f"  数据库路径: {stats['persist_dir']}")
        print(f"  嵌入模型: {stats['embedding_model']}")
        print(f"  提供商: {stats['embedding_provider']}")
        print(f"{'='*50}\n")
        
        if not args.query:
            return
    
    # 执行检索
    print(f"检索: {args.query}")
    print(f"参数: top_k={args.top_k}, min_similarity={args.min_sim}")
    if args.type:
        print(f"类型过滤: {args.type}")
    print()
    
    results = client.search(
        query=args.query,
        top_k=args.top_k,
        filter_type=args.type,
        min_similarity=args.min_sim
    )
    
    if not results:
        print("未找到相关结果")
        print("提示: 尝试降低 --min-sim 参数值")
        return
    
    print(f"找到 {len(results)} 条相关结果:\n")
    print("=" * 60)
    
    for i, r in enumerate(results, 1):
        similarity = r.get('similarity', 0)
        chunk_type = r.get('metadata', {}).get('chunk_type', 'unknown')
        encoding = r.get('metadata', {}).get('detected_encoding', 'unknown')
        content = r['content']
        
        # 截断过长内容
        if len(content) > 300:
            content = content[:300] + "..."
        
        print(f"\n【结果 {i}】相似度: {similarity:.3f} | 类型: {chunk_type} | 编码: {encoding}")
        print("-" * 60)
        print(content)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
