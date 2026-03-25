#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检索测试 - 交互式测试RAG检索功能
Retrieval Test - Interactive test for RAG retrieval

使用方法 / Usage:
    python test_retrieval.py                  # 交互式测试
    python test_retrieval.py --auto           # 自动测试（使用预设查询）
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def run_search(client, query, top_k=3, min_sim=0.3):
    """执行检索并显示结果"""
    print(f"\n🔍 查询: {query}")
    print("-" * 50)
    
    results = client.search(
        query=query,
        top_k=top_k,
        min_similarity=min_sim
    )
    
    if not results:
        print("   未找到相关结果")
        return 0
    
    for i, r in enumerate(results, 1):
        similarity = r.get('similarity', 0)
        chunk_type = r.get('metadata', {}).get('chunk_type', 'unknown')
        content = r['content']
        
        # 截断
        if len(content) > 150:
            content = content[:150] + "..."
        
        print(f"\n   [{i}] 相似度: {similarity:.3f} | 类型: {chunk_type}")
        print(f"       {content}")
    
    return len(results)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="测试RAG检索功能"
    )
    parser.add_argument(
        "--db", "-d",
        default="./rag_index",
        help="数据库目录 (默认: ./rag_index)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动测试模式（使用预设查询）"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=3,
        help="每次返回结果数 (默认: 3)"
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.3,
        help="最小相似度 (默认: 0.3)"
    )
    
    args = parser.parse_args()
    
    # 检查数据库
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"错误: 数据库不存在: {args.db}")
        print("请先运行 create_index.py 创建索引")
        sys.exit(1)
    
    # 导入模块
    print("正在加载索引...")
    from style_rag import StyleRAGClient
    
    client = StyleRAGClient(db_path=args.db)
    stats = client.get_stats()
    
    print(f"\n{'='*60}")
    print(f"RAG 检索测试")
    print(f"{'='*60}")
    print(f"   数据库: {args.db}")
    print(f"   片段数: {stats['total_chunks']}")
    print(f"   模型: {stats['embedding_model']}")
    print(f"{'='*60}")
    
    if stats['total_chunks'] == 0:
        print("\n⚠️  索引为空，请先添加文件")
        sys.exit(0)
    
    if args.auto:
        # 自动测试模式
        test_queries = [
            "战斗场景",
            "月下相遇",
            "离别伤感",
            "修炼突破",
            "对话场景",
        ]
        
        print(f"\n🤖 自动测试模式 - {len(test_queries)} 个预设查询\n")
        
        total_results = 0
        for query in test_queries:
            count = run_search(client, query, args.top_k, args.min_sim)
            total_results += count
        
        print(f"\n{'='*60}")
        print(f"测试完成: {len(test_queries)} 个查询, 共 {total_results} 条结果")
        print(f"{'='*60}\n")
        
    else:
        # 交互式模式
        print("\n📝 交互式测试模式")
        print("   输入查询内容进行检索，输入 'q' 或 'quit' 退出\n")
        
        while True:
            try:
                query = input("查询> ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['q', 'quit', 'exit', '退出']:
                    print("\n再见！")
                    break
                
                run_search(client, query, args.top_k, args.min_sim)
                print()
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except EOFError:
                break


if __name__ == "__main__":
    main()
