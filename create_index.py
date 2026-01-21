#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建RAG索引 - 从input目录创建新的向量索引
Create RAG Index - Create new vector index from input directory

使用方法 / Usage:
    python create_index.py                    # 使用默认设置
    python create_index.py --delete           # 索引后删除源文件
    python create_index.py --clear            # 先清空再索引
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="从input目录创建RAG索引"
    )
    parser.add_argument(
        "--input", "-i",
        default="./input",
        help="输入目录 (默认: ./input)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./rag_index",
        help="数据库目录 (默认: ./rag_index)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="创建前清空现有索引"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="索引成功后删除源文件"
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {args.input}")
        print(f"请创建 '{args.input}' 目录并放入要索引的文本文件")
        sys.exit(1)
    
    # 导入模块
    print("正在初始化...")
    from style_rag import StyleRAGClient
    from style_rag.indexing.article_loader import ArticleLoader
    
    # 统计文件
    loader = ArticleLoader()
    total_files = loader.count_files(args.input, recursive=True)
    
    if total_files == 0:
        print(f"警告: 目录 '{args.input}' 中没有找到 .txt 或 .md 文件")
        sys.exit(0)
    
    print(f"\n{'='*50}")
    print(f"创建RAG索引")
    print(f"{'='*50}")
    print(f"  输入目录: {args.input}")
    print(f"  输出目录: {args.output}")
    print(f"  文件数量: {total_files}")
    if args.delete:
        print(f"  ⚠️  索引后将删除源文件!")
    print(f"{'='*50}\n")
    
    # 创建客户端
    client = StyleRAGClient(db_path=args.output)
    
    # 清空现有索引
    if args.clear:
        print("清空现有索引...")
        client.clear_index()
    
    # 进度显示
    start_time = time.time()
    
    def progress(current, total, message):
        percent = (current / total * 100) if total > 0 else 0
        remaining = total - current
        elapsed = time.time() - start_time
        
        if current > 0:
            eta = (elapsed / current) * remaining
            eta_str = f"ETA: {eta:.0f}s" if eta < 60 else f"ETA: {eta/60:.1f}min"
        else:
            eta_str = "ETA: --"
        
        bar_len = 25
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        
        filename = message.replace("Processing: ", "")[:25]
        print(f"\r[{bar}] {percent:5.1f}% | {current}/{total} | 剩余:{remaining} | {eta_str} | {filename:<25}", end="", flush=True)
    
    # 开始索引
    result = client.index_directory(
        articles_dir=args.input,
        recursive=True,
        progress_callback=progress,
        delete_after_index=args.delete
    )
    
    # 计算用时
    total_time = time.time() - start_time
    time_str = f"{total_time:.1f}秒" if total_time < 60 else f"{total_time/60:.1f}分钟"
    
    # 显示结果
    print(f"\n\n{'='*50}")
    print("✅ 索引创建完成!")
    print(f"{'='*50}")
    print(f"  处理文件: {result.get('total_files', 0)}")
    print(f"  生成片段: {result.get('total_chunks', 0)}")
    print(f"  失败文件: {result.get('failed_files', 0)}")
    if args.delete:
        print(f"  已删除: {result.get('deleted_files', 0)} 个文件")
    print(f"  总用时: {time_str}")
    
    # 显示统计
    stats = client.get_stats()
    print(f"\n  当前索引总片段数: {stats['total_chunks']}")
    print(f"  嵌入模型: {stats['embedding_model']}")


if __name__ == "__main__":
    main()
