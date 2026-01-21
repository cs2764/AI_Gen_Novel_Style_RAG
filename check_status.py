#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查看索引状态 - 显示RAG索引的详细统计信息
Check Index Status - Display detailed statistics of RAG index

使用方法 / Usage:
    python check_status.py                    # 显示索引状态
    python check_status.py --db ./my_rag_db   # 指定数据库路径
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_dir_size(path):
    """计算目录总大小"""
    total = 0
    for p in Path(path).rglob('*'):
        if p.is_file():
            total += p.stat().st_size
    return total


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="查看RAG索引状态"
    )
    parser.add_argument(
        "--db", "-d",
        default="./rag_index",
        help="数据库目录 (默认: ./rag_index)"
    )
    parser.add_argument(
        "--input", "-i",
        default="./input",
        help="输入目录 (默认: ./input)"
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    input_path = Path(args.input)
    
    print(f"\n{'='*60}")
    print(f"Style-RAG 索引状态")
    print(f"{'='*60}\n")
    
    # 检查输入目录
    print("📁 输入目录状态:")
    print("-" * 40)
    if input_path.exists():
        from style_rag.indexing.article_loader import ArticleLoader
        loader = ArticleLoader()
        file_count = loader.count_files(str(input_path), recursive=True)
        print(f"   路径: {input_path.absolute()}")
        print(f"   待索引文件: {file_count} 个 (.txt, .md)")
    else:
        print(f"   路径: {input_path} (不存在)")
        print(f"   提示: 请创建该目录并放入文本文件")
    
    print()
    
    # 检查数据库
    print("🗄️  数据库状态:")
    print("-" * 40)
    
    if not db_path.exists():
        print(f"   路径: {args.db} (不存在)")
        print(f"   状态: 未创建")
        print(f"   提示: 运行 create_index.py 创建索引")
    else:
        # 计算数据库大小
        db_size = get_dir_size(db_path)
        print(f"   路径: {db_path.absolute()}")
        print(f"   大小: {format_size(db_size)}")
        
        try:
            from style_rag import StyleRAGClient
            client = StyleRAGClient(db_path=args.db)
            stats = client.get_stats()
            
            print(f"\n📊 索引统计:")
            print("-" * 40)
            print(f"   总片段数: {stats['total_chunks']}")
            print(f"   集合名称: {stats['collection_name']}")
            print(f"   嵌入模型: {stats['embedding_model']}")
            print(f"   嵌入提供商: {stats['embedding_provider']}")
            
            # 简单功能测试
            if stats['total_chunks'] > 0:
                print(f"\n✅ 索引状态: 正常")
            else:
                print(f"\n⚠️  索引状态: 空索引")
                print(f"   提示: 运行 add_files.py 添加文件")
                
        except Exception as e:
            print(f"\n❌ 读取索引失败: {e}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
