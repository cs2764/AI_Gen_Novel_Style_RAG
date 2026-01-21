#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重置索引 - 清空或删除RAG索引数据库
Reset Index - Clear or delete RAG index database

使用方法 / Usage:
    python reset_index.py                     # 清空索引数据（保留数据库结构）
    python reset_index.py --delete-db         # 完全删除数据库目录
    python reset_index.py --force             # 不询问确认
"""

import sys
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="重置RAG索引"
    )
    parser.add_argument(
        "--db", "-d",
        default="./rag_index",
        help="数据库目录 (默认: ./rag_index)"
    )
    parser.add_argument(
        "--delete-db",
        action="store_true",
        help="完全删除数据库目录（而非仅清空）"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="不询问确认"
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    
    # 检查数据库是否存在
    if not db_path.exists():
        print(f"数据库目录不存在: {args.db}")
        print("无需重置")
        sys.exit(0)
    
    # 获取当前状态
    if not args.delete_db:
        try:
            from style_rag import StyleRAGClient
            client = StyleRAGClient(db_path=args.db)
            stats = client.get_stats()
            current_chunks = stats['total_chunks']
        except Exception:
            current_chunks = "未知"
    else:
        current_chunks = "N/A"
    
    # 显示信息
    print(f"\n{'='*50}")
    print(f"重置RAG索引")
    print(f"{'='*50}")
    print(f"  数据库路径: {args.db}")
    print(f"  当前片段数: {current_chunks}")
    print(f"  操作类型: {'删除数据库' if args.delete_db else '清空数据'}")
    print(f"{'='*50}\n")
    
    # 确认操作
    if not args.force:
        if args.delete_db:
            prompt = f"确定要删除整个数据库目录 '{args.db}' 吗? [y/N]: "
        else:
            prompt = f"确定要清空索引数据吗? [y/N]: "
        
        response = input(prompt).strip().lower()
        if response not in ['y', 'yes']:
            print("操作已取消")
            sys.exit(0)
    
    # 执行重置
    if args.delete_db:
        # 删除整个目录
        try:
            shutil.rmtree(db_path)
            print(f"\n✅ 数据库目录已删除: {args.db}")
        except Exception as e:
            print(f"\n❌ 删除失败: {e}")
            sys.exit(1)
    else:
        # 仅清空数据
        try:
            from style_rag import StyleRAGClient
            client = StyleRAGClient(db_path=args.db)
            success = client.clear_index()
            
            if success:
                stats = client.get_stats()
                print(f"\n✅ 索引已清空!")
                print(f"   当前片段数: {stats['total_chunks']}")
            else:
                print("\n❌ 清空失败")
                sys.exit(1)
        except Exception as e:
            print(f"\n❌ 清空失败: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
