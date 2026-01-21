#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导入索引 - 从备份目录导入RAG索引
Import Index - Import RAG index from backup directory

使用方法 / Usage:
    python import_index.py --input ./backup   # 从指定目录导入
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="导入RAG索引"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="导入源目录（包含导出的索引）"
    )
    parser.add_argument(
        "--db", "-d",
        default="./rag_index",
        help="目标数据库目录 (默认: ./rag_index)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="覆盖现有索引（不询问确认）"
    )
    
    args = parser.parse_args()
    
    # 检查导入源
    import_path = Path(args.input)
    if not import_path.exists():
        print(f"错误: 导入源不存在: {args.input}")
        sys.exit(1)
    
    chroma_dir = import_path / 'chroma_db'
    if not chroma_dir.exists():
        print(f"错误: 无效的导出目录，未找到 chroma_db")
        sys.exit(1)
    
    # 检查元数据
    metadata_file = import_path / 'metadata.json'
    if metadata_file.exists():
        import json
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = None
    
    db_path = Path(args.db)
    
    print(f"\n{'='*50}")
    print(f"导入RAG索引")
    print(f"{'='*50}")
    print(f"  导入源: {args.input}")
    if metadata:
        print(f"  导出时间: {metadata.get('exported_at', 'N/A')}")
        stats = metadata.get('stats', {})
        print(f"  片段数量: {stats.get('total_chunks', 'N/A')}")
        print(f"  嵌入模型: {metadata.get('embedding_model', 'N/A')}")
    print(f"  目标位置: {args.db}")
    print(f"{'='*50}\n")
    
    # 确认覆盖
    if db_path.exists() and not args.force:
        response = input("目标位置已存在索引，是否覆盖? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("操作已取消")
            sys.exit(0)
    
    # 导入模块
    from style_rag import StyleRAGClient
    
    # 创建客户端
    print("正在导入...")
    client = StyleRAGClient(db_path=args.db)
    
    # 执行导入
    success = client.import_index(args.input)
    
    if success:
        stats = client.get_stats()
        print(f"\n✅ 导入成功!")
        print(f"   当前片段数: {stats['total_chunks']}")
        print(f"   嵌入模型: {stats['embedding_model']}")
    else:
        print("\n❌ 导入失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
