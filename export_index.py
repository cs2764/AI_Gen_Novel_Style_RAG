#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出索引 - 将RAG索引导出到指定目录用于备份或分享
Export Index - Export RAG index to a directory for backup or sharing

使用方法 / Usage:
    python export_index.py                    # 导出到默认目录
    python export_index.py --output ./backup  # 导出到指定目录
"""

import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="导出RAG索引"
    )
    parser.add_argument(
        "--db", "-d",
        default="./rag_index",
        help="数据库目录 (默认: ./rag_index)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="导出目录 (默认: ./exports/rag_backup_日期时间)"
    )
    
    args = parser.parse_args()
    
    # 检查数据库
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"错误: 数据库不存在: {args.db}")
        sys.exit(1)
    
    # 导入模块
    from style_rag import StyleRAGClient
    
    # 创建客户端
    print("正在加载索引...")
    client = StyleRAGClient(db_path=args.db)
    stats = client.get_stats()
    
    # 生成导出路径
    if args.output:
        export_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"./exports/rag_backup_{timestamp}"
    
    export_dir = Path(export_path)
    
    print(f"\n{'='*50}")
    print(f"导出RAG索引")
    print(f"{'='*50}")
    print(f"  源数据库: {args.db}")
    print(f"  片段数量: {stats['total_chunks']}")
    print(f"  嵌入模型: {stats['embedding_model']}")
    print(f"  导出路径: {export_path}")
    print(f"{'='*50}\n")
    
    # 执行导出
    print("正在导出...")
    success = client.export_index(export_path)
    
    if success:
        # 计算导出大小
        total_size = 0
        for p in export_dir.rglob('*'):
            if p.is_file():
                total_size += p.stat().st_size
        
        size_mb = total_size / (1024 * 1024)
        
        print(f"\n✅ 导出成功!")
        print(f"   路径: {export_dir.absolute()}")
        print(f"   大小: {size_mb:.2f} MB")
        print(f"\n   使用以下命令导入:")
        print(f"   python import_index.py --input \"{export_path}\"")
    else:
        print("\n❌ 导出失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
