#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Style-RAG 管理程序 - 交互式菜单
Style-RAG Manager - Interactive Menu

使用方法 / Usage:
    python run.py
"""

import sys
import os
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 加载模型配置
try:
    from model_config import *
except ImportError:
    print("\n❌ 未找到配置文件 model_config.py")
    print("   请从 model_config.py.example 复制并创建 model_config.py")
    print("   Please copy model_config.py.example to model_config.py")
    sys.exit(1)

# 全局客户端实例 (单例模式，避免重复加载模型)
_CLIENT = None


def get_client(force_new: bool = False) -> 'StyleRAGClient':
    """
    获取StyleRAGClient单例实例
    
    Args:
        force_new: 是否强制创建新实例（用于重置后重新初始化）
    
    Returns:
        StyleRAGClient实例
    """
    global _CLIENT
    if _CLIENT is None or force_new:
        from style_rag import StyleRAGClient
        from style_rag.core.embedding_config import EmbeddingConfig, EmbeddingProvider
        
        print("\n⏳ 正在初始化RAG客户端...")
        
        # 根据配置选择嵌入模型类型
        if EMBEDDING_PROVIDER == "lm_studio":
            print(f"   📦 使用LM Studio嵌入模型: {LM_STUDIO_MODEL}")
            print(f"   🔗 API地址: {LM_STUDIO_URL}")
            embedding_config = EmbeddingConfig(
                provider=EmbeddingProvider.LM_STUDIO,
                lm_studio_url=LM_STUDIO_URL,
                lm_studio_model=LM_STUDIO_MODEL,
                batch_size=EMBEDDING_BATCH_SIZE
            )
        elif EMBEDDING_PROVIDER == "openrouter":
            print(f"   📦 使用OpenRouter嵌入模型: {OPENROUTER_MODEL}")
            print(f"   🚀 并发数: {OPENROUTER_MAX_CONCURRENCY}, 批大小: {EMBEDDING_BATCH_SIZE}")
            if not OPENROUTER_API_KEY:
                print("   ⚠️  警告: 未设置OPENROUTER_API_KEY")
            embedding_config = EmbeddingConfig(
                provider=EmbeddingProvider.OPENROUTER,
                api_key=OPENROUTER_API_KEY,
                api_model=OPENROUTER_MODEL,
                max_concurrency=OPENROUTER_MAX_CONCURRENCY,
                enable_concurrency=True,
                batch_size=EMBEDDING_BATCH_SIZE
            )
        elif EMBEDDING_PROVIDER == "siliconflow":
            print(f"   📦 使用SiliconFlow嵌入模型: {SILICONFLOW_MODEL}")
            print(f"   🚀 并发数: {SILICONFLOW_MAX_CONCURRENCY}, 批大小: {EMBEDDING_BATCH_SIZE}")
            if not SILICONFLOW_API_KEY:
                print("   ⚠️  警告: 未设置SILICONFLOW_API_KEY")
            embedding_config = EmbeddingConfig(
                provider=EmbeddingProvider.SILICONFLOW,
                api_key=SILICONFLOW_API_KEY,
                api_model=SILICONFLOW_MODEL,
                max_concurrency=SILICONFLOW_MAX_CONCURRENCY,
                enable_concurrency=True,
                batch_size=EMBEDDING_BATCH_SIZE
            )
        elif EMBEDDING_PROVIDER == "local_gguf":
            print(f"   📦 使用GGUF量化模型: {GGUF_MODEL_PATH}")
            embedding_config = EmbeddingConfig(
                provider=EmbeddingProvider.LOCAL_GGUF,
                gguf_model_path=GGUF_MODEL_PATH,
                gguf_n_gpu_layers=GGUF_N_GPU_LAYERS,
                batch_size=EMBEDDING_BATCH_SIZE
            )
        else:  # "local" 或其他
            print("   📦 使用sentence-transformers模型")
            embedding_config = EmbeddingConfig(
                provider=EmbeddingProvider.LOCAL,
                batch_size=EMBEDDING_BATCH_SIZE
            )
        
        _CLIENT = StyleRAGClient(
            db_path=DEFAULT_DB,
            embedding_config=embedding_config
        )
        print("✅ RAG客户端已就绪\n")
    return _CLIENT


def cleanup_client():
    """清理客户端实例，释放内存和文件句柄"""
    global _CLIENT
    if _CLIENT is not None:
        # 尝试显式释放模型资源
        try:
            # 清理向量存储（ChromaDB）
            if hasattr(_CLIENT, 'vector_store') and _CLIENT.vector_store is not None:
                vs = _CLIENT.vector_store
                # 关闭ChromaDB客户端连接
                if hasattr(vs, '_client') and vs._client is not None:
                    try:
                        # ChromaDB PersistentClient 需要显式关闭
                        if hasattr(vs._client, '_system') and vs._client._system is not None:
                            vs._client._system.stop()
                    except Exception:
                        pass
                    vs._client = None
                vs._collection = None
            
            if hasattr(_CLIENT, 'embedding_manager') and _CLIENT.embedding_manager is not None:
                em = _CLIENT.embedding_manager
                # 清理sentence-transformers模型
                if hasattr(em, '_local_model') and em._local_model is not None:
                    del em._local_model
                    em._local_model = None
                if hasattr(em, '_fallback_model') and em._fallback_model is not None:
                    del em._fallback_model
                    em._fallback_model = None
                # 清理GGUF模型
                if hasattr(em, '_gguf_model') and em._gguf_model is not None:
                    del em._gguf_model
                    em._gguf_model = None
                # 清理API客户端
                if hasattr(em, '_api_client') and em._api_client is not None:
                    em._api_client = None
            
            del _CLIENT
            _CLIENT = None
            
            # 清理GPU缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass  # torch未安装时跳过
            
            import gc
            gc.collect()
            print("🧹 已释放模型内存")
        except Exception as e:
            print(f"⚠️  清理时出错: {e}")
            _CLIENT = None


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """打印头部"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 18 + "Style-RAG 管理系统" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    print()


def print_status():
    """打印当前状态"""
    from style_rag.indexing.article_loader import ArticleLoader
    
    loader = ArticleLoader()
    input_path = Path(DEFAULT_INPUT)
    db_path = Path(DEFAULT_DB)
    
    # 输入目录状态
    if input_path.exists():
        file_count = loader.count_files(str(input_path), recursive=True)
        input_status = f"{file_count} 个文件待处理"
    else:
        input_status = "目录不存在"
    
    # 数据库状态
    if db_path.exists():
        try:
            client = get_client()
            stats = client.get_stats()
            db_status = f"{stats['total_chunks']} 个片段"
        except Exception:
            db_status = "读取失败"
    else:
        db_status = "未创建"
    
    print(f"  📁 输入目录: {DEFAULT_INPUT} ({input_status})")
    print(f"  🗄️  索引数据: {DEFAULT_DB} ({db_status})")
    print()


def print_menu():
    """打印菜单"""
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │                    选择操作                         │")
    print("  ├─────────────────────────────────────────────────────┤")
    print("  │  1. 创建索引      从input目录创建新索引             │")
    print("  │  2. 添加文件      将新文件添加到现有索引             │")
    print("  │  3. 查询索引      输入关键词进行检索                 │")
    print("  │  4. 检索测试      运行自动检索测试                   │")
    print("  │  5. 查看状态      显示详细索引统计                   │")
    print("  │  6. 导出索引      备份索引到exports目录              │")
    print("  │  7. 导入索引      从备份恢复索引                     │")
    print("  │  8. 重置索引      清空所有索引数据                   │")
    print("  │  9. 启动服务      启动HTTP API服务 (端口8086)        │")
    print("  │  0. 退出                                            │")
    print("  └─────────────────────────────────────────────────────┘")
    print()


def wait_for_enter():
    """等待回车继续"""
    input("\n按 Enter 键继续...")


def create_index():
    """创建索引"""
    from style_rag.indexing.article_loader import ArticleLoader
    
    input_path = Path(DEFAULT_INPUT)
    if not input_path.exists():
        print(f"\n❌ 输入目录不存在: {DEFAULT_INPUT}")
        print(f"   请创建该目录并放入文本文件")
        return
    
    loader = ArticleLoader()
    total_files = loader.count_files(DEFAULT_INPUT, recursive=True)
    
    if total_files == 0:
        print(f"\n⚠️  输入目录中没有 .txt 或 .md 文件")
        return
    
    print(f"\n找到 {total_files} 个文件")
    
    # 询问是否清空
    clear_first = input("是否先清空现有索引? [y/N]: ").strip().lower() == 'y'
    
    # 询问是否删除源文件
    delete_files = input("索引后是否删除源文件? [y/N]: ").strip().lower() == 'y'
    
    if delete_files:
        confirm = input("⚠️  确定要删除源文件吗? [y/N]: ").strip().lower() == 'y'
        if not confirm:
            delete_files = False
    
    # 强制创建新客户端（确保数据库连接有效）
    client = get_client(force_new=True)
    
    if clear_first:
        db_path = Path(DEFAULT_DB)
        if db_path.exists():
            print("清空现有索引...")
            client.clear_index()
        else:
            print("索引目录不存在，将创建新索引")
    
    # 进度显示
    start_time = time.time()
    last_update_time = [start_time]  # 使用列表以便在闭包中修改
    processed_chunks = [0]
    current_file_info = {'start_time': start_time, 'last_file': 0}  # 跟踪当前文件
    
    def progress(current, total, message):
        now = time.time()
        elapsed = now - start_time
        
        # 计算进度
        percent = (current / total * 100) if total > 0 else 0
        bar_len = 30
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        
        # 解析进度信息 (新版本传入字典，旧版本传入字符串)
        if isinstance(message, dict):
            filename = message.get('filename', 'unknown')
            file_chunks = message.get('file_chunks', 0)
            total_chunks_so_far = message.get('total_chunks_so_far', 0)
        else:
            filename = message.replace("Processing: ", "") if isinstance(message, str) else "unknown"
            file_chunks = 0
            total_chunks_so_far = 0
        
        # 获取已处理的分块数
        processed_chunks = message.get('processed_chunks', 0) if isinstance(message, dict) else 0
        status = message.get('status', 'processing') if isinstance(message, dict) else 'processing'
        
        # 截断长文件名
        if len(filename) > 25:
            filename = filename[:22] + "..."
        
        # 计算速度和预计完成时间
        if current > 1 and elapsed > 0:
            speed = current / elapsed  # 文件/秒
            remaining_files = total - current
            eta_seconds = remaining_files / speed if speed > 0 else 0
            
            # 格式化剩余时间
            if eta_seconds < 60:
                eta_str = f"剩余:{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"剩余:{eta_seconds/60:.1f}m"
            else:
                eta_str = f"剩余:{eta_seconds/3600:.1f}h"
            
            # 预计完成时间
            from datetime import datetime, timedelta
            finish_time = datetime.now() + timedelta(seconds=eta_seconds)
            finish_str = finish_time.strftime("%H:%M:%S")
        else:
            speed = 0
            eta_str = "计算中..."
            finish_str = "--:--:--"
        
        # 格式化已用时间
        if elapsed < 60:
            elapsed_str = f"{elapsed:.0f}s"
        elif elapsed < 3600:
            elapsed_str = f"{elapsed/60:.1f}m"
        else:
            elapsed_str = f"{elapsed/3600:.1f}h"
        
        # 跟踪当前文件的处理时间
        if current != current_file_info['last_file']:
            current_file_info['last_file'] = current
            current_file_info['start_time'] = now
        file_elapsed = now - current_file_info['start_time']
        
        # 格式化文件处理时间
        if file_elapsed < 60:
            file_time_str = f"{file_elapsed:.1f}s"
        else:
            file_time_str = f"{file_elapsed/60:.1f}m"
        
        # 状态图标
        status_icon = "🔄" if status == "embedding" else "📝"
        
        # 格式化输出 - 三行显示
        # 第一行：总进度
        print(f"\r[{bar}] {percent:5.1f}%", end="")
        print(f" | {current}/{total}文件", end="")
        print(f" | 用时:{elapsed_str}", end="")
        print(f" | {eta_str}", end="")
        print(f" | 完成:{finish_str}", end="")
        print("     ", end="")  # 清除残留
        
        # 第二行：当前文件信息 (带文件处理时间)
        print(f"\n   📄 {filename} | ⏱已用时:{file_time_str}", end="")
        print("                    ", end="")  # 清除残留
        
        # 第三行：实时分块进度
        if file_chunks > 0:
            chunk_percent = (processed_chunks / file_chunks * 100) if file_chunks > 0 else 0
            chunk_bar_len = 12
            chunk_filled = int(chunk_bar_len * processed_chunks / file_chunks) if file_chunks > 0 else 0
            chunk_bar = "▓" * chunk_filled + "░" * (chunk_bar_len - chunk_filled)
            print(f"\n   {status_icon} [{chunk_bar}] {processed_chunks}/{file_chunks}块({chunk_percent:.0f}%) | 累计:{total_chunks_so_far}块", end="")
        else:
            print(f"\n   {status_icon} 分块中... | 累计:{total_chunks_so_far}块", end="")
        print("          ", end="")  # 清除残留
        
        # 回到第一行
        print("\033[A\033[A", end="", flush=True)
    
    print("开始索引...\n")
    result = client.index_directory(
        articles_dir=DEFAULT_INPUT,
        recursive=True,
        progress_callback=progress,
        delete_after_index=delete_files
    )
    
    total_time = time.time() - start_time
    
    # 清除进度显示（移动到新行）
    print("\n")
    print(f"✅ 索引完成!")
    print(f"   处理文件: {result.get('total_files', 0)}")
    print(f"   生成片段: {result.get('total_chunks', 0)}")
    if result.get('failed_files', 0) > 0:
        print(f"   ⚠️ 失败文件: {result.get('failed_files', 0)}")
    if delete_files:
        print(f"   已删除文件: {result.get('deleted_files', 0)}")
    print(f"   用时: {total_time:.1f}秒")
    if result.get('total_files', 0) > 0:
        avg_time = total_time / result.get('total_files', 1)
        print(f"   平均: {avg_time:.2f}秒/文件")


def add_files():
    """添加文件"""
    from style_rag.indexing.article_loader import ArticleLoader
    
    input_path = Path(DEFAULT_INPUT)
    if not input_path.exists():
        print(f"\n❌ 输入目录不存在: {DEFAULT_INPUT}")
        return
    
    loader = ArticleLoader()
    total_files = loader.count_files(DEFAULT_INPUT, recursive=True)
    
    if total_files == 0:
        print(f"\n⚠️  输入目录中没有新文件")
        return
    
    print(f"\n找到 {total_files} 个新文件")
    
    delete_files = input("添加后是否删除源文件? [y/N]: ").strip().lower() == 'y'
    
    print("\n正在添加...")
    client = get_client()
    stats_before = client.get_stats()
    
    start_time = time.time()
    
    def progress(current, total, message):
        percent = (current / total * 100) if total > 0 else 0
        bar_len = 25
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {percent:5.1f}% | {current}/{total}", end="", flush=True)
    
    result = client.index_directory(
        articles_dir=DEFAULT_INPUT,
        recursive=True,
        progress_callback=progress,
        delete_after_index=delete_files
    )
    
    stats_after = client.get_stats()
    new_chunks = stats_after['total_chunks'] - stats_before['total_chunks']
    
    print(f"\n\n✅ 添加完成!")
    print(f"   新增片段: {new_chunks}")
    print(f"   当前总片段: {stats_after['total_chunks']}")


def query_index():
    """查询索引"""
    db_path = Path(DEFAULT_DB)
    if not db_path.exists():
        print(f"\n❌ 索引不存在，请先创建索引")
        return
    
    client = get_client()
    
    stats = client.get_stats()
    if stats['total_chunks'] == 0:
        print(f"\n⚠️  索引为空，请先添加文件")
        return
    
    print(f"\n当前索引: {stats['total_chunks']} 个片段")
    print("输入 q 退出查询\n")
    
    while True:
        query = input("查询> ").strip()
        if not query:
            continue
        if query.lower() in ['q', 'quit', 'exit', '退出']:
            break
        
        results = client.search(query, top_k=3, min_similarity=0.3)
        
        if not results:
            print("  未找到相关结果\n")
            continue
        
        print(f"\n  找到 {len(results)} 条结果:")
        for i, r in enumerate(results, 1):
            sim = r.get('similarity', 0)
            content = r['content'][:100] + "..." if len(r['content']) > 100 else r['content']
            print(f"  [{i}] 相似度: {sim:.3f}")
            print(f"      {content}\n")


def run_test():
    """运行检索测试"""
    db_path = Path(DEFAULT_DB)
    if not db_path.exists():
        print(f"\n❌ 索引不存在")
        return
    
    client = get_client()
    
    stats = client.get_stats()
    if stats['total_chunks'] == 0:
        print(f"\n⚠️  索引为空，请先添加文件")
        return
    
    print(f"\n当前索引: {stats['total_chunks']} 个片段")
    print("\n选择测试模式:")
    print("  1. 预设测试查询")
    print("  2. 自定义查询内容")
    print("  0. 返回")
    
    choice = input("\n选择: ").strip()
    
    if choice == "1":
        # 预设测试查询
        test_queries = ["战斗场景", "月下相遇", "离别伤感", "修炼突破", "对话场景"]
        print(f"\n运行 {len(test_queries)} 个测试查询...\n")
        
        total_results = 0
        for query in test_queries:
            results = client.search(query, top_k=2, min_similarity=0.3)
            total_results += len(results)
            print(f"  🔍 {query}: {len(results)} 条结果")
        
        print(f"\n✅ 测试完成，共找到 {total_results} 条结果")
        
    elif choice == "2":
        # 自定义查询
        print("\n输入查询内容 (输入 q 返回):\n")
        
        while True:
            query = input("测试查询> ").strip()
            if not query:
                continue
            if query.lower() in ['q', 'quit', 'exit', '退出']:
                break
            
            # 询问返回数量
            top_k = input(f"  返回结果数量 [默认{API_SEARCH_TOP_K}]: ").strip()
            top_k = int(top_k) if top_k.isdigit() else API_SEARCH_TOP_K
            
            results = client.search(query, top_k=top_k, min_similarity=0.3)
            
            if not results:
                print("  未找到相关结果\n")
                continue
            
            print(f"\n  找到 {len(results)} 条结果:")
            for i, r in enumerate(results, 1):
                sim = r.get('similarity', 0)
                content = r['content']
                source = r.get('metadata', {}).get('source', '未知')
                print(f"\n  [{i}] 相似度: {sim:.3f} | 来源: {source}")
                print(f"  {'─' * 50}")
                print(f"  {content}")
                print(f"  {'─' * 50}")
    else:
        print("已取消")


def show_status():
    """显示详细状态"""
    from style_rag.indexing.article_loader import ArticleLoader
    
    print("\n" + "=" * 50)
    
    # 输入目录
    print("\n📁 输入目录:")
    input_path = Path(DEFAULT_INPUT)
    if input_path.exists():
        loader = ArticleLoader()
        file_count = loader.count_files(DEFAULT_INPUT, recursive=True)
        print(f"   路径: {input_path.absolute()}")
        print(f"   待处理文件: {file_count}")
    else:
        print(f"   不存在")
    
    # 数据库
    print("\n🗄️  索引数据库:")
    db_path = Path(DEFAULT_DB)
    if db_path.exists():
        client = get_client()
        stats = client.get_stats()
        print(f"   路径: {db_path.absolute()}")
        print(f"   片段数: {stats['total_chunks']}")
        print(f"   嵌入模型: {stats['embedding_model']}")
        print(f"   提供商: {stats['embedding_provider']}")
    else:
        print(f"   未创建")
    
    # 导出目录
    print("\n📦 导出目录:")
    exports_path = Path(DEFAULT_EXPORTS)
    if exports_path.exists():
        backups = list(exports_path.iterdir())
        print(f"   路径: {exports_path.absolute()}")
        print(f"   备份数量: {len(backups)}")
    else:
        print(f"   不存在")
    
    print("\n" + "=" * 50)


def export_index():
    """导出索引"""
    from datetime import datetime
    
    db_path = Path(DEFAULT_DB)
    if not db_path.exists():
        print(f"\n❌ 索引不存在")
        return
    
    client = get_client()
    
    # 生成导出路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = f"{DEFAULT_EXPORTS}/rag_backup_{timestamp}"
    
    print(f"\n导出到: {export_path}")
    
    success = client.export_index(export_path)
    
    if success:
        print(f"\n✅ 导出成功!")
    else:
        print(f"\n❌ 导出失败")


def import_index():
    """导入索引"""
    exports_path = Path(DEFAULT_EXPORTS)
    
    if not exports_path.exists():
        print(f"\n❌ 导出目录不存在")
        return
    
    # 列出可用备份
    backups = [d for d in exports_path.iterdir() if d.is_dir()]
    
    if not backups:
        print(f"\n⚠️  没有可用的备份")
        return
    
    print("\n可用备份:")
    for i, backup in enumerate(backups, 1):
        print(f"  {i}. {backup.name}")
    
    try:
        choice = int(input("\n选择备份编号: "))
        if 1 <= choice <= len(backups):
            backup_path = backups[choice - 1]
        else:
            print("无效选择")
            return
    except ValueError:
        print("无效输入")
        return
    
    client = get_client()
    
    print(f"\n从 {backup_path} 导入...")
    success = client.import_index(str(backup_path))
    
    if success:
        stats = client.get_stats()
        print(f"\n✅ 导入成功! 当前片段数: {stats['total_chunks']}")
    else:
        print(f"\n❌ 导入失败")


def reset_index():
    """重置索引"""
    db_path = Path(DEFAULT_DB)
    
    if not db_path.exists():
        print(f"\n⚠️  索引不存在，无需重置")
        return
    
    print("\n选择重置方式:")
    print("  1. 清空数据（保留数据库结构）")
    print("  2. 删除数据库（完全删除）")
    print("  0. 取消")
    
    choice = input("\n选择: ").strip()
    
    if choice == "1":
        confirm = input("确定要清空所有数据吗? [y/N]: ").strip().lower()
        if confirm == 'y':
            client = get_client()
            client.clear_index()
            print("\n✅ 索引已清空")
    elif choice == "2":
        confirm = input("⚠️  确定要删除整个数据库吗? [y/N]: ").strip().lower()
        if confirm == 'y':
            # 先清理客户端实例（释放文件句柄）
            cleanup_client()
            import shutil
            import time
            
            # Windows上需要等待文件句柄完全释放
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    time.sleep(0.5)  # 等待文件句柄释放
                    shutil.rmtree(db_path)
                    print("\n✅ 数据库已删除")
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        print(f"\r   等待文件释放... ({attempt + 1}/{max_retries})", end="", flush=True)
                        time.sleep(1)
                    else:
                        print(f"\n❌ 无法删除数据库，文件被占用: {e}")
                        print("   请关闭所有相关程序后手动删除目录: " + str(db_path))
    else:
        print("已取消")


def start_server():
    """启动HTTP服务"""
    print(f"\n{'='*50}")
    print(f"启动 HTTP 服务")
    print(f"{'='*50}")
    print(f"  地址: http://0.0.0.0:8086 (允许外部访问)")
    print(f"  数据库: {DEFAULT_DB}")
    print(f"  API嵌入模型: {API_EMBEDDING_PROVIDER}")
    print(f"  API文档: http://localhost:8086/docs")
    print(f"{'='*50}")
    print(f"\n按 Ctrl+C 停止服务\n")
    
    try:
        import uvicorn
        import os
        
        # 设置数据库路径
        os.environ['STYLE_RAG_DB_PATH'] = DEFAULT_DB
        
        # 设置嵌入模型配置 (使用API专属配置)
        os.environ['STYLE_RAG_EMBEDDING_PROVIDER'] = API_EMBEDDING_PROVIDER
        os.environ['STYLE_RAG_EMBEDDING_BATCH_SIZE'] = str(API_EMBEDDING_BATCH_SIZE)
        os.environ['STYLE_RAG_SEARCH_TOP_K'] = str(API_SEARCH_TOP_K)
        
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
            host="0.0.0.0",
            port=8086,
            log_level="info"
        )
    except ImportError:
        print("❌ 需要安装 uvicorn:")
        print("   uv pip install uvicorn")
    except KeyboardInterrupt:
        print("\n\n服务已停止")


def main():
    """主程序"""
    # 确保目录存在
    Path(DEFAULT_INPUT).mkdir(exist_ok=True)
    Path(DEFAULT_EXPORTS).mkdir(exist_ok=True)
    
    try:
        while True:
            clear_screen()
            print_header()
            print_status()
            print_menu()
            
            choice = input("  请选择 [0-9]: ").strip()
            
            if choice == "1":
                create_index()
                wait_for_enter()
            elif choice == "2":
                add_files()
                wait_for_enter()
            elif choice == "3":
                query_index()
                wait_for_enter()
            elif choice == "4":
                run_test()
                wait_for_enter()
            elif choice == "5":
                show_status()
                wait_for_enter()
            elif choice == "6":
                export_index()
                wait_for_enter()
            elif choice == "7":
                import_index()
                wait_for_enter()
            elif choice == "8":
                reset_index()
                wait_for_enter()
            elif choice == "9":
                start_server()
                wait_for_enter()
            elif choice == "0":
                print("\n再见!")
                break
            else:
                print("\n无效选择")
                time.sleep(1)
    finally:
        # 程序退出时清理资源
        cleanup_client()


if __name__ == "__main__":
    main()
