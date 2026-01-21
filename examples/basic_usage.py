"""
Style-RAG 使用示例 - 基础用法
Basic Usage Example for Style-RAG
"""

from style_rag import StyleRAGClient, EmbeddingConfig, EmbeddingProvider


def example_local_embedding():
    """示例1: 使用本地Embedding（默认）"""
    print("=" * 60)
    print("示例1: 使用本地Embedding")
    print("=" * 60)
    
    # 初始化客户端
    client = StyleRAGClient(
        db_path="./example_rag_db",
        embedding_model="BAAI/bge-large-zh-v1.5"
    )
    
    # 索引一些示例文本
    texts = [
        "林风站在山门前，望着眼前巍峨的建筑群，心中涌起一股莫名的激动。这里，就是他梦寐以求的天玄宗。",
        "「你就是今年的新弟子？」一个冷淡的声音从身后传来。林风转身，只见一位身着青色道袍的年轻人正打量着自己。",
        "月光洒落，她静静站在窗前，思绪飘向远方。那个人，不知如今身在何处？",
        "一道剑光划破夜空，带着凛冽的杀意直奔林风而来。他瞳孔骤缩，几乎是本能地侧身躲避。",
        "「我等你很久了。」她微微一笑，眸中带着几分期待，几分羞涩。",
    ]
    
    metadatas = [
        {"type": "description", "scene": "初入山门"},
        {"type": "dialogue", "scene": "初遇"},
        {"type": "description", "scene": "思念"},
        {"type": "action", "scene": "战斗"},
        {"type": "dialogue", "scene": "相遇"},
    ]
    
    result = client.index_texts(texts, metadatas)
    print(f"索引完成: {result['total_chunks']} 个片段\n")
    
    # 检索示例
    print("检索: '战斗场面'")
    results = client.search("战斗场面", top_k=2)
    for i, r in enumerate(results, 1):
        print(f"  结果{i} (相似度: {r['similarity']:.2f}): {r['content'][:50]}...")
    
    print("\n检索: '浪漫相遇'")
    results = client.search_by_scene(
        scene_description="浪漫相遇",
        emotion="浪漫",
        top_k=2
    )
    for i, r in enumerate(results, 1):
        print(f"  结果{i} (相似度: {r['similarity']:.2f}): {r['content'][:50]}...")
    
    # 获取统计
    stats = client.get_stats()
    print(f"\n索引统计: {stats['total_chunks']} 个片段")
    
    # 清理
    client.clear_index()
    print("示例结束，已清理索引\n")


def example_cloud_embedding():
    """示例2: 使用云端Embedding (需要API Key)"""
    print("=" * 60)
    print("示例2: 使用云端Embedding (需要配置API Key)")
    print("=" * 60)
    
    # 注意: 需要替换为真实的API Key
    API_KEY = "your-api-key-here"
    
    if API_KEY == "your-api-key-here":
        print("跳过: 请先配置API Key\n")
        return
    
    client = StyleRAGClient(
        db_path="./example_rag_db",
        embedding_config=EmbeddingConfig(
            provider=EmbeddingProvider.SILICONFLOW,
            api_key=API_KEY,
            api_model="BAAI/bge-large-zh-v1.5"
        )
    )
    
    print("云端Embedding客户端初始化成功")
    print(f"提供商: {client.embedding_manager.provider_name}")
    print(f"模型: {client.embedding_manager.model_name}\n")


def example_directory_indexing():
    """示例3: 索引目录"""
    print("=" * 60)
    print("示例3: 索引目录")
    print("=" * 60)
    
    # 创建客户端
    client = StyleRAGClient(db_path="./example_rag_db")
    
    # 检查是否有示例目录
    import os
    if os.path.exists("./example_articles"):
        def progress(current, total, msg):
            print(f"  [{current}/{total}] {msg}")
        
        result = client.index_directory(
            "./example_articles",
            recursive=True,
            progress_callback=progress
        )
        print(f"\n索引完成: {result['total_files']} 个文件, {result['total_chunks']} 个片段")
    else:
        print("跳过: 示例目录 ./example_articles 不存在")
        print("创建该目录并放入一些 .txt 或 .md 文件后可运行此示例")
    
    print()


def example_format_references():
    """示例4: 格式化引用文本"""
    print("=" * 60)
    print("示例4: 格式化引用文本")
    print("=" * 60)
    
    client = StyleRAGClient(db_path="./example_rag_db")
    
    # 添加一些测试数据
    texts = [
        "他缓缓抽出长剑，寒光闪烁，映照出他坚毅的面容。",
        "「来吧，让我看看你的实力！」他大喝一声，剑气纵横。",
    ]
    client.index_texts(texts)
    
    # 检索并格式化
    results = client.search("战斗", top_k=2)
    
    formatted = client.format_references(results, max_length=100)
    print(formatted)
    
    client.clear_index()
    print("示例结束\n")


if __name__ == "__main__":
    print("\nStyle-RAG 使用示例\n")
    
    try:
        example_local_embedding()
    except Exception as e:
        print(f"示例1 出错: {e}\n")
    
    example_cloud_embedding()
    example_directory_indexing()
    
    try:
        example_format_references()
    except Exception as e:
        print(f"示例4 出错: {e}\n")
    
    print("所有示例完成!")
