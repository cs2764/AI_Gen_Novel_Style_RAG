"""Quick test for Style-RAG functionality"""
from style_rag import StyleRAGClient
import tempfile
import os

# 使用临时目录测试
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = os.path.join(tmpdir, 'test_db')
    
    print('1. 初始化客户端...')
    client = StyleRAGClient(db_path=db_path)
    print(f'   成功! 模型: {client.embedding_manager.model_name}')
    
    print('2. 索引测试文本...')
    texts = [
        '林风站在山门前，望着眼前巍峨的建筑群，心中涌起一股莫名的激动。这里，就是他梦寐以求的天玄宗，修仙界最负盛名的宗门之一。',
        '一道剑光划破夜空，带着凛冽的杀意直奔林风而来。他瞳孔骤缩，几乎是本能地侧身躲避，锋利的剑气擦着他的面颊呼啸而过，在身后的石壁上留下一道深深的痕迹。',
        '「你就是今年的新弟子？」一个冷淡的声音从身后传来。林风转身，只见一位身着青色道袍的年轻人正打量着自己，眸中带着几分审视。',
    ]
    result = client.index_texts(texts)
    print(f'   成功! 索引片段数: {result["total_chunks"]}')
    
    print('3. 语义检索测试...')
    results = client.search('战斗场面', top_k=2, min_similarity=0.3)
    print(f'   成功! 找到 {len(results)} 个结果')
    for i, r in enumerate(results, 1):
        print(f'   - 结果{i}: 相似度={r["similarity"]:.2f}')
    
    print('4. 获取统计信息...')
    stats = client.get_stats()
    print(f'   成功! 总片段数: {stats["total_chunks"]}')
    
    print('5. 清空索引测试...')
    client.clear_index()
    stats = client.get_stats()
    print(f'   成功! 清空后片段数: {stats["total_chunks"]}')

print('\n所有测试通过!')
