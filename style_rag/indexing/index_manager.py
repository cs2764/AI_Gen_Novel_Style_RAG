"""
索引管理器 - 管理文章索引的构建和更新
Index Manager - Managing Article Index Building and Updates
"""

import json
import logging
import shutil
from typing import List, Dict, Optional, Callable
from pathlib import Path
from datetime import datetime

from style_rag.core.embeddings import EmbeddingManager
from style_rag.core.chunker import StyleAwareChunker
from style_rag.core.vector_store import StyleVectorStore
from style_rag.indexing.article_loader import ArticleLoader
from style_rag.indexing.preprocessor import ArticlePreprocessor

logger = logging.getLogger(__name__)


class IndexManager:
    """
    索引管理器
    Index Manager
    
    负责构建、更新和管理文章索引
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: StyleVectorStore,
        chunker: Optional[StyleAwareChunker] = None,
        loader: Optional[ArticleLoader] = None,
        preprocessor: Optional[ArticlePreprocessor] = None
    ):
        """
        初始化索引管理器
        
        Args:
            embedding_manager: Embedding管理器
            vector_store: 向量存储
            chunker: 分块器（可选，使用默认配置）
            loader: 文章加载器（可选）
            preprocessor: 预处理器（可选）
        """
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.chunker = chunker or StyleAwareChunker()
        self.loader = loader or ArticleLoader()
        self.preprocessor = preprocessor or ArticlePreprocessor()
    
    def index_directory(
        self,
        articles_dir: str,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        delete_after_index: bool = False
    ) -> Dict:
        """
        索引目录下的所有文章
        
        Args:
            articles_dir: 文章目录路径
            recursive: 是否递归子目录
            file_patterns: 文件匹配模式
            progress_callback: 进度回调函数 (current, total, message)
            delete_after_index: 索引后删除源文件
            
        Returns:
            索引结果统计
        """
        logger.info(f"Indexing directory: {articles_dir}")
        
        # 统计文件数量
        total_files = self.loader.count_files(articles_dir, recursive)
        if total_files == 0:
            logger.warning("No files found to index")
            return {'total_files': 0, 'total_chunks': 0, 'success': True}
        
        stats = {
            'total_files': 0,
            'total_chunks': 0,
            'failed_files': 0,
            'deleted_files': 0,
            'start_time': datetime.now().isoformat(),
            'errors': []
        }
        
        current_file = 0
        total_chunks_processed = 0
        
        # 遍历并索引文件
        for article in self.loader.load_directory(
            articles_dir, recursive, file_patterns
        ):
            current_file += 1
            file_path = article.get('file_path')
            filename = article['metadata'].get('filename', 'unknown')
            
            # 先进行分块以获取分块数量
            try:
                processed = self.preprocessor.preprocess(article)
                chunks = self.chunker.chunk_article(
                    processed['content'],
                    processed['metadata']
                )
                chunk_count = len(chunks) if chunks else 0
            except Exception as e:
                chunk_count = 0
                chunks = []
            
            # 创建实时进度回调的包装器
            def make_chunk_progress_callback(cur_file, tot_files, fname, file_chunks, chunks_so_far):
                def callback(processed_chunks, total_file_chunks):
                    if progress_callback:
                        progress_info = {
                            'current_file': cur_file,
                            'total_files': tot_files,
                            'filename': fname,
                            'file_chunks': file_chunks,
                            'processed_chunks': processed_chunks,
                            'total_chunks_so_far': chunks_so_far + processed_chunks,
                            'status': 'embedding'
                        }
                        progress_callback(cur_file, tot_files, progress_info)
                return callback
            
            chunk_progress = make_chunk_progress_callback(
                current_file, total_files, filename, chunk_count, total_chunks_processed
            )
            
            # 初始状态回调
            if progress_callback:
                progress_info = {
                    'current_file': current_file,
                    'total_files': total_files,
                    'filename': filename,
                    'file_chunks': chunk_count,
                    'processed_chunks': 0,
                    'total_chunks_so_far': total_chunks_processed,
                    'status': 'chunking'
                }
                progress_callback(current_file, total_files, progress_info)
            
            try:
                if chunks:
                    # 直接使用已分好的块进行索引，传入进度回调
                    result = self._index_chunks(chunks, processed['metadata'], chunk_progress)
                    stats['total_files'] += 1
                    stats['total_chunks'] += result['chunks_indexed']
                    total_chunks_processed += result['chunks_indexed']
                else:
                    stats['total_files'] += 1
                
                # 索引成功后删除文件
                if delete_after_index and file_path:
                    try:
                        Path(file_path).unlink()
                        stats['deleted_files'] += 1
                        logger.debug(f"Deleted file: {file_path}")
                    except Exception as del_err:
                        logger.warning(f"Failed to delete {file_path}: {del_err}")
                        
            except Exception as e:
                stats['failed_files'] += 1
                stats['errors'].append({
                    'file': article.get('file_path', 'unknown'),
                    'error': str(e)
                })
                logger.error(f"Failed to index {article.get('file_path')}: {e}")
        
        stats['end_time'] = datetime.now().isoformat()
        stats['success'] = stats['failed_files'] == 0
        
        logger.info(
            f"Indexing complete: {stats['total_files']} files, "
            f"{stats['total_chunks']} chunks"
        )
        if delete_after_index:
            logger.info(f"Deleted {stats['deleted_files']} files")
        
        return stats
    
    def index_files(
        self,
        file_paths: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict:
        """
        索引指定的文件列表
        
        Args:
            file_paths: 文件路径列表
            progress_callback: 进度回调函数
            
        Returns:
            索引结果统计
        """
        stats = {
            'total_files': 0,
            'total_chunks': 0,
            'failed_files': 0,
            'errors': []
        }
        
        for i, file_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(i + 1, len(file_paths), f"Processing: {file_path}")
            
            article = self.loader.load_file(file_path)
            if article is None:
                stats['failed_files'] += 1
                stats['errors'].append({
                    'file': file_path,
                    'error': 'Failed to load file'
                })
                continue
            
            try:
                result = self._index_article(article)
                stats['total_files'] += 1
                stats['total_chunks'] += result['chunks_indexed']
            except Exception as e:
                stats['failed_files'] += 1
                stats['errors'].append({
                    'file': file_path,
                    'error': str(e)
                })
        
        return stats
    
    def index_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> Dict:
        """
        直接索引文本内容
        
        Args:
            texts: 文本列表
            metadatas: 对应的元数据列表
            
        Returns:
            索引结果统计
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have same length")
        
        stats = {
            'total_texts': len(texts),
            'total_chunks': 0
        }
        
        all_chunks = []
        
        for text, metadata in zip(texts, metadatas):
            # 分块
            chunks = self.chunker.chunk_article(text, metadata)
            for chunk in chunks:
                all_chunks.append({
                    'content': chunk.content,
                    'metadata': chunk.metadata
                })
        
        if all_chunks:
            # 生成embeddings
            contents = [c['content'] for c in all_chunks]
            embeddings = self.embedding_manager.embed(contents)
            
            # 添加到向量库
            self.vector_store.add_chunks(all_chunks, embeddings.tolist())
            stats['total_chunks'] = len(all_chunks)
        
        return stats
    
    def _index_article(self, article: Dict) -> Dict:
        """索引单篇文章"""
        # 预处理
        processed = self.preprocessor.preprocess(article)
        
        # 分块
        chunks = self.chunker.chunk_article(
            processed['content'],
            processed['metadata']
        )
        
        if not chunks:
            return {'chunks_indexed': 0}
        
        return self._index_chunks(chunks, processed['metadata'])
    
    def _index_chunks(self, chunks, metadata: Dict = None, progress_callback=None) -> Dict:
        """
        索引预分块的内容
        
        Args:
            chunks: 分块列表
            metadata: 元数据
            progress_callback: 进度回调函数 (processed_chunks, total_chunks)
        """
        if not chunks:
            return {'chunks_indexed': 0}
        
        # 转换为字典格式
        chunk_dicts = [
            {
                'content': chunk.content,
                'metadata': chunk.metadata
            }
            for chunk in chunks
        ]
        
        total_chunks = len(chunk_dicts)
        batch_size = self.embedding_manager.config.batch_size
        processed = 0
        all_embeddings = []
        
        # 分批生成embeddings以便报告进度
        for i in range(0, total_chunks, batch_size):
            batch = chunk_dicts[i:i + batch_size]
            contents = [c['content'] for c in batch]
            
            # 生成这批的embeddings
            batch_embeddings = self.embedding_manager.embed(contents)
            
            # 确保embeddings是列表格式
            if hasattr(batch_embeddings, 'tolist'):
                batch_embeddings = batch_embeddings.tolist()
            
            all_embeddings.extend(batch_embeddings)
            processed += len(batch)
            
            # 报告进度
            if progress_callback:
                progress_callback(processed, total_chunks)
        
        # 添加到向量库
        self.vector_store.add_chunks(chunk_dicts, all_embeddings)
        
        return {'chunks_indexed': len(chunk_dicts)}
    
    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        return self.vector_store.get_stats()
    
    def clear(self) -> bool:
        """清空索引"""
        return self.vector_store.clear()
    
    def export_index(self, export_path: str) -> bool:
        """
        导出索引到指定路径
        
        Args:
            export_path: 导出路径
            
        Returns:
            是否成功
        """
        try:
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制ChromaDB目录
            src_dir = self.vector_store.persist_dir
            shutil.copytree(src_dir, export_dir / 'chroma_db', dirs_exist_ok=True)
            
            # 保存元数据
            meta = {
                'exported_at': datetime.now().isoformat(),
                'stats': self.get_stats(),
                'embedding_model': self.embedding_manager.model_name,
                'embedding_provider': self.embedding_manager.provider_name
            }
            
            with open(export_dir / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Index exported to: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def import_index(self, import_path: str) -> bool:
        """
        从指定路径导入索引
        
        Args:
            import_path: 导入路径
            
        Returns:
            是否成功
        """
        try:
            import_dir = Path(import_path)
            
            if not import_dir.exists():
                logger.error(f"Import path not found: {import_path}")
                return False
            
            chroma_dir = import_dir / 'chroma_db'
            if not chroma_dir.exists():
                logger.error("chroma_db directory not found in import path")
                return False
            
            # 复制到目标位置
            dest_dir = self.vector_store.persist_dir
            shutil.rmtree(dest_dir, ignore_errors=True)
            shutil.copytree(chroma_dir, dest_dir)
            
            # 重新初始化向量存储
            self.vector_store._initialize()
            
            logger.info(f"Index imported from: {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False
