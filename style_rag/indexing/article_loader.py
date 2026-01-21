"""
文章加载器 - 从文件和目录加载文章
Article Loader - Loading Articles from Files and Directories
"""

import logging
from typing import List, Dict, Optional, Generator
from pathlib import Path

logger = logging.getLogger(__name__)


class ArticleLoader:
    """
    文章加载器
    Article Loader
    
    支持从文件和目录加载文章，自动处理编码
    """
    
    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = {'.txt', '.md'}
    
    # 中文常用编码列表（按优先级排序）
    # Common Chinese encodings (sorted by priority)
    ENCODINGS = [
        'utf-8',        # Unicode
        'utf-8-sig',    # UTF-8 with BOM
        'gb18030',      # 国标扩展（兼容GBK和GB2312）
        'gbk',          # 国标扩展
        'gb2312',       # 简体中文
        'big5',         # 繁体中文
        'big5hkscs',    # 香港繁体
        'utf-16',       # Unicode 16-bit
        'utf-16-le',    # Little Endian
        'utf-16-be',    # Big Endian
        'hz',           # HZ编码
        'iso-2022-cn',  # ISO中文
        'cp936',        # Windows中文
        'latin-1',      # 最后尝试Latin-1
    ]
    
    def __init__(
        self,
        supported_extensions: Optional[List[str]] = None,
        use_chardet: bool = True
    ):
        """
        初始化加载器
        
        Args:
            supported_extensions: 支持的文件扩展名列表
            use_chardet: 是否使用chardet库自动检测编码
        """
        if supported_extensions:
            self.supported_extensions = set(
                ext if ext.startswith('.') else f'.{ext}'
                for ext in supported_extensions
            )
        else:
            self.supported_extensions = self.SUPPORTED_EXTENSIONS
        
        self.use_chardet = use_chardet
        self._chardet_available = False
        
        # 尝试导入chardet
        if use_chardet:
            try:
                import chardet
                self._chardet_available = True
            except ImportError:
                logger.debug("chardet not available, using fallback encoding detection")
    
    def load_file(self, file_path: str) -> Optional[Dict]:
        """
        加载单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含 'content', 'file_path', 'metadata' 的字典，或None
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        if path.suffix.lower() not in self.supported_extensions:
            logger.warning(f"Unsupported file type: {path.suffix}")
            return None
        
        content, detected_encoding = self._read_file(path)
        if content is None:
            return None
        
        metadata = self._extract_file_metadata(path, content)
        metadata['detected_encoding'] = detected_encoding
        
        return {
            'content': content,
            'file_path': str(path.absolute()),
            'metadata': metadata
        }
    
    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None
    ) -> Generator[Dict, None, None]:
        """
        加载目录下的所有文章
        
        Args:
            directory: 目录路径
            recursive: 是否递归子目录
            file_patterns: 文件匹配模式（如 ["*.txt", "*.md"]）
            
        Yields:
            包含 'content', 'file_path', 'metadata' 的字典
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return
        
        if not dir_path.is_dir():
            logger.error(f"Not a directory: {directory}")
            return
        
        # 确定要搜索的模式
        if file_patterns:
            patterns = file_patterns
        else:
            patterns = [f"*{ext}" for ext in self.supported_extensions]
        
        # 遍历文件
        for pattern in patterns:
            if recursive:
                files = dir_path.rglob(pattern)
            else:
                files = dir_path.glob(pattern)
            
            for file_path in files:
                if file_path.is_file():
                    result = self.load_file(str(file_path))
                    if result:
                        # 添加相对路径信息
                        try:
                            result['metadata']['relative_path'] = str(
                                file_path.relative_to(dir_path)
                            )
                        except ValueError:
                            result['metadata']['relative_path'] = file_path.name
                        
                        yield result
    
    def load_files(self, file_paths: List[str]) -> List[Dict]:
        """
        加载多个文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            加载结果列表
        """
        results = []
        for fp in file_paths:
            result = self.load_file(fp)
            if result:
                results.append(result)
        return results
    
    def _detect_encoding_with_chardet(self, raw_bytes: bytes) -> Optional[str]:
        """使用chardet检测编码"""
        try:
            import chardet
            result = chardet.detect(raw_bytes)
            if result and result.get('encoding'):
                confidence = result.get('confidence', 0)
                encoding = result['encoding']
                logger.debug(f"chardet detected: {encoding} (confidence: {confidence:.2f})")
                if confidence > 0.5:
                    return encoding
        except Exception as e:
            logger.debug(f"chardet detection failed: {e}")
        return None
    
    def _read_file(self, path: Path) -> tuple:
        """
        读取文件内容，智能检测编码
        
        Returns:
            (content, detected_encoding) 或 (None, None)
        """
        try:
            raw_bytes = path.read_bytes()
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None, None
        
        # 1. 检查BOM标记
        bom_encodings = [
            (b'\xef\xbb\xbf', 'utf-8-sig'),
            (b'\xff\xfe', 'utf-16-le'),
            (b'\xfe\xff', 'utf-16-be'),
        ]
        for bom, encoding in bom_encodings:
            if raw_bytes.startswith(bom):
                try:
                    content = raw_bytes.decode(encoding)
                    logger.debug(f"Detected BOM encoding: {encoding}")
                    return content, encoding
                except UnicodeDecodeError:
                    pass
        
        # 2. 使用chardet自动检测
        if self._chardet_available:
            detected = self._detect_encoding_with_chardet(raw_bytes)
            if detected:
                try:
                    content = raw_bytes.decode(detected)
                    return content, detected
                except (UnicodeDecodeError, LookupError):
                    pass
        
        # 3. 按优先级尝试各种编码
        for encoding in self.ENCODINGS:
            try:
                content = raw_bytes.decode(encoding)
                # 简单验证：检查是否有过多替换字符
                if content.count('\ufffd') < len(content) * 0.01:
                    logger.debug(f"Successfully decoded with: {encoding}")
                    return content, encoding
            except (UnicodeDecodeError, LookupError):
                continue
        
        # 4. 最后使用errors='replace'强制解码
        try:
            content = raw_bytes.decode('utf-8', errors='replace')
            logger.warning(f"Force decoded {path} with utf-8 (some characters may be lost)")
            return content, 'utf-8-forced'
        except Exception:
            pass
        
        logger.error(f"Unable to decode file: {path}")
        return None, None
    
    def _extract_file_metadata(self, path: Path, content: str) -> Dict:
        """提取文件元数据"""
        stats = path.stat()
        
        return {
            'filename': path.name,
            'extension': path.suffix.lower(),
            'file_size': stats.st_size,
            'char_count': len(content),
            'line_count': content.count('\n') + 1,
            'modified_time': stats.st_mtime,
            'parent_dir': path.parent.name
        }
    
    def count_files(
        self,
        directory: str,
        recursive: bool = True
    ) -> int:
        """统计目录中的文件数量"""
        count = 0
        dir_path = Path(directory)
        
        if not dir_path.exists() or not dir_path.is_dir():
            return 0
        
        for ext in self.supported_extensions:
            pattern = f"*{ext}"
            if recursive:
                count += len(list(dir_path.rglob(pattern)))
            else:
                count += len(list(dir_path.glob(pattern)))
        
        return count
