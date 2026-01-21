"""
智能分块器 - 按语义和结构分割文章
Smart Chunker - Splitting Articles by Semantics and Structure
"""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ArticleChunk:
    """文章片段 / Article Chunk"""
    content: str
    metadata: Dict
    chunk_type: str  # paragraph, dialogue, description, action, inner_monologue


class StyleAwareChunker:
    """
    风格感知分块器
    Style-Aware Chunker
    
    按语义和结构智能分割文章，识别对话、描写、动作等不同类型内容
    """
    
    # 中文对话标记 / Chinese dialogue markers
    DIALOGUE_PATTERNS = [
        r'[「『""].*?[」』""]',  # 全角引号
        r'".*?"',                 # 半角双引号
        r"'.*?'",                 # 半角单引号
    ]
    
    # 章节标记 / Chapter markers
    CHAPTER_PATTERNS = [
        r'^第[一二三四五六七八九十百千万零\d]+[章节回]',
        r'^Chapter\s*\d+',
        r'^CHAPTER\s*\d+',
    ]
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50
    ):
        """
        初始化分块器
        
        Args:
            chunk_size: 目标分块大小（字符数）
            chunk_overlap: 分块重叠（字符数）
            min_chunk_size: 最小分块大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_article(
        self, 
        content: str, 
        metadata: Optional[Dict] = None
    ) -> List[ArticleChunk]:
        """
        分块文章
        
        Args:
            content: 文章内容
            metadata: 文章元数据
            
        Returns:
            分块列表
        """
        if metadata is None:
            metadata = {}
        
        chunks = []
        
        # 1. 先按段落分割
        paragraphs = self._split_by_paragraph(content)
        
        for para in paragraphs:
            if len(para.strip()) < self.min_chunk_size:
                continue
            
            # 2. 分类段落类型
            chunk_type = self._classify_chunk(para)
            
            # 3. 根据类型处理
            if chunk_type == 'dialogue':
                chunks.extend(self._process_dialogue(para, metadata))
            else:
                chunks.extend(self._process_prose(para, chunk_type, metadata))
        
        return chunks
    
    def _split_by_paragraph(self, content: str) -> List[str]:
        """按段落分割 / Split by paragraphs"""
        # 使用多个换行符分割
        paragraphs = re.split(r'\n\s*\n+', content)
        
        # 清理每个段落
        result = []
        for para in paragraphs:
            para = para.strip()
            if para:
                result.append(para)
        
        return result
    
    def _classify_chunk(self, text: str) -> str:
        """
        分类片段类型
        
        Returns:
            chunk_type: dialogue, inner_monologue, action, description
        """
        # 检查对话占比
        dialogue_chars = 0
        for pattern in self.DIALOGUE_PATTERNS:
            matches = re.findall(pattern, text)
            dialogue_chars += sum(len(m) for m in matches)
        
        dialogue_ratio = dialogue_chars / max(len(text), 1)
        
        if dialogue_ratio > 0.4:
            return 'dialogue'
        
        # 检查内心独白
        if re.search(r'(他|她|我)(想|心想|暗道|暗想|思索|寻思)', text):
            return 'inner_monologue'
        
        # 检查动作场景
        if re.search(r'(一道|只见|忽然|突然|猛然|霎时|刹那|瞬间)', text):
            return 'action'
        
        return 'description'
    
    def _process_dialogue(
        self, 
        text: str, 
        metadata: Dict
    ) -> List[ArticleChunk]:
        """处理对话片段 / Process dialogue chunks"""
        chunks = []
        
        # 提取完整对话交互
        # 匹配对话及其上下文
        dialogue_pattern = r'[^。！？\n]*?[「『""].*?[」』""][^。！？\n]*[。！？]?'
        dialogues = re.findall(dialogue_pattern, text)
        
        current_chunk = ""
        for dialogue in dialogues:
            if len(dialogue) < 20:
                continue
            
            if len(current_chunk) + len(dialogue) <= self.chunk_size:
                current_chunk += dialogue + "\n"
            else:
                if current_chunk:
                    chunks.append(ArticleChunk(
                        content=current_chunk.strip(),
                        metadata={**metadata, 'chunk_type': 'dialogue'},
                        chunk_type='dialogue'
                    ))
                current_chunk = dialogue + "\n"
        
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(ArticleChunk(
                content=current_chunk.strip(),
                metadata={**metadata, 'chunk_type': 'dialogue'},
                chunk_type='dialogue'
            ))
        
        # 如果没有提取到对话，按普通方式处理
        if not chunks:
            chunks = self._process_prose(text, 'dialogue', metadata)
        
        return chunks
    
    def _process_prose(
        self, 
        text: str, 
        chunk_type: str, 
        metadata: Dict
    ) -> List[ArticleChunk]:
        """
        处理散文/叙事片段 - 智能分块
        Process prose/narrative chunks - Smart chunking
        
        改进:
        1. 绝不在句子中间断开
        2. 重叠以完整句子为单位
        3. 允许弹性大小以保证句子完整性
        """
        chunks = []
        
        # 如果文本小于chunk_size，直接返回
        if len(text) <= self.chunk_size:
            return [ArticleChunk(
                content=text,
                metadata={**metadata, 'chunk_type': chunk_type},
                chunk_type=chunk_type
            )]
        
        # 按句子分割
        sentences = self._split_sentences(text)
        
        if not sentences:
            # 如果无法分割句子，按原文本处理
            return [ArticleChunk(
                content=text,
                metadata={**metadata, 'chunk_type': chunk_type},
                chunk_type=chunk_type
            )]
        
        current_sentences = []  # 当前块的句子列表
        current_length = 0
        overlap_sentences = []  # 用于重叠的句子
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # 如果单个句子超过chunk_size，单独作为一个chunk（不截断）
            if sentence_len > self.chunk_size:
                # 先保存当前块
                if current_sentences:
                    chunk_text = ''.join(current_sentences)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(ArticleChunk(
                            content=chunk_text.strip(),
                            metadata={**metadata, 'chunk_type': chunk_type},
                            chunk_type=chunk_type
                        ))
                        # 计算重叠句子
                        overlap_sentences = self._get_overlap_sentences(current_sentences)
                    current_sentences = []
                    current_length = 0
                
                # 长句单独作为一个chunk
                chunks.append(ArticleChunk(
                    content=sentence.strip(),
                    metadata={**metadata, 'chunk_type': chunk_type},
                    chunk_type=chunk_type
                ))
                overlap_sentences = [sentence] if len(sentence) <= self.chunk_overlap * 3 else []
                continue
            
            # 检查添加这个句子后是否超过目标大小
            if current_length + sentence_len <= self.chunk_size:
                current_sentences.append(sentence)
                current_length += sentence_len
            else:
                # 当前块已满，保存并开始新块
                if current_sentences:
                    chunk_text = ''.join(current_sentences)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(ArticleChunk(
                            content=chunk_text.strip(),
                            metadata={**metadata, 'chunk_type': chunk_type},
                            chunk_type=chunk_type
                        ))
                        # 计算重叠句子（基于句子数量而非字符数）
                        overlap_sentences = self._get_overlap_sentences(current_sentences)
                
                # 开始新块，添加重叠句子
                current_sentences = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_sentences)
                overlap_sentences = []
        
        # 处理最后一个chunk
        if current_sentences:
            chunk_text = ''.join(current_sentences)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(ArticleChunk(
                    content=chunk_text.strip(),
                    metadata={**metadata, 'chunk_type': chunk_type},
                    chunk_type=chunk_type
                ))
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """
        获取重叠句子 - 从末尾取句子直到达到重叠目标
        Get overlap sentences from the end
        """
        if not sentences or self.chunk_overlap <= 0:
            return []
        
        overlap = []
        total_len = 0
        
        # 从后向前取句子
        for sentence in reversed(sentences):
            if total_len + len(sentence) > self.chunk_overlap:
                break
            overlap.insert(0, sentence)
            total_len += len(sentence)
            # 最多取3个句子作为重叠
            if len(overlap) >= 3:
                break
        
        return overlap
    
    def _split_sentences(self, text: str) -> List[str]:
        """按句子分割 / Split by sentences"""
        # 中英文句子结束标记
        sentence_endings = r'([。！？!?]+|\.{3}|…)'
        
        # 分割但保留标点
        parts = re.split(f'({sentence_endings})', text)
        
        sentences = []
        current = ""
        for i, part in enumerate(parts):
            current += part
            # 如果是句子结束标记后面，添加到结果
            if i % 2 == 1:  # 标点符号部分
                if current.strip():
                    sentences.append(current)
                current = ""
        
        if current.strip():
            sentences.append(current)
        
        return sentences
    
    def extract_chapter_info(self, content: str) -> Optional[Dict]:
        """
        提取章节信息
        
        Returns:
            包含章节号和标题的字典，或None
        """
        for pattern in self.CHAPTER_PATTERNS:
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                chapter_line = content[match.start():].split('\n')[0]
                return {
                    'chapter_marker': match.group(),
                    'chapter_title': chapter_line.strip()
                }
        return None
