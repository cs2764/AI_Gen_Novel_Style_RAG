"""
文章预处理器 - 清洗和标准化文章
Article Preprocessor - Cleaning and Normalizing Articles
"""

import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ArticlePreprocessor:
    """
    文章预处理器
    Article Preprocessor
    
    负责文章的清洗、标准化和元数据提取
    """
    
    # 风格关键词 / Style keywords
    STYLE_KEYWORDS = {
        'xianxia': ['修炼', '灵气', '丹药', '仙人', '境界', '灵力', '天道', '渡劫'],
        'wuxia': ['武功', '江湖', '侠客', '剑法', '内力', '掌门', '门派'],
        'romance': ['爱情', '暧昧', '心动', '甜蜜', '拥抱', '亲吻'],
        'urban': ['总裁', '都市', '公司', '职场', '白领'],
        'fantasy': ['魔法', '精灵', '龙', '魔王', '异世界', '冒险'],
        'scifi': ['飞船', '星际', '机器人', '科技', '未来', '太空'],
        'system': ['穿越', '系统', '重生', '金手指', '任务', '升级'],
        'historical': ['皇帝', '皇后', '朝廷', '后宫', '太监', '嫔妃'],
    }
    
    def __init__(
        self,
        remove_empty_lines: bool = True,
        normalize_punctuation: bool = True,
        remove_urls: bool = True,
        remove_author_notes: bool = True
    ):
        """
        初始化预处理器
        
        Args:
            remove_empty_lines: 是否移除多余空行
            normalize_punctuation: 是否标准化标点符号
            remove_urls: 是否移除URL
            remove_author_notes: 是否移除作者注释
        """
        self.remove_empty_lines = remove_empty_lines
        self.normalize_punctuation = normalize_punctuation
        self.remove_urls = remove_urls
        self.remove_author_notes = remove_author_notes
    
    def preprocess(self, article: Dict) -> Dict:
        """
        预处理文章
        
        Args:
            article: 包含 'content', 'file_path', 'metadata' 的字典
            
        Returns:
            预处理后的文章字典
        """
        content = article['content']
        metadata = article.get('metadata', {}).copy()
        
        # 1. 清洗内容
        content = self._clean_content(content)
        
        # 2. 提取额外元数据
        extra_metadata = self._extract_metadata(content)
        metadata.update(extra_metadata)
        
        return {
            'content': content,
            'file_path': article.get('file_path', ''),
            'metadata': metadata
        }
    
    def _clean_content(self, content: str) -> str:
        """清洗文章内容"""
        # 移除URL
        if self.remove_urls:
            content = re.sub(
                r'https?://[^\s<>"{}|\\^`\[\]]+',
                '',
                content
            )
        
        # 移除作者注释
        if self.remove_author_notes:
            # 常见的作者注释模式
            patterns = [
                r'【作者.*?】',
                r'（作者.*?）',
                r'\(作者.*?\)',
                r'PS[：:].{0,100}',
                r'——+\s*作者有话说.{0,200}',
            ]
            for pattern in patterns:
                content = re.sub(pattern, '', content, flags=re.DOTALL)
        
        # 标准化标点
        if self.normalize_punctuation:
            content = self._normalize_punctuation(content)
        
        # 移除多余空行
        if self.remove_empty_lines:
            content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # 清理首尾空白
        content = content.strip()
        
        return content
    
    def _normalize_punctuation(self, content: str) -> str:
        """标准化标点符号"""
        # 英文标点转中文标点（可选）
        replacements = {
            '...': '……',
            '..': '……',
            '--': '——',
        }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # 规范化引号
        # content = re.sub(r'"([^"]*)"', r'「\1」', content)
        
        return content
    
    def _extract_metadata(self, content: str) -> Dict:
        """提取文章元数据"""
        metadata = {}
        
        # 检测风格
        metadata['detected_style'] = self._detect_style(content)
        
        # 检测是否有对话
        metadata['has_dialogue'] = self._has_dialogue(content)
        
        # 提取章节信息
        chapter_info = self._extract_chapter_info(content)
        if chapter_info:
            metadata.update(chapter_info)
        
        return metadata
    
    def _detect_style(self, content: str) -> str:
        """检测文章风格"""
        scores = {}
        
        for style, keywords in self.STYLE_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                count = content.count(keyword)
                score += count
            if score > 0:
                scores[style] = score
        
        if scores:
            return max(scores, key=scores.get)
        return 'general'
    
    def _has_dialogue(self, content: str) -> bool:
        """检查是否包含对话"""
        dialogue_patterns = [
            r'[「『""].*?[」』""]',
            r'".*?"',
        ]
        
        for pattern in dialogue_patterns:
            if re.search(pattern, content):
                return True
        return False
    
    def _extract_chapter_info(self, content: str) -> Optional[Dict]:
        """提取章节信息"""
        chapter_patterns = [
            r'第([一二三四五六七八九十百千万零\d]+)[章节回][\s：:]*(.{0,50})',
            r'Chapter\s*(\d+)[:\s]*(.{0,50})?',
        ]
        
        for pattern in chapter_patterns:
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                groups = match.groups()
                return {
                    'chapter_number': groups[0] if groups else None,
                    'chapter_title': groups[1].strip() if len(groups) > 1 and groups[1] else None
                }
        
        return None
    
    def preprocess_batch(self, articles: List[Dict]) -> List[Dict]:
        """批量预处理文章"""
        return [self.preprocess(article) for article in articles]
