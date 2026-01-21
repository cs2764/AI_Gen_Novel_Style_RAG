"""
查询构建器 - 从创作上下文生成检索查询
Query Builder - Generating Retrieval Queries from Creative Context
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass

from style_rag.retrieval.retriever import RetrievalQuery


@dataclass
class CreativeContext:
    """创作上下文 / Creative Context"""
    storyline: str                          # 故事线
    chapter_summary: Optional[str] = None   # 章节摘要
    character_states: Optional[Dict] = None # 人物状态
    writing_phase: Optional[str] = None     # 写作阶段


class QueryBuilder:
    """
    查询构建器
    Query Builder
    
    从创作上下文自动构建检索查询
    """
    
    # 情感关键词映射 / Emotion keyword mapping
    EMOTION_KEYWORDS = {
        '激动': ['战斗', '冲突', '愤怒', '激动', '震惊', '爆发'],
        '温馨': ['温暖', '爱', '拥抱', '微笑', '幸福', '甜蜜'],
        '紧张': ['危险', '逃跑', '追杀', '紧张', '恐惧', '威胁'],
        '悲伤': ['离别', '死亡', '泪', '悲', '痛苦', '失去'],
        '神秘': ['秘密', '未知', '古老', '神秘', '隐藏', '谜团'],
        '浪漫': ['月光', '星空', '相遇', '心动', '暧昧', '表白'],
        '幽默': ['笑', '搞笑', '有趣', '滑稽', '调侃'],
    }
    
    # 写作阶段映射 / Writing phase mapping
    PHASE_TYPE_MAP = {
        'opening': 'description',
        'development': 'description',
        'dialogue': 'dialogue',
        'climax': 'action',
        'ending': 'description',
        'action': 'action',
        'romance': 'description',
    }
    
    def build_query_from_context(
        self,
        context: CreativeContext
    ) -> RetrievalQuery:
        """
        从创作上下文构建检索查询
        
        Args:
            context: 创作上下文
            
        Returns:
            检索查询
        """
        # 提取场景描述
        scene_description = self._extract_scene(
            context.storyline,
            context.chapter_summary
        )
        
        # 提取人物名称
        character_names = []
        if context.character_states:
            character_names = list(context.character_states.keys())[:3]
        
        # 推断情感基调
        emotion_tone = self._infer_emotion(context.storyline)
        
        # 推断写作类型
        writing_type = self._infer_writing_type(context.writing_phase)
        
        return RetrievalQuery(
            text=scene_description,
            scene_description=scene_description,
            character_names=character_names,
            emotion_tone=emotion_tone,
            writing_type=writing_type
        )
    
    def build_query_from_text(
        self,
        text: str,
        writing_type: Optional[str] = None
    ) -> RetrievalQuery:
        """
        从文本构建检索查询
        
        Args:
            text: 输入文本
            writing_type: 写作类型（可选）
            
        Returns:
            检索查询
        """
        # 推断情感
        emotion = self._infer_emotion(text)
        
        # 推断写作类型
        if writing_type is None:
            writing_type = self._detect_writing_type(text)
        
        return RetrievalQuery(
            text=text,
            scene_description=self._extract_key_sentences(text),
            emotion_tone=emotion,
            writing_type=writing_type
        )
    
    def _extract_scene(
        self,
        storyline: str,
        summary: Optional[str]
    ) -> str:
        """提取场景描述"""
        # 取故事线的关键描述
        text = storyline + (summary or "")
        sentences = text.replace('。', '。\n').replace('！', '！\n').split('\n')
        
        # 取最具信息量的句子
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 10][:3]
        return '。'.join(key_sentences)
    
    def _extract_key_sentences(self, text: str, max_sentences: int = 2) -> str:
        """提取关键句子"""
        sentences = re.split(r'[。！？]', text)
        key = [s.strip() for s in sentences if len(s.strip()) > 10][:max_sentences]
        return '。'.join(key)
    
    def _infer_emotion(self, text: str) -> str:
        """推断情感基调"""
        scores = {}
        
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = sum(text.count(kw) for kw in keywords)
            if score > 0:
                scores[emotion] = score
        
        if scores:
            return max(scores, key=scores.get)
        return '平稳'
    
    def _infer_writing_type(self, phase: Optional[str]) -> Optional[str]:
        """推断写作类型"""
        if phase is None:
            return None
        return self.PHASE_TYPE_MAP.get(phase.lower(), 'description')
    
    def _detect_writing_type(self, text: str) -> str:
        """检测文本的写作类型"""
        # 检查对话比例
        dialogue_patterns = [r'[「『""].*?[」』""]', r'".*?"']
        dialogue_count = sum(
            len(re.findall(p, text)) for p in dialogue_patterns
        )
        
        if dialogue_count > 2:
            return 'dialogue'
        
        # 检查动作关键词
        action_keywords = ['突然', '猛然', '一道', '只见', '霎时', '瞬间']
        if any(kw in text for kw in action_keywords):
            return 'action'
        
        return 'description'
    
    def build_embellishment_query(
        self,
        draft_content: str,
        target_style: Optional[str] = None
    ) -> List[RetrievalQuery]:
        """
        为润色阶段构建检索查询
        
        Args:
            draft_content: 草稿内容
            target_style: 目标风格
            
        Returns:
            检索查询列表（可能包含多种类型）
        """
        queries = []
        
        # 检测内容类型
        has_dialogue = bool(re.search(r'[「『""]', draft_content))
        
        if has_dialogue:
            # 对话润色查询
            queries.append(RetrievalQuery(
                text=draft_content[:500],
                writing_type='dialogue',
                emotion_tone=self._infer_emotion(draft_content),
                style_preference=target_style
            ))
        
        # 描写润色查询
        queries.append(RetrievalQuery(
            text=draft_content[:500],
            writing_type='description',
            emotion_tone=self._infer_emotion(draft_content),
            style_preference=target_style
        ))
        
        return queries
