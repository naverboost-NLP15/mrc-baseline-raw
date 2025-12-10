"""
문서 로딩 모듈
"""
import os
import json
from typing import Dict, List, Tuple


class WikipediaLoader:
    """Wikipedia 문서 로더"""
    
    def __init__(self, data_path: str, context_path: str = "wikipedia_documents.json"):
        self.data_path = data_path
        self.context_path = context_path
        self.full_path = os.path.join(data_path, context_path)
    
    def load(self) -> Dict:
        """Wikipedia 문서를 로드합니다."""
        with open(self.full_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        return wiki
    
    def load_as_lists(self) -> Tuple[List[str], List[str], List[int]]:
        """
        Wikipedia 문서를 텍스트, 제목, 문서ID 리스트로 반환합니다.
        
        Returns:
            texts: 문서 텍스트 리스트
            titles: 문서 제목 리스트
            doc_ids: 문서 ID 리스트
        """
        wiki = self.load()
        
        texts = []
        titles = []
        doc_ids = []
        seen_texts = set()
        
        for v in wiki.values():
            text, title, doc_id = v["text"], v["title"], v["document_id"]
            if text in seen_texts:
                continue
            seen_texts.add(text)
            texts.append(text)
            titles.append(title)
            doc_ids.append(doc_id)
        
        return texts, titles, doc_ids
