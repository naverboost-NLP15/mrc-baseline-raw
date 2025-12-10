"""
텍스트 전처리 모듈 (청킹, 정규화, 토크나이징)
"""
from typing import List
from kiwipiepy import Kiwi
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextPreprocessor:
    """텍스트 전처리기 (청킹, 토크나이징)"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kiwi = Kiwi()
    
    def split_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할합니다.
        Kiwi로 문장 분리 후 RecursiveCharacterTextSplitter로 청킹
        """
        if not text:
            return []
        
        try:
            sents = [sent.text for sent in self.kiwi.split_into_sents(text)]
        except Exception:
            sents = text.split(". ")
        
        preprocessed_text = "\n\n".join(sents)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
        
        return text_splitter.split_text(preprocessed_text)
    
    def kiwi_tokenize(self, text: str) -> List[str]:
        """
        Kiwi 형태소 분석기를 사용하여 텍스트를 토큰화합니다.
        명사(N*), 동사(V*), 숫자(SN), 영어(SL), 한자(SH)만 추출
        """
        tokens = self.kiwi.tokenize(text)
        return [
            t.form
            for t in tokens
            if t.tag.startswith("N")
            or t.tag.startswith("V")
            or t.tag.startswith("SN")
            or t.tag.startswith("SL")
            or t.tag.startswith("SH")
        ]
    
    def chunk_documents(
        self, texts: List[str], titles: List[str], doc_ids: List[int]
    ) -> tuple:
        """
        문서 리스트를 청킹하여 반환합니다.
        
        Returns:
            chunk_texts: 청크 텍스트 리스트
            chunk_titles: 청크별 제목 리스트
            chunk_doc_ids: 청크별 문서 ID 리스트
            chunk_ids: 청크 고유 ID 리스트
        """
        chunk_texts = []
        chunk_titles = []
        chunk_doc_ids = []
        chunk_ids = []
        
        for text, title, doc_id in zip(texts, titles, doc_ids):
            chunks = self.split_text(text)
            for chunk in chunks:
                chunk_texts.append(chunk)
                chunk_titles.append(title)
                chunk_doc_ids.append(doc_id)
                chunk_ids.append(len(chunk_ids))
        
        return chunk_texts, chunk_titles, chunk_doc_ids, chunk_ids
