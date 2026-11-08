import jieba
from PyPDF2 import PdfReader
from rank_bm25 import BM25Okapi

class PaperRetriever:
    def __init__(self, pdf_path, chunk_size=600, overlap=120):
        self.chunks = []
        self.chunk_pages = []
        self._load_pdf(pdf_path, chunk_size, overlap)
        self._build_bm25()

    def _load_pdf(self, pdf_path, chunk_size, overlap):
        """读取PDF，按字数切分块并记录页码"""
        reader = PdfReader(pdf_path)
        full_text = ""
        page_markers = []  # 记录每页文本的起始位置

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_markers.append((len(full_text), page_idx + 1))
            full_text += page_text

        # 滑动窗口切分
        start = 0
        text_len = len(full_text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = full_text[start:end]
            # 找到这个块对应的页码
            page_num = 1
            for pos, p in page_markers:
                if start >= pos:
                    page_num = p
            self.chunks.append(chunk)
            self.chunk_pages.append(page_num)
            start += (chunk_size - overlap)

    def _build_bm25(self):
        """构建BM25检索索引"""
        tokenized_corpus = [list(jieba.cut(chunk)) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=3):
        """检索最相关的文本块，返回带页码的结果"""
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "page": self.chunk_pages[idx],
                "content": self.chunks[idx]
            })
        return results
