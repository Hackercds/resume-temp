"""
文本分块服务 - 面试核心问题：
1. 分块策略为什么选滑动窗口？简单可控，效率高
2. chunk_size=400 为什么？中文约200词，10块=4000字，留足够空间给prompt+答案
3. overlap=50 为什么？关键句刚好被切断时，前后块各有50字重叠能完整检索
4. CSV 为什么按行分块？每行是独立的语义单元（一条记录），保持语义完整
"""
from typing import List, Dict

from internal.model.config import get_config


class ChunkService:
    """文本分块服务"""

    def __init__(self, chunk_size: int = None, overlap: int = None):
        cfg = get_config()
        chunk_cfg = cfg.chunk
        self.chunk_size = chunk_size or chunk_cfg.chunk_size
        self.overlap = overlap or chunk_cfg.overlap

    def chunk_text(self, text: str, file_name: str) -> List[Dict]:
        """
        滑动窗口分块 - 面试点：句子边界对齐
        策略：
        1. 按 chunk_size 切分
        2. 如果切断行/句，向前找最近的标点符号（。！？；\n）截断
        3. 窗口向后滑动 overlap 个字符
        """
        chunks = []
        start = 0
        chunk_index = 0
        text_len = len(text)

        if text_len == 0:
            return chunks

        while start < text_len:
            end = start + self.chunk_size

            # 句子边界对齐：如果切断句子，向前找最近的标点
            if end < text_len:
                chunk_text_at_end = text[start:end]
                # 找最近的句号/问号/感叹号/分号/换行
                sentence_breaks = '。！？；\n'
                positions = []
                for p in sentence_breaks:
                    pos = chunk_text_at_end.rfind(p)
                    if pos >= 0:
                        positions.append(pos)

                if positions:
                    last_punct = max(positions)
                    # 只有标点在后半段时才截断，避免块太短
                    if last_punct > self.chunk_size // 2:
                        end = start + last_punct + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": f"{file_name}_{chunk_index}",
                    "content": chunk_text,
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "char_count": len(chunk_text)
                })
                chunk_index += 1

            # 滑动窗口，重叠 overlap 字符
            start = end - self.overlap

        return chunks

    def chunk_csv(self, content: str, file_name: str) -> List[Dict]:
        """
        CSV 按行分块 - 面试点：CSV 为什么按行分块而不是按字数？
        答：CSV 每行通常是一个独立的语义单元（一条记录），
           按行分块保持每行语义完整，检索时能精确匹配到具体行
        """
        lines = content.strip().split('\n')
        chunks = []
        for chunk_index, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            chunks.append({
                "chunk_id": f"{file_name}_row_{chunk_index}",
                "content": line,
                "file_name": file_name,
                "chunk_index": chunk_index,
                "char_count": len(line)
            })
        return chunks

    def chunk_by_paragraph(self, text: str, file_name: str,
                           max_chunk_size: int = None) -> List[Dict]:
        """
        段落分块（可选方案）- 面试点：什么时候用段落分块？
        答：报告/论文等自然段落分明的文档，段落分块语义更完整
        """
        if max_chunk_size is None:
            max_chunk_size = self.chunk_size

        paragraphs = text.split('\n\n')
        chunks = []
        chunk_index = 0
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前块 + 新段落不超限，合并
            if len(current_chunk) + len(para) <= max_chunk_size:
                current_chunk = (current_chunk + '\n\n' + para).strip()
            else:
                # 当前块已满，保存并开始新块
                if current_chunk:
                    chunks.append({
                        "chunk_id": f"{file_name}_{chunk_index}",
                        "content": current_chunk,
                        "file_name": file_name,
                        "chunk_index": chunk_index,
                        "char_count": len(current_chunk)
                    })
                    chunk_index += 1
                current_chunk = para

        # 最后一块
        if current_chunk:
            chunks.append({
                "chunk_id": f"{file_name}_{chunk_index}",
                "content": current_chunk,
                "file_name": file_name,
                "chunk_index": chunk_index,
                "char_count": len(current_chunk)
            })

        return chunks
