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

    def chunk_text(self, text: str, file_name: str, strategy: str = None) -> List[Dict]:
        """
        文本分块 - 支持三种策略：
        - fixed: 滑动窗口（默认）
        - semantic: 按 Markdown/PDF 标题语义分块
        - hybrid: 标题边界 + 滑动窗口兜底
        """
        if strategy is None:
            strategy = getattr(get_config().chunk, 'strategy', 'fixed')

        if strategy == 'semantic':
            return self._chunk_by_headings(text, file_name)
        if strategy == 'hybrid':
            return self._chunk_hybrid(text, file_name)
        return self._chunk_fixed(text, file_name)

    def _chunk_fixed(self, text: str, file_name: str) -> List[Dict]:
        """
        滑动窗口分块 - 面试点：句子边界对齐
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
                    "char_count": len(chunk_text),
                    "section_title": self._infer_section_title(text, start),
                })
                chunk_index += 1

            # 滑动窗口，重叠 overlap 字符
            start = end - self.overlap

        return chunks

    def _chunk_by_headings(self, text: str, file_name: str) -> List[Dict]:
        """
        按标题语义分块。
        识别 text 中由【标题】标记的各级标题，把标题下内容作为一个 chunk。
        如果某个 chunk 超过 chunk_size，再用滑动窗口细分。
        """
        import re

        heading_pattern = re.compile(r'^【标题】(\S.*?)$', re.MULTILINE)
        matches = list(heading_pattern.finditer(text))

        chunks = []
        chunk_index = 0

        if not matches:
            # 无标题，回退到固定窗口
            return self._chunk_fixed(text, file_name)

        # 处理标题前的引言
        if matches[0].start() > 0:
            intro = text[:matches[0].start()].strip()
            if intro:
                chunks.append({
                    "chunk_id": f"{file_name}_{chunk_index}",
                    "content": intro,
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "char_count": len(intro),
                    "section_title": "",
                })
                chunk_index += 1

        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if not section_text:
                continue

            # 如果章节太长，拆成多个 chunk 但保留同一 section_title
            if len(section_text) > self.chunk_size:
                sub_chunks = self._split_section(section_text, title, file_name, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            else:
                chunks.append({
                    "chunk_id": f"{file_name}_{chunk_index}",
                    "content": f"【{title}】\n{section_text}",
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "char_count": len(section_text),
                    "section_title": title,
                })
                chunk_index += 1

        return chunks

    def _chunk_hybrid(self, text: str, file_name: str) -> List[Dict]:
        """
        混合分块：优先按标题切分，标题之间内容过大时再用滑动窗口。
        与 _chunk_by_headings 类似，但不再在 chunk 内容前加【标题】前缀。
        """
        import re

        heading_pattern = re.compile(r'^【标题】(\S.*?)$', re.MULTILINE)
        matches = list(heading_pattern.finditer(text))

        if not matches:
            return self._chunk_fixed(text, file_name)

        chunks = []
        chunk_index = 0

        def add_chunk(content, title):
            nonlocal chunk_index
            content = content.strip()
            if not content:
                return
            chunks.append({
                "chunk_id": f"{file_name}_{chunk_index}",
                "content": content,
                "file_name": file_name,
                "chunk_index": chunk_index,
                "char_count": len(content),
                "section_title": title,
            })
            chunk_index += 1

        # 标题前引言
        if matches[0].start() > 0:
            add_chunk(text[:matches[0].start()], "")

        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if not section_text:
                continue

            if len(section_text) > self.chunk_size:
                sub_chunks = self._split_section(section_text, title, file_name, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            else:
                add_chunk(section_text, title)

        return chunks

    def _split_section(self, section_text: str, section_title: str, file_name: str, start_index: int) -> List[Dict]:
        """对一个过长的章节按滑动窗口拆分，保留 section_title"""
        chunks = []
        start = 0
        text_len = len(section_text)
        chunk_index = start_index

        while start < text_len:
            end = start + self.chunk_size
            if end < text_len:
                part = section_text[start:end]
                sentence_breaks = '。！？；\n'
                positions = []
                for p in sentence_breaks:
                    pos = part.rfind(p)
                    if pos >= 0:
                        positions.append(pos)
                if positions:
                    last_punct = max(positions)
                    if last_punct > self.chunk_size // 2:
                        end = start + last_punct + 1

            chunk_text = section_text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": f"{file_name}_{chunk_index}",
                    "content": chunk_text,
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "char_count": len(chunk_text),
                    "section_title": section_title,
                })
                chunk_index += 1
            start = end - self.overlap

        return chunks

    def _infer_section_title(self, text: str, position: int) -> str:
        """从 position 向前查找最近的【标题】，作为 section_title"""
        import re
        prefix = text[:position]
        matches = list(re.finditer(r'^【标题】(\S.*?)$', prefix, re.MULTILINE))
        if matches:
            return matches[-1].group(1).strip()
        return ""

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
