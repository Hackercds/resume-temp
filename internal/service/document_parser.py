"""
文档解析服务 - 面试核心问题：
1. 为什么拆成 parser + chunk 两层？职责分离，parser 负责提取文本，chunk 负责分块
2. PDF 解析为什么用 PyMuPDF 而不是 pdfplumber？速度快（C实现），内存占用低
3. 如何检测文件类型？扩展名 + Magic Number 双重校验防止恶意伪装
4. Markdown 解析策略：去除语法噪音（#、```、|）但保留语义结构（标题分级、代码块、列表）
"""
import os
import io
import re
from typing import Optional

from internal.pkg.errors import UnsupportedFileTypeError


class DocumentParser:
    """文档解析服务 - 支持 PDF / TXT / CSV / MD"""

    def parse(self, content: bytes, file_name: str) -> str:
        """
        根据文件类型调用对应解析器
        面试点：扩展名检测，生产环境应加 Magic Number 校验
        """
        ext = os.path.splitext(file_name)[1].lower()

        if ext == '.pdf':
            return self._parse_pdf(content)
        elif ext == '.txt':
            return self._parse_txt(content)
        elif ext == '.csv':
            return self._parse_csv(content)
        elif ext in ('.md', '.markdown'):
            return self._parse_markdown(content)
        else:
            raise UnsupportedFileTypeError(
                f"不支持的文件格式: {ext}，仅支持 PDF/TXT/CSV/MD"
            )

    def _parse_pdf(self, content: bytes) -> str:
        """
        PDF 解析 - 面试点：PyMuPDF (fitz) vs pdfplumber vs pdfminer
        - PyMuPDF: 速度快（C实现），文本提取准，本项目首选
        - pdfplumber: 表格处理好，但速度较慢
        - pdfminer: 精确但速度最慢
        """
        import fitz  # PyMuPDF

        doc = fitz.open(stream=content, filetype="pdf")
        text_parts = []

        try:
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(page_text.strip())
        finally:
            doc.close()

        text = "\n".join(text_parts)
        return self._clean_text(text)

    def _parse_txt(self, content: bytes) -> str:
        """TXT 解析 - 尝试多种编码"""
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
            try:
                text = content.decode(encoding)
                return self._clean_text(text)
            except (UnicodeDecodeError, LookupError):
                continue
        # 最后的兜底
        text = content.decode('utf-8', errors='ignore')
        return self._clean_text(text)

    def _parse_csv(self, content: bytes) -> str:
        """
        CSV 解析 - 返回原始文本，交给 chunk_service 按行分块
        面试点：为什么 CSV 不在这里做分块？
        答：职责分离，parser 不关心下游如何处理文本
        """
        for encoding in ['utf-8', 'gbk', 'gb2312']:
            try:
                return content.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        return content.decode('utf-8', errors='ignore')

    def _parse_markdown(self, content: bytes) -> str:
        """
        Markdown 解析 - 面试点：为什么不直接当 TXT 处理？
        答：MD 含有语法噪音（#、*、```、|），如果直接丢进 embedding 会影响检索质量。
        这里把语法符号去掉，但保留语义结构：
          - 标题转成"【标题】xxx"形式，方便后续按 heading 分块
          - 代码块用 "代码: " 前缀保留，与正文区分
          - 表格行保留（去掉 | 边界）
          - 链接保留文本部分

        注意：parser 只输出纯文本 + 结构标记，chunk 策略由 chunk_service 决定
        """
        for encoding in ['utf-8', 'gbk', 'utf-8-sig']:
            try:
                text = content.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            text = content.decode('utf-8', errors='ignore')

        lines = text.split('\n')
        out_lines = []
        in_code_block = False

        for raw_line in lines:
            line = raw_line.rstrip()

            # 围栏代码块：``` 或 ~~~
            if re.match(r'^\s*```', line) or re.match(r'^\s*~~~', line):
                in_code_block = not in_code_block
                # 保留代码块开始/结束标记，让 chunk 知道这是结构边界
                out_lines.append('[代码块]' if in_code_block else '[/代码块]')
                continue

            if in_code_block:
                # 代码块内：行首加 "    " 表示保留
                out_lines.append('    ' + line)
                continue

            # 标题：# ～ ######  → 【标题】xxx
            m = re.match(r'^(#{1,6})\s+(.*)$', line)
            if m:
                title = m.group(2).strip()
                # 去掉尾部 # 闭合符（GFM）
                title = re.sub(r'\s+#+\s*$', '', title)
                if title:
                    out_lines.append(f'【标题】{title}')
                continue

            # 引用：> xxx →  xxx
            if line.startswith('>'):
                line = re.sub(r'^>\s*', '', line)
                if line:
                    out_lines.append(line)
                continue

            # 水平线
            if re.match(r'^\s*([-*_])\1{2,}\s*$', line):
                out_lines.append('')  # 留空行作为分块边界
                continue

            # 表格行：| ... | 或 |---|---|
            if '|' in line and re.match(r'^\s*\|', line):
                # 跳过纯分隔行 | --- | --- |
                if re.match(r'^\s*\|?\s*[-:|\s]+\|?\s*$', line):
                    continue
                # 去掉首尾 | 和多余空白
                cells = [c.strip() for c in line.strip().strip('|').split('|')]
                out_lines.append('  '.join(c for c in cells if c))
                continue

            # 列表项：- /* /1. →  去掉前缀符号
            line = re.sub(r'^\s*[-*+]\s+', '  • ', line)
            line = re.sub(r'^\s*\d+\.\s+', '  • ', line)

            # 行内格式：加粗、斜体、代码、链接
            # 链接 [text](url) → text
            line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)
            # 图片 ![alt](url) → alt
            line = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[\1]', line)
            # 行内代码 `code` → "code"
            line = re.sub(r'`([^`]+)`', r'\1', line)
            # 加粗 **text** / __text__ → text
            line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
            line = re.sub(r'__([^_]+)__', r'\1', line)
            # 斜体 *text* / _text_ → text
            line = re.sub(r'\*([^*]+)\*', r'\1', line)
            line = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', line)

            # HTML 标签：<br> <kbd> 等
            line = re.sub(r'<[^>]+>', '', line)

            out_lines.append(line)

        return self._clean_text('\n'.join(out_lines))

    def _clean_text(self, text: str) -> str:
        """
        文本清洗 - 面试点：为什么要清洗？
        1. 去除控制字符（\x00-\x1f），保留换行
        2. 规范化多行空白（多个\n → 最多2个）
        3. 去除行首尾空白
        4. 保留中文标点和英文标点
        """
        # 去除控制字符（保留换行符 \n、\t）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # 规范化换行：3个以上连续换行 → 2个
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 去除行首尾空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)

        # 规范化空白字符（多个空格 → 1个）
        text = re.sub(r' {2,}', ' ', text)

        return text

    def parse_file(self, file_path: str) -> str:
        """从文件路径解析（用于本地文件）"""
        file_name = os.path.basename(file_path)
        with open(file_path, 'rb') as f:
            content = f.read()
        return self.parse(content, file_name)
