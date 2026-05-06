"""
文档解析服务 - 面试核心问题：
1. 为什么拆成 parser + chunk 两层？职责分离，parser 负责提取文本，chunk 负责分块
2. PDF 解析为什么用 PyMuPDF 而不是 pdfplumber？速度快（C实现），内存占用低
3. 如何检测文件类型？扩展名 + Magic Number 双重校验防止恶意伪装
"""
import os
import io
import re
from typing import Optional

from internal.pkg.errors import UnsupportedFileTypeError


class DocumentParser:
    """文档解析服务 - 支持 PDF / TXT / CSV"""

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
        else:
            raise UnsupportedFileTypeError(
                f"不支持的文件格式: {ext}，仅支持 PDF/TXT/CSV"
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
