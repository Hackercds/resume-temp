"""
Embedding 服务单元测试 - 面试点：表格驱动测试法
遵循 MICRO_SERVICE_SPEC.md 测试规范
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestChunkService:
    """文本分块服务测试"""

    def test_chunk_text_basic(self):
        """基本分块：短文本只产生一个 chunk"""
        from internal.service.chunk_service import ChunkService
        service = ChunkService(chunk_size=400, overlap=50)
        text = "Hello World 测试"
        chunks = service.chunk_text(text, "test.txt")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Hello World 测试"
        assert chunks[0]["file_name"] == "test.txt"
        assert chunks[0]["chunk_index"] == 0

    def test_chunk_text_long(self):
        """长文本分块：超过 chunk_size 应分多块"""
        from internal.service.chunk_service import ChunkService
        service = ChunkService(chunk_size=100, overlap=20)
        text = "测试内容" * 50  # 200 字
        chunks = service.chunk_text(text, "test.txt")
        assert len(chunks) >= 2
        # 验证 chunk_id 唯一
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_text_sentence_boundary(self):
        """句子边界对齐测试：块边界应在标点处"""
        from internal.service.chunk_service import ChunkService
        service = ChunkService(chunk_size=200, overlap=50)
        # 构造跨边界的文本
        text = "A" * 180 + "。" + "B" * 200
        chunks = service.chunk_text(text, "test.txt")
        # 第一块结尾应该在句号后
        assert chunks[0]["content"].endswith("。")

    def test_chunk_text_empty(self):
        """空文本不应产生 chunk"""
        from internal.service.chunk_service import ChunkService
        service = ChunkService()
        chunks = service.chunk_text("", "test.txt")
        assert len(chunks) == 0

    def test_chunk_csv(self):
        """CSV 按行分块"""
        from internal.service.chunk_service import ChunkService
        service = ChunkService()
        csv_content = "name,age,city\nAlice,25,Beijing\nBob,30,Shanghai"
        chunks = service.chunk_csv(csv_content, "test.csv")
        assert len(chunks) == 3
        assert chunks[0]["chunk_id"] == "test.csv_row_0"
        assert chunks[1]["chunk_id"] == "test.csv_row_1"

    def test_chunk_csv_empty_lines(self):
        """CSV 空行应被跳过"""
        from internal.service.chunk_service import ChunkService
        service = ChunkService()
        csv_content = "line1\n\n\nline2"
        chunks = service.chunk_csv(csv_content, "test.csv")
        assert len(chunks) == 2

    def test_chunk_by_paragraph(self):
        """段落分块"""
        from internal.service.chunk_service import ChunkService
        service = ChunkService()
        text = "段落1内容\n\n段落2内容\n\n段落3内容"
        chunks = service.chunk_by_paragraph(text, "test.txt", max_chunk_size=200)
        assert len(chunks) >= 1


class TestDocumentParser:
    """文档解析器测试"""

    def test_parse_txt_utf8(self):
        """TXT UTF-8 编码解析"""
        from internal.service.document_parser import DocumentParser
        parser = DocumentParser()
        content = "测试中文内容\n第二行".encode('utf-8')
        text = parser.parse(content, "test.txt")
        assert "测试中文内容" in text
        assert "第二行" in text

    def test_parse_unsupported_format(self):
        """不支持的格式应抛出异常"""
        from internal.service.document_parser import DocumentParser
        from internal.pkg.errors import UnsupportedFileTypeError
        parser = DocumentParser()
        with pytest.raises(UnsupportedFileTypeError):
            parser.parse(b"content", "test.docx")

    def test_clean_text_removes_control_chars(self):
        """文本清洗应去除控制字符"""
        from internal.service.document_parser import DocumentParser
        parser = DocumentParser()
        # 包含 \x00 控制字符的文本
        dirty = "正常文本\x00\x01\x08正常"
        cleaned = parser._clean_text(dirty)
        assert "\x00" not in cleaned
        assert "正常文本" in cleaned

    def test_clean_text_normalizes_newlines(self):
        """文本清洗应规范化换行"""
        from internal.service.document_parser import DocumentParser
        parser = DocumentParser()
        text = "行1\n\n\n\n\n行2\n\n\n行3"
        cleaned = parser._clean_text(text)
        # 3+个换行 → 2个换行
        assert "\n\n\n\n" not in cleaned


class TestEmbeddingServiceMocks:
    """Embedding 服务 Mock 测试

    面试点：为什么要 mock 外部依赖？
    答：1) 不依赖模型下载和 ES 连接就能跑测试
       2) CI/CD 环境不需要 GPU
       3) 测试速度快
    """

    @patch('sentence_transformers.SentenceTransformer')
    def test_load_model(self, mock_st):
        """模拟模型加载"""
        from internal.service.embedding_service import EmbeddingService

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st.return_value = mock_model

        service = EmbeddingService(model_name="BAAI/bge-small-zh-v1.5")
        service.load_model()

        assert service.is_loaded
        assert service.dim == 1024

    @patch('sentence_transformers.SentenceTransformer')
    def test_encode_single(self, mock_st):
        """单条向量化"""
        from internal.service.embedding_service import EmbeddingService

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_model.encode.return_value = np.random.randn(1, 1024).astype(np.float32)
        mock_st.return_value = mock_model

        service = EmbeddingService(model_name="BAAI/bge-small-zh-v1.5")
        service.load_model()
        result = service.encode(["测试文本"])

        assert result.shape == (1, 1024)
        mock_model.encode.assert_called_once()

    @patch('sentence_transformers.SentenceTransformer')
    def test_encode_query(self, mock_st):
        """查询向量化"""
        from internal.service.embedding_service import EmbeddingService

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_model.encode.return_value = np.random.randn(1, 1024).astype(np.float32)
        mock_st.return_value = mock_model

        service = EmbeddingService(model_name="BAAI/bge-small-zh-v1.5")
        service.load_model()
        result = service.encode_query("测试问题")

        assert result.shape == (1024,)
