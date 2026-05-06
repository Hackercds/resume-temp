"""测试配置 - 面试点：Mock 外部依赖，纯单元测试"""
import sys
import os
import pytest

# 确保项目根在 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_text():
    """模拟简历文本"""
    return """张成都，5年测试开发经验。
    熟悉Python、Go语言开发，曾使用FastAPI构建个性化日报服务。
    使用Python + Redis实现DCS缓存系统，将MES系统某Web页面的平均响应时间从200ms优化到50ms。
    熟悉Robot Framework自动化测试框架。
    了解Kubernetes和Docker容器化技术。"""


@pytest.fixture
def sample_chunks():
    """模拟分块结果"""
    return [
        {
            "chunk_id": "test.pdf_0",
            "content": "张成都，5年测试开发经验。熟悉Python、Go语言开发",
            "file_name": "test.pdf",
            "chunk_index": 0,
            "char_count": 30
        },
        {
            "chunk_id": "test.pdf_1",
            "content": "曾使用FastAPI构建个性化日报服务",
            "file_name": "test.pdf",
            "chunk_index": 1,
            "char_count": 18
        },
        {
            "chunk_id": "test.pdf_2",
            "content": "使用Python + Redis实现DCS缓存系统",
            "file_name": "test.pdf",
            "chunk_index": 2,
            "char_count": 22
        }
    ]


@pytest.fixture
def sample_vectors():
    """模拟1024维向量"""
    import numpy as np
    return np.random.randn(3, 1024).astype(np.float32)
