# 简历 RAG 智能问答系统 - API 文档

## 统一响应格式

所有接口遵循统一响应格式：

```json
{
    "code": 0,
    "message": "success",
    "data": {},
    "requestId": "abc12345",
    "timestamp": 1746134400000
}
```

## 错误码规范

| 区间 | 用途 |
|------|------|
| 0 | 成功 |
| 40001-40099 | 参数错误 |
| 40401-40499 | 资源不存在 |
| 50001-50099 | 内部错误 |

---

## 1. GET /health

健康检查接口

**响应示例：**
```json
{
    "code": 0,
    "message": "success",
    "data": {
        "status": "ok",
        "version": "1.0.0",
        "embedding_loaded": true,
        "es_connected": true
    }
}
```

---

## 2. POST /api/query

RAG 智能问答

**请求：**
```json
{
    "question": "张成都有哪些Python开发经验？",
    "api_key": "sk-xxxx",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "top_k": 5
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| question | string | 是 | 用户问题，1-2000字 |
| api_key | string | 是 | LLM API Key，后端不存储 |
| provider | string | 否 | LLM Provider，默认 openai |
| model | string | 否 | 模型名，默认 gpt-4o-mini |
| top_k | int | 否 | 检索Top-K，默认5，最大20 |

**响应：**
```json
{
    "code": 0,
    "message": "success",
    "data": {
        "answer": "张成都熟悉Python开发，曾使用FastAPI构建...",
        "sources": [
            {
                "content": "熟悉Python、Go语言开发...",
                "file_name": "张成都-简历.pdf",
                "score": 0.89
            }
        ],
        "trace_id": "abc123",
        "timing": {
            "embedding_ms": 85.3,
            "search_ms": 23.1,
            "llm_s": 1.2
        }
    }
}
```

---

## 3. POST /api/knowledge/upload

上传文档入库

**请求：** `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | 是 | 文件，支持 PDF/TXT/CSV，最大10MB |
| chunk_size | int | 否 | 分块大小，默认400 |
| overlap | int | 否 | 块间重叠，默认50 |

**响应：**
```json
{
    "code": 0,
    "message": "success",
    "data": {
        "file_name": "简历.pdf",
        "file_type": "pdf",
        "chunks_created": 45,
        "total_chunks": 45
    }
}
```

---

## 4. GET /api/knowledge/documents

列出知识库所有文档

**响应：**
```json
{
    "code": 0,
    "message": "success",
    "data": {
        "documents": [
            {
                "file_name": "简历.pdf",
                "file_type": "pdf",
                "chunk_count": 45,
                "upload_time": "2026-05-06T10:00:00"
            }
        ]
    }
}
```

---

## 5. DELETE /api/knowledge/documents/{file_name}

删除文档（含所有关联 chunk）

**路径参数：**
- `file_name`：文件名（需 URL 编码）

**响应：**
```json
{
    "code": 0,
    "message": "success",
    "data": {
        "deleted_chunks": 45,
        "remaining_chunks": 0
    }
}
```

---

## 6. GET /api/stats

知识库统计信息

**响应：**
```json
{
    "code": 0,
    "message": "success",
    "data": {
        "total_chunks": 216,
        "total_documents": 5,
        "documents": [...]
    }
}
```
