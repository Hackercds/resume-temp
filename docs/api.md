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

## 流式响应事件（SSE）

`/api/query/stream` 接口以 SSE 推送事件，每条 `data:` 帧为 JSON 字符串，事件类型：

| 事件 type | 关键字段 | 说明 |
|-----------|---------|------|
| `token` | `content` | LLM 流式输出片段（仅 content，过滤 thinking/reasoning） |
| `done` | `answer/sources/trace/timing/trace_id` | 流式结束，返回完整结果 |
| `error` | `message/suggestion/empty_retrieval/trace_id` | 流式错误（含可操作建议） |

---

## 1. GET /health

健康检查接口

**响应示例：**
```json
{
    "code": 0,
    "data": {
        "status": "ok",
        "version": "1.1.0",
        "embedding_loaded": true,
        "es_connected": true,
        "default_api_key_configured": false
    }
}
```

---

## 2. GET /api/config

返回前端公开配置（不含敏感信息）

**响应：**
```json
{
    "code": 0,
    "data": {
        "llm_presets": [
            {"name": "OpenAI GPT-4o-mini", "provider": "openai", "model": "gpt-4o-mini", "base_url": ""},
            {"name": "硅基流动 DeepSeek-V3", "provider": "openai", "model": "deepseek-chat", "base_url": "https://api.siliconflow.cn/v1"},
            {"name": "MiniMax", "provider": "openai", "model": "abab6.5s-chat", "base_url": "https://api.minimax.chat/v1"}
        ],
        "default_api_key_configured": false,
        "chunk_strategy": "hybrid"
    }
}
```

---

## 3. POST /api/query

RAG 智能问答（同步）

**请求：**
```json
{
    "question": "张成都有哪些Python开发经验？",
    "api_key": "sk-xxxx",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "top_k": 5,
    "history": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "sources": [{"file_name": "x.pdf", "score": 0.9}]}
    ],
    "session_id": "s_abc",
    "retrieve_full_doc": false
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| question | string | 是 | 用户问题，1-2000字 |
| api_key | string | 是 | LLM API Key，后端不存储 |
| provider | string | 否 | LLM Provider，默认 openai |
| model | string | 否 | 模型名，默认 gpt-4o-mini |
| base_url | string | 否 | 自定义 API 地址 |
| top_k | int | 否 | 检索Top-K，默认5，最大20 |
| history | list | 否 | 历史对话消息（含 sources 用于跨轮来源记忆） |
| session_id | string | 否 | 会话ID |
| retrieve_full_doc | bool | 否 | 是否强制召回整篇文档，默认 false |

**响应：**
```json
{
    "code": 0,
    "data": {
        "answer": "张成都熟悉Python开发，曾使用FastAPI构建...",
        "sources": [
            {
                "content": "熟悉Python、Go语言开发...",
                "file_name": "张成都-简历.pdf",
                "score": 0.89,
                "section_title": "项目经历",
                "chunk_index": 3,
                "is_full_doc": false
            }
        ],
        "trace_id": "abc123",
        "timing": {"embedding_ms": 85.3, "search_ms": 23.1, "llm_s": 1.2},
        "trace": {
            "intent": "follow_up",
            "original_question": "他的项目有什么",
            "expanded_question": "张成都的项目有什么",
            "context_question": "基于之前关于「张成都是谁」的讨论，当前追问：张成都的项目有什么",
            "rewrite_method": "intent+entity",
            "candidates_before_boost": 10,
            "candidates_after_boost": 5,
            "source_boosts": {"张成都-简历.pdf": 0.26},
            "full_doc_requested": false,
            "error": null
        },
        "suggestion": "",
        "empty_retrieval": false
    }
}
```

---

## 4. POST /api/query/stream

RAG 流式问答（SSE）

请求参数同 `/api/query`。返回 `text/event-stream`，按上文事件流推送。

---

## 5. POST /api/knowledge/upload

上传文档入库

**请求：** `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | 是 | 文件，支持 PDF/TXT/CSV/MD，最大10MB |
| chunk_size | int | 否 | 分块大小，默认400 |
| overlap | int | 否 | 块间重叠，默认50 |

**响应：**
```json
{
    "code": 0,
    "data": {
        "file_name": "简历.pdf",
        "file_type": "pdf",
        "chunks_created": 45,
        "total_chunks": 45,
        "replaced": false,
        "replaced_chunks": 0
    }
}
```

---

## 6. GET /api/knowledge/documents

列出知识库所有文档

**响应：**
```json
{
    "code": 0,
    "data": {
        "documents": [
            {"file_name": "简历.pdf", "file_type": "pdf", "chunk_count": 45, "upload_time": "2026-05-06T10:00:00"}
        ]
    }
}
```

---

## 7. DELETE /api/knowledge/documents/{file_name}

删除文档（含所有关联 chunk）

**路径参数：**
- `file_name`：文件名（需 URL 编码）

**响应：**
```json
{
    "code": 0,
    "data": {"deleted_chunks": 45, "remaining_chunks": 0}
}
```

---

## 8. GET /api/stats

知识库统计信息

**响应：**
```json
{
    "code": 0,
    "data": {
        "total_chunks": 216,
        "total_documents": 5,
        "documents": [...]
    }
}
```

---

## 配置相关说明

### 意图识别（v1.1 新增）

后端自动判断用户问题意图，分类为：

| 意图 | 说明 | 处理策略 |
|------|------|---------|
| `new_topic` | 与上文无关 | 直接用当前问题检索 |
| `follow_up` | 追问/指代上文 | 实体继承 + 带上下文检索 |
| `summarize` | 总结/对比/概括 | 扩大 top_k 召回更多 |
| `clarify` | 要求解释/举例 | 当前检索结果 + 历史上下文 |

### 来源加权（v1.1 新增）

历史 assistant 消息中引用过的 `file_name`，在后续检索中加权：
- 权重 = `source_boost * (1 + log(引用次数)) * exp(-轮数 / source_boost_decay)`
- 默认 `source_boost = 0.15`，`source_boost_decay = 5`

### 分块策略（v1.1 新增）

`chunk.strategy` 配置项：
- `fixed`：滑动窗口（默认）
- `semantic`：按标题语义分块（需 Markdown 结构）
- `hybrid`：标题边界 + 滑动窗口兜底

### 全文文档召回

- 用户手动：发送 `retrieve_full_doc: true`
- LLM 自动：LLM 在答案中输出 `{{retrieve_full_doc:文件名}}` 标记，后端自动召回

### 团队部署 - 默认 API Key

通过环境变量 `DEFAULT_API_KEY` 或配置 `app.default_api_key` 注入团队 Key，前端用户无需填写：
```bash
DEFAULT_API_KEY=sk-team-key python main.py
```
