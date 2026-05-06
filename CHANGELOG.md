# CHANGELOG

## [1.0.0] - 2026-05-06

### Added
- 完整 RAG 智能问答系统：本地 BGE embedding + ES 混合检索 + 在线 LLM
- FastAPI 后端服务（端口 8080），遵循 DDD 分层架构
- Vue3 前端界面（端口 5000），支持智能问答、知识库管理、系统统计
- POST /api/query - RAG 问答接口
- POST /api/knowledge/upload - 文档上传入库
- GET /api/knowledge/documents - 文档列表查询
- DELETE /api/knowledge/documents/{file_name} - 文档删除
- GET /api/stats - 知识库统计
- GET /health - 健康检查
- Embedding 服务：BGE-small-zh-v1.5，本地 CPU 推理，1024维
- ES 混合检索：向量 cosineSimilarity + BM25，RRF 融合
- 文本分块：滑动窗口 400字/50字重叠，CSV 按行分块
- 文档解析：支持 PDF (PyMuPDF)、TXT、CSV
- 多 Provider LLM 支持：OpenAI + Anthropic
- 可选 Rerank 重排：cross-encoder 精排
- 并发限流：令牌桶控制 embedding 推理并发
- 异常处理：BM25降级、空检索兜底、LLM失败友好提示
- 蓝绿部署：全量索引重建零停机
- Docker 镜像构建，多阶段构建
- 单元测试覆盖核心模块
- 统一响应格式：{code, message, data, requestId, timestamp}
- 结构化日志：traceId 贯穿全链路
- API 文档：docs/api.md
