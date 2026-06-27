# CHANGELOG

## [1.1.0] - 2026-06-28

### Added
- **对话质量升级**：
  - 新增 `internal/service/intent_service.py`，规则版意图识别（new_topic/follow_up/summarize/clarify）+ 可选 LLM 增强
  - 查询重写 2.0：实体继承 + 双查询并行检索（`expanded` 与 `context`）
  - 历史压缩（最近 2 轮完整 + 更早历史摘要）
  - 来源加权衰减：`source_boost * (1 + log(引用次数)) * exp(-轮数/衰减窗口)`
- **错误体验升级**：
  - 所有错误响应增加 `suggestion` 字段（可操作建议）
  - 空检索时返回 `empty_retrieval: true`
  - 前端显示“上传文档/换个问法/重试”按钮
- **Markdown 上传与渲染**：支持 `.md`/`.markdown` 文件，答案渲染 Markdown + 代码高亮
- **流式问答**：SSE 流式输出 `/api/query/stream`，过滤 LLM thinking/reasoning
- **整篇文档召回**：用户手动 `retrieve_full_doc` 或 LLM 自动输出 `{{retrieve_full_doc:...}}` 标记触发
- **会话管理**：多会话切换、本地持久化、空会话过滤、删除确认
- **标题感知分块**：`chunk.strategy` 支持 `fixed`/`semantic`/`hybrid`
- **来源卡片 2.0**：显示文件名、章节标签、chunk 序号、相似度
- **检索 Trace**：后端返回 `trace` 对象（意图/重写/加权/全文召回），前端可折叠查看
- **全文追加不覆盖**：查看完整文档改为追加新消息
- **首次使用优化**：
  - 空会话 onboarding 引导页 + 示例问题胶囊
  - 回答后追问建议胶囊
  - 模型预设下拉（OpenAI / 硅基流动 DeepSeek / MiniMax）
  - "记住 API Key" 复选框
  - 后端 `default_api_key` 团队部署支持
- **公开配置接口**：`GET /api/config` 返回 LLM 预设、默认 Key 状态、分块策略
- **PWA 支持**：`manifest.json` + `sw.js` 离线缓存静态资源
- **移动端响应式**：底部固定 Tab 栏、输入框全宽、消息气泡加宽
- **一键启动脚本**：`bin/start.sh`（Linux/Mac）、`bin/start.ps1`（Windows），自动检测依赖与 ES

### Changed
- `QueryResult` 增加 `trace/suggestion/empty_retrieval/fallback_context` 字段
- `SourceItem` 增加 `section_title/chunk_index/is_full_doc` 字段
- ES mapping 增加 `section_title`（keyword），兼容旧索引自动添加
- `chunk` 配置增加 `strategy` 字段，默认 `hybrid`
- `conversation` 配置增加 `source_boost_decay/enable_intent_llm/intent_rule_confidence_threshold/intent_model`
- `app` 配置增加 `default_api_key`
- `llm` 配置增加 `presets`（3 条默认预设）

### Fixed
- 422 错误：Vue 事件对象被当成参数传入 `doQuery()`
- LLMAPI NameError：`List` 缺失导入
- 长问句被误判为 new_topic 的意图识别问题
- 空会话堆积、不可删除问题
- 查看完整文档覆盖原答案

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
- Embedding 服务：BGE-small-zh-v1.5，本地 CPU 推理，512维
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

