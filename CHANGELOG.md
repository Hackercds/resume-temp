# CHANGELOG

## [1.4.0] - 2026-07-02

### 实体文档扩展 + 前端流式渲染优化

#### 问题诊断
- 库里有两篇张成都的论文，问「张成都是谁」只提到一篇：向量检索偏向语义最相近的一篇，
  第二篇论文没进入候选池；多实体分解要求≥2实体，「张成都是谁」只提取出1个实体不触发
- 前端流式输出每 token 重新解析 Markdown，内容长了卡顿，表现为「首段后整段输出」
- 「记住 Key」复选框在 API Key 列内使该列变高，输入框与预设模型/Provider/Model 不对齐
- 眼睛按钮落在 1.2fr 宽列，占比过大

#### 实体文档扩展（单实体多文档召回）
- 新增 `_extract_entity_terms`：严格提取专有名词（人名/技术名/产品名），过滤普通名词（项目/总结/作者）
- 新增 `_expand_entity_documents`：对核心实体做 BM25，把「包含实体但未进候选池的文档」
  各取一个代表 chunk 注入候选。每文档取分数最高的代表，不挤占 primary 名额
- query/query_stream Step 4.5 接入，单实体也触发（不要求≥2实体）
- `_NON_ENTITY_WORDS` 过滤集：项目/经验/总结/可行性研究/大型语言模型 等泛称不扩展

#### 前端流式渲染优化
- 流式期间用纯文本+闪烁光标显示（`answer-streaming`），完成后切换到 Markdown 渲染
- assistantMsg 增加 `streaming` 标志，done/error 时置 false
- 避免每 token 调用 `renderMarkdown` 重新解析，消除「首段后整段输出」卡顿

#### 前端配置区对齐修复
- 「记住 Key」从 API Key 列内移出，作为独立行跨所有列（`grid-column: 1 / -1`）
- config-card 网格改为 5 列 `1.6fr 1fr 1fr 1fr 40px`，眼睛按钮固定 40px 宽
- 输入框与预设模型/Provider/Model 严格对齐（同一行同高）
- 移动端适配同步更新

### Added
- `rag_service._extract_entity_terms`、`_expand_entity_documents`、`_NON_ENTITY_WORDS`
- query/query_stream Step 4.5 实体文档扩展
- 前端 `.answer-streaming`、`.stream-cursor` 光标动画 CSS
- 5 个专项测试：实体过滤、新文档注入、代表选择、端到端单实体多文档召回

### Tests
- 全量 102 个测试通过（原 97 + 新增 5），无回归

## [1.3.0] - 2026-07-02

### 文档多样性策略升级（回答「同一文档该用多少内容 / 其他文档为何不能一次引用」）

#### 问题诊断
- 旧流程在重排后 `candidates[:top_k]` 一刀切，单文档多个 chunk 会垄断 top_k，挤掉其他文档
- 邻域扩展后追加的邻居会被最终 `[:top_k]` 切掉，**邻居其实没真正进入上下文**（v1.2 潜在 bug）
- 同一文档多次出现时，给多少内容无控制，可能撑爆模型上下文窗口

#### 策略升级
- **文档多样性选择（MMR 简化版）** `_diversify_by_document`：贪心按分数选入，每文档不超过 `max_chunks_per_doc`（默认2），留位置给其他文档；候选不足时回退补齐。O(n) 复杂度，海量数据无需改算法
- **primary / context 分离**：primary 是多样化命中（作为 `sources` 返回），context_candidates = primary + 邻居（用于拼上下文）。修复邻居被切掉的 bug，邻居真正进入上下文，但 sources 干净不混杂 score=0 的邻居
- **上下文内容预算** `context_char_budget`（默认6000字符）：按文档均分额度，单文档超额度截断并提示「内容已按文档预算截断」；完整文档（用户手动召回）不参与截断。回答「该用多少内容」
- 重排保留 `top_k*3` 候选供多样性选择，不一刀切到 top_k

#### 海量文章可扩展性
当前策略 O(n) 与候选规模线性，百倍数据量也无需改算法。后续可叠加：
- 两阶段检索：先按 file_name 聚合召回 Top 文档，再在文档内取 Top chunk（候选数爆炸时启用）
- chunk 级 MMR：仅对同文档内 chunk 做相似度去重（进一步降冗余）
- ES 层：HNSW 已启用（dense_vector index=true）；可调 number_of_shards 横向扩展

### Added
- `config.yaml`：`retrieval.enable_doc_diversity`、`retrieval.max_chunks_per_doc`、`retrieval.context_char_budget`
- `rag_service._diversify_by_document`、`_sort_candidates_by_score`
- `_build_context` 文档预算截断逻辑
- `QueryTrace` 增加 `primary_count`、`primary_docs` 字段
- 12 个专项测试：多样性选择、per-doc cap、内容预算、邻居进上下文回归、端到端多文档

### Fixed
- 邻域扩展后被 `[:top_k]` 切掉、邻居无法进入上下文的潜在 bug（primary/context 分离后修复）
- 单文档垄断 top_k、其他文档无法一次引用的问题

### Tests
- 全量 97 个测试通过（原 85 + 新增 12），无回归

## [1.2.1] - 2026-07-02

### Fixed
- **关键修复：父文档过滤误排除常规 chunk**。v1.2.0 用 `term: is_full_doc=false` 过滤父文档，
  但常规 chunk 入库时未写 `is_full_doc` 字段（值缺失），ES 中 `term:false` 不命中字段缺失文档，
  导致常规 chunk 被误排除、检索结果为空。改为 `must_not: [{term: {is_full_doc: true}}]`，
  只排除父文档，保留所有常规 chunk（含字段缺失的历史数据）。
  - `search_by_vector` / `search_by_keyword` / `search_neighbor_chunks` 三处过滤全部修正
  - `bulk_insert` 入库时显式写 `is_full_doc: false`，保证新数据字段完整
  - 新增回归测试 `test_regular_chunks_without_field_are_not_excluded` 锁定该数据语义

## [1.2.0] - 2026-07-02

### 检索召回质量升级（从技术栈原理底层优化）

#### ES 中文分词
- 索引映射升级为 `ik_max_word`（写入细粒度）+ `ik_smart`（查询智能切分）
- 启动时探测 `analysis-ik` 插件，未安装自动降级 `standard` 并告警，保证「有 ik 用 ik，没 ik 不崩」
- 旧索引动态补字段时同步使用 ik 分析器
- 全文 `full_text` 字段同样启用 ik 分词

#### 切片全量召回策略
- **邻域上下文扩展（neighbor expansion）**：命中某 chunk 后，按同 `file_name` + 相邻 `chunk_index` 召回前后 N 个块（默认 N=1），拼成扩展上下文，解决跨块「如前所述」断章取义。一次 `terms` 查询完成，开销极低
- **全文父文档防污染**：`is_full_doc=true` 的父文档不再参与向量/关键词常规检索（`exclude_full_doc` 过滤），避免首段向量误匹配挤占 top_k；父文档仅通过 `retrieve_full_document` 按 file_name 精确召回
- **CSV 表头继承**：CSV 分块时把第 0 行表头注入每条数据行（`name: 张三 | skill: Python`），每行自包含完整语义；单列文本自动识别不误当表头；正确处理引号内逗号
- context 拼接：邻域块标注「【上下文补充N】」并按文件+chunk_index 排序，便于 LLM 衔接理解

#### 连续对话质量升级
- **追问向量去污染**：`follow_up`/`clarify` 时主向量用 `expanded`（纯实体消解，如「张成都的项目」），而非 `context`（含「基于之前关于…」套话）——后者会把语义空间拉向「讨论/上下文」类词，降低对原文档的召回精度；`context` 仅用于 BM25（关键词检索对套话不敏感，反而能借上下文多命中实体）
- **查询向量 LRU 缓存**：相同问题文本复用向量，避免重试/相似追问/多路检索重复推理（CPU ~50ms → 0ms），可配置容量与开关
- **指代消解增强**：支持多代词/多次出现替换（「他的项目和他的经验」→ 两处「他」都替换），单实体全替换、多实体保守只替换句首；保护「其他」不被误替换
- **历史压缩保留 assistant 答案**：旧版只留 user 问句导致长对话失忆；改为「问→答」配对压缩摘要，保留关键结论
- **LLM 意图识别真实可用**：修复 `_llm_classify` 的 `NotImplementedError` 死代码，实现规则优先 + LLM 四分类兜底（JSON 输出 + 容错解析），api_key 透传

#### 体验优化
- **空检索动态建议**：空检索时返回知识库实际文档名（「当前知识库包含：简历.pdf、项目.md」），让用户「知道能问什么」，替代静态文本
- **Prompt 质量优化**：强化引用规范（标注来源编号）、明确邻域片段衔接说明、防幻觉约束、列举条理化要求

### Added
- `config.yaml` 新增：`chunk.csv_inject_header`、`retrieval.enable_neighbor_expansion`、`retrieval.neighbor_window`、`conversation.follow_up_vector_use_expanded`、`conversation.enable_query_vector_cache`、`conversation.query_vector_cache_size`
- `es_repository`：`_detect_ik_available`、`search_neighbor_chunks`、`search_by_vector(exclude_full_doc)`、`search_by_keyword(exclude_full_doc)`
- `es_service`：`expand_neighbors`
- `embedding_service`：查询向量 LRU 缓存（`encode_query` 缓存 + `clear_query_cache`）
- `rag_service`：`_build_empty_retrieval_suggestion`、追问向量去污染编排、邻域扩展接入
- `intent_service`：真实可用的 `_llm_classify`、`resolve_references` 多代词增强
- `tests/test_retrieval_upgrade.py`：39 个专项单元测试覆盖全部升级点

### Changed
- `search_by_vector`/`search_by_keyword` 默认 `exclude_full_doc=True`
- `create_index` 索引映射使用 ik 分词器（探测后降级）
- `_build_context` 标注邻域块并按文件+chunk_index 排序
- `_build_prompt` 强化引用与防幻觉约束
- `_compress_history` 保留 assistant 答案摘要

### Tests
- 全量 84 个测试通过（原 45 + 新增 39），无回归

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

