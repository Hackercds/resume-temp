# CHANGELOG

## [1.5.0] - 2026-07-03

### 「查看完整文档」按钮语义修正 + 全文召回降级 + 深度回答合并

#### 问题诊断（用户反馈）
- 「查看完整文档」按钮字面是「跳转查看完整文档」，实际却走 AI 生成：
  ES 中无 `is_full_doc:true` 父文档（存量数据），`_retrieve_full_documents` 返回空，
  `view_only` 分支被跳过，代码跌入常规 RAG 链路 → 调 LLM → 无 Key 时直接报错。
  按钮「字面意思」与「实际行为」不一致，且「出错」。
- 两个「深度回答」按钮（单文档「基于此文档深入回答」+ 多文档「召回所选文档生成深度回答」）
  语义雷同，造成困惑。

#### 修复
- **全文召回双路 + 优雅降级** `es_repository.get_full_document`：
  主路取 `is_full_doc:true` 父文档（含完整 full_text）；无命中则降级路按 `file_name`
  聚合所有常规 chunk（chunk_index 升序）拼回全文。存量老数据零迁移即可用「查看完整文档 / 深度回答」。
- **API Key 按需校验** `dto.QueryRequest`：`api_key` 改为可选，`model_validator` 仅在
  `view_only=False`（深度回答）时强制。「查看完整文档」对无 Key 用户也可用。
- **按钮语义对齐**：每张来源卡只留「📄 查看完整文档」（直接交付原文，不调 LLM、无需 Key）；
  下方一个「🔍 召回所选文档生成深度回答」按钮覆盖 1~N 文档（选1即单文档深入，选N即跨文档综合），
  移除冗余的「基于此文档深入回答」按钮。
- **来源去重（展示层）** `dedupedSources`：同文档多切片合并成1张卡（保留最高分，标注片段数），
  仅在展示层合并，不改后端召回与 primary 多样性——其他文档仍占席位，不会「5切片合并成1」挤掉排名靠后的文档。

#### 价值（面试点）
- 按钮字面意思 = 实际行为：「查看完整文档」纯展示零成本，「深度回答」才推理计成本。
- 降级路让特性对存量数据可用，零迁移成本——工程上常见的「老数据兼容」考量。
- 一个深度回答入口覆盖单/多文档，避免雷同按钮的体验割裂。

### Added
- `es_repository._reconstruct_full_from_chunks`：父文档缺失时按 chunk 聚合重建全文
- `dto.QueryRequest._check_api_key_when_llm`：按 view_only 按需校验 API Key
- 前端 `dedupedSources` 来源去重展示
- 6 个专项测试：view_only 空 Key 合法、深度回答缺 Key 报错、重建降级三场景

## [1.4.1] - 2026-07-02

### 实体文档优先覆盖 + 流式渲染节流

#### 问题诊断
- 「张成都是谁」三篇相关文档（涉警论文、计算机论文、简历）仍有缺失：
  - 注入的实体代表 chunk 用 BM25 裸分（5-7），压垮原有 RRF 分数（0.03）排序，
    让面试题代表挤掉真实论文
  - `_diversify_by_document` 按纯分数贪心，涉警论文5个 chunk 被 cap=2 后仍竞争不过面试题
  - 面试题里 BM25 命中「张成都」（当示例引用），与真实论文难区分
- 前端流式 token 快速到达时 Vue 响应式逐次 diff 长文本，累积卡顿表现为「后面一段突然全出」

#### 修复
- **实体文档精准识别** `_identify_entity_documents`：name_match（文件名含实体）∪ BM25 top-3，
  排除面试题噪声。`_expand_entity_documents` 和 `_mark_entity_matched_chunks` 复用此判定
- **注入代表分数归一化**：实体代表 score 设为 RRF 量级（0.04），靠 entity_match 标记优先，
  不靠 BM25 裸分压垮排序
- **多样性两阶段 + entity_match 优先**：`_diversify_by_document` 第一轮每个不同文档各取1代表，
  entity_match 文档排序靠前优先占席位；第二轮 max_per_doc 内按分数补齐
- **前端流式 rAF 节流**：token 累积到 buffer，requestAnimationFrame 每帧合并刷新一次，
  避免逐 token 响应式 diff 长文本卡顿

### Added
- `rag_service._identify_entity_documents`、`_mark_entity_matched_chunks`
- `_diversify_by_document` 两阶段覆盖优先 + entity_match 排序
- 前端流式 rAF 节流（streamBuf + requestAnimationFrame）
- 2 个专项测试：entity_match 优先覆盖、注入代表分数归一化

### Tests
- 全量 103 个测试通过（原 102 + 新增 1）

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


## [1.5.0] - 2026-07-02

### 召回完整文档卡死 + 流式体验修复

#### 问题诊断
- 「召回完整文档」耗时极长、前端卡死（滚动被反复拉回）：
  - 后端 retrieve_full_doc 把完整文档当 context 调 LLM，几万字 context 处理 10-30s
  - 前端 retrieveFullDoc 对多文件串行 await，逐个等 LLM 完成才开始下一个
  - 每个 token 触发 scrollToBottom 强制滚到底，用户上滑阅读被反复拉回
- 流式「首句正常，后面整段出现」仍存在：
  - done 事件 answer 覆盖 content，全文重生成时 answer 与流式 token 不一致导致跳动

#### 修复
- **完整文档召回短路**：retrieve_full_doc=True 时直接分块流式返回文档原文（_split_for_stream
  按段落/句子边界切分，每块~200字），不调 LLM。零 LLM 延迟，前端即时显示
- **autoScroll 策略**：scrollToBottom 只在用户已接近底部（≤120px）时自动滚；
  forceScrollToBottom 仅用户主动发送/切会话时强制滚。上滑阅读不被打断
- **滚动节流**：流式 token 快速到达时 rAF 合并到一帧一次 scrollToBottom，避免卡顿
- **retrieveFullDoc 并发**：多文件 Promise.all 并发召回，不串行 await 阻塞 UI
- **done 校准优化**：仅当 answer 比已流式 content 更长才覆盖，避免一致时覆盖导致内容跳动

### Added
- `rag_service._split_for_stream` 按句子边界切分长文本
- `query_stream` 完整文档召回短路（直接 yield 原文，不调 LLM）
- 前端 `forceScrollToBottom`、autoScroll 阈值、rAF 滚动节流
- 3 个专项测试：split_for_stream 边界/短文本、完整文档召回短路不调 LLM

### Tests
- 全量 107 个测试通过（原 104 + 新增 3）

## [1.6.0] - 2026-07-02

### 修复「5个答案」+ 流式标记泄露

#### 问题诊断
- 用户问多部分问题，返回 5 个答案拼接：
  - 旧设计让 LLM 输出 {{retrieve_full_doc:文件名}} 标记，后端召回完整文档后重新生成整个答案
  - 多文档/多部分问题让 LLM 多次输出标记，级联重新生成，产生多个答案拼接
  - 每次重新生成耗时 10s+，5 个答案 = ~50s，且混乱
- 流式 token 中标记被拆开（如 `{{retrieve` + `_full_doc:x.pdf}}`）：
  - 逐 token 用完整正则无法匹配，标记片段泄露给用户

#### 修复
- **移除自动全文召回标记特性**：prompt 不再指示 LLM 输出 {{retrieve_full_doc}} 标记；
  query/query_stream 移除自动检测标记重新生成的代码块。初始检索（向量+BM25+邻域扩展+
  实体文档扩展）已提供足够上下文；用户需要全文可手动点「召回完整文档」按钮
  （retrieve_full_doc=true 走短路直接返回文档原文，不调 LLM）
- **流式标记过滤状态机** `_filter_stream_marks`：缓存跨 token 拆开的未闭合 {{，
  拼齐后判断是否合法标记，合法则丢弃，普通 {{ 文本则释放。流结束 flush 时
  未闭合的标记前缀丢弃，不泄露给用户
- `_strip_retrieve_marks` 保留作 done 时的兜底双重保险

### Added
- `rag_service._filter_stream_marks`、`_flush_stream_mark_buf`
- 7 个专项测试：完整标记、跨2/3 token 拆开、普通花括号、flush 丢弃标记前缀

### Changed
- `llm_service._build_prompt` 移除 {{retrieve_full_doc}} 标记指令
- query/query_stream 移除自动重新生成代码块
- 流式 token 用 `_filter_stream_marks` 替代 `_strip_retrieve_marks`

### Tests
- 全量 114 个测试通过（原 107 + 新增 7）

### Fixed
- 多部分问题返回多个答案拼接（5个答案）的级联重新生成
- 流式标记被拆开后片段泄露给用户（如 `{{retrieve_full_doc`）

## [1.7.0] - 2026-07-02

### 单文档按需操作：查看全文 vs 深入回答

#### 问题诊断
- 旧「召回完整文档」按钮一次性召回所有来源文档，每个文档生成一条答案，
  多文档问题出现多个答案拼接；且无法按需查看单个文档
- 缺乏「只看原文不调LLM」的轻量入口（核对细节/找原文出处不需推理，应省成本）

#### 设计（经得住面试官追问的价值论证）
RAG 本质是检索+生成，但「完整文档」场景有两种截然不同的用户意图：
- 查看原文：用户要核对细节/找原文出处 → 不需要 LLM 推理，直接流式返回原文。
  价值：零 API 成本、即时显示、不丢细节、不幻觉
- 深入回答：当前 chunk 答得太浅，要基于完整文档的详细回答 → 调 LLM，但只针对
  单文档（多文档拼 context 会超长、稀释焦点、丢失细节，单文档聚焦质量更高）

#### 实现
- 后端 query/query_stream 新增 `full_doc_files`（指定召回文件）与 `view_only` 参数：
  - view_only=True：流式返回文档原文，不调 LLM（按需加载源文件）
  - view_only=False：单文档全文作 context 调 LLM 生成 1 条详细答案
- 前端移除底部「召回完整文档」按钮，改为每个来源卡片两个按钮：
  - 📄 查看全文（view_only=true）
  - 🔍 基于此文档深入回答（view_only=false，单文档聚焦）
- DTO/api_handler 透传 full_doc_files、view_only

### Added
- DTO: QueryRequest 新增 full_doc_files、view_only
- rag_service: query/query_stream 支持 full_doc_files 指定召回文件 + view_only 模式
- 前端 viewFullSource / deepAnswer / _priorQueryText 方法
- 2 个专项测试：view_only 不调 LLM、deep_answer 单文档调 LLM

### Changed
- 移除前端底部「召回完整文档」按钮（一次性所有文档）
- 来源卡片新增「查看全文」「深入回答」两个按钮

### Tests
- 全量 117 个测试通过（原 115 + 新增 2）
