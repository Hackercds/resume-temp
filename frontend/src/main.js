const { createApp, ref, reactive, computed, watch, onMounted, nextTick } = Vue;
import { ApiClient } from './api/client.js';

const app = createApp({
    setup() {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('sw.js').catch(() => {});
        }

        const activeTab = ref('chat');

        const apiConfig = reactive({
            apiKey: localStorage.getItem('rag_api_key') || '',
            provider: localStorage.getItem('rag_provider') || 'openai',
            model: localStorage.getItem('rag_model') || 'gpt-4o-mini',
            baseUrl: localStorage.getItem('rag_base_url') || ''
        });
        const rememberKey = ref(localStorage.getItem('rag_remember_key') !== 'false');
        // 召回深度：与 config-card 子组件通过 retrieval-depth-changed 事件保持同步
        const retrievalDepth = ref(Number(localStorage.getItem('rag_retrieval_depth') || 10));
        const llmPresets = ref([]);
        const defaultApiKeyConfigured = ref(false);

        watch(() => apiConfig.apiKey, v => { if (rememberKey.value) localStorage.setItem('rag_api_key', v); });
        watch(() => apiConfig.provider, v => localStorage.setItem('rag_provider', v));
        watch(() => apiConfig.model, v => localStorage.setItem('rag_model', v));
        watch(() => apiConfig.baseUrl, v => localStorage.setItem('rag_base_url', v));
        watch(rememberKey, v => {
            localStorage.setItem('rag_remember_key', v ? 'true' : 'false');
            if (v) localStorage.setItem('rag_api_key', apiConfig.apiKey);
            else localStorage.removeItem('rag_api_key');
        });

        const modelHints = {
            openai: ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1', 'gpt-3.5-turbo'],
            anthropic: ['claude-sonnet-4-6', 'claude-3-haiku-20240307', 'claude-3-opus-20240229'],
            custom: ['deepseek-chat', 'deepseek-reasoner', 'doubao-pro-32k']
        };
        const modelDatalist = computed(() => [...new Set([...(modelHints[apiConfig.provider] || []), apiConfig.model])]);

        const health = reactive({ embedding_loaded: false, es_connected: false, checking: true });
        let healthTimer = null;
        const isMobile = ref(window.innerWidth <= 640);
        window.addEventListener('resize', () => { isMobile.value = window.innerWidth <= 640; });

        async function checkHealth() {
            try {
                const h = await ApiClient.healthCheck();
                health.embedding_loaded = h.embedding_loaded;
                health.es_connected = h.es_connected;
                health.default_api_key_configured = h.default_api_key_configured;
            } catch (e) { /* 忽略 */ }
            health.checking = false;
        }

        async function loadPublicConfig() {
            try {
                const cfg = await ApiClient.getPublicConfig();
                llmPresets.value = cfg.llm_presets || [];
                defaultApiKeyConfigured.value = cfg.default_api_key_configured;
                if (defaultApiKeyConfigured.value && !apiConfig.apiKey) {
                    apiConfig.apiKey = 'DEFAULT_API_KEY';
                }
            } catch (e) { /* 忽略 */ }
        }

        onMounted(() => {
            checkHealth();
            loadPublicConfig();
            healthTimer = setInterval(checkHealth, 15000);
        });

        const notification = ref(null);
        const showNotification = (msg, type = 'info') => {
            notification.value = { msg, type };
            setTimeout(() => notification.value = null, 4000);
        };

        return {
            activeTab, apiConfig, rememberKey, modelDatalist, llmPresets,
            defaultApiKeyConfigured, health, isMobile, notification, showNotification
        };
    },

    template: `
    <div>
        <!-- Header -->
        <header class="masthead">
            <div class="masthead-title">
                <span class="masthead-title-icon">✦</span>
                简历 RAG 智能问答
            </div>
            <div class="masthead-meta">
                <div class="status-pill" :class="health.es_connected ? 'ok' : 'bad'">
                    ES · {{ health.es_connected ? '在线' : '离线' }}
                </div>
                <div class="status-pill" :class="health.embedding_loaded ? 'ok' : 'warn'">
                    Embedding · {{ health.embedding_loaded ? '就绪' : '加载中' }}
                </div>
            </div>
        </header>

        <div v-if="notification" class="toast" :class="notification.type">
            {{ notification.msg }}
        </div>

        <api-key-config :api-config="apiConfig" :remember-key="rememberKey"
            :model-datalist="modelDatalist" :llm-presets="llmPresets"
            :default-api-key-configured="defaultApiKeyConfigured"
            @update:remember-key="rememberKey = $event" />

        <nav class="tabs">
            <button class="tab-btn" :class="{ active: activeTab === 'chat' }" @click="activeTab = 'chat'">
                💬 问答
            </button>
            <button class="tab-btn" :class="{ active: activeTab === 'knowledge' }" @click="activeTab = 'knowledge'">
                📚 知识库
            </button>
            <button class="tab-btn" :class="{ active: activeTab === 'stats' }" @click="activeTab = 'stats'">
                📊 统计
            </button>
        </nav>

        <div v-show="activeTab === 'chat'">
            <chat-panel :api-config="apiConfig" :health="health" :retrieval-depth="retrievalDepth"
                @notify="showNotification" @switch-tab="activeTab = $event"
                @retrieval-depth-changed="retrievalDepth = $event" />
        </div>
        <div v-show="activeTab === 'knowledge'">
            <knowledge-panel :health="health" @notify="showNotification" />
        </div>
        <div v-show="activeTab === 'stats'">
            <stats-panel :health="health" />
        </div>
    </div>
    `,
});

// ==================== API Key 配置 ====================
app.component('api-key-config', {
    props: ['apiConfig', 'rememberKey', 'modelDatalist', 'llmPresets', 'defaultApiKeyConfigured'],
    emits: ['update:rememberKey'],
    template: `
    <div class="config-card">
        <div>
            <label>API Key</label>
            <input :type="showKey ? 'text' : 'password'"
                :value="apiConfig.apiKey" @input="update('apiKey', $event.target.value)"
                :placeholder="defaultApiKeyConfigured ? '已由后端配置默认 Key' : 'sk-...'"
                :disabled="defaultApiKeyConfigured && apiConfig.apiKey === 'DEFAULT_API_KEY'" />
        </div>
        <div>
            <label>预设模型</label>
            <select @change="applyPreset($event.target.value)">
                <option value="">自定义</option>
                <option v-for="p in llmPresets" :key="p.name" :value="p.name">{{ p.name }}</option>
            </select>
        </div>
        <div>
            <label>Provider</label>
            <select :value="apiConfig.provider" @change="update('provider', $event.target.value)">
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="custom">Custom</option>
            </select>
        </div>
        <div>
            <label>Model</label>
            <input :value="apiConfig.model" @input="update('model', $event.target.value)"
                :list="'ml-' + apiConfig.provider" placeholder="gpt-4o-mini" />
            <datalist :id="'ml-' + apiConfig.provider">
                <option v-for="m in modelDatalist" :value="m" />
            </datalist>
        </div>
        <!-- 召回深度档：用户控制 RAG top_k（5 轻量 / 10 标准 / 15 深度 / 20 极深）
             值越大多样性覆盖越广、上下文越全，但 LLM token 也越多 -->
        <div class="depth-row">
            <label>召回深度</label>
            <select :value="retrievalDepth" @change="retrievalDepth = Number($event.target.value); saveRetrievalDepth()">
                <option :value="5">轻量 (5 条 · 快速)</option>
                <option :value="10">标准 (10 条 · 均衡)</option>
                <option :value="15">深度 (15 条 · 详细)</option>
                <option :value="20">极深 (20 条 · 全面)</option>
            </select>
        </div>
        <!-- 自定义 Base URL：custom provider 或选了带 base_url 的预设时露出。
             没填时后端走 OpenAI/Anthropic 官方；填了就走用户指定（OpenAI 兼容协议）。 -->
        <div v-if="showBaseUrl" class="baseurl-row">
            <label>
                Base URL（自定义 API 地址）
                <span class="baseurl-hint">{{ apiConfig.baseUrl ? '已启用' : '未填则走默认' }}</span>
            </label>
            <input :value="apiConfig.baseUrl"
                @input="update('baseUrl', $event.target.value)"
                placeholder="https://api.deepseek.com/v1"
                spellcheck="false" autocomplete="off" />
        </div>
        <button class="eye-btn" @click="showKey = !showKey" :title="showKey ? '隐藏' : '显示'">
            {{ showKey ? '🙈' : '👁' }}
        </button>
        <label class="remember-row">
            <input type="checkbox" :checked="rememberKey"
                @change="$emit('update:rememberKey', $event.target.checked)" />
            记住 Key
        </label>
    </div>
    `,
    data() { return { showKey: false, retrievalDepth: Number(localStorage.getItem('rag_retrieval_depth') || 10) }; },
    computed: {
        // Base URL 露出条件：custom provider，或当前选了带 base_url 的预设
        showBaseUrl() {
            if (this.apiConfig.provider === 'custom') return true;
            const cur = (this.llmPresets || []).find(p =>
                p.name && p.name !== '自定义' && p.provider === this.apiConfig.provider
                && (!this.apiConfig.model || p.model === this.apiConfig.model)
            );
            return !!(cur && cur.base_url);
        }
    },
    methods: {
        update(key, val) { this.apiConfig[key] = val; },
        saveRetrievalDepth() {
            localStorage.setItem('rag_retrieval_depth', String(this.retrievalDepth));
            this.$emit('retrieval-depth-changed', this.retrievalDepth);
        },
        applyPreset(name) {
            const p = (this.llmPresets || []).find(x => x.name === name);
            if (!p) return;
            this.apiConfig.provider = p.provider || 'openai';
            this.apiConfig.model = p.model || '';
            // 预设带 base_url 就回填（用户可继续手改），不带就清空
            this.apiConfig.baseUrl = p.base_url || '';
        }
    }
});

// ==================== 问答面板 ====================
app.component('chat-panel', {
    props: ['apiConfig', 'health', 'retrievalDepth'],
    emits: ['notify', 'switch-tab'],
    template: `
    <div>
        <!-- 会话工具栏 -->
        <div class="chat-bar">
            <span class="chat-bar-label">💬 对话</span>
            <button class="btn-new-chat" @click="newSession">＋ 新建</button>
            <div class="btn-search-bar">
                🔍 <input v-model="sessionSearch" placeholder="搜索对话…" />
            </div>
            <div v-for="s in filteredSessions" :key="s.id"
                 :class="['session-chip', { active: s.id === currentSessionId }]"
                 @click="loadSession(s.id)">
                <span class="session-title">{{ s.title }}</span>
                <span class="session-x" @click.stop="deleteSession(s.id)">✕</span>
            </div>
        </div>

        <!-- 消息列表 / 空状态 -->
        <div v-if="messages.length || loading" class="chat-card">
            <div v-if="messages.length" class="message-list" ref="messageList">
                <div v-for="(msg, mIdx) in messages" :key="msg.id"
                     :class="['message', msg.role]">
                    <div class="bubble">
                        <div v-if="msg.role === 'user'">{{ msg.content }}</div>
                        <div v-else>
                            <!-- 流式期间用纯文本+光标，避免每 token 重新解析 Markdown 导致卡顿 -->
                            <div v-if="msg.streaming" class="answer-box answer-streaming">{{ msg.content }}<span class="stream-cursor">▋</span></div>
                            <div v-else class="answer-box" v-html="renderMarkdown(msg.content)"></div>
                            <div v-if="msg.timing" class="bubble-meta">
                                <span><b>{{ msg.timing.embedding_ms }}ms</b> embed</span>
                                <span><b>{{ msg.timing.search_ms }}ms</b> search</span>
                                <span><b>{{ msg.timing.llm_s }}s</b> llm</span>
                                <span v-if="msg.usage"><b>{{ msg.usage.total_tokens.toLocaleString() }}</b> tokens
                                    <span class="meta-sub">(prompt {{ msg.usage.prompt_tokens.toLocaleString() }} · comp {{ msg.usage.completion_tokens.toLocaleString() }})</span>
                                </span>
                            </div>
                            <div v-if="msg.sources && msg.sources.length" class="source-list">
                                <div class="source-list-title">引用来源 ({{ dedupedSources(msg).length }} 个文档，{{ msg.sources.length }} 个片段)</div>
                                <div v-for="(s, i) in dedupedSources(msg)" :key="s.file_name + i"
                                    class="source-item" :class="{ open: expandedMsgIdx === mIdx && expandedIdx === i }"
                                    @click="toggleSource(mIdx, i)">
                                    <div class="source-header">
                                        <div class="source-meta">
                                            <label class="src-check" @click.stop>
                                                <input type="checkbox" v-model="msg.selectedDocs" :value="s.file_name" />
                                            </label>
                                            <span class="source-file">{{ s.file_name }}</span>
                                            <span v-if="s.chunk_count > 1" class="source-section">· {{ s.chunk_count }} 片段 ·</span>
                                            <span class="source-index">#{{ s.chunk_index }}</span>
                                        </div>
                                        <span class="source-score">{{ s.score }}</span>
                                    </div>
                                    <div class="source-content" v-html="renderMarkdown(s.content)"></div>
                                    <!-- 查看完整文档：直接交付原文，不调 LLM（无需 API Key） -->
                                    <div v-if="!msg.isFullDoc" class="source-actions" @click.stop>
                                        <button class="btn-source-act" @click="viewFullSource(msg, s.file_name)">📄 查看完整文档</button>
                                    </div>
                                </div>
                                <!-- 深度回答：勾选 1~N 个文档，合并完整内容调 LLM 生成 1 条综合答案 -->
                                <button v-if="!msg.isFullDoc && dedupedSources(msg).length >= 1"
                                    class="btn-multi-deep" :disabled="!(msg.selectedDocs && msg.selectedDocs.length)"
                                    @click="deepAnswerMulti(msg)">
                                    🔍 召回所选文档生成深度回答{{ msg.selectedDocs && msg.selectedDocs.length ? '（' + msg.selectedDocs.length + ' 个）' : '（未选择）' }}
                                </button>
                            </div>
                            <div v-if="msg.ungrounded" class="grounded-warn" title="答案中包含具体数字/日期但未引用任何来源（heuristic），可能为 LLM 编造">
                                ⚠ 本回答包含具体数字/日期但未引用任何来源，请核实。
                            </div>
                            <div v-if="msg.trace && msg.trace.history_quoted_files && msg.trace.history_quoted_files.length"
                                class="history-quoted" title="本次检索参考了你之前对话里引用过的文档，帮助保持多轮连贯性">
                                📎 本次召回参考了历史引用：{{ msg.trace.history_quoted_files.join('、') }}
                            </div>
                            <button v-if="msg.trace" class="trace-toggle" @click="msg.showTrace = !msg.showTrace">
                                {{ msg.showTrace ? '— 收起检索过程' : '+ 查看检索过程' }}
                            </button>
                            <pre v-if="msg.trace && msg.showTrace" class="trace-content">{{ JSON.stringify(msg.trace, null, 2) }}</pre>
                            <div v-if="msg.fullDocs && msg.fullDocs.length" class="full-doc-card">
                                <div v-for="(fd, fi) in msg.fullDocs" :key="fi">
                                    <div class="full-doc-header">
                                        <span>{{ fd.file_name }}</span>
                                        <button class="btn-full-doc-close" @click="msg.fullDocs.splice(fi, 1)">收起</button>
                                    </div>
                                    <div class="answer-box" v-html="renderMarkdown(fd.content)"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div v-if="loading" class="loading">
                    <div class="spinner"></div>
                    检索中 · {{ statusText }}
                </div>
            </div>

            <div v-if="error" class="error-msg">
                <div>{{ error }}</div>
                <div v-if="errorSuggestion" class="error-suggestion">{{ errorSuggestion }}</div>
                <div v-if="emptyRetrieval" class="error-actions">
                    <button class="btn-empty" @click="$emit('switch-tab', 'knowledge')">上传文档</button>
                    <button class="btn-empty" @click="retryQuestion">换个问法</button>
                </div>
                <div v-else-if="errorRetryable" class="error-actions">
                    <button class="btn-empty" @click="retryLastQuestion">重新发送</button>
                </div>
            </div>
        </div>

        <!-- 空状态 / Onboarding -->
        <div v-else class="chat-card empty-state">
            <div v-if="!apiConfig.apiKey">
                <div class="onboarding-title">先填 API Key</div>
                <div class="onboarding-sub">在上方配置区填入 OpenAI / Anthropic / 自定义 Provider 的 API Key。</div>
            </div>
            <div v-else-if="documents.length === 0">
                <div class="onboarding-title">欢迎使用</div>
                <div class="onboarding-sub">基于本地 Embedding + ES 混合检索 + 在线 LLM 的简历/文档问答系统。</div>
                <div class="onboarding-steps">
                    <div class="onboarding-step">
                        <span class="onboarding-step-num">01</span>
                        <div><strong>上传文档</strong><br /><span>支持 PDF · TXT · CSV · Markdown</span></div>
                    </div>
                    <div class="onboarding-step">
                        <span class="onboarding-step-num">02</span>
                        <div><strong>输入问题</strong><br /><span>基于知识库内容获得带引用来源的回答</span></div>
                    </div>
                </div>
                <button class="btn-onboarding" @click="$emit('switch-tab', 'knowledge')">上传第一份文档</button>
            </div>
            <div v-else>
                <div class="onboarding-title">准备好了</div>
                <div class="onboarding-sub">知识库中已有 {{ documents.length }} 份文档，开始提问吧。</div>
            </div>

            <div v-if="suggestedQuestions.length" class="suggested-questions">
                <div class="suggested-questions-label">试试这样问</div>
                <div class="question-chips">
                    <button v-for="sq in suggestedQuestions" :key="sq"
                        class="question-chip" @click="question = sq; doQuery()">{{ sq }}</button>
                </div>
            </div>
        </div>

        <!-- 追问建议 -->
        <div v-if="followUpQuestions.length && !loading && messages.length" class="follow-up-bar">
            <span class="follow-up-label">继续问</span>
            <button v-for="fq in followUpQuestions" :key="fq"
                class="question-chip" @click="question = fq; doQuery()">{{ fq }}</button>
        </div>

        <!-- 输入区 -->
        <div class="input-area">
            <!-- 历史记忆管理（A+B：全局 toggle + 逐条勾选排除） -->
            <div class="history-bar">
                <button class="history-toggle-chip"
                    :class="{ disabled: !useHistory }"
                    @click="useHistory = !useHistory"
                    :title="useHistory ? '点击关闭历史上下文（下次发送将不带 history）' : '点击开启历史上下文'">
                    <span class="dot"></span>
                    📌 使用历史上下文
                    <span v-if="useHistory" class="history-count">{{ historyPreview.length }} 条</span>
                    <span v-else>（已关闭）</span>
                </button>
                <button v-if="useHistory && historyPreview.length" class="history-toggle-chip"
                    @click="historyPanelOpen = !historyPanelOpen"
                    title="逐条勾选/排除历史">
                    {{ historyPanelOpen ? '收起' : '管理' }}
                </button>
            </div>
            <div v-if="useHistory && historyPanelOpen" class="history-panel">
                <div class="history-panel-header">
                    <span>本次将发送给模型的对话历史（取消勾选=本轮不发）</span>
                    <span class="ops">
                        <button @click="historyExcludeAll">反选</button>
                        <button @click="historyIncludeAll">全选</button>
                    </span>
                </div>
                <div v-if="!historyPreview.length" class="history-panel-empty">
                    当前会话暂无历史消息
                </div>
                <div v-for="(m, hi) in historyPreview" :key="m.id || hi"
                    class="history-item" :class="{ excluded: m.excluded }">
                    <input type="checkbox"
                        :checked="!m.excluded"
                        @change="toggleHistoryItem(hi, $event.target.checked)" />
                    <div class="history-item-body">
                        <div class="history-item-role">{{ m.role === 'user' ? '用户' : '助手' }}</div>
                        <div class="history-item-content" :class="{ assistant: m.role === 'assistant' }">
                            {{ m.preview }}
                        </div>
                    </div>
                </div>
            </div>

            <textarea v-model="question" placeholder="输入问题，Ctrl + Enter 发送"
                @keydown.ctrl.enter="doQuery" @keydown.meta.enter="doQuery"
                :disabled="loading" rows="2"></textarea>
            <div class="input-area-actions">
                <span class="input-area-hint">基于知识库内容回答</span>
                <button class="btn-send" @click="doQuery"
                    :disabled="loading || !question.trim() || !apiConfig.apiKey">
                    {{ loading ? '发送中' : '发送' }}
                </button>
            </div>
        </div>
    </div>
    `,
    data() {
        return {
            question: '',
            messages: [],
            sessions: this._loadSessions(),
            currentSessionId: this._newSessionId(),
            loading: false,
            statusText: '等待响应',
            error: '',
            errorSuggestion: '',
            errorRetryable: false,
            emptyRetrieval: false,
            expandedIdx: -1,
            expandedMsgIdx: -1,
            maxHistory: 5,
            // 历史记忆（A+B 设计）
            useHistory: true,                 // 全局开关：是否带上历史
            historyPanelOpen: false,          // 是否展开逐条管理面板
            historyExclude: {},               // { msgId: true } 本轮排除的消息 id
            lastQuestion: '',
            followUpQuestions: [],
            documents: [],
            suggestedQuestions: [],
            sessionSearch: ''
        };
    },
    computed: {
        filteredSessions() {
            const q = this.sessionSearch.trim().toLowerCase();
            if (!q) return this.sessions;
            return this.sessions.filter(s =>
                (s.title || '').toLowerCase().includes(q) ||
                (s.messages || []).some(m => (m.content || '').toLowerCase().includes(q))
            );
        },
        /**
         * 历史预览：面板里展示给用户看的列表
         * 面试点：面板直接复用 messages，不存第二份真相；
         * excluded 状态由 historyExclude 这个 id→bool 映射单独存，刷新会话自动重置。
         */
        historyPreview() {
            const list = [];
            for (let i = 0; i < this.messages.length; i++) {
                const m = this.messages[i];
                if (m.role !== 'user' && m.role !== 'assistant') continue;
                list.push({
                    id: m.id || i,
                    role: m.role,
                    preview: (m.content || '').slice(0, 80),
                    excluded: !!this.historyExclude[m.id || i]
                });
            }
            return list;
        }
    },
    mounted() {
        if (this.messages.length > 0) {
            this.saveSession(true);
        } else {
            this.sessions = this._loadSessions();
        }
        this.loadDocuments();
        this.generateSuggestedQuestions();
    },
    methods: {
        renderMarkdown(text) {
            if (!text) return '';
            const raw = marked.parse(String(text), { gfm: true, breaks: true, headerIds: false });
            const clean = DOMPurify.sanitize(raw, { USE_PROFILES: { html: true } });
            nextTick(() => {
                this.$el.querySelectorAll('.answer-box pre code, .source-content pre code')
                    .forEach(block => hljs.highlightElement(block));
            });
            return clean;
        },
        toggleSource(mIdx, i) {
            if (this.expandedMsgIdx === mIdx && this.expandedIdx === i) {
                this.expandedMsgIdx = -1;
                this.expandedIdx = -1;
            } else {
                this.expandedMsgIdx = mIdx;
                this.expandedIdx = i;
            }
        },
        async doQuery(retrieveFullDoc = false) {
            if (typeof retrieveFullDoc !== 'boolean') retrieveFullDoc = false;
            if (!this.question.trim() || !this.apiConfig.apiKey) return;
            const q = this.question.trim();
            this.lastQuestion = q;
            this.messages.push({ id: Date.now(), role: 'user', content: q, timestamp: Date.now() });
            this.question = '';
            this.loading = true; this.statusText = '向量化中';
            this.error = ''; this.errorSuggestion = ''; this.errorRetryable = false; this.emptyRetrieval = false;
            this.followUpQuestions = [];
            this.forceScrollToBottom();  // 用户发送，强制滚到底

            // 全局 toggle 关 → 直接传空 history；开 → 按 historyExclude 过滤逐条
            const rawHistory = this.messages
                .filter(m => m.role === 'user' || m.role === 'assistant')
                .slice(-(this.maxHistory * 2));
            const history = this.useHistory
                ? rawHistory
                    .filter(m => !this.historyExclude[m.id])
                    .map(m => ({
                        role: m.role,
                        content: m.content,
                        sources: (m.sources || []).map(s => ({ file_name: s.file_name, score: s.score }))
                    }))
                : [];

            const assistantMsg = {
                id: Date.now() + 1,
                role: 'assistant',
                content: '',
                sources: [],
                timing: null,
                trace_id: '',
                trace: null,
                showTrace: false,
                fullDocs: [],
                selectedDocs: [],  // 勾选的多文档（用于多文档深度召回）
                streaming: true,  // 流式期间用纯文本显示，完成后渲染 Markdown
                timestamp: Date.now()
            };
            this.messages.push(assistantMsg);
            // 关键修复：Vue3 push 后数组里存的是 reactive proxy，但 assistantMsg 仍指向原始对象。
            // 直接改原始对象不触发响应式更新（表现为流式 token 累积但不显示，直到 done
            // 触发组件重渲染才一次性全出，含来源信息一起出现）。取回 reactive 引用再修改。
            const assistantMsgRef = this.messages[this.messages.length - 1];

            try {
                const body = {
                    question: q,
                    api_key: this.apiConfig.apiKey,
                    provider: this.apiConfig.provider,
                    model: this.apiConfig.model || null,
                    top_k: this.retrievalDepth || 10,
                    history,
                    session_id: this.currentSessionId,
                    retrieve_full_doc: retrieveFullDoc
                };
                if (this.apiConfig.baseUrl) body.base_url = this.apiConfig.baseUrl;

                // 流式：token 到达即追加显示，done 时校准 answer 并切 Markdown。
                // 不做 rAF 节流/限量释放——LLM 本身按 token 流式返回，前端忠实显示即可。
                // 若某些模型/代理批量返回，那是后端特性，前端伪造流式反而引入 bug
                // （光标残留、内容丢失）。简单 = 可靠。
                // 滚动节流：多个 token 快速到达时合并到一帧一次 scrollToBottom，避免卡顿
                let scrollRaf = null;
                const throttledScroll = () => {
                    if (scrollRaf === null) {
                        scrollRaf = requestAnimationFrame(() => {
                            scrollRaf = null;
                            this.scrollToBottom();
                        });
                    }
                };
                await ApiClient.queryStream(
                    body,
                    (token) => {
                        assistantMsgRef.content += token;
                        this.statusText = '生成中';
                        throttledScroll();
                    },
                    (data) => {
                        // 校准 content：仅当 answer 比已流式内容更长时才覆盖
                        const ans = data.answer || '';
                        if (ans && ans.length > (assistantMsgRef.content || '').length) {
                            assistantMsgRef.content = ans;
                        } else if (!assistantMsgRef.content) {
                            assistantMsgRef.content = ans;
                        }
                        assistantMsgRef.sources = data.sources || [];
                        assistantMsgRef.timing = data.timing || null;
                        assistantMsgRef.trace_id = data.trace_id || '';
                        assistantMsgRef.trace = data.trace || null;
                        assistantMsgRef.ungrounded = !!data.ungrounded;
                        assistantMsgRef.streaming = false;  // 立即切 Markdown 渲染
                        this.followUpQuestions = this._generateFollowUpQuestions(data.sources || []);
                        this.saveSession();
                    },
                    (err) => {
                        this.error = err.message;
                        this.errorSuggestion = err.suggestion || '';
                        this.emptyRetrieval = !!err.emptyRetrieval;
                        this.errorRetryable = !this.emptyRetrieval;
                        assistantMsgRef.streaming = false;
                        const idx = this.messages.findIndex(m => m.id === assistantMsgRef.id);
                        if (idx >= 0) this.messages.splice(idx, 1);
                        this.$emit('notify', err.message, 'error');
                    }
                );
            } catch (e) {
                this.error = e.message;
                this.errorSuggestion = e.suggestion || '';
                this.emptyRetrieval = !!e.emptyRetrieval;
                this.errorRetryable = !this.emptyRetrieval;
                assistantMsgRef.streaming = false;
                const idx = this.messages.findIndex(m => m.id === assistantMsgRef.id);
                if (idx >= 0) this.messages.splice(idx, 1);
                this.$emit('notify', e.message, 'error');
            } finally {
                this.loading = false;
                this.scrollToBottom();
            }
        },
        retryLastQuestion() {
            if (this.lastQuestion) {
                this.question = this.lastQuestion;
                this.doQuery();
            }
        },
        retryQuestion() {
            if (this.lastQuestion) {
                let q = this.lastQuestion.replace(/[吗呢？?]/g, '');
                if (q === this.lastQuestion) q = '请介绍' + this.lastQuestion;
                this.question = q;
                this.doQuery();
            }
        },
        async retrieveFullDoc(lastMsg) {
            // 保留旧入口兼容（若被外部调用），内部转 viewFullSource 全部
            if (!lastMsg || !lastMsg.sources || !lastMsg.sources.length) return;
            const files = [...new Set(lastMsg.sources.map(s => s.file_name))];
            for (const fn of files) await this.viewFullSource(lastMsg, fn);
        },
        /**
         * 查看完整源文件：不调 LLM，流式返回文档原文。
         * 设计价值（面试点）：原文交付不需要 LLM 推理，零 API 成本、即时显示；
         * 用户核对细节/找原文出处时，原文比 LLM 总结更可靠（不丢细节、不幻觉）。
         */
        async viewFullSource(lastMsg, fileName) {
            if (!fileName) return;
            const idx = this.messages.indexOf(lastMsg);
            if (idx < 0) return;
            const queryText = this._priorQueryText(idx);

            const fullDocMsg = {
                id: Date.now() + Math.random(),
                role: 'assistant',
                content: '', sources: [], timing: null, trace_id: '',
                trace: null, showTrace: false, fullDocs: [],
                isFullDoc: true, targetFile: fileName, mode: 'view',
                streaming: true, timestamp: Date.now()
            };
            this.messages.splice(idx + 1, 0, fullDocMsg);
            const ref = this.messages[idx + 1];
            this.loading = true; this.error = '';

            let scrollRaf = null;
            const throttledScroll = () => {
                if (scrollRaf === null) {
                    scrollRaf = requestAnimationFrame(() => { scrollRaf = null; this.scrollToBottom(); });
                }
            };

            const body = {
                question: queryText, api_key: this.apiConfig.apiKey,
                provider: this.apiConfig.provider, model: this.apiConfig.model || null,
                top_k: this.retrievalDepth || 10, history: [], session_id: this.currentSessionId,
                retrieve_full_doc: true, full_doc_files: [fileName], view_only: true
            };
            if (this.apiConfig.baseUrl) body.base_url = this.apiConfig.baseUrl;

            try {
                await ApiClient.queryStream(body,
                    (token) => { ref.content += token; throttledScroll(); },
                    (data) => {
                        const ans = data.answer || '';
                        if (ans && ans.length > (ref.content || '').length) ref.content = ans;
                        else if (!ref.content) ref.content = ans;
                        ref.sources = data.sources || [];
                        ref.timing = data.timing || null;
                        ref.ungrounded = !!data.ungrounded;
                        ref.streaming = false;
                        this.saveSession();
                    },
                    (err) => {
                        ref.content = `⚠ ${err.message}`;
                        ref.streaming = false;
                        this.$emit('notify', err.message, 'error');
                    }
                );
            } catch (e) {
                ref.content = `⚠ ${e.message}`;
                ref.streaming = false;
            }
            this.loading = false;
            this.scrollToBottom();
        },
        _priorQueryText(idx) {
            // 取 idx 之前最近的用户问题作为查询文本
            const priorUserMsgs = this.messages.slice(0, idx).filter(m => m.role === 'user');
            if (priorUserMsgs.length >= 2) {
                const lastTwo = priorUserMsgs.slice(-2);
                return `${lastTwo[0].content}；${lastTwo[1].content}`;
            }
            return priorUserMsgs.length === 1 ? priorUserMsgs[0].content : '';
        },
        /**
         * 来源去重：同文档多切片合并成1个，保留分数最高的切片，标注片段数。
         * 面试点：去重只在展示层，不改变后端召回与 primary 多样性结果——
         * 其他文档仍占席位，不会「5切片合并成1」挤掉排名靠后的文档。
         */
        dedupedSources(msg) {
            const srcs = msg.sources || [];
            const byFile = {};
            const order = [];
            for (const s of srcs) {
                const fn = s.file_name || '';
                if (!byFile[fn]) {
                    byFile[fn] = { ...s, chunk_count: 1 };
                    order.push(fn);
                } else {
                    byFile[fn].chunk_count += 1;
                    // 保留分数最高的切片（score 可能是字符串/数字，统一转数字比较）
                    const cur = Number(byFile[fn].score) || 0;
                    const now = Number(s.score) || 0;
                    if (now > cur) {
                        byFile[fn] = { ...s, chunk_count: byFile[fn].chunk_count };
                    }
                }
            }
            return order.map(fn => byFile[fn]);
        },
        /**
         * 深度回答：勾选的文档（1~N 个）合并为 1 个 context，调 LLM 生成 1 条综合答案。
         * 选 1 个即单文档深入回答，选 N 个即跨文档综合——一个入口覆盖两种需求，
         * 避免单文档/多文档两个雷同按钮造成困惑。区别于「查看完整文档」：那个是原文交付不推理。
         */
        async deepAnswerMulti(lastMsg) {
            const files = (lastMsg.selectedDocs || []).filter(Boolean);
            if (files.length < 1) {
                this.$emit('notify', '请先勾选至少1个文档', 'error');
                return;
            }
            const idx = this.messages.indexOf(lastMsg);
            if (idx < 0) return;
            const queryText = this._priorQueryText(idx);

            const ansMsg = {
                id: Date.now() + Math.random(),
                role: 'assistant',
                content: '', sources: [], timing: null, trace_id: '',
                trace: null, showTrace: false, fullDocs: [],
                isFullDoc: true, targetFile: files.join(', '), mode: 'deep_multi',
                streaming: true, timestamp: Date.now()
            };
            this.messages.splice(idx + 1, 0, ansMsg);
            const ref = this.messages[idx + 1];
            this.loading = true; this.error = '';

            let scrollRaf = null;
            const throttledScroll = () => {
                if (scrollRaf === null) {
                    scrollRaf = requestAnimationFrame(() => { scrollRaf = null; this.scrollToBottom(); });
                }
            };

            const body = {
                question: queryText, api_key: this.apiConfig.apiKey,
                provider: this.apiConfig.provider, model: this.apiConfig.model || null,
                top_k: this.retrievalDepth || 10, history: [], session_id: this.currentSessionId,
                retrieve_full_doc: true, full_doc_files: files, view_only: false
            };
            if (this.apiConfig.baseUrl) body.base_url = this.apiConfig.baseUrl;

            try {
                await ApiClient.queryStream(body,
                    (token) => { ref.content += token; throttledScroll(); },
                    (data) => {
                        const ans = data.answer || '';
                        if (ans && ans.length > (ref.content || '').length) ref.content = ans;
                        else if (!ref.content) ref.content = ans;
                        ref.sources = data.sources || [];
                        ref.timing = data.timing || null;
                        ref.trace_id = data.trace_id || '';
                        ref.trace = data.trace || null;
                        ref.streaming = false;
                        this.saveSession();
                    },
                    (err) => {
                        ref.content = `⚠ ${err.message}`;
                        ref.streaming = false;
                        this.$emit('notify', err.message, 'error');
                    }
                );
            } catch (e) {
                ref.content = `⚠ ${e.message}`;
                ref.streaming = false;
            }
            this.loading = false;
            this.scrollToBottom();
        },
        newSession() {
            if (this.messages.length === 0) return;
            this.currentSessionId = this._newSessionId();
            this.messages = [];
            this.error = '';
            this.question = '';
            this.followUpQuestions = [];
            this.expandedIdx = -1;
            this.expandedMsgIdx = -1;
            this.historyExclude = {};  // 新会话清空历史排除
        },
        /**
         * 历史逐条勾选/排除（本轮请求级别，不影响消息本身）
         * 面试点：这种"本轮作用域"的 UI 状态用 id 映射，不要污染 messages 对象。
         * 换会话时 historyExclude 自动失去意义（id 对不上新 messages），所以新会话清空。
         */
        toggleHistoryItem(idx, checked) {
            const m = this.historyPreview[idx];
            if (!m) return;
            // 反映回原始消息：读 messages 找到对应 id（顺序与 historyPreview 一致）
            let cur = 0;
            for (const mm of this.messages) {
                if (mm.role !== 'user' && mm.role !== 'assistant') continue;
                if (cur === idx) {
                    if (checked) delete this.historyExclude[mm.id];
                    else this.historyExclude[mm.id] = true;
                    return;
                }
                cur++;
            }
        },
        historyExcludeAll() {
            // 反选：所有 excluded=false 的变成 true
            for (const m of this.historyPreview) {
                if (!m.excluded) this.historyExclude[m.id] = true;
            }
        },
        historyIncludeAll() {
            this.historyExclude = {};
        },
        deleteSession(id) {
            if (!confirm('删除该对话？')) return;
            const sessions = this._loadSessions().filter(s => s.id !== id);
            localStorage.setItem('rag_sessions', JSON.stringify(sessions.slice(0, 50)));
            this.sessions = sessions.slice(0, 50);
            if (this.currentSessionId === id) {
                this.messages = [];
                this.error = '';
                this.question = '';
                this.followUpQuestions = [];
                this.expandedIdx = -1;
                this.expandedMsgIdx = -1;
                this.currentSessionId = this._newSessionId();
            }
        },
        loadSession(id) {
            const s = this.sessions.find(x => x.id === id);
            if (s) {
                this.currentSessionId = id;
                this.messages = s.messages || [];
                this.error = '';
                this.expandedIdx = -1;
                this.expandedMsgIdx = -1;
                this.historyExclude = {};  // 切会话清空旧 exclude（id 已对不上）
            }
        },
        saveSession(isNew = false) {
            if (this.messages.length === 0) {
                this.sessions = this._loadSessions();
                return;
            }
            const sessions = this._loadSessions();
            const idx = sessions.findIndex(s => s.id === this.currentSessionId);
            const session = {
                id: this.currentSessionId,
                title: this._genTitle(this.messages),
                messages: this.messages,
                update_time: Date.now()
            };
            if (idx >= 0) sessions[idx] = session;
            else sessions.unshift(session);
            sessions.sort((a, b) => b.update_time - a.update_time);
            localStorage.setItem('rag_sessions', JSON.stringify(sessions.slice(0, 50)));
            this.sessions = sessions.slice(0, 50);
        },
        _loadSessions() {
            try { return JSON.parse(localStorage.getItem('rag_sessions') || '[]'); }
            catch (e) { return []; }
        },
        _newSessionId() {
            return 's_' + Math.random().toString(36).slice(2, 10) + '_' + Date.now();
        },
        _genTitle(messages) {
            const first = messages.find(m => m.role === 'user');
            return first ? first.content.slice(0, 20) + (first.content.length > 20 ? '…' : '') : '新对话';
        },
        scrollToBottom() {
            nextTick(() => {
                const el = this.$refs.messageList;
                if (!el) return;
                // autoScroll：只在用户已接近底部时才自动滚到底
                // （用户主动上滑阅读时，不被流式新内容反复拉回底部）
                const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
                if (dist <= 120) el.scrollTop = el.scrollHeight;
            });
        },
        forceScrollToBottom() {
            // 用户发送消息、切换会话等主动操作时强制滚到底
            nextTick(() => {
                const el = this.$refs.messageList;
                if (el) el.scrollTop = el.scrollHeight;
            });
        },
        async loadDocuments() {
            try {
                const d = await ApiClient.listDocuments();
                this.documents = d.documents || [];
            } catch (e) { /* 忽略 */ }
        },
        generateSuggestedQuestions() {
            const samples = [
                '张成都是谁？',
                '介绍一下他的项目经历',
                '他熟悉哪些技术栈？',
                '总结一下核心能力',
                '文档里有哪些 Python 相关的内容？'
            ];
            if (this.documents.length > 0) {
                this.suggestedQuestions = samples.sort(() => 0.5 - Math.random()).slice(0, 3);
            } else {
                this.suggestedQuestions = ['这个系统怎么用？', '支持哪些文件格式？'];
            }
        },
        _generateFollowUpQuestions(sources) {
            const questions = [];
            const files = [...new Set(sources.map(s => s.file_name))].slice(0, 2);
            for (const fn of files) {
                const base = fn.replace(/\.[^.]+$/, '').replace(/^[\d.]+/, '');
                questions.push(`详细介绍 ${base}`);
                questions.push(`${base} 里有哪些关键技能？`);
            }
            return questions.slice(0, 3);
        }
    }
});

// ==================== 知识库面板 ====================
app.component('knowledge-panel', {
    props: ['health'],
    emits: ['notify'],
    template: `
    <div>
        <div class="knowledge-card">
            <h3 class="section-title">上传文档</h3>
            <div class="upload-zone" :class="{ over: isDragOver }"
                @dragover.prevent="isDragOver = true" @dragleave="isDragOver = false"
                @drop.prevent="handleDrop" @click="$refs.fi.click()">
                <div class="upload-icon">⬆</div>
                <p>点击或拖拽 PDF / TXT / CSV / Markdown</p>
                <p class="upload-limit">最大 10MB · 支持多文件</p>
                <input type="file" ref="fi" accept=".pdf,.txt,.csv,.md,.markdown" multiple @change="handleFileSelect" hidden />
            </div>
            <div v-if="uploading">
                <div class="progress-bar"><div class="progress-fill" :style="{width:uploadProgress+'%'}"></div></div>
                <p class="upload-status">{{ uploadProgress }}% · {{ uploadStatusText }}</p>
            </div>
        </div>

        <div class="knowledge-card">
            <h3 class="section-title">
                文档列表
                <span class="section-title-count">· {{ filteredDocs.length }} / {{ documents.length }}</span>
            </h3>
            <div class="btn-search-bar" style="margin-bottom:14px;">
                🔍 <input v-model="docSearch" placeholder="搜索文档名…" />
            </div>
            <div v-if="loading" class="loading">
                <div class="spinner"></div>
                加载中
            </div>
            <div v-else-if="filteredDocs.length === 0" class="empty-state" style="padding:24px 0;">
                {{ docSearch ? '没有匹配的文档' : '暂无文档，先上传第一份吧' }}
            </div>
            <table v-else class="doc-table">
                <thead><tr><th>文件名</th><th>类型</th><th>分块</th><th>时间</th><th>操作</th></tr></thead>
                <tbody>
                    <tr v-for="d in filteredDocs" :key="d.file_name">
                        <td>{{ d.file_name }}</td>
                        <td><span class="doc-type">{{ d.file_type }}</span></td>
                        <td>{{ d.chunk_count }}</td>
                        <td>{{ fmt(d.upload_time) }}</td>
                        <td><button class="btn-delete" @click="remove(d.file_name)">删除</button></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    `,
    data() {
        return {
            documents: [], loading: false, uploading: false,
            uploadProgress: 0, uploadStatusText: '准备中',
            isDragOver: false, docSearch: ''
        };
    },
    computed: {
        filteredDocs() {
            const q = (this.docSearch || '').trim().toLowerCase();
            if (!q) return this.documents;
            return this.documents.filter(d => (d.file_name || '').toLowerCase().includes(q));
        }
    },
    mounted() { this.load(); },
    methods: {
        async load() {
            this.loading = true;
            try { const d = await ApiClient.listDocuments(); this.documents = d.documents || []; }
            catch (e) { /* 索引为空时忽略 */ }
            this.loading = false;
        },
        handleFileSelect(e) {
            const files = Array.from(e.target.files || []);
            if (files.length) this.uploadFiles(files);
            e.target.value = '';
        },
        handleDrop(e) {
            this.isDragOver = false;
            const files = Array.from(e.dataTransfer.files || []);
            if (files.length) this.uploadFiles(files);
        },
        async uploadFiles(files) {
            const validExts = ['.pdf','.txt','.csv','.md','.markdown'];
            const validFiles = files.filter(f => {
                const ext = '.' + f.name.split('.').pop().toLowerCase();
                if (!validExts.includes(ext)) {
                    this.$emit('notify', `跳过「${f.name}」：仅支持 PDF/TXT/CSV/Markdown`, 'error');
                    return false;
                }
                if (f.size > 10 * 1024 * 1024) {
                    this.$emit('notify', `跳过「${f.name}」：超过 10MB 限制`, 'error');
                    return false;
                }
                return true;
            });
            if (!validFiles.length) return;

            this.uploading = true; this.uploadProgress = 0;
            this.uploadStatusText = '解析与向量化中';

            let successCount = 0;
            let failCount = 0;
            const total = validFiles.length;

            for (let i = 0; i < total; i++) {
                const file = validFiles[i];
                try {
                    const fd = new FormData(); fd.append('file', file);
                    const r = await ApiClient.uploadDocument(fd, (pct) => {
                        // 真实上传进度（XMLHttpRequest）
                        const fileBase = Math.floor((i / total) * 100);
                        const filePart = Math.floor(pct / total);
                        this.uploadProgress = Math.min(99, fileBase + filePart);
                    });
                    successCount++;
                    this.uploadProgress = Math.min(99, Math.floor(((i + 1) / total) * 100));
                    const msg = r.replaced
                        ? `已更新「${r.file_name}」(${r.replaced_chunks} → ${r.chunks_created})`
                        : `已上传「${r.file_name}」· ${r.chunks_created} 分块`;
                    this.$emit('notify', msg);
                } catch (e) {
                    failCount++;
                    this.$emit('notify', `「${file.name}」失败: ${e.message}`, 'error');
                }
            }

            this.uploadProgress = 100;
            if (successCount) await this.load();
            this.$emit('notify', failCount === 0
                ? `完成：${successCount} 个文件`
                : `完成：${successCount} 成功，${failCount} 失败`,
                failCount === 0 ? 'info' : 'error');
            setTimeout(() => { this.uploading = false; this.uploadProgress = 0; }, 800);
        },
        async remove(name) {
            if (!confirm('删除「' + name + '」？所有分块将被删除。')) return;
            try { const r = await ApiClient.deleteDocument(name); this.$emit('notify', `已删除 ${r.deleted_chunks} 个分块`); await this.load(); }
            catch (e) { this.$emit('notify', '删除失败: ' + e.message, 'error'); }
        },
        fmt(t) { return t ? new Date(t).toLocaleString('zh-CN') : '—'; }
    }
});

// ==================== 统计面板 ====================
app.component('stats-panel', {
    props: ['health'],
    emits: ['notify'],
    template: `
    <div class="knowledge-card">
        <h3 class="section-title">知识库统计</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ stats.total_chunks || 0 }}</div>
                <div class="stat-label">分块总数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ stats.total_documents || 0 }}</div>
                <div class="stat-label">文档数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" :class="health.es_connected ? '' : 'mono'">{{ health.es_connected ? '✓' : '—' }}</div>
                <div class="stat-label">ES 状态</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" :class="health.embedding_loaded ? '' : 'mono'">{{ health.embedding_loaded ? '✓' : '—' }}</div>
                <div class="stat-label">Embedding</div>
            </div>
        </div>
    </div>
    `,
    data() { return { stats: {} }; },
    async mounted() {
        try { this.stats = await ApiClient.getStats(); } catch (e) {}
    }
});

app.mount('#app');
