const { createApp, ref, reactive, computed, watch, onMounted, nextTick } = Vue;
import { ApiClient } from './api/client.js';

const app = createApp({
    setup() {
        // 注册 Service Worker
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('sw.js').catch(() => {});
        }

        const activeTab = ref('chat');

        // API 配置
        const apiConfig = reactive({
            apiKey: localStorage.getItem('rag_api_key') || '',
            provider: localStorage.getItem('rag_provider') || 'openai',
            model: localStorage.getItem('rag_model') || 'gpt-4o-mini',
            baseUrl: localStorage.getItem('rag_base_url') || ''
        });
        const rememberKey = ref(localStorage.getItem('rag_remember_key') !== 'false');
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

        // 模型建议
        const modelHints = {
            openai: ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1', 'gpt-3.5-turbo'],
            anthropic: ['claude-sonnet-4-6', 'claude-3-haiku-20240307', 'claude-3-opus-20240229'],
            custom: ['deepseek-chat', 'deepseek-reasoner', 'doubao-pro-32k']
        };
        const modelDatalist = computed(() => [...new Set([...(modelHints[apiConfig.provider] || []), apiConfig.model])]);

        // 系统状态（全局轮询）
        const health = reactive({ embedding_loaded: false, es_connected: false, checking: true });
        let healthTimer = null;
        const isMobile = ref(window.innerWidth <= 640);
        const showMobileNav = ref(false);
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
            healthTimer = setInterval(checkHealth, 15000); // 15s 刷新
        });
        // cleanup not needed in SPA but included for completeness
        // onUnmounted(() => clearInterval(healthTimer));

        const notification = ref(null);
        const showNotification = (msg, type = 'info') => {
            notification.value = { msg, type };
            setTimeout(() => notification.value = null, 4000);
        };

        return { activeTab, apiConfig, rememberKey, modelDatalist, llmPresets, defaultApiKeyConfigured, health, isMobile, showMobileNav, notification, showNotification };
    },

    template: `
    <div>
        <div class="app-nav" v-if="isMobile">
            <button class="menu-btn" @click="showMobileNav = !showMobileNav">☰</button>
            <span class="header" style="flex:1; text-align:left; padding:0; font-size:18px; font-weight:700;">简历 RAG</span>
            <span class="status-dot" :class="health.es_connected ? 'ok' : 'fail'" :title="health.es_connected ? 'ES 已连接' : 'ES 未连接'"></span>
        </div>

        <div class="header" v-if="!isMobile">
            <h1>简历 RAG 智能问答系统</h1>
            <p class="subtitle">本地 Embedding + ES 混合检索 + 在线 LLM</p>
            <div class="status-bar">
                <span class="status-dot" :class="health.es_connected ? 'ok' : 'fail'"></span>
                ES {{ health.checking ? '检测中...' : (health.es_connected ? '已连接' : '未连接') }}
                <span style="margin:0 8px">|</span>
                <span class="status-dot" :class="health.embedding_loaded ? 'ok' : 'warn'"></span>
                Embedding {{ health.embedding_loaded ? '已加载' : '加载中...' }}
            </div>
        </div>

        <div v-if="notification" class="toast" :class="notification.type">
            {{ notification.msg }}
        </div>

        <api-key-config v-if="!isMobile || activeTab === 'chat'" :api-config="apiConfig" :remember-key="rememberKey" :model-datalist="modelDatalist" :llm-presets="llmPresets" :default-api-key-configured="defaultApiKeyConfigured" @update:remember-key="rememberKey = $event" />

        <div v-show="activeTab === 'chat'">
            <chat-panel :api-config="apiConfig" :health="health" @notify="showNotification" @switch-tab="activeTab = $event" />
        </div>
        <div v-show="activeTab === 'knowledge'">
            <knowledge-panel :health="health" @notify="showNotification" />
        </div>
        <div v-show="activeTab === 'stats'">
            <stats-panel :health="health" />
        </div>

        <div class="tabs">
            <button class="tab-btn" :class="{ active: activeTab === 'chat' }" @click="activeTab = 'chat'">💬 问答</button>
            <button class="tab-btn" :class="{ active: activeTab === 'knowledge' }" @click="activeTab = 'knowledge'">📚 知识库</button>
            <button class="tab-btn" :class="{ active: activeTab === 'stats' }" @click="activeTab = 'stats'">📊 统计</button>
        </div>
    </div>
    `,
});

// ==================== API Key 配置 ====================
app.component('api-key-config', {
    props: ['apiConfig', 'rememberKey', 'modelDatalist', 'llmPresets', 'defaultApiKeyConfigured'],
    emits: ['update:rememberKey'],
    template: `
    <div class="card config-card">
        <div class="config-grid">
            <div class="field" style="grid-column: span 2;">
                <label>🔑 API Key</label>
                <input :type="showKey ? 'text' : 'password'"
                    :value="apiConfig.apiKey" @input="update('apiKey', $event.target.value)"
                    :placeholder="defaultApiKeyConfigured ? '后端已配置默认 Key，可留空' : 'sk-...'"
                    :disabled="defaultApiKeyConfigured && apiConfig.apiKey === 'DEFAULT_API_KEY'" />
                <label class="remember-label">
                    <input type="checkbox" :checked="rememberKey" @change="$emit('update:rememberKey', $event.target.checked)" />
                    记住 Key
                </label>
            </div>
            <div class="field">
                <label>模型预设</label>
                <select @change="applyPreset($event.target.value)">
                    <option value="">自定义</option>
                    <option v-for="p in llmPresets" :key="p.name" :value="p.name">{{ p.name }}</option>
                </select>
            </div>
            <div class="field">
                <label>Provider</label>
                <select :value="apiConfig.provider" @change="update('provider', $event.target.value)">
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="custom">Custom（兼容）</option>
                </select>
            </div>
            <div class="field">
                <label>Model</label>
                <input :value="apiConfig.model" @input="update('model', $event.target.value)"
                    :list="'ml-' + apiConfig.provider" placeholder="自由输入" />
                <datalist :id="'ml-' + apiConfig.provider">
                    <option v-for="m in modelDatalist" :value="m" />
                </datalist>
            </div>
            <div class="field">
                <label>API 地址（可选）</label>
                <input :value="apiConfig.baseUrl" @input="update('baseUrl', $event.target.value)"
                    placeholder="自动使用官方地址" />
            </div>
            <button class="btn-eye" @click="showKey = !showKey" :title="showKey ? '隐藏' : '显示'">
                {{ showKey ? '🙈' : '👁' }}
            </button>
        </div>
    </div>
    `,
    data() { return { showKey: false }; },
    methods: {
        update(key, val) { this.apiConfig[key] = val; },
        applyPreset(name) {
            const p = (this.llmPresets || []).find(x => x.name === name);
            if (!p) return;
            this.apiConfig.provider = p.provider || 'openai';
            this.apiConfig.model = p.model || '';
            this.apiConfig.baseUrl = p.base_url || '';
        }
    }
});

// ==================== 问答面板 ====================
app.component('chat-panel', {
    props: ['apiConfig', 'health'],
    emits: ['notify'],
    template: `
    <div>
        <!-- 提示卡片 -->
        <div v-if="!apiConfig.apiKey" class="card hint-card">
            ⚠️ 请先在上方填入 API Key 才能开始提问
        </div>

        <!-- 会话工具栏 -->
        <div class="card chat-toolbar">
            <button class="btn-new-chat" @click="newSession">➕ 新建对话</button>
            <div class="session-bar">
                <div v-for="s in sessions" :key="s.id"
                     :class="['session-chip', { active: s.id === currentSessionId }]"
                     @click="loadSession(s.id)">
                    <span class="session-title">{{ s.title }}</span>
                    <span class="session-delete" @click.stop="deleteSession(s.id)">✕</span>
                </div>
            </div>
        </div>

        <div class="card chat-card">
            <!-- 消息列表 -->
            <div v-if="messages.length" class="message-list" ref="messageList">
                <div v-for="(msg, mIdx) in messages" :key="msg.id"
                     :class="['message', msg.role]">
                    <div class="message-bubble">
                        <div v-if="msg.role === 'user'">{{ msg.content }}</div>
                        <div v-else>
                            <div class="answer-box" v-html="renderMarkdown(msg.content)"></div>
                            <div v-if="msg.timing" class="timing-badge">
                                <span>🔍 向量化 {{ msg.timing.embedding_ms }}ms</span>
                                <span>📡 检索 {{ msg.timing.search_ms }}ms</span>
                                <span>🤖 生成 {{ msg.timing.llm_s }}s</span>
                            </div>
                            <div v-if="msg.sources && msg.sources.length" class="source-list">
                                <h4>📄 引用来源 ({{ msg.sources.length }})</h4>
                                <div v-for="(s, i) in msg.sources" :key="i"
                                    class="source-item" :class="{ open: expandedMsgIdx === mIdx && expandedIdx === i }"
                                    @click="toggleSource(mIdx, i)">
                                    <div class="source-header">
                                        <div class="source-meta">
                                            <span class="source-file">{{ s.file_name }}</span>
                                            <span v-if="s.section_title" class="source-section">{{ s.section_title }}</span>
                                            <span class="source-index">#{{ s.chunk_index }}</span>
                                        </div>
                                        <span class="badge">相关度 {{ s.score }}</span>
                                    </div>
                                    <div class="source-content" v-html="renderMarkdown(s.content)"></div>
                                </div>
                            </div>
                            <div v-if="msg.trace" class="trace-box">
                                <button class="btn-trace-toggle" @click="msg.showTrace = !msg.showTrace">🔍 查看检索过程</button>
                                <pre v-if="msg.showTrace" class="trace-content">{{ JSON.stringify(msg.trace, null, 2) }}</pre>
                            </div>
                            <button v-if="msg.sources && msg.sources.length && !msg.isFullDoc"
                                class="btn-full-doc" @click="retrieveFullDoc(msg)">
                                📄 查看完整文档
                            </button>
                        </div>
                    </div>
                </div>
                <div v-if="loading" class="loading">
                    <div class="spinner"></div>
                    <p>正在检索知识库...</p>
                </div>
            </div>

            <!-- 空状态 -->
            <div v-else-if="!loading" class="empty-state">
                <div v-if="documents.length === 0" class="onboarding">
                    <div class="onboarding-icon">🚀</div>
                    <h3>欢迎使用简历 RAG 智能问答系统</h3>
                    <p>基于上传的简历/文档，AI 会结合知识库内容回答您的问题。</p>
                    <div class="onboarding-steps">
                        <div class="step"><span>1</span> 上传 PDF / TXT / CSV / Markdown 文档</div>
                        <div class="step"><span>2</span> 配置 API Key（或让管理员后端注入）</div>
                        <div class="step"><span>3</span> 输入问题，获得带来源引用的回答</div>
                    </div>
                    <button class="btn-onboarding" @click="$emit('switch-tab', 'knowledge')">📤 立即上传文档</button>
                    <div v-if="suggestedQuestions.length" class="suggested-questions">
                        <p>您也可以试试：</p>
                        <div class="question-chips">
                            <button v-for="sq in suggestedQuestions" :key="sq"
                                class="question-chip" @click="question = sq; doQuery()">{{ sq }}</button>
                        </div>
                    </div>
                </div>
                <div v-else-if="apiConfig.apiKey">
                    <p>💡 输入问题，基于知识库智能回答</p>
                    <div v-if="suggestedQuestions.length" class="suggested-questions">
                        <p class="input-hint">试试这样问：</p>
                        <div class="question-chips">
                            <button v-for="sq in suggestedQuestions" :key="sq"
                                class="question-chip" @click="question = sq; doQuery()">{{ sq }}</button>
                        </div>
                    </div>
                </div>
                <div v-else>
                    <p>⚠️ 请先在上方填入 API Key 才能开始提问</p>
                </div>
            </div>
            <div v-if="error" class="error-msg">
                <div>{{ error }}</div>
                <div v-if="errorSuggestion" class="error-suggestion">{{ errorSuggestion }}</div>
                <div v-if="emptyRetrieval" class="error-actions">
                    <button class="btn-retry" @click="switchTabToKnowledge">📤 上传文档</button>
                    <button class="btn-retry" @click="retryQuestion">🔄 换个问法</button>
                </div>
                <div v-else-if="errorRetryable" class="error-actions">
                    <button class="btn-retry" @click="retryLastQuestion">🔄 重试</button>
                </div>
            </div>
        </div>

        <!-- 追问建议 -->
        <div v-if="followUpQuestions.length && !loading" class="follow-up-bar">
            <span class="follow-up-label">继续问：</span>
            <button v-for="fq in followUpQuestions" :key="fq"
                class="question-chip" @click="question = fq; doQuery()">{{ fq }}</button>
        </div>

        <!-- 输入区 -->
        <div class="card">
            <div class="input-row">
                <textarea v-model="question" placeholder="输入问题..." @keydown.ctrl.enter="doQuery"
                    :disabled="loading" rows="2"></textarea>
                <button class="btn-send" @click="doQuery"
                    :disabled="loading || !question.trim() || !apiConfig.apiKey"
                    :title="!apiConfig.apiKey ? '请先填写 API Key' : ''">
                    {{ loading ? '检索中...' : '发送' }}
                </button>
            </div>
            <div class="input-hint">Ctrl+Enter 发送 · 基于知识库内容回答</div>
        </div>
    </div>
    `,
    data() {
        const sid = this._newSessionId();
        return {
            question: '',
            messages: [],
            sessions: this._loadSessions(),
            currentSessionId: sid,
            loading: false,
            error: '',
            errorSuggestion: '',
            errorRetryable: false,
            emptyRetrieval: false,
            expandedIdx: -1,
            expandedMsgIdx: -1,
            maxHistory: 5,
            lastQuestion: '',
            followUpQuestions: [],
            documents: [],
            suggestedQuestions: []
        };
    },
    mounted() {
        // 不要保存空会话，避免产生一堆"新对话"
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
            const raw = marked.parse(String(text), {
                gfm: true,
                breaks: true,
                headerIds: false
            });
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
            // 防止 Vue 事件对象被当成参数传入
            if (typeof retrieveFullDoc !== 'boolean') retrieveFullDoc = false;
            if (!this.question.trim() || !this.apiConfig.apiKey) return;
            const q = this.question.trim();
            this.lastQuestion = q;
            this.messages.push({ id: Date.now(), role: 'user', content: q, timestamp: Date.now() });
            this.question = '';
            this.loading = true; this.error = ''; this.errorSuggestion = ''; this.errorRetryable = false; this.emptyRetrieval = false;
            this.scrollToBottom();

            const history = this.messages
                .filter(m => m.role === 'user' || m.role === 'assistant')
                .slice(-(this.maxHistory * 2))
                .map(m => ({
                    role: m.role,
                    content: m.content,
                    sources: (m.sources || []).map(s => ({
                        file_name: s.file_name,
                        score: s.score
                    }))
                }));

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
                timestamp: Date.now()
            };
            this.messages.push(assistantMsg);

            try {
                const body = {
                    question: q,
                    api_key: this.apiConfig.apiKey,
                    provider: this.apiConfig.provider,
                    model: this.apiConfig.model || null,
                    top_k: 5,
                    history,
                    session_id: this.currentSessionId,
                    retrieve_full_doc: retrieveFullDoc
                };
                if (this.apiConfig.baseUrl) body.base_url = this.apiConfig.baseUrl;

                await ApiClient.queryStream(
                    body,
                    (token) => {
                        assistantMsg.content += token;
                        this.scrollToBottom();
                    },
                    (data) => {
                        assistantMsg.content = data.answer || assistantMsg.content;
                        assistantMsg.sources = data.sources || [];
                        assistantMsg.timing = data.timing || null;
                        assistantMsg.trace_id = data.trace_id || '';
                        assistantMsg.trace = data.trace || null;
                        this.followUpQuestions = this._generateFollowUpQuestions(data.sources || []);
                        this.saveSession();
                    },
                    (err) => {
                        this.error = err.message;
                        this.errorSuggestion = err.suggestion || '';
                        this.emptyRetrieval = !!err.emptyRetrieval;
                        this.errorRetryable = !this.emptyRetrieval;
                        // 移除已添加的空白 assistant 消息
                        const idx = this.messages.findIndex(m => m.id === assistantMsg.id);
                        if (idx >= 0) this.messages.splice(idx, 1);
                        this.$emit('notify', err.message, 'error');
                    }
                );
            } catch (e) {
                this.error = e.message;
                this.errorSuggestion = e.suggestion || '';
                this.emptyRetrieval = !!e.emptyRetrieval;
                this.errorRetryable = !this.emptyRetrieval;
                const idx = this.messages.findIndex(m => m.id === assistantMsg.id);
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
                // 简单改写：去掉疑问词，换个说法
                let q = this.lastQuestion.replace(/[吗呢？?]/g, '');
                if (q === this.lastQuestion) q = '请介绍一下' + this.lastQuestion;
                this.question = q;
                this.doQuery();
            }
        },
        switchTabToKnowledge() {
            this.$emit('switch-tab', 'knowledge');
        },
        async retrieveFullDoc(lastMsg) {
            if (!lastMsg || !lastMsg.sources || !lastMsg.sources.length) return;
            const idx = this.messages.indexOf(lastMsg);
            if (idx < 0) return;

            const priorUserMsgs = this.messages.slice(0, idx).filter(m => m.role === 'user');
            let queryText = '';
            if (priorUserMsgs.length >= 2) {
                const lastTwo = priorUserMsgs.slice(-2);
                queryText = `${lastTwo[0].content}；${lastTwo[1].content}`;
            } else if (priorUserMsgs.length === 1) {
                queryText = priorUserMsgs[0].content;
            }
            if (!queryText) return;

            const targetFiles = [...new Set(lastMsg.sources.map(s => s.file_name))];
            if (!targetFiles.length) return;

            this.loading = true; this.error = ''; this.errorSuggestion = ''; this.errorRetryable = false; this.emptyRetrieval = false;

            for (const fileName of targetFiles) {
                const fullDocMsg = {
                    id: Date.now() + Math.random(),
                    role: 'assistant',
                    content: '',
                    sources: [],
                    timing: null,
                    trace_id: '',
                    trace: null,
                    showTrace: false,
                    fullDocs: [],
                    isFullDoc: true,
                    targetFile: fileName,
                    timestamp: Date.now()
                };
                // 追加到当前消息下方，不覆盖原答案
                this.messages.splice(idx + 1, 0, fullDocMsg);

                try {
                    const body = {
                        question: queryText,
                        api_key: this.apiConfig.apiKey,
                        provider: this.apiConfig.provider,
                        model: this.apiConfig.model || null,
                        top_k: 5,
                        history: [],
                        session_id: this.currentSessionId,
                        retrieve_full_doc: true
                    };
                    if (this.apiConfig.baseUrl) body.base_url = this.apiConfig.baseUrl;

                    await ApiClient.queryStream(
                        body,
                        (token) => {
                            fullDocMsg.content += token;
                            this.scrollToBottom();
                        },
                        (data) => {
                            fullDocMsg.content = data.answer || fullDocMsg.content;
                            fullDocMsg.sources = data.sources || [];
                            fullDocMsg.timing = data.timing || null;
                            fullDocMsg.trace_id = data.trace_id || '';
                            fullDocMsg.trace = data.trace || null;
                            this.saveSession();
                        },
                        (err) => {
                            this.error = err.message;
                            this.errorSuggestion = err.suggestion || '';
                            this.emptyRetrieval = !!err.emptyRetrieval;
                            this.errorRetryable = !this.emptyRetrieval;
                            fullDocMsg.content = `⚠️ ${err.message}`;
                            this.$emit('notify', err.message, 'error');
                        }
                    );
                } catch (e) {
                    this.error = e.message;
                    this.errorSuggestion = e.suggestion || '';
                    this.emptyRetrieval = !!e.emptyRetrieval;
                    this.errorRetryable = !this.emptyRetrieval;
                    fullDocMsg.content = `⚠️ ${e.message}`;
                    this.$emit('notify', e.message, 'error');
                }
            }

            this.$emit('notify', `已召回完整文档：${targetFiles.join(', ')}`);
            this.loading = false;
            this.scrollToBottom();
        },
        newSession() {
            // 如果当前会话为空，直接复用不新建
            if (this.messages.length === 0) return;
            this.currentSessionId = this._newSessionId();
            this.messages = [];
            this.error = '';
            this.question = '';
            this.followUpQuestions = [];
            this.expandedIdx = -1;
            this.expandedMsgIdx = -1;
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
                // 重新生成一个空会话 ID，避免后续消息误入已删会话
                this.currentSessionId = this._newSessionId();
            }
        },
        loadSession(id) {
            const s = this.sessions.find(s => s.id === id);
            if (s) {
                this.currentSessionId = id;
                this.messages = s.messages || [];
                this.error = '';
                this.expandedIdx = -1;
                this.expandedMsgIdx = -1;
            }
        },
        saveSession(isNew = false) {
            // 不保存空会话
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
            return first ? first.content.slice(0, 20) + (first.content.length > 20 ? '...' : '') : '新对话';
        },
        scrollToBottom() {
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
                '张成都有几年的工作经验？',
                '介绍一下张成都的项目经历',
                '张成都熟悉哪些技术栈？',
                '总结一下文档中的核心能力',
                '文档中有哪些关于 Python 的内容？'
            ];
            // 如果已上传文档，随机展示 3 个；否则展示 2 个通用引导问题
            if (this.documents.length > 0) {
                this.suggestedQuestions = samples.sort(() => 0.5 - Math.random()).slice(0, 3);
            } else {
                this.suggestedQuestions = ['如何使用这个系统？', '支持上传哪些文件格式？'];
            }
        },
        _generateFollowUpQuestions(sources) {
            // 基于来源文件名和章节生成简单追问
            const questions = [];
            const files = [...new Set(sources.map(s => s.file_name))].slice(0, 2);
            for (const fn of files) {
                const base = fn.replace(/\.[^.]+$/, '');
                questions.push(`请详细介绍一下 ${base} 的内容`);
                questions.push(`${base} 中有哪些关键技能？`);
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
        <div class="card">
            <h3>📤 上传文档</h3>
            <div class="upload-zone" :class="{ over: isDragOver }"
                @dragover.prevent="isDragOver = true" @dragleave="isDragOver = false"
                @drop.prevent="handleDrop" @click="$refs.fi.click()">
                <div class="upload-icon">📄</div>
                <p>点击或拖拽 PDF / TXT / CSV / Markdown 文件到此处</p>
                <p class="upload-limit">最大 10MB · 支持多个文件</p>
                <input type="file" ref="fi" accept=".pdf,.txt,.csv,.md,.markdown" multiple @change="handleFileSelect" hidden />
            </div>
            <div v-if="uploading">
                <div class="progress-bar"><div class="progress-fill" :style="{width:uploadProgress+'%'}"></div></div>
                <p class="upload-status">正在解析、分块、向量化...请等待</p>
            </div>
        </div>

        <div class="card">
            <h3>📚 文档列表 ({{ documents.length }})</h3>
            <div v-if="loading">加载中...</div>
            <div v-else-if="documents.length === 0" class="empty-state">暂无文档</div>
            <table v-else class="doc-table">
                <thead><tr><th>文件名</th><th>类型</th><th>分块数</th><th>时间</th><th>操作</th></tr></thead>
                <tbody>
                    <tr v-for="d in documents" :key="d.file_name">
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
    data() { return { documents: [], loading: false, uploading: false, uploadProgress: 0, isDragOver: false }; },
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
            e.target.value = ''; // 允许重复选择同一文件
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

            this.uploading = true; this.uploadProgress = 5;
            const t = setInterval(() => { if (this.uploadProgress < 85) this.uploadProgress += 8; }, 600);

            let successCount = 0;
            let failCount = 0;
            for (const file of validFiles) {
                try {
                    const fd = new FormData(); fd.append('file', file);
                    const r = await ApiClient.uploadDocument(fd);
                    successCount++;
                    const msg = r.replaced
                        ? `✓ 已更新「${r.file_name}」（替换 ${r.replaced_chunks} → ${r.chunks_created} 个分块）`
                        : `✓ 上传成功「${r.file_name}」，${r.chunks_created} 个分块已入库`;
                    this.$emit('notify', msg);
                } catch (e) {
                    failCount++;
                    this.$emit('notify', `「${file.name}」上传失败: ${e.message}`, 'error');
                }
            }
            clearInterval(t); this.uploadProgress = 100;
            if (successCount) await this.load();
            if (failCount === 0) {
                this.$emit('notify', `全部完成：${successCount} 个文件上传成功`);
            } else {
                this.$emit('notify', `完成：${successCount} 成功，${failCount} 失败`, 'error');
            }
            this.uploading = false;
        },
        async remove(name) {
            if (!confirm('删除「' + name + '」？所有分块将被删除。')) return;
            try { const r = await ApiClient.deleteDocument(name); this.$emit('notify', `已删除 ${r.deleted_chunks} 个分块`); await this.load(); }
            catch (e) { this.$emit('notify', '删除失败: ' + e.message, 'error'); }
        },
        fmt(t) { return t ? new Date(t).toLocaleString('zh-CN') : '-'; }
    }
});

// ==================== 统计面板 ====================
app.component('stats-panel', {
    props: ['health'],
    template: `
    <div>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ stats.total_chunks }}</div>
                <div class="stat-label">总分块数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ stats.total_documents }}</div>
                <div class="stat-label">文档数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" :style="{color: health.es_connected ? 'var(--success)' : 'var(--danger)'}">
                    {{ health.es_connected ? '✓' : '✗' }}
                </div>
                <div class="stat-label">Elasticsearch</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" :style="{color: health.embedding_loaded ? 'var(--success)' : 'var(--warning)'}">
                    {{ health.embedding_loaded ? '✓' : '⏳' }}
                </div>
                <div class="stat-label">Embedding</div>
            </div>
        </div>

        <div class="card" v-if="stats.documents && stats.documents.length">
            <h3>文档详情</h3>
            <table class="doc-table">
                <thead><tr><th>文件名</th><th>类型</th><th>分块</th><th>时间</th></tr></thead>
                <tbody>
                    <tr v-for="d in stats.documents" :key="d.file_name">
                        <td>{{ d.file_name }}</td>
                        <td><span class="doc-type">{{ d.file_type }}</span></td>
                        <td>{{ d.chunk_count }}</td>
                        <td>{{ fmt(d.upload_time) }}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    `,
    data() { return { stats: { total_chunks:0, total_documents:0, documents:[] }, loading: true }; },
    mounted() { this.load(); },
    methods: {
        async load() {
            this.loading = true;
            try { this.stats = await ApiClient.getStats(); } catch (e) { /* ok */ }
            this.loading = false;
        },
        fmt(t) { return t ? new Date(t).toLocaleString('zh-CN') : '-'; }
    }
});

app.mount('#app');
