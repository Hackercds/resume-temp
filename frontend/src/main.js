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
        <!-- Editorial masthead -->
        <header class="masthead">
            <h1 class="masthead-title">简历 <em>RAG</em><br />智能问答</h1>
            <div class="masthead-meta">
                <div class="status-pill" :class="health.es_connected ? 'ok' : 'bad'">
                    ES · {{ health.es_connected ? 'online' : 'offline' }}
                </div>
                <div style="margin-top:6px;">
                    <div class="status-pill" :class="health.embedding_loaded ? 'ok' : 'warn'">
                        Embedding · {{ health.embedding_loaded ? 'ready' : 'loading' }}
                    </div>
                </div>
            </div>
        </header>

        <div v-if="notification" class="toast" :class="notification.type">
            {{ notification.msg }}
        </div>

        <!-- Config - 编辑式配置条 -->
        <api-key-config :api-config="apiConfig" :remember-key="rememberKey"
            :model-datalist="modelDatalist" :llm-presets="llmPresets"
            :default-api-key-configured="defaultApiKeyConfigured"
            @update:remember-key="rememberKey = $event" />

        <!-- Tabs -->
        <nav class="tabs">
            <button class="tab-btn" :class="{ active: activeTab === 'chat' }" @click="activeTab = 'chat'">
                问答<span class="tab-num">01</span>
            </button>
            <button class="tab-btn" :class="{ active: activeTab === 'knowledge' }" @click="activeTab = 'knowledge'">
                知识库<span class="tab-num">02</span>
            </button>
            <button class="tab-btn" :class="{ active: activeTab === 'stats' }" @click="activeTab = 'stats'">
                统计<span class="tab-num">03</span>
            </button>
        </nav>

        <div v-show="activeTab === 'chat'">
            <chat-panel :api-config="apiConfig" :health="health"
                @notify="showNotification" @switch-tab="activeTab = $event" />
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
            <label class="remember-row">
                <input type="checkbox" :checked="rememberKey"
                    @change="$emit('update:rememberKey', $event.target.checked)" />
                记住 Key
            </label>
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
        <button class="eye-btn" @click="showKey = !showKey" :title="showKey ? '隐藏' : '显示'">
            {{ showKey ? '🙈' : '👁' }}
        </button>
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
    emits: ['notify', 'switch-tab'],
    template: `
    <div>
        <!-- 会话工具栏 -->
        <div class="chat-bar">
            <span class="chat-bar-label">对话</span>
            <button class="btn-new-chat" @click="newSession">＋ 新建</button>
            <div v-for="s in sessions" :key="s.id"
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
                            <div class="answer-box" v-html="renderMarkdown(msg.content)"></div>
                            <div v-if="msg.timing" class="bubble-meta">
                                <span><b>{{ msg.timing.embedding_ms }}ms</b> embed</span>
                                <span><b>{{ msg.timing.search_ms }}ms</b> search</span>
                                <span><b>{{ msg.timing.llm_s }}s</b> llm</span>
                            </div>
                            <div v-if="msg.sources && msg.sources.length" class="source-list">
                                <div class="source-list-title">引用来源 ({{ msg.sources.length }})</div>
                                <div v-for="(s, i) in msg.sources" :key="i"
                                    class="source-item" :class="{ open: expandedMsgIdx === mIdx && expandedIdx === i }"
                                    @click="toggleSource(mIdx, i)">
                                    <div class="source-header">
                                        <div class="source-meta">
                                            <span class="source-file">{{ s.file_name }}</span>
                                            <span v-if="s.section_title" class="source-section">· {{ s.section_title }} ·</span>
                                            <span class="source-index">#{{ s.chunk_index }}</span>
                                        </div>
                                        <span class="source-score">{{ s.score }}</span>
                                    </div>
                                    <div class="source-content" v-html="renderMarkdown(s.content)"></div>
                                </div>
                            </div>
                            <button v-if="msg.trace" class="trace-toggle" @click="msg.showTrace = !msg.showTrace">
                                {{ msg.showTrace ? '— 收起检索过程' : '+ 查看检索过程' }}
                            </button>
                            <pre v-if="msg.trace && msg.showTrace" class="trace-content">{{ JSON.stringify(msg.trace, null, 2) }}</pre>
                            <button v-if="msg.sources && msg.sources.length && !msg.isFullDoc"
                                class="btn-full-doc" @click="retrieveFullDoc(msg)">
                                召回完整文档
                            </button>
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
            lastQuestion: '',
            followUpQuestions: [],
            documents: [],
            suggestedQuestions: []
        };
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
            this.scrollToBottom();

            const history = this.messages
                .filter(m => m.role === 'user' || m.role === 'assistant')
                .slice(-(this.maxHistory * 2))
                .map(m => ({
                    role: m.role,
                    content: m.content,
                    sources: (m.sources || []).map(s => ({ file_name: s.file_name, score: s.score }))
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
                        this.statusText = '生成中';
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
                let q = this.lastQuestion.replace(/[吗呢？?]/g, '');
                if (q === this.lastQuestion) q = '请介绍' + this.lastQuestion;
                this.question = q;
                this.doQuery();
            }
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
                            fullDocMsg.content = `⚠ ${err.message}`;
                            this.$emit('notify', err.message, 'error');
                        }
                    );
                } catch (e) {
                    fullDocMsg.content = `⚠ ${e.message}`;
                    this.$emit('notify', e.message, 'error');
                }
            }

            this.$emit('notify', `已召回完整文档：${targetFiles.join(', ')}`);
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
                <div class="upload-icon">⤴</div>
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
            <h3 class="section-title">文档列表 <span style="color:var(--ink-mute); font-weight:400;">· {{ documents.length }}</span></h3>
            <div v-if="loading" class="loading">
                <div class="spinner"></div>
                加载中
            </div>
            <div v-else-if="documents.length === 0" class="empty-state" style="padding:20px 0;">暂无文档</div>
            <table v-else class="doc-table">
                <thead><tr><th>文件名</th><th>类型</th><th>分块</th><th>时间</th><th>操作</th></tr></thead>
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
    data() { return { documents: [], loading: false, uploading: false, uploadProgress: 0, uploadStatusText: '准备中', isDragOver: false }; },
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
