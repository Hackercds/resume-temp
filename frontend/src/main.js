const { createApp, ref, reactive, computed, watch, onMounted, nextTick } = Vue;
import { ApiClient } from './api/client.js';

const app = createApp({
    setup() {
        const activeTab = ref('chat');

        // API 配置
        const apiConfig = reactive({
            apiKey: localStorage.getItem('rag_api_key') || '',
            provider: localStorage.getItem('rag_provider') || 'openai',
            model: localStorage.getItem('rag_model') || 'gpt-4o-mini',
            baseUrl: localStorage.getItem('rag_base_url') || ''
        });
        watch(() => apiConfig.apiKey, v => localStorage.setItem('rag_api_key', v));
        watch(() => apiConfig.provider, v => localStorage.setItem('rag_provider', v));
        watch(() => apiConfig.model, v => localStorage.setItem('rag_model', v));
        watch(() => apiConfig.baseUrl, v => localStorage.setItem('rag_base_url', v));

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

        async function checkHealth() {
            try {
                const h = await ApiClient.healthCheck();
                health.embedding_loaded = h.embedding_loaded;
                health.es_connected = h.es_connected;
            } catch (e) { /* 忽略 */ }
            health.checking = false;
        }

        onMounted(() => {
            checkHealth();
            healthTimer = setInterval(checkHealth, 15000); // 15s 刷新
        });
        // cleanup not needed in SPA but included for completeness
        // onUnmounted(() => clearInterval(healthTimer));

        const notification = ref(null);
        const showNotification = (msg, type = 'info') => {
            notification.value = { msg, type };
            setTimeout(() => notification.value = null, 4000);
        };

        return { activeTab, apiConfig, modelDatalist, health, notification, showNotification };
    },

    template: `
    <div>
        <div class="header">
            <h1>简历 RAG 智能问答系统</h1>
            <p class="subtitle">本地 Embedding + ES 混合检索 + 在线 LLM</p>
            <!-- 系统状态条 -->
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

        <div class="tabs">
            <button class="tab-btn" :class="{ active: activeTab === 'chat' }" @click="activeTab = 'chat'">💬 问答</button>
            <button class="tab-btn" :class="{ active: activeTab === 'knowledge' }" @click="activeTab = 'knowledge'">📚 知识库</button>
            <button class="tab-btn" :class="{ active: activeTab === 'stats' }" @click="activeTab = 'stats'">📊 统计</button>
        </div>

        <api-key-config :api-config="apiConfig" :model-datalist="modelDatalist" />

        <div v-show="activeTab === 'chat'">
            <chat-panel :api-config="apiConfig" :health="health" @notify="showNotification" />
        </div>
        <div v-show="activeTab === 'knowledge'">
            <knowledge-panel :health="health" @notify="showNotification" />
        </div>
        <div v-show="activeTab === 'stats'">
            <stats-panel :health="health" />
        </div>
    </div>
    `
});

// ==================== API Key 配置 ====================
app.component('api-key-config', {
    props: ['apiConfig', 'modelDatalist'],
    emits: ['update:apiConfig'],
    template: `
    <div class="card config-card">
        <div class="config-grid">
            <div class="field">
                <label>🔑 API Key</label>
                <input :type="showKey ? 'text' : 'password'"
                    :value="apiConfig.apiKey" @input="update('apiKey', $event.target.value)"
                    placeholder="sk-..." />
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
        // 直接修改父组件的 reactive 对象（同一个引用）
        update(key, val) { this.apiConfig[key] = val; }
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

        <div class="card">
            <!-- 加载中 -->
            <div v-if="loading" class="loading">
                <div class="spinner"></div>
                <p>正在检索知识库...</p>
            </div>

            <!-- 答案 -->
            <div v-else-if="answer">
                <div v-if="timing" class="timing-badge">
                    <span>🔍 向量化 {{ timing.embedding_ms }}ms</span>
                    <span>📡 检索 {{ timing.search_ms }}ms</span>
                    <span>🤖 生成 {{ timing.llm_s }}s</span>
                </div>
                <div class="answer-box">{{ answer }}</div>

                <div v-if="sources.length" class="source-list">
                    <h4>📄 引用来源 ({{ sources.length }})</h4>
                    <div v-for="(s, i) in sources" :key="i"
                        class="source-item" :class="{ open: expandedIdx === i }"
                        @click="expandedIdx = expandedIdx === i ? -1 : i">
                        <div class="source-header">
                            <span>{{ s.file_name }}</span>
                            <span class="badge">相关度 {{ s.score }}</span>
                        </div>
                        <div class="source-content">{{ s.content }}</div>
                    </div>
                </div>
            </div>

            <!-- 空状态 -->
            <div v-else class="empty-state">💡 输入问题，基于知识库智能回答</div>
            <div v-if="error" class="error-msg">{{ error }}</div>
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
    data() { return { question: '', answer: '', sources: [], timing: null, loading: false, error: '', expandedIdx: -1 }; },
    methods: {
        async doQuery() {
            if (!this.question.trim() || !this.apiConfig.apiKey) return;
            this.loading = true; this.error = ''; this.answer = ''; this.sources = []; this.expandedIdx = -1;
            try {
                const body = { question: this.question, api_key: this.apiConfig.apiKey, provider: this.apiConfig.provider, model: this.apiConfig.model || null, top_k: 5 };
                if (this.apiConfig.baseUrl) body.base_url = this.apiConfig.baseUrl;
                const r = await ApiClient.query(body);
                this.answer = r.answer; this.sources = r.sources || []; this.timing = r.timing || null;
            } catch (e) { this.error = e.message; this.$emit('notify', e.message, 'error'); }
            finally { this.loading = false; }
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
                <p>点击或拖拽 PDF / TXT / CSV 文件到此处</p>
                <p class="upload-limit">最大 10MB</p>
                <input type="file" ref="fi" accept=".pdf,.txt,.csv" @change="handleFileSelect" hidden />
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
        handleFileSelect(e) { const f = e.target.files[0]; if (f) this.upload(f); },
        handleDrop(e) { this.isDragOver = false; const f = e.dataTransfer.files[0]; if (f) this.upload(f); },
        async upload(file) {
            const ext = '.' + file.name.split('.').pop().toLowerCase();
            if (!['.pdf','.txt','.csv'].includes(ext)) { this.$emit('notify', '仅支持 PDF/TXT/CSV', 'error'); return; }
            if (file.size > 10*1024*1024) { this.$emit('notify', '超过 10MB 限制', 'error'); return; }
            this.uploading = true; this.uploadProgress = 5;
            const t = setInterval(() => { if (this.uploadProgress < 85) this.uploadProgress += 8; }, 600);
            try {
                const fd = new FormData(); fd.append('file', file);
                const r = await ApiClient.uploadDocument(fd);
                clearInterval(t); this.uploadProgress = 100;
                const msg = r.replaced
                    ? `✓ 已更新「${r.file_name}」（替换 ${r.replaced_chunks} → ${r.chunks_created} 个分块）`
                    : `✓ 上传成功，${r.chunks_created} 个分块已入库`;
                this.$emit('notify', msg);
                await this.load();
            } catch (e) { clearInterval(t); this.$emit('notify', '上传失败: ' + e.message, 'error'); }
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
