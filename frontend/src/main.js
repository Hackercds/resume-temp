/**
 * 简历 RAG 智能问答系统 - 应用入口
 * Vue3 App 初始化
 */
import { createApp, ref, reactive, computed, watch, nextTick } from 'vue';
import { ApiClient } from './api/client.js';

const app = createApp({
    setup() {
        // ---------- 全局状态 ----------
        const activeTab = ref('chat');  // chat | knowledge | stats

        // API 配置（localStorage 持久化）
        const apiConfig = reactive({
            apiKey: localStorage.getItem('rag_api_key') || '',
            provider: localStorage.getItem('rag_provider') || 'openai',
            model: localStorage.getItem('rag_model') || 'gpt-4o-mini'
        });
        watch(() => apiConfig.apiKey, v => localStorage.setItem('rag_api_key', v));
        watch(() => apiConfig.provider, v => localStorage.setItem('rag_provider', v));
        watch(() => apiConfig.model, v => localStorage.setItem('rag_model', v));

        // 模型选项
        const providerModels = {
            openai: ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'],
            anthropic: ['claude-3-haiku-20240307', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229']
        };
        const modelOptions = computed(() => providerModels[apiConfig.provider] || []);

        // 自动切换模型
        watch(() => apiConfig.provider, (p) => {
            const models = providerModels[p] || [];
            if (models.length && !models.includes(apiConfig.model)) {
                apiConfig.model = models[0];
            }
        });

        // 提示消息
        const notification = ref(null);
        const showNotification = (msg, type = 'info') => {
            notification.value = { msg, type };
            setTimeout(() => notification.value = null, 4000);
        };

        return {
            activeTab, apiConfig, providerModels, modelOptions, notification,
            showNotification
        };
    },

    // ---------- 模板 ----------
    template: `
    <div>
        <!-- Header -->
        <div class="header">
            <h1>简历 RAG 智能问答系统</h1>
            <p class="subtitle">本地 Embedding + Elasticsearch 混合检索 + 在线 LLM</p>
        </div>

        <!-- Notification -->
        <div v-if="notification" class="error-msg" style="background:#E8F5E9; border-color:#A5D6A7; color:#2E7D32;"
             v-bind:class="{'error-msg': notification.type === 'error'}">
            {{ notification.msg }}
        </div>

        <!-- Tabs -->
        <div class="tabs">
            <button class="tab-btn" :class="{ active: activeTab === 'chat' }" @click="activeTab = 'chat'">
                💬 智能问答
            </button>
            <button class="tab-btn" :class="{ active: activeTab === 'knowledge' }" @click="activeTab = 'knowledge'">
                📚 知识库管理
            </button>
            <button class="tab-btn" :class="{ active: activeTab === 'stats' }" @click="activeTab = 'stats'">
                📊 系统统计
            </button>
        </div>

        <!-- API Key 配置 -->
        <api-key-config
            :api-config="apiConfig"
            :model-options="modelOptions"
        />

        <!-- Chat Tab -->
        <div v-show="activeTab === 'chat'">
            <chat-panel :api-config="apiConfig" @notify="showNotification" />
        </div>

        <!-- Knowledge Tab -->
        <div v-show="activeTab === 'knowledge'">
            <knowledge-panel @notify="showNotification" />
        </div>

        <!-- Stats Tab -->
        <div v-show="activeTab === 'stats'">
            <stats-panel />
        </div>
    </div>
    `
});

// ---------- 组件 ----------

// API Key 配置组件
app.component('api-key-config', {
    props: ['apiConfig', 'modelOptions'],
    template: `
    <div class="card">
        <div class="api-key-row">
            <div class="field">
                <label>🔑 API Key（仅本地存储，不发送到后端）</label>
                <input
                    :type="showKey ? 'text' : 'password'"
                    :value="apiConfig.apiKey"
                    @input="$emit('update:apiConfig', {...apiConfig, apiKey: $event.target.value})"
                    placeholder="sk-xxxx"
                />
            </div>
            <div class="field" style="max-width: 160px;">
                <label>Provider</label>
                <select
                    :value="apiConfig.provider"
                    @change="$emit('update:apiConfig', {...apiConfig, provider: $event.target.value})"
                >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                </select>
            </div>
            <div class="field" style="max-width: 200px;">
                <label>Model</label>
                <select
                    :value="apiConfig.model"
                    @change="$emit('update:apiConfig', {...apiConfig, model: $event.target.value})"
                >
                    <option v-for="m in modelOptions" :key="m" :value="m">{{ m }}</option>
                </select>
            </div>
            <div style="padding-bottom: 4px;">
                <button style="background:transparent; border:none; cursor:pointer; font-size:18px;"
                        @click="showKey = !showKey"
                        :title="showKey ? '隐藏' : '显示'">
                    {{ showKey ? '🙈' : '👁️' }}
                </button>
            </div>
        </div>
    </div>
    `,
    data() { return { showKey: false }; }
});

// 问答面板组件
app.component('chat-panel', {
    props: ['apiConfig'],
    emits: ['notify'],
    template: `
    <div>
        <!-- 答案区域 -->
        <div class="card">
            <div v-if="loading" class="loading">⏳ 检索中...</div>

            <div v-else-if="answer" class="chat-area">
                <div v-if="timing" class="timing-badge">
                    <span>Embedding: {{ timing.embedding_ms }}ms</span>
                    <span>ES检索: {{ timing.search_ms }}ms</span>
                    <span>LLM生成: {{ timing.llm_s }}s</span>
                </div>
                <div class="answer-box">{{ answer }}</div>

                <!-- 引用来源 -->
                <div v-if="sources && sources.length > 0" class="source-list">
                    <h4>📄 引用来源 ({{ sources.length }}条)</h4>
                    <div
                        v-for="(src, idx) in sources" :key="idx"
                        class="source-item"
                        :class="{ expanded: expandedIdx === idx }"
                        @click="expandedIdx = expandedIdx === idx ? -1 : idx"
                    >
                        <div class="source-header">
                            <span class="source-file">📄 {{ src.file_name }}</span>
                            <span class="source-score">相似度: {{ src.score }}</span>
                        </div>
                        <div class="source-content">{{ src.content }}</div>
                    </div>
                </div>
            </div>

            <div v-else class="answer-box empty">
                输入问题开始智能问答之旅
            </div>

            <!-- 错误 -->
            <div v-if="error" class="error-msg">{{ error }}</div>
        </div>

        <!-- 问题输入 -->
        <div class="card">
            <div class="question-row">
                <textarea
                    v-model="question"
                    placeholder="请输入问题，例如：张成都有哪些Python开发经验？"
                    @keydown.enter.ctrl="doQuery"
                    :disabled="loading"
                ></textarea>
                <button class="btn-send" @click="doQuery" :disabled="loading || !question.trim() || !apiConfig.apiKey">
                    {{ loading ? '检索中...' : '发送' }}
                </button>
            </div>
        </div>
    </div>
    `,
    data() {
        return {
            question: '',
            answer: '',
            sources: [],
            timing: null,
            loading: false,
            error: '',
            expandedIdx: -1
        };
    },
    methods: {
        async doQuery() {
            if (!this.question.trim() || !this.apiConfig.apiKey) return;

            this.loading = true;
            this.error = '';
            this.answer = '';
            this.sources = [];
            this.expandedIdx = -1;

            try {
                const result = await ApiClient.query({
                    question: this.question,
                    api_key: this.apiConfig.apiKey,
                    provider: this.apiConfig.provider,
                    model: this.apiConfig.model,
                    top_k: 5
                });
                this.answer = result.answer;
                this.sources = result.sources || [];
                this.timing = result.timing || null;
            } catch (e) {
                this.error = e.message;
                this.$emit('notify', e.message, 'error');
            } finally {
                this.loading = false;
            }
        }
    }
});

// 知识库管理面板组件
app.component('knowledge-panel', {
    emits: ['notify'],
    template: `
    <div>
        <!-- 上传区域 -->
        <div class="card">
            <h3 style="margin-bottom: 16px;">📤 上传文档</h3>
            <div
                class="upload-zone"
                :class="{ 'drag-over': isDragOver }"
                @dragover.prevent="isDragOver = true"
                @dragleave="isDragOver = false"
                @drop.prevent="handleDrop"
                @click="$refs.fileInput.click()"
            >
                <button class="btn-upload">📄 选择文件上传</button>
                <p>支持 PDF / TXT / CSV 文件，最大 10MB</p>
                <input
                    type="file"
                    ref="fileInput"
                    accept=".pdf,.txt,.csv"
                    @change="handleFileSelect"
                    style="display: none"
                />
            </div>

            <!-- 进度条 -->
            <div v-if="uploading" class="progress-bar">
                <div class="progress-fill" :style="{ width: uploadProgress + '%' }"></div>
            </div>
            <div v-if="uploading" style="text-align:center; font-size:13px; color:var(--text-secondary);">
                正在上传并向量化...
            </div>
        </div>

        <!-- 文档列表 -->
        <div class="card">
            <h3 style="margin-bottom: 16px;">📚 已入库文档 ({{ documents.length }})</h3>
            <div v-if="loading" class="loading">加载中...</div>
            <div v-else-if="documents.length === 0" class="empty">暂无文档，请上传文件</div>
            <table v-else class="doc-table">
                <thead>
                    <tr>
                        <th>文件名</th>
                        <th>类型</th>
                        <th>Chunks</th>
                        <th>入库时间</th>
                        <th>操作</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="doc in documents" :key="doc.file_name">
                        <td class="doc-name">📄 {{ doc.file_name }}</td>
                        <td><span class="doc-type">{{ doc.file_type }}</span></td>
                        <td>{{ doc.chunk_count }}</td>
                        <td>{{ formatTime(doc.upload_time) }}</td>
                        <td>
                            <button class="btn-delete" @click="deleteDoc(doc.file_name)">删除</button>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    `,
    data() {
        return {
            documents: [],
            loading: false,
            uploading: false,
            uploadProgress: 0,
            isDragOver: false
        };
    },
    mounted() { this.loadDocuments(); },
    methods: {
        async loadDocuments() {
            this.loading = true;
            try {
                const data = await ApiClient.listDocuments();
                this.documents = data.documents || [];
            } catch (e) {
                this.$emit('notify', '加载文档列表失败: ' + e.message, 'error');
            }
            this.loading = false;
        },
        handleFileSelect(e) {
            const file = e.target.files[0];
            if (file) this.uploadFile(file);
        },
        handleDrop(e) {
            this.isDragOver = false;
            const file = e.dataTransfer.files[0];
            if (file) this.uploadFile(file);
        },
        async uploadFile(file) {
            const allowed = ['.pdf', '.txt', '.csv'];
            const ext = '.' + file.name.split('.').pop().toLowerCase();
            if (!allowed.includes(ext)) {
                this.$emit('notify', '仅支持 PDF / TXT / CSV 文件', 'error');
                return;
            }
            if (file.size > 10 * 1024 * 1024) {
                this.$emit('notify', '文件大小超过 10MB 限制', 'error');
                return;
            }

            this.uploading = true;
            this.uploadProgress = 0;

            // 模拟进度
            const timer = setInterval(() => {
                if (this.uploadProgress < 90) this.uploadProgress += 10;
            }, 300);

            const formData = new FormData();
            formData.append('file', file);

            try {
                const result = await ApiClient.uploadDocument(formData);
                clearInterval(timer);
                this.uploadProgress = 100;
                this.$emit('notify', `上传成功！创建了 ${result.chunks_created} 个 chunks`);
                this.loadDocuments();
            } catch (e) {
                clearInterval(timer);
                this.$emit('notify', '上传失败: ' + e.message, 'error');
            }
            this.uploading = false;
        },
        async deleteDoc(fileName) {
            if (!confirm(`确定要删除文档「${fileName}」吗？所有相关 chunks 都会被删除。`)) {
                return;
            }
            try {
                const result = await ApiClient.deleteDocument(fileName);
                this.$emit('notify', `删除成功，删除了 ${result.deleted_chunks} 个 chunks`);
                this.loadDocuments();
            } catch (e) {
                this.$emit('notify', '删除失败: ' + e.message, 'error');
            }
        },
        formatTime(t) {
            if (!t) return '-';
            return new Date(t).toLocaleString('zh-CN');
        }
    }
});

// 系统统计面板
app.component('stats-panel', {
    template: `
    <div>
        <div v-if="loading" class="loading">加载统计信息...</div>
        <div v-else>
            <!-- 统计卡片 -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ stats.total_chunks }}</div>
                    <div class="stat-label">总 Chunks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ stats.total_documents }}</div>
                    <div class="stat-label">总文档数</div>
                </div>
            </div>

            <!-- 健康检查 -->
            <div class="card">
                <h3 style="margin-bottom: 12px;">💚 系统状态</h3>
                <div style="font-size:14px; display:flex; gap:20px;">
                    <span>Embedding: <strong :style="{ color: health.embedding_loaded ? 'var(--success)' : 'var(--danger)' }">
                        {{ health.embedding_loaded ? '✓ 已加载' : '✗ 未加载' }}
                    </strong></span>
                    <span>ES: <strong :style="{ color: health.es_connected ? 'var(--success)' : 'var(--danger)' }">
                        {{ health.es_connected ? '✓ 已连接' : '✗ 未连接' }}
                    </strong></span>
                </div>
            </div>

            <!-- 文档详情 -->
            <div class="card" v-if="stats.documents && stats.documents.length > 0">
                <h3 style="margin-bottom: 12px;">📋 文档详情</h3>
                <table class="doc-table">
                    <thead>
                        <tr>
                            <th>文件名</th>
                            <th>类型</th>
                            <th>Chunks</th>
                            <th>入库时间</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="doc in stats.documents" :key="doc.file_name">
                            <td class="doc-name">📄 {{ doc.file_name }}</td>
                            <td><span class="doc-type">{{ doc.file_type }}</span></td>
                            <td>{{ doc.chunk_count }}</td>
                            <td>{{ formatTime(doc.upload_time) }}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    `,
    data() {
        return {
            stats: { total_chunks: 0, total_documents: 0, documents: [] },
            health: { embedding_loaded: false, es_connected: false },
            loading: true
        };
    },
    mounted() { this.load(); },
    methods: {
        async load() {
            this.loading = true;
            try {
                const [stats, health] = await Promise.all([
                    ApiClient.getStats(),
                    ApiClient.healthCheck()
                ]);
                this.stats = stats;
                this.health = health;
            } catch (e) {
                console.error('加载统计失败:', e);
            }
            this.loading = false;
        },
        formatTime(t) {
            if (!t) return '-';
            return new Date(t).toLocaleString('zh-CN');
        }
    }
});

// 挂载应用
app.mount('#app');
