/**
 * HTTP 请求封装层
 * 统一处理错误、统一响应格式解析
 */
// 自动检测 API 地址：
//   nginx 代理模式 → 相对路径 ''（同域 /api/ → nginx → backend）
//   独立前端模式 → 用 ?api= 参数或默认同域
const API_BASE = '';

class ApiClient {
    /**
     * POST /api/query - RAG 问答
     */
    static async query(body) {
        const payload = {
            question: body.question,
            api_key: body.api_key,
            provider: body.provider,
            model: body.model,
            base_url: body.base_url,
            top_k: body.top_k || 5,
            history: body.history || [],
            session_id: body.session_id || null,
            retrieve_full_doc: body.retrieve_full_doc || false
        };
        const res = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        return ApiClient._handleResponse(res);
    }

    /**
     * POST /api/query/stream - 流式 RAG 问答
     */
    static async queryStream(body, onToken, onDone, onError) {
        const payload = {
            question: body.question,
            api_key: body.api_key,
            provider: body.provider,
            model: body.model,
            base_url: body.base_url,
            top_k: body.top_k || 5,
            history: body.history || [],
            session_id: body.session_id || null,
            retrieve_full_doc: body.retrieve_full_doc || false
        };

        // 兜底：即使后端漏过滤，前端再清一遍 {{retrieve_full_doc:...}} 标记
        const STRIP_RETRIEVE = /\{\{retrieve_full_doc:[^}]+\}\}\s*/g;

        const res = await fetch(`${API_BASE}/api/query/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok || !res.body) {
            const data = await res.json().catch(() => ({}));
            throw new Error(data.message || `请求失败 (${res.status})`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || !trimmed.startsWith('data: ')) continue;
                const dataStr = trimmed.slice(6);
                if (dataStr === '[DONE]') continue;
                try {
                    const event = JSON.parse(dataStr);
                    if (event.type === 'token') {
                        // 流式：过滤 retrieve_full_doc 标记
                        const cleaned = (event.content || '').replace(STRIP_RETRIEVE, '');
                        if (cleaned && onToken) onToken(cleaned);
                    } else if (event.type === 'error') {
                        const err = new Error(event.message || '请求失败');
                        err.suggestion = event.suggestion || '';
                        err.emptyRetrieval = !!event.empty_retrieval;
                        err.traceId = event.trace_id || '';
                        if (onError) onError(err);
                    } else if (event.type === 'done') {
                        // done：兜底清理 answer 中的残留标记
                        if (event.answer) {
                            event.answer = event.answer.replace(STRIP_RETRIEVE, '').trim();
                        }
                        if (onDone) onDone(event);
                    }
                } catch (e) { /* 忽略解析失败的行 */ }
            }
        }
    }

    /**
     * POST /api/knowledge/upload - 上传文档
     * onProgress: 0-100 上传进度
     */
    static async uploadDocument(formData, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            if (typeof onProgress === 'function') {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
                });
            }
            xhr.addEventListener('load', () => {
                try {
                    const data = JSON.parse(xhr.responseText);
                    if (data.code === 0) resolve(data.data);
                    else reject(new Error(data.message || `上传失败 (${xhr.status})`));
                } catch (e) {
                    reject(new Error('响应解析失败'));
                }
            });
            xhr.addEventListener('error', () => reject(new Error('网络错误')));
            xhr.addEventListener('abort', () => reject(new Error('已取消')));
            xhr.open('POST', `${API_BASE}/api/knowledge/upload`);
            xhr.send(formData);
        });
    }

    /**
     * GET /api/knowledge/documents - 文档列表
     */
    static async listDocuments() {
        const res = await fetch(`${API_BASE}/api/knowledge/documents`);
        return ApiClient._handleResponse(res);
    }

    /**
     * DELETE /api/knowledge/documents/{name} - 删除文档
     */
    static async deleteDocument(fileName) {
        const res = await fetch(`${API_BASE}/api/knowledge/documents/${encodeURIComponent(fileName)}`, {
            method: 'DELETE'
        });
        return ApiClient._handleResponse(res);
    }

    /**
     * GET /api/stats - 知识库统计
     */
    static async getStats() {
        const res = await fetch(`${API_BASE}/api/stats`);
        return ApiClient._handleResponse(res);
    }

    /**
     * GET /api/config - 公开配置（LLM 预设、默认 Key 状态）
     */
    static async getPublicConfig() {
        const res = await fetch(`${API_BASE}/api/config`);
        return ApiClient._handleResponse(res);
    }

    /**
     * GET /health - 健康检查
     */
    static async healthCheck() {
        const res = await fetch(`${API_BASE}/health`);
        return ApiClient._handleResponse(res);
    }

    /**
     * 统一处理响应
     */
    static async _handleResponse(res) {
        const data = await res.json();
        if (data.code === 0) {
            return data.data;
        }
        throw new Error(data.message || `请求失败 (${data.code})`);
    }
}

export { ApiClient };
