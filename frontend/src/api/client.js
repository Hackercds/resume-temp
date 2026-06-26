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
                        if (onToken) onToken(event.content);
                    } else if (event.type === 'done') {
                        if (onDone) onDone(event);
                    } else if (event.type === 'error') {
                        if (onError) onError(new Error(event.message));
                    }
                } catch (e) { /* 忽略解析失败的行 */ }
            }
        }
    }

    /**
     * POST /api/knowledge/upload - 上传文档
     */
    static async uploadDocument(formData) {
        const res = await fetch(`${API_BASE}/api/knowledge/upload`, {
            method: 'POST',
            body: formData  // FormData, 不设 Content-Type 让浏览器自动加 boundary
        });
        return ApiClient._handleResponse(res);
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
