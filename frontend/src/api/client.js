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
        const res = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        return ApiClient._handleResponse(res);
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
