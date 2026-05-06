# ============================================================
# 简历 RAG 智能问答系统 - Docker 镜像
# ============================================================
# 分层策略：轻量依赖（变化少）→ 重量依赖（ML）→ 源码（变化频）
# 配合 --cache-from 使 Jenkins 构建时层缓存生效，避免重复下载
#
# 使用：
#   docker build -t resume-rag -f docker/Dockerfile .
#   docker run -d --name rag -p 8080:8080 -e ES_HOST=http://ES地址:9200 resume-rag
# ============================================================

FROM python:3.11-slim AS builder
WORKDIR /build

# ---- Layer 1: 轻量依赖（fastapi/es 等，几乎不变 → 层缓存）----
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# ---- Layer 2: 重量 ML 依赖（torch/sentence-transformers）----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# ---- Layer 3: 预下载 Embedding 模型（避免首次请求等几十秒）----
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-zh-v1.5', cache_folder='/build/models')" \
    2>/dev/null || echo "模型预下载跳过（非关键）"

# ============================================================
FROM python:3.11-slim
WORKDIR /app

# 从构建阶段复制 Python 包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 从构建阶段复制预下载的模型
COPY --from=builder /build/models /app/models

# 复制应用代码
COPY main.py .
COPY config/ ./config/
COPY internal/ ./internal/
COPY frontend/ ./frontend/

# 非 root 用户
RUN useradd -m app && chown -R app:app /app
RUN mkdir -p /app/log /app/uploads && chown -R app:app /app/log /app/uploads

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "main.py"]
