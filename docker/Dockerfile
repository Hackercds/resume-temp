# ============================================================
# 简历 RAG 智能问答系统 - Docker 镜像
# ============================================================
# 使用：
#   # 基本构建
#   docker build -t resume-rag -f docker/Dockerfile .
#
#   # 挂代理（国内网络加速 pip 和 huggingface）
#   docker build --build-arg HTTP_PROXY=http://192.168.3.2:20172 \
#                --build-arg HTTPS_PROXY=http://192.168.3.2:20172 \
#                -t resume-rag -f docker/Dockerfile .
#
#   # 切换镜像源
#   docker build --build-arg PIP_INDEX=https://mirrors.aliyun.com/pypi/simple \
#                -t resume-rag -f docker/Dockerfile .
# ============================================================

FROM python:3.11-slim AS builder
WORKDIR /build

# ---------- 构建参数 ----------
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# 代理 + 模型缓存路径（与运行时一致，避免重复下载）
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    SENTENCE_TRANSFORMERS_HOME=/build/models

# ---- Layer 1: 轻量依赖 ----
COPY requirements-base.txt .
RUN if [ -n "${HTTP_PROXY}" ]; then \
        pip install --proxy "${HTTP_PROXY}" --no-cache-dir -r requirements-base.txt \
            -i ${PIP_INDEX} --trusted-host ${PIP_TRUSTED_HOST}; \
    else \
        pip install --no-cache-dir -r requirements-base.txt \
            -i ${PIP_INDEX} --trusted-host ${PIP_TRUSTED_HOST}; \
    fi

# ---- Layer 2: ML 依赖（torch/sentence-transformers ~2GB）----
COPY requirements.txt .
RUN if [ -n "${HTTP_PROXY}" ]; then \
        pip install --proxy "${HTTP_PROXY}" --no-cache-dir -r requirements.txt \
            -i ${PIP_INDEX} --trusted-host ${PIP_TRUSTED_HOST}; \
    else \
        pip install --no-cache-dir -r requirements.txt \
            -i ${PIP_INDEX} --trusted-host ${PIP_TRUSTED_HOST}; \
    fi

# ---- Layer 3: 预下载 BGE 模型（走 SENTENCE_TRANSFORMERS_HOME 路径）----
RUN echo ">>> 预下载 BGE-small-zh 模型到 ${SENTENCE_TRANSFORMERS_HOME} ..." && \
    python -c "import os; p=os.environ.get('HTTP_PROXY',''); print('代理:', p if p else '无'); print('模型路径:', os.environ.get('SENTENCE_TRANSFORMERS_HOME','未设置')); from sentence_transformers import SentenceTransformer; m=SentenceTransformer('BAAI/bge-small-zh-v1.5'); print('下载成功, 维度:', m.get_sentence_embedding_dimension())" \
    && echo ">>> 模型文件:" && ls -la ${SENTENCE_TRANSFORMERS_HOME}/ \
    || echo ">>> [WARN] 模型预下载失败，首次请求时将自动下载（可能需要几十秒）"

# ============================================================
FROM python:3.11-slim
WORKDIR /app

# sentence-transformers 模型查找路径（与 config.yaml 中 cache_folder 一致）
ENV SENTENCE_TRANSFORMERS_HOME=/app/models

# 复制 Python 包 + 预下载模型
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /build/models /app/models

# 应用代码
COPY main.py .
COPY config/ ./config/
COPY internal/ ./internal/
COPY frontend/ ./frontend/

RUN useradd -m app && chown -R app:app /app
RUN mkdir -p /app/log /app/uploads && chown -R app:app /app/log /app/uploads

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "main.py"]
