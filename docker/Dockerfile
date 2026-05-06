# ============================================================
# 简历 RAG 智能问答系统 - Docker 镜像
# ============================================================
# 多阶段构建，最小化镜像体积
# 遵循 MICRO_SERVICE_SPEC.md 规范
#
# === 使用方式 ===
#
# 1. 构建：
#    docker build -t resume-rag -f docker/Dockerfile .
#
# 2. 启动（连已有 ES）：
#    docker run -d --name rag \
#      -p 8080:8080 \
#      -e ES_HOST=http://你的ES地址:9200 \
#      resume-rag
#
# 3. 启动（需要启动 ES 容器）：
#    docker network create rag-net
#    docker run -d --name rag-es --network rag-net --network-alias elasticsearch \
#      -p 9200:9200 \
#      -e "discovery.type=single-node" -e "xpack.security.enabled=false" \
#      docker.elastic.co/elasticsearch/elasticsearch:8.12.0
#    docker run -d --name rag --network rag-net \
#      -p 8080:8080 \
#      -e ES_HOST=http://elasticsearch:9200 \
#      resume-rag
#
# 4. 健康检查：
#    curl http://localhost:8080/health
#
# 5. 修改应用配置：
#    编辑 config/config.yaml 后重新构建镜像
# ============================================================

FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

FROM python:3.11-slim
# 个人展示项目，未使用 digest 锁
WORKDIR /app

# 从构建阶段复制 Python 包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 复制应用代码（config.yaml 在 config/ 目录下，是唯一配置源）
COPY main.py .
COPY config/ ./config/
COPY bin/ ./bin/
COPY internal/ ./internal/
COPY frontend/ ./frontend/

# 创建非 root 用户
RUN useradd -m app && chown -R app:app /app && \
    chmod +x /app/bin/start.sh 2>/dev/null || true

# 创建运行时目录
RUN mkdir -p /app/log /app/models /app/uploads && \
    chown -R app:app /app/log /app/models /app/uploads

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "main.py"]
