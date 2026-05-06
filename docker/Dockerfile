# 简历 RAG 智能问答系统 - Docker 镜像
# 多阶段构建：减小镜像体积
# 遵循 MICRO_SERVICE_SPEC.md 规范

FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

FROM python:3.11-slim
# 这里没搞digest锁，个人展示，不麻烦了
WORKDIR /app

# 从构建阶段复制 Python 包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 复制应用代码
COPY main.py .
COPY config/ ./config/
COPY bin/ ./bin/
COPY scripts/ ./scripts/
COPY internal/ ./internal/
COPY frontend/ ./frontend/

# 创建非 root 用户
RUN useradd -m app && chown -R app:app /app && \
    chmod +x /app/bin/start.sh /app/bin/stop.sh 2>/dev/null || true

# 创建必要的目录
RUN mkdir -p /app/log /app/models /app/uploads && \
    chown -R app:app /app/log /app/models /app/uploads

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "main.py"]
