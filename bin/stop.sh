#!/bin/bash
# 简历 RAG 智能问答系统 - 停止脚本
APP_NAME="${APP_NAME:-resume-rag-service}"

echo "正在停止 $APP_NAME..."

# 尝试 Docker 方式停止
if docker ps -q -f name=$APP_NAME 2>/dev/null | grep -q .; then
    docker stop $APP_NAME 2>/dev/null || true
    docker rm $APP_NAME 2>/dev/null || true
    echo "Docker 容器已停止"
fi

# 尝试进程方式停止
PID=$(pgrep -f "python main.py" 2>/dev/null || true)
if [ -n "$PID" ]; then
    kill -TERM $PID 2>/dev/null || true
    echo "进程已停止 (PID: $PID)"
fi

echo "服务已停止"
