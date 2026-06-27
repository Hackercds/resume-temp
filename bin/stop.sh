#!/bin/bash
# 简历 RAG 智能问答系统 - 优雅停止脚本
# 兼容 Docker 容器与本地 python 进程

APP_NAME="${APP_NAME:-resume-rag-service}"
APP_PID="${1:-$(pgrep -f 'python main.py')}"

echo "正在停止 $APP_NAME..."

# 尝试 Docker 方式停止
if docker ps -q -f name=$APP_NAME 2>/dev/null | grep -q .; then
    docker stop $APP_NAME 2>/dev/null || true
    docker rm $APP_NAME 2>/dev/null || true
    echo "Docker 容器已停止"
fi

# 尝试进程方式停止（SIGTERM 优雅退出）
if [ -z "$APP_PID" ]; then
    APP_PID=$(pgrep -f "python main.py" 2>/dev/null || true)
fi
if [ -n "$APP_PID" ]; then
    kill -TERM $APP_PID 2>/dev/null || true
    echo "已发送 SIGTERM，等待优雅退出 (PID: $APP_PID)..."

    # 最多等待 30 秒
    for i in $(seq 1 30); do
        if ! kill -0 $APP_PID 2>/dev/null; then
            echo "服务已优雅退出"
            exit 0
        fi
        sleep 1
    done

    # 强制终止
    echo "超时未退出，强制终止"
    kill -KILL $APP_PID 2>/dev/null || true
fi

echo "服务已停止"