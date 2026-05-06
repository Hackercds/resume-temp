#!/bin/bash
# 简历 RAG 智能问答系统 - 优雅退出脚本
# 面试点：捕获 SIGTERM，关闭连接后才能退出

APP_PID="${1:-$(pgrep -f 'python main.py')}"

if [ -z "$APP_PID" ]; then
    echo "未找到运行中的服务进程"
    exit 0
fi

echo "正在优雅关闭服务 (PID: $APP_PID)..."

# 发送 SIGTERM 信号
kill -TERM $APP_PID 2>/dev/null || true

# 等待最多 30 秒
for i in $(seq 1 30); do
    if ! kill -0 $APP_PID 2>/dev/null; then
        echo "服务已优雅退出"
        exit 0
    fi
    echo "等待服务退出... ($i/30)"
    sleep 1
done

# 强制退出
echo "服务超时未退出，强制终止"
kill -KILL $APP_PID 2>/dev/null || true
