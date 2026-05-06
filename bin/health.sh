#!/bin/bash
# 简历 RAG 智能问答系统 - 健康检查
APP_PORT="${APP_PORT:-8080}"
HEALTH_URL="http://localhost:${APP_PORT}/health"

echo "健康检查: $HEALTH_URL"

response=$(curl -sf "$HEALTH_URL" 2>&1)
if [ $? -eq 0 ]; then
    echo "✓ 服务正常"
    echo "$response"
    exit 0
else
    echo "✗ 服务异常"
    exit 1
fi
