#!/bin/bash
# 简历 RAG 智能问答系统 - 启动脚本
set -e

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$APP_DIR"

echo "=========================================="
echo "  简历 RAG 智能问答系统 v1.0.0"
echo "=========================================="

# 创建必要目录
mkdir -p log models uploads

# 启动服务
export APP_MODE="${APP_MODE:-release}"
export APP_PORT="${APP_PORT:-8080}"

echo "启动模式: $APP_MODE"
echo "监听端口: $APP_PORT"
echo "=========================================="

exec python main.py
