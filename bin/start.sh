#!/bin/bash
# 简历 RAG 智能问答系统 - 一键启动脚本
# 自动检测依赖、ES 连接、模型加载
set -e

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$APP_DIR"

echo "=========================================="
echo "  简历 RAG 智能问答系统 v1.1.0"
echo "=========================================="

# 1. Python 版本检查
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v $PYTHON_BIN >/dev/null 2>&1; then
    echo "❌ Python 未安装"
    exit 1
fi
PY_VERSION=$($PYTHON_BIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "✅ Python $PY_VERSION"

# 2. 创建必要目录
mkdir -p log models uploads

# 3. 检查依赖
if ! $PYTHON_BIN -c "import fastapi, elasticsearch, sentence_transformers" 2>/dev/null; then
    echo "⚠️  依赖未安装，正在执行 pip install -r requirements.txt..."
    pip install -r requirements.txt
fi
echo "✅ 依赖已就绪"

# 4. ES 连接检查
ES_HOST="${ES_HOST:-http://localhost:9200}"
echo "检查 ES 连接: $ES_HOST"
if curl -sf "$ES_HOST" -o /dev/null --max-time 5 2>/dev/null; then
    echo "✅ ES 已连接"
else
    echo "⚠️  ES 未连接 ($ES_HOST)"
    echo "   提示: 可通过 ES_HOST=http://your-es:9200 启动"
    echo "   或先启动 ES: docker run -d -p 9200:9200 -e discovery.type=single-node elasticsearch:8.12.0"
fi

# 5. 启动服务
export APP_MODE="${APP_MODE:-release}"
export APP_PORT="${APP_PORT:-8080}"
echo "=========================================="
echo "启动模式: $APP_MODE"
echo "监听端口: $APP_PORT"
echo "前端入口: frontend/index.html"
echo "API 文档: http://localhost:$APP_PORT/docs"
echo "=========================================="

exec $PYTHON_BIN main.py
