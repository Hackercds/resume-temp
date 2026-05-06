#!/bin/bash
# ============================================================
# 简历 RAG 智能问答系统 - 一键部署脚本
# ============================================================
# 使用方式：
#   # 全栈部署（自动启动 ES）
#   ./bin/deploy.sh
#
#   # 跳过 ES（本机已有 ES 时使用）
#   SKIP_ES=true ES_HOST=http://192.168.3.184:9200 ./bin/deploy.sh
#
#   # 部署到远程 Docker
#   DOCKER_HOST=tcp://192.168.3.184:2375 ./bin/deploy.sh
#
#   # 自定义所有端口
#   BACKEND_PORT=9090 FRONTEND_PORT=8080 ES_PORT=19200 ./bin/deploy.sh
# ============================================================
set -e

# ---------- 可配置变量（全部支持环境变量覆盖）----------
PROJECT_NAME="${PROJECT_NAME:-resume-rag-service}"
NETWORK="${NETWORK:-rag-network}"

# 容器名（与 docker-compose.yml / Jenkinsfile 统一）
CN_ES="rag-es"
CN_BACKEND="rag-backend"
CN_FRONTEND="rag-frontend"

# 主机侧端口映射（容器内部端口固定）
BACKEND_PORT="${BACKEND_PORT:-8080}"      # 主机 → 容器:8080
FRONTEND_PORT="${FRONTEND_PORT:-5000}"    # 主机 → 容器:80
ES_PORT="${ES_PORT:-9200}"               # 主机 → 容器:9200

# ES 配置
SKIP_ES="${SKIP_ES:-false}"              # true = 不启动 ES 容器
ES_HOST="${ES_HOST:-http://elasticsearch:9200}"  # 后端连哪个 ES
ES_MEMORY="${ES_MEMORY:-2g}"
ES_IMAGE="docker.elastic.co/elasticsearch/elasticsearch:8.12.0"

# 后端配置
BACKEND_MEMORY="${BACKEND_MEMORY:-2g}"
APP_MODE="${APP_MODE:-release}"

# 构建标签
BUILD_TAG="${BUILD_TAG:-latest}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  简历 RAG 智能问答系统 - 部署脚本"
echo "  目标 Docker: ${DOCKER_HOST:-localhost}"
echo "  SKIP_ES: $SKIP_ES"
echo "  ES_HOST: $ES_HOST"
echo "=========================================="

# 1. 检查 Docker 连接
echo ">>> 检查 Docker 连接"
docker info > /dev/null 2>&1 || {
    echo "✗ 无法连接 Docker。请检查："
    echo "  1. Docker 是否已启动"
    echo "  2. DOCKER_HOST 环境变量是否正确"
    echo "  3. Docker TCP 端口是否开放（默认 2375）"
    exit 1
}
echo "✓ Docker 连接正常"

# 2. 构建镜像
echo ""
echo ">>> 构建镜像"
docker build -t ${CN_BACKEND}:${BUILD_TAG} -f docker/Dockerfile .
docker build -t ${CN_FRONTEND}:${BUILD_TAG} -f docker/Dockerfile.frontend .

# 3. 创建网络
echo ""
echo ">>> 创建网络"
docker network create ${NETWORK} 2>/dev/null || true

# 4. 停止旧容器
echo ""
echo ">>> 停止旧容器"
docker rm -f ${CN_ES} \
           ${CN_BACKEND} \
           ${CN_FRONTEND} 2>/dev/null || true
sleep 2

# 5. 启动 ES（可跳过）
if [ "$SKIP_ES" = "true" ]; then
    echo ""
    echo ">>> [SKIP] 跳过 ES 启动，使用外部 ES: $ES_HOST"

    # 检查外部 ES 是否可达
    if curl -sf "${ES_HOST}/_cluster/health" 2>/dev/null | grep -q '"status"'; then
        echo "✓ 外部 ES 连接正常: $ES_HOST"
    else
        echo "⚠ 警告：无法连接到 $ES_HOST，后端可能无法正常工作"
        echo "  继续部署（10秒后取消请按 Ctrl+C）..."
        sleep 10
    fi
else
    echo ""
    echo ">>> 启动 Elasticsearch"
    docker run -d \
        --name ${CN_ES} \
        --network ${NETWORK} \
        --network-alias elasticsearch \
        -p ${ES_PORT}:9200 \
        -e "discovery.type=single-node" \
        -e "xpack.security.enabled=false" \
        -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
        --memory=${ES_MEMORY} \
        ${ES_IMAGE}

    # 等待 ES 就绪
    echo ">>> 等待 ES 就绪..."
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:${ES_PORT}/_cluster/health" 2>/dev/null | grep -q '"status"'; then
            echo "✓ ES 已就绪"
            break
        fi
        printf "."
        sleep 3
    done
    echo ""
fi

# 6. 启动后端
echo ""
echo ">>> 启动后端 (主机端口: $BACKEND_PORT)"
docker run -d \
    --name ${CN_BACKEND} \
    --network ${NETWORK} \
    --network-alias backend \
    -p ${BACKEND_PORT}:8080 \
    -e ES_HOST=${ES_HOST} \
    -e APP_MODE=${APP_MODE} \
    -e APP_PORT=8080 \
    --memory=${BACKEND_MEMORY} \
    --restart=unless-stopped \
    ${CN_BACKEND}:${BUILD_TAG}

# 7. 启动前端
echo ""
echo ">>> 启动前端 (主机端口: $FRONTEND_PORT)"
docker run -d \
    --name ${CN_FRONTEND} \
    --network ${NETWORK} \
    -p ${FRONTEND_PORT}:80 \
    --memory=256m \
    --restart=unless-stopped \
    ${CN_FRONTEND}:${BUILD_TAG}

# 8. 健康检查
echo ""
echo ">>> 等待服务就绪..."
sleep 10

BACKEND_OK=0
FRONTEND_OK=0

for i in $(seq 1 30); do
    if [ $BACKEND_OK -eq 0 ]; then
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            http://localhost:${BACKEND_PORT}/health 2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then
            echo "✓ 后端就绪 (:${BACKEND_PORT})"
            BACKEND_OK=1
        fi
    fi

    if [ $FRONTEND_OK -eq 0 ]; then
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            http://localhost:${FRONTEND_PORT} 2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then
            echo "✓ 前端就绪 (:${FRONTEND_PORT})"
            FRONTEND_OK=1
        fi
    fi

    if [ $BACKEND_OK -eq 1 ] && [ $FRONTEND_OK -eq 1 ]; then
        break
    fi
    sleep 3
done

echo ""
echo "=========================================="
echo "  部署完成！"
echo "  后端 API: http://localhost:${BACKEND_PORT}"
echo "  Swagger:  http://localhost:${BACKEND_PORT}/docs"
echo "  前端页面: http://localhost:${FRONTEND_PORT}"
echo "  ES 状态:  $( [ "$SKIP_ES" = "true" ] && echo "外部 $ES_HOST" || echo "容器 :${ES_PORT}" )"
echo "=========================================="
echo ""
docker ps --filter "name=${PROJECT_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
