pipeline {
    agent any

    // ============================================================
    // 参数说明：
    //   SKIP_ES=true → 不启动 ES 容器，使用外部 ES_HOSTe
    //   端口冲突时 → 修改 BACKEND_PORT / FRONTEND_PORT / ES_PORT
    //   所有默认端口与 docker-compose.yml / deploy.sh 保持一致
    // ============================================================
    parameters {
        booleanParam(name: 'SKIP_ES', defaultValue: true,
                     description: '已有 ES 时选 true，不启动 ES 容器。端口不会被占用。')
        string(name: 'ES_HOST', defaultValue: 'http://maco.hackercd.cn:9200',
               description: 'ES 地址。SKIP_ES=true 时填外部实际地址，容器内走弹性网络。')
        string(name: 'BACKEND_PORT', defaultValue: '8080',
               description: '后端主机端口（避免与已有服务冲突可改为 9090 等）')
        string(name: 'FRONTEND_PORT', defaultValue: '5000',
               description: '前端主机端口')
        string(name: 'ES_PORT', defaultValue: '9200',
               description: 'ES 主机端口。SKIP_ES=true 时忽略此参数')
        string(name: 'DEPLOY_HOST', defaultValue: 'maco.hackercd.cn',
               description: '部署目标 IP（留空 = localhost）')
        string(name: 'DOCKER_HOST_URI', defaultValue: 'tcp://192.168.3.188:2375',
               description: '远程 Docker TCP 地址，如 tcp://192.168.3.188:2375。留空 = 本地 Docker。')
    }

    environment {
        PROJECT_NAME = 'resume-rag-service'
        NETWORK = 'rag-network'
        APP_MODE = 'release'
        // 容器名前缀统一与 docker-compose 一致
        CN_ES = 'rag-es'
        CN_BACKEND = 'rag-backend'
        CN_FRONTEND = 'rag-frontend'
        // 有外部 DOCKER_HOST 时才设置
        DOCKER_HOST = "${params.DOCKER_HOST_URI}"
        HTTP_PROXY = 'http://192.168.3.2:20172'
        //这个http_proxy是因为，我这边下载huggingface模型的原因，你们可以去掉的，另外就是国内的话modelscope也挺好
    }

    stages {
        stage('信息确认') {
            steps {
                script {
                    echo "============================================"
                    echo "  部署参数"
                    echo "  SKIP_ES:       ${params.SKIP_ES}"
                    echo "  ES_HOST:       ${params.ES_HOST}"
                    echo "  后端端口:       ${params.BACKEND_PORT}"
                    echo "  前端端口:       ${params.FRONTEND_PORT}"
                    echo "  ES 端口:        ${params.ES_PORT}"
                    echo "  目标 Docker:    ${params.DOCKER_HOST_URI ?: '本地'}"
                    echo "============================================"
                }
            }
        }

        stage('镜像构建') {
            steps {
                sh '''
                    # 拉取上次镜像做层缓存
                    docker pull ${PROJECT_NAME}-backend:latest 2>/dev/null || true
                    docker pull ${PROJECT_NAME}-frontend:latest 2>/dev/null || true

                    # 镜像源 + 代理（Jenkins 环境变量或参数传入）
                    PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
                    PIP_HOST="${PIP_HOST:-pypi.tuna.tsinghua.edu.cn}"
                    HTTP_PROXY="${HTTP_PROXY:-}"

                    # 构造代理 --build-arg
                    PROXY_ARGS=""
                    if [ -n "${HTTP_PROXY}" ]; then
                        echo ">>> 使用代理: ${HTTP_PROXY}"
                        PROXY_ARGS="--build-arg HTTP_PROXY=${HTTP_PROXY} --build-arg HTTPS_PROXY=${HTTP_PROXY}"
                    fi

                    echo ">>> 构建后端（镜像: ${PIP_MIRROR}, 代理: ${HTTP_PROXY:-无}）"
                    docker build \
                        --build-arg PIP_INDEX="${PIP_MIRROR}" \
                        --build-arg PIP_TRUSTED_HOST="${PIP_HOST}" \
                        ${PROXY_ARGS} \
                        --cache-from ${PROJECT_NAME}-backend:latest \
                        -t ${PROJECT_NAME}-backend:${BUILD_NUMBER} \
                        -t ${PROJECT_NAME}-backend:latest \
                        -f docker/Dockerfile .

                    echo ">>> 构建前端"
                    docker build \
                        --cache-from ${PROJECT_NAME}-frontend:latest \
                        -t ${PROJECT_NAME}-frontend:${BUILD_NUMBER} \
                        -t ${PROJECT_NAME}-frontend:latest \
                        -f docker/Dockerfile.frontend .
                '''
            }
        }

        stage('部署服务') {
            steps {
                sh '''
                    set -e

                    echo ">>> 创建网络"
                    docker network create ${NETWORK} 2>/dev/null || true

                    # ---- 停止旧容器（精确匹配容器名，不误伤其他服务）----
                    echo ">>> 停止旧容器"
                    for cn in ${CN_ES} ${CN_BACKEND} ${CN_FRONTEND}; do
                        if docker ps -a --format "{{.Names}}" 2>/dev/null | grep -qx "$cn"; then
                            echo "  停止: $cn"
                            docker rm -f "$cn" 2>/dev/null || true
                        fi
                    done
                    sleep 2

                    # ---- ES ----
                    # 健康检查用 IP：远程部署用 DEPLOY_HOST，本地用 localhost
                    HOST="${DEPLOY_HOST:-localhost}"

                    if [ "${SKIP_ES}" = "true" ]; then
                        echo ">>> [SKIP] 跳过 ES，使用外部: ${ES_HOST}"
                        if curl -sf "${ES_HOST}/_cluster/health" 2>/dev/null | grep -q "status"; then
                            echo "✓ 外部 ES 可达"
                        else
                            echo "⚠ 无法连接 ${ES_HOST}，请确认后再部署"
                        fi
                    else
                        echo ">>> 启动 ES (${HOST}:${ES_PORT})"
                        docker run -d \
                            --name ${CN_ES} \
                            --network ${NETWORK} \
                            --network-alias elasticsearch \
                            -p ${ES_PORT}:9200 \
                            -e "discovery.type=single-node" \
                            -e "xpack.security.enabled=false" \
                            -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
                            --memory=2g \
                            --restart=unless-stopped \
                            docker.elastic.co/elasticsearch/elasticsearch:8.12.0

                        echo ">>> 等待 ES 就绪 (http://${HOST}:${ES_PORT})"
                        for i in $(seq 1 60); do
                            if curl -sf "http://${HOST}:${ES_PORT}/_cluster/health" 2>/dev/null; then
                                echo "✓ ES 就绪"
                                break
                            fi
                            printf "."
                            sleep 3
                        done
                        echo ""
                    fi

                    # ---- 后端 ----
                    echo ">>> 启动后端 (:${BACKEND_PORT})"
                    docker run -d \
                        --name ${CN_BACKEND} \
                        --network ${NETWORK} \
                        --network-alias backend \
                        -p ${BACKEND_PORT}:8080 \
                        -e ES_HOST=${ES_HOST} \
                        -e APP_MODE=${APP_MODE} \
                        -e APP_PORT=8080 \
                        --memory=2g \
                        --restart=unless-stopped \
                        ${PROJECT_NAME}-backend:${BUILD_NUMBER}

                    # ---- 前端 ----
                    echo ">>> 启动前端 (:${FRONTEND_PORT})"
                    docker run -d \
                        --name ${CN_FRONTEND} \
                        --network ${NETWORK} \
                        -p ${FRONTEND_PORT}:80 \
                        --memory=256m \
                        --restart=unless-stopped \
                        ${PROJECT_NAME}-frontend:${BUILD_NUMBER}

                    echo ""
                    echo ">>> 运行中容器"
                    docker ps --filter "name=${CN_BACKEND}" --filter "name=${CN_FRONTEND}" --filter "name=${CN_ES}"
                '''
            }
        }

        stage('健康检查') {
            steps {
                sh '''
                    HOST="${DEPLOY_HOST:-localhost}"
                    echo ">>> 检查后端 http://${HOST}:${BACKEND_PORT}/health"
                    sleep 15

                    for i in $(seq 1 30); do
                        code=$(curl -s -o /dev/null -w "%{http_code}" \
                            "http://${HOST}:${BACKEND_PORT}/health" 2>/dev/null || echo "000")
                        if [ "$code" = "200" ]; then
                            echo "✓ 后端 OK"
                            curl -s "http://${HOST}:${BACKEND_PORT}/health"
                            break
                        fi
                        echo "  等待后端 ($i/30) HTTP=$code"
                        sleep 5
                    done

                    echo ""
                    for i in $(seq 1 10); do
                        code=$(curl -s -o /dev/null -w "%{http_code}" \
                            "http://${HOST}:${FRONTEND_PORT}" 2>/dev/null || echo "000")
                        if [ "$code" = "200" ]; then
                            echo "✓ 前端 OK"
                            break
                        fi
                        echo "  等待前端 ($i/10)"
                        sleep 2
                    done

                    echo ""
                    echo "=========================================="
                    echo "  后端: http://${HOST}:${BACKEND_PORT}"
                    echo "  API:  http://${HOST}:${BACKEND_PORT}/docs"
                    echo "  前端: http://${HOST}:${FRONTEND_PORT}"
                    echo "=========================================="
                '''
            }
        }
    }

    post {
        success {
            echo "✓ 部署成功 ${BUILD_NUMBER}"
        }
        failure {
            echo "✗ 部署失败，清理本项目容器..."
            sh '''
                # 只清理本项目的容器，不碰 SKIP_ES 时的外部 ES
                docker rm -f ${CN_BACKEND} ${CN_FRONTEND} 2>/dev/null || true
                # 只在非 SKIP_ES 时才删 ES 容器
                if [ "${SKIP_ES}" != "true" ]; then
                    docker rm -f ${CN_ES} 2>/dev/null || true
                fi
            '''
        }
        always {
            cleanWs()
        }
    }
}
