pipeline {
    agent any

    // 构建参数：支持已有 ES 的场景
    parameters {
        booleanParam(name: 'SKIP_ES', defaultValue: false,
                     description: '本机已有 ES 时设为 true，不启动 ES 容器')
        string(name: 'ES_HOST', defaultValue: 'http://elasticsearch:9200',
               description: 'ES 地址。已有 ES 时填实际地址，如 http://192.168.3.184:9200')
        string(name: 'BACKEND_PORT', defaultValue: '8080',
               description: '后端主机端口')
        string(name: 'FRONTEND_PORT', defaultValue: '5000',
               description: '前端主机端口')
        string(name: 'ES_PORT', defaultValue: '9200',
               description: 'ES 主机端口（仅 SKIP_ES=false 时生效）')
        string(name: 'HOST_IP', defaultValue: '192.168.3.184',
               description: '部署目标 IP')
        string(name: 'DOCKER_HOST', defaultValue: 'tcp://192.168.3.184:2375',
               description: 'Docker TCP 地址')
    }

    environment {
        PROJECT_NAME = 'resume-rag-service'
        NETWORK = 'rag-network'
        APP_MODE = 'release'
    }

    stages {
        stage('代码检出') {
            steps {
                checkout scm
                echo "分支: ${env.BRANCH_NAME}, SKIP_ES: ${params.SKIP_ES}"
            }
        }

        stage('Docker 镜像构建') {
            steps {
                sh '''
                    echo ">>> 构建后端镜像"
                    docker build -t ${PROJECT_NAME}-backend:${BUILD_NUMBER} \
                        -f docker/Dockerfile .

                    echo ">>> 构建前端镜像"
                    docker build -t ${PROJECT_NAME}-frontend:${BUILD_NUMBER} \
                        -f docker/Dockerfile.frontend .
                '''
            }
        }

        stage('部署服务') {
            steps {
                sh '''
                    set -e

                    echo ">>> 创建 Docker 网络"
                    docker network create ${NETWORK} 2>/dev/null || true

                    echo ">>> 停止旧容器"
                    docker rm -f ${PROJECT_NAME}-es \
                               ${PROJECT_NAME}-backend \
                               ${PROJECT_NAME}-frontend 2>/dev/null || true
                    sleep 3

                    # ---------- ES ----------
                    if [ "${SKIP_ES}" = "true" ]; then
                        echo ">>> [SKIP] 不启动 ES 容器，使用外部 ES: ${ES_HOST}"

                        # 检查外部 ES 可达性
                        if curl -sf "${ES_HOST}/_cluster/health" 2>/dev/null | grep -q '"status"'; then
                            echo "✓ 外部 ES 连接正常"
                        else
                            echo "⚠ 警告：无法连接 ${ES_HOST}，请确认 ES 地址正确"
                        fi
                    else
                        echo ">>> 启动 Elasticsearch"
                        docker run -d \
                            --name ${PROJECT_NAME}-es \
                            --network ${NETWORK} \
                            --network-alias elasticsearch \
                            -p ${ES_PORT}:9200 \
                            -e "discovery.type=single-node" \
                            -e "xpack.security.enabled=false" \
                            -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
                            --memory=2g \
                            docker.elastic.co/elasticsearch/elasticsearch:8.12.0

                        echo ">>> 等待 ES 就绪 (最多 3 分钟)"
                        for i in $(seq 1 60); do
                            if curl -sf http://${HOST_IP}:${ES_PORT}/_cluster/health 2>/dev/null; then
                                echo "✓ ES 已就绪"
                                break
                            fi
                            echo "等待 ES... ($i/60)"
                            sleep 3
                        done
                    fi

                    # ---------- 后端 ----------
                    echo ">>> 启动后端 (端口: ${BACKEND_PORT})"
                    docker run -d \
                        --name ${PROJECT_NAME}-backend \
                        --network ${NETWORK} \
                        --network-alias backend \
                        -p ${BACKEND_PORT}:8080 \
                        -e ES_HOST=${ES_HOST} \
                        -e APP_MODE=${APP_MODE} \
                        -e APP_PORT=8080 \
                        --memory=2g \
                        ${PROJECT_NAME}-backend:${BUILD_NUMBER}

                    # ---------- 前端 ----------
                    echo ">>> 启动前端 (端口: ${FRONTEND_PORT})"
                    docker run -d \
                        --name ${PROJECT_NAME}-frontend \
                        --network ${NETWORK} \
                        -p ${FRONTEND_PORT}:80 \
                        --memory=256m \
                        ${PROJECT_NAME}-frontend:${BUILD_NUMBER}

                    echo ">>> 容器状态"
                    docker ps --filter "name=${PROJECT_NAME}"
                '''
            }
        }

        stage('健康检查') {
            steps {
                sh '''
                    echo ">>> 等待服务就绪..."
                    sleep 30

                    # 后端
                    for i in $(seq 1 30); do
                        http_code=$(curl -s -o /dev/null -w "%{http_code}" \
                            http://${HOST_IP}:${BACKEND_PORT}/health 2>/dev/null || echo "000")
                        if [ "$http_code" = "200" ]; then
                            echo "✓ 后端健康检查通过"
                            break
                        fi
                        echo "等待后端... ($i/30) HTTP $http_code"
                        sleep 5
                    done

                    # 前端
                    for i in $(seq 1 10); do
                        http_code=$(curl -s -o /dev/null -w "%{http_code}" \
                            http://${HOST_IP}:${FRONTEND_PORT} 2>/dev/null || echo "000")
                        if [ "$http_code" = "200" ]; then
                            echo "✓ 前端健康检查通过"
                            break
                        fi
                        echo "等待前端... ($i/10)"
                        sleep 2
                    done

                    echo "=========================================="
                    echo "  后端: http://${HOST_IP}:${BACKEND_PORT}"
                    echo "  API:  http://${HOST_IP}:${BACKEND_PORT}/docs"
                    echo "  前端: http://${HOST_IP}:${FRONTEND_PORT}"
                    echo "=========================================="
                '''
            }
        }
    }

    post {
        success {
            echo "✓ 部署成功！构建号: ${BUILD_NUMBER}"
        }
        failure {
            echo "✗ 部署失败，清理容器..."

            sh '''
                docker rm -f ${PROJECT_NAME}-backend \
                           ${PROJECT_NAME}-frontend \
                           ${PROJECT_NAME}-es 2>/dev/null || true
            '''
        }
        always {
            cleanWs()
        }
    }
}
