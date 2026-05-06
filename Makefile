.PHONY: build run stop health test clean docker-build docker-run dev deploy deploy-full deploy-no-es compose-up compose-down compose-full-up

APP_NAME=resume-rag-service
PORT=8080
FRONTEND_PORT=5000
ES_PORT=9200

build:
	pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

run:
	python main.py

dev:
	APP_MODE=dev python main.py

stop:
	docker rm -f rag-es rag-backend rag-frontend 2>/dev/null || true

health:
	@echo -n "Backend: "; curl -sf http://localhost:$(PORT)/health && echo " OK" || echo " FAILED"
	@echo -n "Frontend: "; curl -sf -o /dev/null -w "%{http_code}" http://localhost:$(FRONTEND_PORT) && echo "" || echo " FAILED"

docker-build:
	docker build \
		--build-arg PIP_INDEX=${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple} \
		--build-arg PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn} \
		-t $(APP_NAME)-backend:latest -f docker/Dockerfile .
	docker build -t $(APP_NAME)-frontend:latest -f docker/Dockerfile.frontend .

docker-run:
	docker rm -f $(APP_NAME) 2>/dev/null || true
	docker run -d \
		--name $(APP_NAME) \
		-p $(PORT):8080 \
		--restart=unless-stopped \
		$(APP_NAME)-backend:latest
	@sleep 5
	@make health

# ========== 部署 ==========

# 全栈部署（ES + 后端 + 前端）
deploy:
	chmod +x bin/deploy.sh 2>/dev/null || true
	bash bin/deploy.sh

# 跳过 ES 部署（已有 ES 时使用）
deploy-no-es:
	SKIP_ES=true ES_HOST=$(ES_HOST) bash bin/deploy.sh

# 全栈部署 + 自定义端口
deploy-full:
	BACKEND_PORT=$(PORT) FRONTEND_PORT=$(FRONTEND_PORT) ES_PORT=$(ES_PORT) bash bin/deploy.sh

# ========== Docker Compose ==========

# 仅后端 + 前端（连外部 ES）→ 已有 ES 时用这个
compose-up:
	docker-compose up -d --build backend frontend
	@sleep 10
	@make health

# 全栈（ES + 后端 + 前端）→ 没有 ES 时用这个
compose-full-up:
	docker-compose --profile full up -d --build
	@sleep 20
	@make health

compose-down:
	docker-compose --profile full down -v

test:
	python -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
