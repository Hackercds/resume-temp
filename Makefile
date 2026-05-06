.PHONY: build run stop health test clean docker-build docker-run dev

APP_NAME=resume-rag-service
PORT=8080

build:
	pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

run:
	python main.py

dev:
	APP_MODE=dev python main.py

stop:
	docker rm -f $(APP_NAME) 2>/dev/null || true

health:
	curl -sf http://localhost:$(PORT)/health && echo " OK" || echo " FAILED"

docker-build:
	docker build -t $(APP_NAME):latest -f docker/Dockerfile .

docker-run:
	docker rm -f $(APP_NAME) 2>/dev/null || true
	docker run -d \
		--name $(APP_NAME) \
		-p $(PORT):8080 \
		--restart=unless-stopped \
		$(APP_NAME):latest
	@echo "Waiting for service..."
	@sleep 5
	@make health

test:
	python -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
