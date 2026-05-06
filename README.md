# 简历上写的 RAG 智能问答系统

> 本地 Embedding + Elasticsearch 混合检索 + 在线 LLM
>
> 面试核心项目：演示完整 RAG 链路，能回答"为什么做本地 embedding、为什么选 BGE、ES 混合检索原理、分块策略"等问题

---

## 配置体系

```
config/config.yaml     ← 【唯一配置源】修改应用行为（模型、分块、检索参数等）
                        运行时环境变量可覆盖（环境变量 > config.yaml > 默认值）

.env（可选）           ← 仅供 docker-compose 读取（端口映射等编排参数）
                        不影响 Dockerfile / deploy.sh / 直接 python 运行
```

**只想改配置 → 编辑 `config/config.yaml`，重新构建镜像即可。**
**已有 ES → 启动时传 `-e ES_HOST=http://你的ES:9200` 即可。**

---

## 五种启动方式

### 方式 1：Dockerfile 一键启动（⭐ 推荐，最纯粹）

```bash
# 构建
docker build -t resume-rag -f docker/Dockerfile .

# === 场景 A：本机已有 ES，连已有 ES ===
docker run -d --name rag \
  -p 8080:8080 \
  -e ES_HOST=http://192.168.3.184:9200 \
  resume-rag

# === 场景 B：本机没有 ES，需要新起一个 ===
docker network create rag-net

docker run -d --name rag-es \
  --network rag-net --network-alias elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.0

# 等 ES 启动后
docker run -d --name rag \
  --network rag-net \
  -p 8080:8080 \
  -e ES_HOST=http://elasticsearch:9200 \
  resume-rag

# === 场景 C：端口冲突，自定义端口 ===
docker run -d --name rag \
  -p 9090:8080 \
  -e ES_HOST=http://192.168.3.184:9200 \
  resume-rag

# 验证
curl http://localhost:8080/health
# 前端：浏览器打开 frontend/index.html
```

**配置修改流程：** 编辑 `config/config.yaml` → `docker build` + `docker run` 即可。

---

### 方式 2：Python 直接运行（开发调试）

```bash
pip install -r requirements.txt

# 本机已有 ES
ES_HOST=http://localhost:9200 python main.py

# 本机没有 ES，需要 Docker 启动一个
docker run -d --name es-dev -p 9200:9200 \
  -e "discovery.type=single-node" -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.0

ES_HOST=http://localhost:9200 python main.py
```

启动后：后端 `http://localhost:8080`，Swagger `http://localhost:8080/docs`。

---

### 方式 3：docker-compose（本地开发全家桶）

```bash
# === 已有 ES → 只启后端+前端 ===
ES_HOST=http://192.168.3.184:9200 docker-compose up -d backend frontend

# === 没有 ES → 全家桶（--profile full 才会启动 ES 容器）===
docker-compose --profile full up -d --build

# 自定义端口（复制 .env.example 为 .env，修改端口）
cp .env.example .env
# 编辑 .env 改 BACKEND_PORT=9090
docker-compose up -d backend frontend
```

---

### 方式 4：部署脚本（一键部署到远程 Docker）

```bash
# 全栈部署
make deploy

# 跳过 ES（已有 ES）
make deploy-no-es

# 部署到远程 Docker
DOCKER_HOST=tcp://192.168.3.184:2375 make deploy

# 或直接调用脚本
SKIP_ES=true ES_HOST=http://192.168.3.184:9200 bash bin/deploy.sh
```

---

### 方式 5：CI/CD 自动部署

**GitLab CI/CD：**
1. GitLab → Settings → CI/CD → Variables 添加：
   - `DEPLOY_HOST` = `192.168.3.184`
   - `DOCKER_PORT` = `2375`
   - `SKIP_ES` = `true`（已有 ES 时）
   - `ES_HOST` = `http://192.168.3.184:9200`
2. `git push` 即自动构建 + 部署

**Jenkins Pipeline：**
1. 创建 Pipeline Job，指向仓库
2. 构建时勾选参数：`SKIP_ES`、填写 `ES_HOST`
3. 执行构建

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/query` | RAG 智能问答 |
| POST | `/api/knowledge/upload` | 上传文档（PDF/TXT/CSV） |
| GET | `/api/knowledge/documents` | 文档列表 |
| DELETE | `/api/knowledge/documents/{name}` | 删除文档 |
| GET | `/api/stats` | 知识库统计 |

完整 API 文档见 `docs/api.md`，或启动后访问 `http://localhost:8080/docs`。

---

## 面试要点速查

| 面试问题 | 答案在代码中 | 关键文件 |
|---------|-------------|---------|
| 为什么做本地 embedding？ | `embedding_service.py` 顶部注释 | `internal/service/embedding_service.py` |
| 为什么选 BGE-small-zh？ | 同上 | 同上 |
| 完整 RAG 链路？ | `rag_service.py` query() 方法 | `internal/service/rag_service.py` |
| ES 混合检索原理？ | `es_service.py` search_hybrid() | `internal/service/es_service.py` |
| RRF 融合算法？ | `es_service.py` _rrf_fusion() | 同上 |
| 分块策略？ | `chunk_service.py` chunk_text() | `internal/service/chunk_service.py` |
| 性能优化？ | batch推理、向量缓存、令牌桶限流 | `embedding_service.py` + `rate_limiter.py` |
| 异常兜底？ | BM25降级、空检索兜底、LLM失败降级 | `rag_service.py` query_with_fallback |
| 蓝绿部署？ | 全量索引重建零停机 | `knowledge_base_service.py` sync_all() |

---

## 项目结构

```
resume-rag-service/
├── config/config.yaml         ← 唯一配置源
├── docker/Dockerfile          ← 核心部署方式
├── main.py                    ← FastAPI 入口
├── internal/
│   ├── handler/               ← API 路由层
│   ├── service/               ← 业务逻辑层（RAG / Embedding / ES / LLM）
│   ├── repository/            ← ES 数据访问层
│   ├── model/                 ← 数据模型 + 配置加载
│   └── pkg/                   ← 日志 + 错误定义
├── frontend/                  ← Vue3 前端（CDN，直接打开 index.html）
├── tests/                     ← 30 个单元测试
├── docs/api.md
└── README.md
```

---

## 前置依赖

- Python 3.11+
- Elasticsearch 8.x（已有或通过 Docker 启动）
- [可选] Docker（用于容器化部署）
- [可选] HuggingFace 模型下载（首次运行自动下载 BGE-small-zh，约 500MB）
<img width="1920" height="1778" alt="image" src="https://github.com/user-attachments/assets/4079d258-0515-4c03-9aa2-86c12f469111" />
<img width="3115" height="1894" alt="image" src="https://github.com/user-attachments/assets/eb161269-1d19-4363-bdc1-5f24775443bd" />

