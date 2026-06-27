# 简历 RAG 智能问答系统 - 项目说明文档

> 本目录（`resume-rag-service/`）是项目代码。

---

## 仓库根目录（`../`）的 4 份规范文档

**不属于本项目**，是面向 AI 编码工具与开发者的指导/规范文档：

| 文档 | 位置 | 用途 |
|------|------|------|
| `MICRO_SERVICE_SPEC.md` | `../MICRO_SERVICE_SPEC.md` | 企业级微服务项目代码规范（v1.0） |
| `RAG_RESUME_PROJECT_GUIDE.md` | `../RAG_RESUME_PROJECT_GUIDE.md` | 项目开发指导书（面向面试） |
| `DEPLOY_TROUBLESHOOTING.md` | `../DEPLOY_TROUBLESHOOTING.md` | Docker 部署踩坑记录 |
| `METADATA_DESIGN.md` | `../METADATA_DESIGN.md` | RAG 知识库元数据方案设计 |

---

## 本项目目录结构

```
resume-rag-service/
├── README.md / CHANGELOG.md / DEV_GUIDE.md    ← 本目录内的文档
├── .env.example                                ← 环境变量示例
├── .gitignore                                  ← Git 忽略规则
├── Makefile                                    ← 本地/CI 构建入口
├── Jenkinsfile / .gitlab-ci.yml                ← 跨平台 CI 配置
├── docker-compose.yml                          ← Compose 一键启动
├── main.py                                     ← FastAPI 入口
├── requirements.txt                            ← 本地 pip 一键安装（合并）
├── requirements-base.txt                       ← Docker 分层 Layer 1（轻量）
├── requirements-ml.txt                         ← Docker 分层 Layer 2（torch/sentence-transformers）
│
├── bin/                                        ← 部署/启动/停止脚本
│   ├── start.sh / start.ps1                    ← 一键启动（Linux/Windows）
│   ├── stop.sh                                 ← 优雅停止（合并自原 scripts/）
│   ├── deploy.sh                               ← Docker 部署
│   └── health.sh                               ← 健康检查
│
├── config/                                     ← 运行时配置
│   └── config.yaml                             ← 唯一配置源
│
├── docker/                                     ← Docker 构建文件
│   ├── Dockerfile                              ← 后端镜像（多层构建）
│   ├── Dockerfile.frontend                     ← 前端镜像
│   └── download_model.py                       ← 模型下载脚本
│
├── docs/                                       ← 项目运行时文档
│   └── api.md                                  ← HTTP API 文档
│
├── frontend/                                   ← 前端（Vue 3 全局 CDN）
│   ├── index.html                              ← 单页入口
│   ├── manifest.json                           ← PWA manifest
│   ├── sw.js                                   ← Service Worker
│   ├── package.json                            ← 项目元信息（无构建）
│   └── src/
│       ├── main.js                             ← Vue 组件入口
│       ├── api/client.js                       ← HTTP 封装
│       └── styles/main.css                     ← 样式
│
├── internal/                                   ← 后端业务代码（DDD 分层）
│   ├── handler/api_handler.py                  ← FastAPI 路由
│   ├── service/                                ← 业务服务层
│   │   ├── rag_service.py                      ← RAG 核心编排
│   │   ├── llm_service.py                      ← LLM Provider 适配
│   │   ├── embedding_service.py                ← BGE 本地推理
│   │   ├── es_service.py                       ← ES 混合检索
│   │   ├── chunk_service.py                    ← 分块（fixed/semantic/hybrid）
│   │   ├── document_parser.py                  ← PDF/TXT/CSV/MD 解析
│   │   ├── knowledge_base_service.py           ← 入库编排
│   │   ├── intent_service.py                   ← 意图识别（v1.1）
│   │   ├── rerank_service.py                   ← 可选重排
│   │   └── rate_limiter.py                     ← 令牌桶限流
│   ├── repository/es_repository.py             ← ES 数据访问层
│   ├── model/                                  ← 数据模型
│   │   ├── config.py                           ← Pydantic 配置
│   │   └── dto.py                              ← Request/Response DTO
│   └── pkg/                                    ← 公共工具
│       ├── logger.py                           ← 结构化日志
│       └── errors.py                           ← 业务异常定义
│
└── tests/                                      ← 单元测试
    ├── conftest.py
    ├── test_embedding.py
    ├── test_rag_evaluation.py
    └── test_rag_service.py
```

---

## 项目内文档（运行时）

- `README.md` - 项目使用说明（快速开始、API 概览、面试要点）
- `docs/api.md` - HTTP API 详细文档
- `CHANGELOG.md` - 变更记录
- `DEV_GUIDE.md` - 本文件（结构说明 + 外部规范索引）