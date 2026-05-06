"""
Elasticsearch 数据访问层 - 面试点：ES 连接管理、索引映射、批量写入
遵循 MICRO_SERVICE_SPEC.md：repository 层只做数据操作，不包含业务逻辑
"""
import time
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from internal.model.config import get_config
from internal.pkg.logger import get_logger
from internal.pkg.errors import ESConnectionError


class ESRepository:
    """ES 底层操作 - 面试点：为什么用 repository 层？
    答：隔离 ES 具体实现，如果切换向量数据库（如 Milvus）只需改这里"""

    def __init__(self, hosts: List[str] = None, index: str = None,
                 vector_dim: int = None, timeout: int = None):
        cfg = get_config()
        es_cfg = cfg.elasticsearch
        self.hosts = hosts or es_cfg.hosts
        self.index = index or es_cfg.index
        self.vector_dim = vector_dim or es_cfg.vector_dim
        self.timeout = timeout or es_cfg.request_timeout
        self.client: Optional[Elasticsearch] = None
        self.logger = get_logger()

    def connect(self) -> Elasticsearch:
        """ES 连接 - 面试点：连接管理为什么要懒加载？
        答：避免 import 时就连接，测试环境可能没有 ES"""
        if self.client is None:
            self.client = Elasticsearch(
                hosts=self.hosts,
                request_timeout=self.timeout,
                max_retries=3,
                retry_on_timeout=True,
            )
            # 验证连接
            if not self.client.ping():
                raise ESConnectionError(f"无法连接到 ES: {self.hosts}")
        return self.client

    def is_connected(self) -> bool:
        try:
            if self.client is None:
                self.client = Elasticsearch(
                    hosts=self.hosts,
                    request_timeout=3,
                )
            return self.client.ping()
        except Exception:
            return False

    # ---------- 索引操作 ----------
    def create_index(self, index_name: str = None) -> Dict:
        """创建带 dense_vector 映射的索引 - 面试点：ES 8.x dense_vector 配置"""
        index = index_name or self.index
        es = self.connect()
        mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "standard",  # 生产环境推荐 ik_max_word
                        "search_analyzer": "standard"
                    },
                    "file_name": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "char_count": {"type": "integer"},
                    "upload_time": {"type": "date"},
                    "vector": {
                        "type": "dense_vector",
                        "element_type": "float",
                        "dims": self.vector_dim,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
        return es.indices.create(index=index, body=mapping, ignore=400)

    def delete_index(self, index_name: str = None) -> Dict:
        es = self.connect()
        return es.indices.delete(index=index_name or self.index, ignore=[400, 404])

    def index_exists(self, index_name: str = None) -> bool:
        es = self.connect()
        return es.indices.exists(index=index_name or self.index)

    # ---------- 数据写入 ----------
    def bulk_insert(self, chunks: List[Dict], vectors: np.ndarray) -> int:
        """
        批量写入 - 面试点：为什么用 bulk 而不是逐条 index？
        答：HTTP 开销大，bulk 批量写入效率高 10-100 倍
        """
        es = self.connect()
        actions = []
        index = self.index
        now = datetime.now().isoformat()

        for chunk, vector in zip(chunks, vectors):
            actions.append({
                "_index": index,
                "_id": chunk.get("chunk_id"),
                "_source": {
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "file_name": chunk["file_name"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "char_count": chunk.get("char_count", len(chunk["content"])),
                    "upload_time": chunk.get("upload_time", now),
                    "vector": vector.tolist()
                }
            })

        success, failed = bulk(es, actions, raise_on_error=False,
                               request_timeout=60)
        if failed:
            self.logger.warn("bulk_insert", "es_repository",
                             f"批量写入部分失败: {len(failed)}条")
        return success

    # ---------- 数据检索 ----------
    def search_by_vector(self, query_vector: np.ndarray, size: int = 10) -> List[Dict]:
        """
        向量语义检索 - 面试点：cosineSimilarity + 1.0 确保分数非负
        ES script_score 需要正分数做排序
        """
        es = self.connect()
        body = {
            "size": size,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_vector.tolist()}
                    }
                }
            },
            "_source": ["chunk_id", "content", "file_name", "chunk_index"]
        }
        response = es.search(index=self.index, body=body)
        return [
            {
                "chunk_id": hit["_source"]["chunk_id"],
                "content": hit["_source"]["content"],
                "file_name": hit["_source"]["file_name"],
                "chunk_index": hit["_source"].get("chunk_index", 0),
                "score": hit["_score"] - 1.0  # 还原为真实余弦相似度
            }
            for hit in response["hits"]["hits"]
        ]

    def search_by_keyword(self, query_text: str, size: int = 10) -> List[Dict]:
        """
        BM25 关键词检索 - 面试点：ES match query 默认使用 BM25 算法
        为什么保留这个？即使 embedding 不可用，基本搜索功能仍可用（降级方案）
        """
        es = self.connect()
        body = {
            "size": size,
            "query": {
                "match": {
                    "content": {
                        "query": query_text,
                        "operator": "or"
                    }
                }
            },
            "_source": ["chunk_id", "content", "file_name", "chunk_index"]
        }
        response = es.search(index=self.index, body=body)
        return [
            {
                "chunk_id": hit["_source"]["chunk_id"],
                "content": hit["_source"]["content"],
                "file_name": hit["_source"]["file_name"],
                "chunk_index": hit["_source"].get("chunk_index", 0),
                "score": hit["_score"]
            }
            for hit in response["hits"]["hits"]
        ]

    # ---------- 数据管理 ----------
    def delete_by_file_name(self, file_name: str) -> int:
        """
        按文件名删除所有 chunk - 面试点：为什么用 delete_by_query？
        答：一条请求删除所有关联 chunk，原子性有保证
        """
        es = self.connect()
        body = {
            "query": {
                "term": {"file_name": file_name}
            }
        }
        response = es.delete_by_query(index=self.index, body=body,
                                       conflicts="proceed")
        return response.get("deleted", 0)

    def list_file_names(self) -> List[Dict]:
        """
        列出所有文档及 chunk 数量 - 面试点：在 ES 端聚合，不拉全量数据
        """
        es = self.connect()
        body = {
            "size": 0,
            "aggs": {
                "documents": {
                    "terms": {
                        "field": "file_name",
                        "size": 100
                    },
                    "aggs": {
                        "first_upload": {
                            "top_hits": {
                                "sort": [{"upload_time": {"order": "asc"}}],
                                "_source": ["upload_time"],
                                "size": 1
                            }
                        }
                    }
                }
            }
        }
        response = es.search(index=self.index, body=body)
        buckets = response["aggregations"]["documents"].get("buckets", [])

        documents = []
        for bucket in buckets:
            first_hit = bucket.get("first_upload", {}).get("hits", {}).get("hits", [])
            upload_time = None
            if first_hit:
                upload_time = first_hit[0].get("_source", {}).get("upload_time")
            file_name = bucket["key"]
            documents.append({
                "file_name": file_name,
                "file_type": file_name.split('.')[-1] if '.' in file_name else "unknown",
                "chunk_count": bucket["doc_count"],
                "upload_time": upload_time
            })
        return documents

    def count(self) -> Dict:
        """返回索引文档总数"""
        es = self.connect()
        return {"total": es.count(index=self.index)["count"]}

    def switch_alias(self, new_index: str, old_index: str = None) -> Dict:
        """
        索引别名切换 - 蓝绿部署用
        面试点：全量重建时服务不中断
        """
        es = self.connect()
        alias = self.index
        # 先删除旧索引的别名
        if old_index:
            try:
                es.indices.update_aliases({
                    "actions": [
                        {"remove": {"index": old_index, "alias": alias}},
                        {"add": {"index": new_index, "alias": alias}}
                    ]
                })
            except Exception:
                # 如果 old_index 没有 alias，直接添加
                es.indices.update_aliases({
                    "actions": [
                        {"add": {"index": new_index, "alias": alias}}
                    ]
                })
        else:
            es.indices.put_alias(index=new_index, name=alias)
        self.index = new_index
        return {"new_index": new_index}


# 全局单例
_repo: Optional[ESRepository] = None


def get_es_repository() -> ESRepository:
    global _repo
    if _repo is None:
        _repo = ESRepository()
    return _repo
