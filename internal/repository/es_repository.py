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
        """检查 ES 是否连通（3秒超时）"""
        try:
            if self.client is None:
                self.client = Elasticsearch(
                    hosts=self.hosts,
                    request_timeout=3,
                )
            return self.client.ping()
        except Exception as e:
            self.logger.warn("health", "es_repository",
                             "ES 连接检查失败", error=str(e)[:100])
            return False

    # ---------- 索引操作 ----------
    # 中文分词器：优先 ik_max_word（细粒度，召回高），查询用 ik_smart（智能切分，精确）。
    # ES 未安装 ik 插件时，mapping 仍可创建，但检索时若 analyzer 不存在会回退到 standard。
    # 为保证健壮性，create_index 时探测 ik 是否可用，不可用则降级 standard 并记录日志。
    _ZH_ANALYZER = "ik_max_word"
    _ZH_SEARCH_ANALYZER = "ik_smart"

    def _detect_ik_available(self, es) -> bool:
        """
        探测 ES 是否安装了 analysis-ik 插件。
        面试点：为什么不硬依赖 ik？
        答：开发/CI 环境常装的是精简版 ES，强制 ik 会让索引创建直接失败。
            探测后降级，保证「有 ik 用 ik，没 ik 不崩」。
        """
        try:
            # GET /_cat/plugins — 含 ik 说明已安装
            plugins = es.cat.plugins(format="json")
            for row in plugins or []:
                if "analysis-ik" in (row.get("component", "") or "") or \
                   "ik" in (row.get("name", "") or ""):
                    return True
            return False
        except Exception as e:
            self.logger.warn("es_repository", "_detect_ik_available",
                             "IK 插件探测失败，降级 standard", error=str(e)[:120])
            return False

    def create_index(self, index_name: str = None) -> Dict:
        """
        创建带 dense_vector 映射的索引 - 面试点：ES 8.x dense_vector 配置
        中文分词：有 ik 插件用 ik_max_word/ik_smart，否则降级 standard。
        全文父文档（is_full_doc=true）通过 filter 在常规检索中排除，避免首段向量污染 top_k。
        """
        index = index_name or self.index
        es = self.connect()
        use_ik = self._detect_ik_available(es)
        analyzer = self._ZH_ANALYZER if use_ik else "standard"
        search_analyzer = self._ZH_SEARCH_ANALYZER if use_ik else "standard"
        if use_ik:
            self.logger.info("create_index", "es_repository",
                             "✓ 启用 IK 中文分词",
                             analyzer=analyzer, search_analyzer=search_analyzer)
        else:
            self.logger.warn("create_index", "es_repository",
                             "IK 插件未安装，使用 standard 分词（中文 BM25 召回较弱）",
                             hint="安装 analysis-ik 以提升中文检索质量")

        mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": analyzer,        # 有 ik 用 ik_max_word，否则 standard
                        "search_analyzer": search_analyzer
                    },
                    "file_name": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "char_count": {"type": "integer"},
                    "section_title": {"type": "keyword"},
                    "upload_time": {"type": "date"},
                    "full_text": {"type": "text", "analyzer": analyzer},
                    "is_full_doc": {"type": "boolean"},
                    "doc_id": {"type": "keyword"},
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

    def _ensure_full_doc_fields(self):
        """
        兼容旧索引：动态添加整篇文档召回及语义分块所需字段。
        ES 8.x 支持向已有 mapping 添加字段，不会丢失数据。
        新增 text 字段时若 ES 支持 ik，使用 ik 分词器。
        """
        es = self.connect()
        try:
            mapping = es.indices.get_mapping(index=self.index)
            props = mapping.get(self.index, {}).get("mappings", {}).get("properties", {})
            use_ik = self._detect_ik_available(es)
            analyzer = self._ZH_ANALYZER if use_ik else "standard"
            search_analyzer = self._ZH_SEARCH_ANALYZER if use_ik else "standard"
            missing = []
            for field, ftype in [("full_text", {"type": "text", "analyzer": analyzer}),
                                 ("is_full_doc", {"type": "boolean"}),
                                 ("doc_id", {"type": "keyword"}),
                                 ("section_title", {"type": "keyword"})]:
                if field not in props:
                    missing.append((field, ftype))
            if missing:
                body = {"properties": {field: ftype for field, ftype in missing}}
                es.indices.put_mapping(index=self.index, body=body)
                self.logger.info("es_repository", "_ensure_full_doc_fields",
                                 f"已动态添加字段: {[f for f, _ in missing]}",
                                 analyzer=analyzer)
        except Exception as e:
            self.logger.warn("es_repository", "_ensure_full_doc_fields",
                             "动态添加字段失败", error=str(e)[:200])

    def delete_index(self, index_name: str = None) -> Dict:
        es = self.connect()
        return es.indices.delete(index=index_name or self.index, ignore=[400, 404])

    def index_exists(self, index_name: str = None) -> bool:
        es = self.connect()
        return es.indices.exists(index=index_name or self.index)

    def _ensure_correct_mapping(self):
        """
        确保索引存在。不自动删已有数据——拿到旧 mapping 时走兼容路径。
        面试点：为什么不一刀切重建？
        答：用户已有数据不能随便丢，兼容处理比强制清空更负责任。
        """
        if not self.index_exists():
            self.create_index()
        else:
            self._ensure_full_doc_fields()
    def bulk_insert(self, chunks: List[Dict], vectors: np.ndarray) -> int:
        """
        批量写入 - 面试点：为什么用 bulk 而不是逐条 index？
        答：HTTP 开销大，bulk 批量写入效率高 10-100 倍

        面试点：为什么写入前要确保索引存在？
        答：ES auto_create_index 会用动态 mapping，导致 file_name 变 text 类型、
            vector 维度未知，后续聚合查询和向量检索都会报错。
        """
        # 确保索引以正确的 mapping 存在（防止 auto_create 用错 mapping）
        self._ensure_correct_mapping()

        es = self.connect()
        actions = []
        index = self.index
        now = datetime.now().isoformat()

        for chunk, vector in zip(chunks, vectors):
            source = {
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "file_name": chunk["file_name"],
                "chunk_index": chunk.get("chunk_index", 0),
                "char_count": chunk.get("char_count", len(chunk["content"])),
                "section_title": chunk.get("section_title", ""),
                "upload_time": chunk.get("upload_time", now),
                # 常规 chunk 显式标记 is_full_doc=false，保证字段完整；
                # 检索时用 must_not(term:true) 排除父文档，兼容历史未写该字段的数据
                "is_full_doc": bool(chunk.get("is_full_doc", False)),
                "vector": vector.tolist()
            }
            if chunk.get("is_full_doc"):
                source["full_text"] = chunk.get("full_text", chunk["content"])
                source["doc_id"] = chunk.get("doc_id", chunk["file_name"])
            actions.append({
                "_index": index,
                "_id": chunk.get("chunk_id"),
                "_source": source
            })

        success, failed = bulk(es, actions, raise_on_error=False,
                               request_timeout=60)
        if failed:
            self.logger.warn("bulk_insert", "es_repository",
                             f"批量写入部分失败: {len(failed)}条")

        # 强制刷新，让新数据立即可搜索
        es.indices.refresh(index=index)
        return success

    def get_full_document(self, file_name: str) -> Optional[Dict]:
        """
        通过 file_name 查询整篇文档。

        面试点：双路召回 + 优雅降级
        - 主路：is_full_doc=true 的父文档（入库时写入，含完整 full_text，最干净）
        - 降级路：父文档不存在（老数据 / 入库版本早于全文特性）时，
          按该 file_name 聚合所有常规 chunk（chunk_index 升序）拼回全文。
        这样「查看全文」「基于文档深度回答」对任何文档都可用，
        而不必强制用户重新上传入库——存量数据零迁移成本。
        """
        es = self.connect()
        agg_field = self._get_agg_field("file_name")

        # 主路：父文档
        body = {
            "size": 1,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"is_full_doc": True}},
                        {"term": {agg_field: file_name}}
                    ]
                }
            },
            "_source": ["chunk_id", "content", "full_text", "file_name", "upload_time", "char_count"]
        }
        response = es.search(index=self.index, body=body)
        hits = response["hits"]["hits"]
        if hits:
            src = hits[0]["_source"]
            return {
                "chunk_id": src.get("chunk_id"),
                "content": src.get("full_text") or src.get("content", ""),
                "file_name": src.get("file_name"),
                "score": 0,
                "is_full_doc": True
            }

        # 降级路：聚合该文档所有常规 chunk 拼回全文
        return self._reconstruct_full_from_chunks(es, file_name, agg_field)

    def _reconstruct_full_from_chunks(self, es, file_name: str, agg_field: str) -> Optional[Dict]:
        """按 chunk_index 升序聚合该 file_name 的所有常规 chunk，拼回完整文档。"""
        body = {
            "size": 1000,
            "query": {
                "bool": {
                    "must": [{"term": {agg_field: file_name}}],
                    "must_not": [{"term": {"is_full_doc": True}}]
                }
            },
            "sort": [{"chunk_index": {"order": "asc"}}],
            "_source": ["chunk_id", "content", "file_name", "chunk_index"]
        }
        response = es.search(index=self.index, body=body)
        hits = response["hits"]["hits"]
        if not hits:
            return None
        parts = [h["_source"].get("content", "") for h in hits]
        full_text = "\n".join(p for p in parts if p)
        return {
            "chunk_id": f"{file_name}__reconstructed",
            "content": full_text,
            "file_name": file_name,
            "score": 0,
            "is_full_doc": True,
            "reconstructed": True,
            "chunk_count": len(hits)
        }

    # ---------- 数据检索 ----------
    def search_by_vector(self, query_vector: np.ndarray, size: int = 10,
                         exclude_full_doc: bool = True) -> List[Dict]:
        """
        向量语义检索 - 面试点：cosineSimilarity + 1.0 确保分数非负
        ES script_score 需要正分数做排序

        exclude_full_doc：排除 is_full_doc=true 的父文档。
        面试点：为什么要排除？
        答：入库时父文档的 vector 只编码了「前 chunk_size 字」的向量，
            它会以首段语义匹配、并挤占 top_k，污染检索结果。
            父文档应只通过 retrieve_full_document 按 file_name 精确召回。

        面试点：为什么用 must_not(term:true) 而不是 term:false？
        答：常规 chunk 入库时未显式写 is_full_doc 字段（值为缺失/null）。
           ES 中 term:false 只命中显式等于 false 的文档，**不命中字段缺失的文档**，
           若用 term:false 会把所有常规 chunk 误排除，导致检索为空。
           must_not(term:true) 只排除父文档，其余（含字段缺失的常规 chunk）全部保留。
        """
        es = self.connect()
        if exclude_full_doc:
            inner_query = {"bool": {"must_not": [{"term": {"is_full_doc": True}}]}}
        else:
            inner_query = {"match_all": {}}

        body = {
            "size": size,
            "query": {
                "script_score": {
                    "query": inner_query,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_vector.tolist()}
                    }
                }
            },
            "_source": ["chunk_id", "content", "file_name", "chunk_index", "section_title", "is_full_doc"]
        }
        response = es.search(index=self.index, body=body)
        return [
            {
                "chunk_id": hit["_source"]["chunk_id"],
                "content": hit["_source"]["content"],
                "file_name": hit["_source"]["file_name"],
                "chunk_index": hit["_source"].get("chunk_index", 0),
                "section_title": hit["_source"].get("section_title", ""),
                "is_full_doc": hit["_source"].get("is_full_doc", False),
                "score": hit["_score"] - 1.0  # 还原为真实余弦相似度
            }
            for hit in response["hits"]["hits"]
        ]

    def search_by_keyword(self, query_text: str, size: int = 10,
                          exclude_full_doc: bool = True) -> List[Dict]:
        """
        BM25 关键词检索 - 面试点：ES match query 默认使用 BM25 算法
        为什么保留这个？即使 embedding 不可用，基本搜索功能仍可用（降级方案）
        exclude_full_doc：同 search_by_vector，用 must_not 排除父文档（兼容字段缺失的常规 chunk）。
        """
        es = self.connect()
        bool_body = {"must": [{"match": {"content": {"query": query_text, "operator": "or"}}}]}
        if exclude_full_doc:
            bool_body["must_not"] = [{"term": {"is_full_doc": True}}]

        body = {
            "size": size,
            "query": {"bool": bool_body},
            "_source": ["chunk_id", "content", "file_name", "chunk_index", "section_title", "is_full_doc"]
        }
        response = es.search(index=self.index, body=body)
        return [
            {
                "chunk_id": hit["_source"]["chunk_id"],
                "content": hit["_source"]["content"],
                "file_name": hit["_source"]["file_name"],
                "chunk_index": hit["_source"].get("chunk_index", 0),
                "section_title": hit["_source"].get("section_title", ""),
                "is_full_doc": hit["_source"].get("is_full_doc", False),
                "score": hit["_score"]
            }
            for hit in response["hits"]["hits"]
        ]

    def search_neighbor_chunks(self, file_name: str, chunk_indices: List[int]) -> List[Dict]:
        """
        邻域上下文扩展：按 file_name + 一组 chunk_index 精确召回相邻 chunk。
        面试点：为什么要做邻域扩展？
        答：分块会切断跨块语义（「如前所述」「上文提到的项目」），
            命中某块后召回其前后相邻块，能让 LLM 看到完整上下文，
            显著降低跨块追问的断章取义。一次 mget/term 查询完成，开销极低。
        """
        if not file_name or not chunk_indices:
            return []
        es = self.connect()
        # 用 terms 一次查回所有目标 chunk_index；is_full_doc 用 must_not 排除父文档
        # （常规 chunk 字段缺失，必须用 must_not 而非 term:false，详见 search_by_vector 注释）
        body = {
            "size": len(chunk_indices) * 2 + 4,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"file_name": file_name}},
                        {"terms": {"chunk_index": chunk_indices}},
                    ],
                    "must_not": [{"term": {"is_full_doc": True}}]
                }
            },
            "_source": ["chunk_id", "content", "file_name", "chunk_index", "section_title"],
            "sort": [{"chunk_index": {"order": "asc"}}]
        }
        try:
            response = es.search(index=self.index, body=body)
            return [
                {
                    "chunk_id": hit["_source"]["chunk_id"],
                    "content": hit["_source"]["content"],
                    "file_name": hit["_source"]["file_name"],
                    "chunk_index": hit["_source"].get("chunk_index", 0),
                    "section_title": hit["_source"].get("section_title", ""),
                    "score": 0.0  # 邻域扩展项不计分，仅作上下文补充
                }
                for hit in response["hits"]["hits"]
            ]
        except Exception as e:
            self.logger.warn("search_neighbor", "es_repository",
                             "邻域扩展查询失败", error=str(e)[:120])
            return []

    # ---------- 数据管理 ----------
    def delete_by_file_name(self, file_name: str) -> int:
        """
        按文件名删除所有 chunk
        """
        es = self.connect()
        agg_field = self._get_agg_field("file_name")
        body = {
            "query": {
                "term": {agg_field: file_name}
            }
        }
        response = es.delete_by_query(index=self.index, body=body,
                                       conflicts="proceed")
        return response.get("deleted", 0)

    def _get_agg_field(self, field_name: str) -> str:
        """探明 ES 实际聚合字段名：keyword 用原名，text 用 .keyword 子字段"""
        try:
            es = self.connect()
            mapping = es.indices.get_mapping(index=self.index)
            props = mapping.get(self.index, {}).get("mappings", {}).get("properties", {})
            ftype = props.get(field_name, {}).get("type", "keyword")
            if ftype == "text":
                return f"{field_name}.keyword"
            return field_name
        except Exception:
            return field_name  # fallback

    def list_file_names(self) -> List[Dict]:
        """
        列出所有文档及 chunk 数量
        兼容两种 mapping：keyword 字段用 file_name，text 字段用 file_name.keyword
        """
        if not self.index_exists():
            return []

        es = self.connect()

        # 先搞清楚 field 名：新索引 keyword → file_name，旧索引 text → file_name.keyword
        agg_field = self._get_agg_field("file_name")

        body = {
            "size": 0,
            "aggs": {
                "documents": {
                    "terms": {
                        "field": agg_field,
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
        if not self.index_exists():
            return {"total": 0}
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
