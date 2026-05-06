"""
知识库管理服务 - 面试核心问题：
1. 文档入库完整流程？解析 → 分块 → 向量化 → ES 写入
2. 文档更新策略？删除旧chunk + 重新解析（delete + rebuild）
3. 全量同步如何不中断服务？蓝绿部署：新索引建好 → 别名切换
"""
import os
import time
from typing import List, Dict

from internal.model.config import get_config
from internal.pkg.logger import get_logger
from internal.service.embedding_service import EmbeddingService
from internal.service.es_service import ESService
from internal.service.chunk_service import ChunkService
from internal.service.document_parser import DocumentParser


class KnowledgeBaseService:
    """知识库管理服务"""

    def __init__(self,
                 embedding_service: EmbeddingService = None,
                 es_service: ESService = None,
                 chunk_service: ChunkService = None,
                 parser: DocumentParser = None):
        self.embedding = embedding_service or EmbeddingService()
        self.es = es_service or ESService()
        self.chunk = chunk_service or ChunkService()
        self.parser = parser or DocumentParser()
        self.logger = get_logger()

    def ingest_content(self, content: bytes, file_name: str,
                       chunk_size: int = None,
                       overlap: int = None) -> Dict:
        """
        文档入库主流程（同名覆盖策略：先删旧后写新）
        0. 同名检测 → 删旧 chunks
        1. 解析文档 → 原始文本
        2. 文本分块 → chunks
        3. 批量向量化 → vectors
        4. 写入 ES
        """
        # Step 0: 同名文件覆盖策略（删旧写新，保证幂等）
        deleted = self.es.delete_by_file_name(file_name)
        if deleted > 0:
            self.logger.info("ingest", "knowledge_base",
                             f"删除同名旧文档: {file_name}", deleted_chunks=deleted)

        # Step 1: 解析文档
        text = self.parser.parse(content, file_name)
        if not text or not text.strip():
            raise ValueError(f"文档内容为空: {file_name}")

        ext = os.path.splitext(file_name)[1].lower()
        self.logger.info("ingest", "knowledge_base",
                         "Step1 文档解析完成",
                         file_name=file_name, text_len=len(text))

        # Step 2: 分块
        if ext == '.csv':
            # CSV 按行分块
            chunks = self.chunk.chunk_csv(text, file_name)
        else:
            # 使用指定参数或默认参数
            if chunk_size is not None:
                self.chunk.chunk_size = chunk_size
            if overlap is not None:
                self.chunk.overlap = overlap
            chunks = self.chunk.chunk_text(text, file_name)

        if not chunks:
            raise ValueError(f"文本分块后无内容: {file_name}")

        self.logger.info("ingest", "knowledge_base",
                         "Step2 文本分块完成",
                         file_name=file_name, chunks=len(chunks))

        # Step 3: 批量向量化
        texts = [c["content"] for c in chunks]
        vectors = self.embedding.encode_batch(texts)

        self.logger.info("ingest", "knowledge_base",
                         "Step3 向量化完成",
                         file_name=file_name, vectors=len(vectors))

        # Step 4: 写入 ES
        inserted = self.es.bulk_insert(chunks, vectors)

        self.logger.info("ingest", "knowledge_base",
                         "Step4 ES写入完成",
                         file_name=file_name, inserted=inserted)

        result = {
            "file_name": file_name,
            "file_type": ext[1:] if ext.startswith('.') else ext,
            "chunks_created": inserted,
            "total_chunks": self.es.count()["total"]
        }
        if deleted > 0:
            result["replaced"] = True
            result["replaced_chunks"] = deleted
        return result

    def ingest_file(self, file_path: str, chunk_size: int = 400,
                    overlap: int = 50) -> Dict:
        """
        从文件路径入库 - 面试点：已有 ES 数据的环境可直接重建
        """
        file_name = os.path.basename(file_path)
        with open(file_path, 'rb') as f:
            content = f.read()
        return self.ingest_content(content, file_name, chunk_size, overlap)

    def delete_document(self, file_name: str) -> Dict:
        """
        删除文档 - 面试点：为什么按 file_name 删除？
        答：一个文件对应多个 chunk，按文件名一次性删除所有相关 chunk
        """
        deleted = self.es.delete_by_file_name(file_name)
        self.logger.info("delete", "knowledge_base",
                         f"文档已删除: {file_name}",
                         deleted_chunks=deleted)
        return {
            "deleted_chunks": deleted,
            "remaining_chunks": self.es.count()["total"]
        }

    def list_documents(self) -> List[Dict]:
        """列出所有文档及 chunk 数量"""
        return self.es.list_file_names()

    def sync_all(self, data_dir: str, chunk_size: int = 400,
                 overlap: int = 50) -> Dict:
        """
        全量同步 - 面试点：蓝绿部署，零停机
        1. 创建新索引（带时间戳后缀）
        2. 切换到新索引
        3. 重建所有数据
        4. 删除旧索引

        为什么用蓝绿部署？新索引建好后才切换，旧索引继续服务
        """
        import time

        # 确保索引存在
        self.es.ensure_index()

        # 创建新索引
        new_index = f"{self.es.index}_v{int(time.time())}"
        self.es.create_index(new_index)

        # 切换到新索引
        old_index = self.es.index
        old_repo_index = self.es.index  # ESRepository 的 index
        self.es.repo.index = new_index
        self.es.index = new_index

        # 重建数据
        total_inserted = 0
        total_files = 0
        errors = []

        for file_name in os.listdir(data_dir):
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in ('.pdf', '.txt', '.csv'):
                continue

            file_path = os.path.join(data_dir, file_name)
            try:
                result = self.ingest_file(file_path, chunk_size, overlap)
                total_inserted += result["chunks_created"]
                total_files += 1
            except Exception as e:
                errors.append({"file": file_name, "error": str(e)})
                self.logger.error("sync_all", "knowledge_base",
                                  f"处理失败: {file_name}",
                                  error=str(e))

        # 切换索引别名
        if total_inserted > 0:
            self.es.switch_alias(new_index, old_index)
            # 删除旧索引
            try:
                self.es.repo.delete_index(old_index)
            except Exception:
                pass

        self.logger.info("sync_all", "knowledge_base",
                         "全量同步完成",
                         files=total_files, inserted=total_inserted,
                         errors=len(errors))

        return {
            "inserted": total_inserted,
            "files_processed": total_files,
            "new_index": new_index,
            "errors": errors
        }

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        count = self.es.count()
        documents = self.list_documents()
        return {
            "total_chunks": count["total"],
            "total_documents": len(documents),
            "documents": documents
        }
