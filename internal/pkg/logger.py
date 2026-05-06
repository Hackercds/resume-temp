"""
结构化日志模块 - 面试点：traceId 贯穿全链路
容器环境始终输出到 stdout（docker logs 可见），level 由 LOG_LEVEL 环境变量控制
"""
import logging
import json
import os
import sys
import re
from datetime import datetime
from typing import Optional


class RAGLogger:
    """结构化日志 — 容器 stdout + 文件双写"""

    def __init__(self, name: str = "resume-rag", level: str = "info",
                 log_path: str = "./log", filename: str = "app.log"):
        self._level_name = level.upper()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # handler 级别单独控制
        self.logger.handlers.clear()

        # 1. stdout handler — 容器 logs 可见
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(getattr(logging, self._level_name, logging.DEBUG))
        sh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(sh)

        # 2. 文件 handler — 持久化
        os.makedirs(log_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_path, filename), encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(fh)

    @property
    def level_name(self) -> str:
        return self._level_name

    def _format(self, level: str, trace_id: str, module: str,
                message: str, **fields) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        fields_str = json.dumps(fields, ensure_ascii=False, default=str) if fields else "{}"
        # API Key 脱敏
        if "api_key" in fields_str:
            fields_str = re.sub(
                r'"api_key":\s*"([^"]{4})[^"]*"',
                r'"api_key": "\1****"',
                fields_str
            )
        return f"{timestamp} | {level:5s} | {trace_id:8s} | {module:16s} | {message} | {fields_str}"

    def debug(self, trace_id: str, module: str, message: str, **fields):
        self.logger.debug(self._format("DEBUG", trace_id, module, message, **fields))

    def info(self, trace_id: str, module: str, message: str, **fields):
        self.logger.info(self._format("INFO", trace_id, module, message, **fields))

    def warn(self, trace_id: str, module: str, message: str, **fields):
        self.logger.warning(self._format("WARN", trace_id, module, message, **fields))

    def error(self, trace_id: str, module: str, message: str, **fields):
        self.logger.error(self._format("ERROR", trace_id, module, message, **fields))


_logger: Optional[RAGLogger] = None


def get_logger() -> RAGLogger:
    global _logger
    if _logger is None:
        from internal.model.config import get_config
        cfg = get_config()
        _logger = RAGLogger(
            level=cfg.log.level,
            log_path=cfg.log.path,
            filename=cfg.log.filename
        )
    return _logger
