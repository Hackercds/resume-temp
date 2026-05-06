"""
结构化日志模块 - 面试点：traceId 贯穿全链路
遵循 MICRO_SERVICE_SPEC.md 规范：时间 | 级别 | traceId | 模块 | 消息 | 附加字段
"""
import logging
import json
import os
import time
from datetime import datetime
from typing import Optional


class RAGLogger:
    """结构化日志封装 - 面试点：为什么不用标准 logging？
    答：标准 logging 不支持结构化字段（traceId/module），需要用 extra 传递"""

    def __init__(self, name: str = "resume-rag", level: str = "info",
                 log_path: str = "./log", filename: str = "app.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.handlers.clear()

        # 确保日志目录存在
        os.makedirs(log_path, exist_ok=True)

        # 文件 handler
        fh = logging.FileHandler(os.path.join(log_path, filename), encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(fh)

        # 开发模式也输出到控制台
        if os.getenv("APP_MODE", "release") == "dev":
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(ch)

    def _format(self, level: str, trace_id: str, module: str,
                message: str, **fields) -> str:
        """格式化日志为：时间 | 级别 | traceId | 模块 | 消息 | 附加字段"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        fields_str = json.dumps(fields, ensure_ascii=False) if fields else "{}"
        # 脱敏处理：API Key 只显示前4位
        if "api_key" in fields_str:
            import re
            fields_str = re.sub(
                r'"api_key":\s*"([^"]{4})[^"]*"',
                r'"api_key": "\1****"',
                fields_str
            )
        return f"{timestamp} | {level} | {trace_id} | {module} | {message} | {fields_str}"

    def debug(self, trace_id: str, module: str, message: str, **fields):
        self.logger.debug(self._format("DEBUG", trace_id, module, message, **fields))

    def info(self, trace_id: str, module: str, message: str, **fields):
        self.logger.info(self._format("INFO", trace_id, module, message, **fields))

    def warn(self, trace_id: str, module: str, message: str, **fields):
        self.logger.warning(self._format("WARN", trace_id, module, message, **fields))

    def error(self, trace_id: str, module: str, message: str, **fields):
        self.logger.error(self._format("ERROR", trace_id, module, message, **fields))


# 全局日志实例
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
