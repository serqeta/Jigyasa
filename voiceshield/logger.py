import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = getattr(record, "data", {})
        run_id = getattr(record, "run_id", None)
        trace_id = getattr(record, "trace_id", None)
        event = getattr(record, "event", record.getMessage())

        log_record = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "module": record.name,
            "level": record.levelname,
            "event": event,
            "data": data,
        }

        if run_id:
            log_record["run_id"] = run_id
        if trace_id:
            log_record["trace_id"] = trace_id

        return json.dumps(log_record)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class StructuredLogger:
    """Wrapper to strictly enforce event names and structured data."""

    def __init__(self, name: str, run_id: Optional[str] = None):
        self._logger = get_logger(name)
        self.run_id = run_id or str(uuid.uuid4())

    def log(
        self, level: int, event: str, data: Dict[str, Any], trace_id: Optional[str] = None
    ) -> None:
        extra = {"event": event, "data": data, "run_id": self.run_id, "trace_id": trace_id}
        self._logger.log(level, event, extra=extra)

    def info(self, event: str, data: Dict[str, Any], trace_id: Optional[str] = None) -> None:
        self.log(logging.INFO, event, data, trace_id)

    def error(self, event: str, data: Dict[str, Any], trace_id: Optional[str] = None) -> None:
        self.log(logging.ERROR, event, data, trace_id)

    def warning(self, event: str, data: Dict[str, Any], trace_id: Optional[str] = None) -> None:
        self.log(logging.WARNING, event, data, trace_id)

    def debug(self, event: str, data: Dict[str, Any], trace_id: Optional[str] = None) -> None:
        self.log(logging.DEBUG, event, data, trace_id)
