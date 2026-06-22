import json
import logging
from io import StringIO

from voiceshield.logger import StructuredLogger, get_logger


def test_logger_emits_json():
    """TEST-S0.4: Logger emits valid JSON lines."""
    # Capture output
    stream = StringIO()
    handler = logging.StreamHandler(stream)

    # We must reset handlers to capture specifically
    logger_instance = get_logger("test_module")
    logger_instance.handlers = [handler]

    # Needs to use the JSONFormatter that gets attached in get_logger
    from voiceshield.logger import JSONFormatter
    handler.setFormatter(JSONFormatter())

    struct_logger = StructuredLogger("test_module", run_id="test_run_123")
    struct_logger._logger = logger_instance

    struct_logger.info("chunk_processed", {"snr": 15.0}, trace_id="trace_abc")

    output = stream.getvalue().strip()
    assert output, "Logger produced no output"

    parsed = json.loads(output)
    assert parsed["module"] == "test_module"
    assert parsed["level"] == "INFO"
    assert parsed["event"] == "chunk_processed"
    assert parsed["data"] == {"snr": 15.0}
    assert parsed["run_id"] == "test_run_123"
    assert parsed["trace_id"] == "trace_abc"
    assert "ts" in parsed
