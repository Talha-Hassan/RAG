import logging
import json
import time
from datetime import datetime , timezone
from functools import wraps
from typing import Callable, Any 


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        if hasattr(record, "extra"):
            log_record.update(record.extra)

        return json.dumps(log_record)
    

def get_logger(name : str = "Prod") -> logging.Logger:
    """Get a logger with JSON formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class MetricsCollector:
    def __init__(self):
        self.requests_total = 0 
        self.errors_total = 0
        self.letency_total = 0.0
        self.letency_count = 0
        self.token_input = 0
        self.token_output = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def record_request(self, latency: float, tokens_in: int, tokens_out: int, cache_hit: bool):
        print(f"Recording request: latency={latency}, tokens_in={tokens_in}, tokens_out={tokens_out}, cache_hit={cache_hit}")
        self.requests_total += 1
        self.letency_total += latency
        self.letency_count += 1
        self.token_input += tokens_in
        self.token_output += tokens_out
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        return None

