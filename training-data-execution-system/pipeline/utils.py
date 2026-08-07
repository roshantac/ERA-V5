"""
Shared utilities: logging, hashing, path setup.
"""

import os
import json
import hashlib
import logging
import datetime
from pipeline import config


# ─── Directory bootstrap ───────────────────────────────────────────────────────
def ensure_dirs():
    for d in [
        config.BASE_DIR,
        config.MANIFESTS_DIR,
        config.LEDGERS_DIR,
        config.CHECKPOINTS_DIR,
        config.REPORTS_DIR,
        config.AUDITS_DIR,
        config.REPLAY_DIR,
        config.TESTS_DIR,
        config.RAW_DATA_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


# ─── Execution logger ──────────────────────────────────────────────────────────
_logger = None

def get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    ensure_dirs()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    file_handler = logging.FileHandler(config.EXECUTION_LOG, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    _logger = logging.getLogger("pipeline")
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()
    _logger.addHandler(file_handler)
    _logger.addHandler(stream_handler)
    return _logger


def log(msg: str):
    get_logger().info(msg)


# ─── Hashing helpers ───────────────────────────────────────────────────────────
def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_obj(obj) -> str:
    """Deterministic hash of any JSON-serialisable object."""
    serialised = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    return sha256_str(serialised)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── JSON helpers ──────────────────────────────────────────────────────────────
def write_json(path: str, obj, indent: int = 2):
    ensure_dirs()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: str, obj):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─── Timestamp ─────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"
