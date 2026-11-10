"""Disk-backed JSONL cache for LLM responses.

Keyed by (provider, model, prompt_hash). Identical re-runs are free.

The cache is append-only: every call writes a single JSONL line, and on load the
file is read once into an in-memory dict. Concurrent writes from a single Python
process are safe (an `fsync` line-by-line append).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import Lock


def prompt_hash(provider: str, model: str, system: str, user: str, **extra) -> str:
    """Stable cache key. Callers MUST pass every parameter that affects the output:
    `max_tokens`, `temperature`, `top_p`, `stop`, `task`, `shots`, and a
    `fewshot_signature` digest of the few-shot exemplar list. The wrappers in
    `tasks/*.py` do this for you — only call `prompt_hash` directly if you know
    what you're doing.
    """
    payload = {
        "provider": provider,
        "model": model,
        "system": system,
        "user": user,
        **extra,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def fewshot_signature(examples) -> str:
    """Deterministic digest of a few-shot exemplar list.

    Used so changing the few-shot pool (different seed, larger pool, edited labels)
    invalidates the cache automatically.
    """
    if not examples:
        return "0"
    blob = json.dumps(
        [(ex.source, ex.mt, ex.label) for ex in examples],
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class ResponseCache:
    """Append-only JSONL store of (key → response_text)."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, str] = {}
        self._lock = Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = rec.get("key")
                v = rec.get("response")
                if k and v is not None:
                    self._mem[k] = v

    def get(self, key: str) -> str | None:
        return self._mem.get(key)

    def put(self, key: str, response: str) -> None:
        with self._lock:
            if key in self._mem and self._mem[key] == response:
                return
            self._mem[key] = response
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def __contains__(self, key: str) -> bool:
        return key in self._mem

    def __len__(self) -> int:
        return len(self._mem)
