"""Tests for the response cache and the prompt-hash helpers."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from indicqe_ape.cache import ResponseCache, fewshot_signature, prompt_hash
from indicqe_ape.prompts import Example


class TestPromptHash:
    def test_stable_for_same_input(self) -> None:
        a = prompt_hash("openai", "gpt-4o", "sys", "user")
        b = prompt_hash("openai", "gpt-4o", "sys", "user")
        assert a == b

    def test_changes_with_provider(self) -> None:
        a = prompt_hash("openai", "gpt-4o", "sys", "user")
        b = prompt_hash("anthropic", "gpt-4o", "sys", "user")
        assert a != b

    def test_changes_with_kwargs(self) -> None:
        a = prompt_hash("openai", "gpt-4o", "sys", "user", temperature=0.0)
        b = prompt_hash("openai", "gpt-4o", "sys", "user", temperature=0.5)
        assert a != b

    def test_changes_with_shots(self) -> None:
        a = prompt_hash("openai", "gpt-4o", "sys", "user", shots=0)
        b = prompt_hash("openai", "gpt-4o", "sys", "user", shots=4)
        assert a != b


class TestFewshotSignature:
    def test_empty_is_zero(self) -> None:
        assert fewshot_signature([]) == "0"
        assert fewshot_signature(None) == "0"

    def test_stable_for_same_examples(self) -> None:
        ex = [Example("s", "m", "85"), Example("s2", "m2", "60")]
        assert fewshot_signature(ex) == fewshot_signature(ex)

    def test_changes_when_label_changes(self) -> None:
        a = [Example("s", "m", "85")]
        b = [Example("s", "m", "60")]
        assert fewshot_signature(a) != fewshot_signature(b)

    def test_changes_when_order_changes(self) -> None:
        a = [Example("s1", "m", "1"), Example("s2", "m", "2")]
        b = [Example("s2", "m", "2"), Example("s1", "m", "1")]
        assert fewshot_signature(a) != fewshot_signature(b)


class TestResponseCache:
    def test_get_put_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "c.jsonl"
            cache = ResponseCache(path)
            assert cache.get("k") is None
            cache.put("k", "value")
            assert cache.get("k") == "value"
            assert "k" in cache

    def test_persists_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "c.jsonl"
            c1 = ResponseCache(path)
            c1.put("k", "value")
            del c1
            c2 = ResponseCache(path)
            assert c2.get("k") == "value"

    def test_idempotent_puts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "c.jsonl"
            cache = ResponseCache(path)
            cache.put("k", "v1")
            cache.put("k", "v1")  # same value — should NOT append twice
            with path.open("r", encoding="utf-8") as fh:
                lines = [l for l in fh if l.strip()]
            assert len(lines) == 1

    def test_overwrite_appends_new_line(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "c.jsonl"
            cache = ResponseCache(path)
            cache.put("k", "v1")
            cache.put("k", "v2")  # different value — appends, in-memory wins
            assert cache.get("k") == "v2"
