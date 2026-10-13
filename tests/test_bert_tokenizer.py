"""BERT 分词器加载兜底：缺 tokenizer.json 时回退慢速 SentencePiece（桩替换，不加载权重）。"""

from __future__ import annotations

import pytest

from injection_guard.defense import prompt_detection as pd


class _FakeTokenizer:
    def __init__(self, name: str) -> None:
        self.name = name


def _patch(monkeypatch, *, fast, slow=None):
    """桩替换 ``AutoTokenizer.from_pretrained``，记录调用参数并按序返回/抛错。"""
    calls: list[dict] = []

    def fake_from_pretrained(path, **kwargs):
        calls.append({"path": path, **kwargs})
        if "use_fast" not in kwargs:
            if isinstance(fast, Exception):
                raise fast
            return fast
        if isinstance(slow, Exception):
            raise slow
        return slow

    monkeypatch.setattr(pd.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    return calls


def test_fast_path_used_when_available(tmp_path, monkeypatch):
    fast = _FakeTokenizer("fast")
    calls = _patch(monkeypatch, fast=fast)

    assert pd.load_bert_tokenizer(str(tmp_path)) is fast
    assert len(calls) == 1 and "use_fast" not in calls[0]


def test_falls_back_to_sentencepiece_when_fast_missing(tmp_path, monkeypatch):
    (tmp_path / "spm.model").write_bytes(b"fake")
    slow = _FakeTokenizer("slow")
    calls = _patch(monkeypatch, fast=ValueError("no tokenizer.json"), slow=slow)

    assert pd.load_bert_tokenizer(str(tmp_path)) is slow
    assert calls[1]["use_fast"] is False
    assert calls[1]["vocab_file"] == str(tmp_path / "spm.model")


def test_reraises_fast_error_without_sentencepiece_vocab(tmp_path, monkeypatch):
    fast_exc = ValueError("Converting from SentencePiece and Tiktoken failed")
    calls = _patch(monkeypatch, fast=fast_exc)

    with pytest.raises(ValueError, match="SentencePiece and Tiktoken"):
        pd.load_bert_tokenizer(str(tmp_path))
    assert len(calls) == 1


def test_slow_error_is_chained_when_sentencepiece_missing(tmp_path, monkeypatch):
    (tmp_path / "tokenizer.model").write_bytes(b"fake")
    fast_exc = ValueError("no tokenizer.json")
    slow_exc = ImportError("DebertaV2Tokenizer requires the SentencePiece library")
    _patch(monkeypatch, fast=fast_exc, slow=slow_exc)

    with pytest.raises(ValueError, match="no tokenizer.json") as info:
        pd.load_bert_tokenizer(str(tmp_path))
    assert info.value.__cause__ is slow_exc
