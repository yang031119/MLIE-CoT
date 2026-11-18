"""Canonical artifacts must have identical hash bytes on every platform."""
from inference.router_data_collect.canonical import (
    canonical_json, sha256_file, sha256_text, write_jsonl_atomic,
)


def test_jsonl_hash_is_independent_of_platform_newlines(tmp_path):
    row = {"prompt": "A red square labeled ①"}
    path = tmp_path / "rows.jsonl"
    write_jsonl_atomic(path, [row])
    assert path.read_bytes() == (canonical_json(row) + "\n").encode("utf-8")
    assert sha256_file(path) == sha256_text(canonical_json(row) + "\n")
