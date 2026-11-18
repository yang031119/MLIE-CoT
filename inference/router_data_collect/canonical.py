"""Canonical serialization and hashing helpers used by router-data artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Iterator, Mapping


def canonical_json(value: Any) -> str:
	"""Return a stable JSON representation suitable for hashing."""

	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
	"""Return the SHA-256 digest of an exact byte sequence."""
	return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
	"""Hash UTF-8 text without newline or whitespace normalization."""
	return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
	"""Hash a value after deterministic canonical-JSON serialization."""
	return sha256_text(canonical_json(value))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
	"""Stream a file into SHA-256 so large artifacts need not enter memory."""
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		while True:
			chunk = handle.read(chunk_size)
			if not chunk:
				break
			digest.update(chunk)
	return digest.hexdigest()


def stable_id(prefix: str, value: Any, length: int = 24) -> str:
	"""Derive a reproducible identifier from a namespace and canonical payload."""
	return f"{prefix}_{sha256_json(value)[:length]}"


def stable_seed(*parts: Any) -> int:
	"""Return a deterministic positive 31-bit seed independent of row order."""

	digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).digest()
	return int.from_bytes(digest[:8], "big") % (2**31 - 1) + 1


def read_json(path: Path) -> Any:
	"""Read one UTF-8 JSON document from disk."""
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
	"""Yield non-empty JSON objects from JSONL with precise line diagnostics."""
	with path.open("r", encoding="utf-8") as handle:
		for line_number, line in enumerate(handle, start=1):
			text = line.strip()
			if not text:
				continue
			try:
				value = json.loads(text)
			except json.JSONDecodeError as exc:
				raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
			if not isinstance(value, dict):
				raise ValueError(f"Expected JSON object at {path}:{line_number}")
			yield value


def _atomic_text(path: Path, text: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
	try:
		with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
			handle.write(text)
			handle.flush()
			os.fsync(handle.fileno())
		os.replace(temporary_name, path)
	except Exception:
		try:
			os.unlink(temporary_name)
		except FileNotFoundError:
			pass
		raise


def write_json_atomic(path: Path, value: Any) -> None:
	"""Atomically replace a JSON document using canonical serialization."""
	_atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
	"""Atomically replace a JSONL file using canonical row serialization."""
	text = "".join(canonical_json(dict(row)) + "\n" for row in rows)
	_atomic_text(path, text)


def write_if_identical_or_absent(path: Path, text: str) -> None:
	"""Write a planned artifact, rejecting incompatible reuse of its path."""

	if path.exists():
		existing = path.read_text(encoding="utf-8")
		if existing != text:
			raise ValueError(f"Refusing to overwrite non-identical artifact: {path}")
		return
	_atomic_text(path, text)


def archive_existing(path: Path) -> Path | None:
	"""Preserve an earlier attempt before an explicitly requested retry."""

	if not path.exists():
		return None
	attempts_dir = path.parent / "attempts"
	attempts_dir.mkdir(parents=True, exist_ok=True)
	index = 1
	while True:
		candidate = attempts_dir / f"{path.stem}_attempt_{index:03d}{path.suffix}"
		if not candidate.exists():
			shutil.copy2(path, candidate)
			return candidate
		index += 1
