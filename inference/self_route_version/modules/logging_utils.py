"""Terminal logging helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, List, Optional, Sequence

from .state import SampleState

_TERMINAL_LOG_STREAM: Optional[Any] = None
_TERMINAL_LOG_ENABLED = False

class TeeTextIO:
	"""Forward writes to the original terminal and an append-only mirror."""
	def __init__(self, stream: Any, mirror: Any):
		"""Store the user-visible stream and line-buffered run log."""
		self._stream = stream
		self._mirror = mirror

	def write(self, data: str) -> int:
		"""Write identical text to both destinations."""
		written = self._stream.write(data)
		self._mirror.write(data)
		return written

	def flush(self) -> None:
		"""Flush both streams so crash diagnostics are not left buffered."""
		self._stream.flush()
		self._mirror.flush()

	def fileno(self) -> int:
		"""Expose the original stream descriptor for library compatibility."""
		return self._stream.fileno()

	def isatty(self) -> bool:
		"""Report the terminal status of the original stream."""
		return bool(getattr(self._stream, "isatty", lambda: False)())

	def writable(self) -> bool:
		"""Advertise the writable text-stream interface."""
		return True

	def __getattr__(self, name: str) -> Any:
		"""Delegate unsupported stream attributes to the original stream."""
		return getattr(self._stream, name)


def log_stage(message: str) -> None:
	"""Print a timestamped, immediately flushed stage-level status line."""
	print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def describe_states_brief(states: Sequence["SampleState"], limit: int = 6) -> str:
	"""Summarize queued sample ids and their scheduler positions for logs."""
	items: List[str] = []
	for state in states[:limit]:
		if state.phase == "micro":
			items.append(
				f"{state.sample_id}(micro edit={state.micro_edit_index + 1}/{max(len(state.micro_plan), 1)} attempt={state.micro_attempt + 1})"
			)
		else:
			items.append(f"{state.sample_id}(global round={state.global_round})")
	if len(states) > limit:
		items.append(f"... +{len(states) - limit} more")
	return ", ".join(items)


def enable_terminal_logging(output_dir: Path) -> Path:
	"""Append a session header and tee stdout/stderr to ``terminal.log``."""
	global _TERMINAL_LOG_STREAM, _TERMINAL_LOG_ENABLED
	log_path = output_dir / "terminal.log"
	if _TERMINAL_LOG_ENABLED:
		return log_path
	output_dir.mkdir(parents=True, exist_ok=True)
	log_stream = log_path.open("a", encoding="utf-8", buffering=1)
	log_stream.write(
		f"\n===== session_start {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} cwd={os.getcwd()} =====\n"
	)
	log_stream.write(f"argv={json.dumps(sys.argv, ensure_ascii=False)}\n")
	log_stream.flush()
	sys.stdout = TeeTextIO(sys.stdout, log_stream)
	sys.stderr = TeeTextIO(sys.stderr, log_stream)
	_TERMINAL_LOG_STREAM = log_stream
	_TERMINAL_LOG_ENABLED = True
	return log_path
