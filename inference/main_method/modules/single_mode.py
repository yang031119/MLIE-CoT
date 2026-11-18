"""Normalize legacy single-backend selections.

These helpers do not bypass V10 diagnosis or reviews.  They only map a
compatibility ``single_model`` option onto the generator/editor family used by
the shared pipeline.
"""

from __future__ import annotations

import argparse
from typing import Optional


def single_model(args: argparse.Namespace) -> str:
	"""Return the normalized compatibility selector, defaulting to ``off``."""
	return str(getattr(args, "single_model", "off") or "off").strip().lower()


def single_local_model(args: argparse.Namespace) -> Optional[str]:
	"""Map a supported single-model selector to its local editor family."""
	model = single_model(args)
	if model == "bagel":
		return "bagel"
	if model == "flux2":
		return "flux2"
	return None


def single_global_model(args: argparse.Namespace) -> Optional[str]:
	"""Map a supported selector to the corresponding global generator family."""
	return single_local_model(args)
