#!/usr/bin/env python3
"""Seed exact initial images from a compatible completed run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping

from .state import STATE_SCHEMA_VERSION, safe_prompt_id


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _link_or_copy(source: Path, target: Path) -> None:
	if target.exists():
		if _sha256(source) != _sha256(target):
			raise ValueError(f"existing seeded image differs: {target}")
		return
	try:
		os.link(source, target)
	except OSError:
		shutil.copy2(source, target)


def _source_by_prompt(source_run: Path, source_image_name: str) -> dict[str, Path]:
	result: dict[str, Path] = {}
	if Path(source_image_name).name != source_image_name:
		raise ValueError(f"source image name must be a plain filename: {source_image_name!r}")
	for prompt_path in source_run.glob("*/prompt.txt"):
		prompt = prompt_path.read_text(encoding="utf-8").strip()
		image = prompt_path.parent / source_image_name
		if not prompt or not image.is_file():
			continue
		if prompt in result:
			raise ValueError(f"duplicate source prompt: {prompt!r}")
		result[prompt] = image
	return result


def _validate_source_policy(
	source_run: Path,
	target_policy: Mapping[str, Any],
	policy_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
	config_path = source_run / "run_config.json"
	if not config_path.is_file():
		raise FileNotFoundError(f"source run_config.json missing: {config_path}")
	config = json.loads(config_path.read_text(encoding="utf-8"))
	source_policy = config.get("l3_policy_contract")
	if isinstance(source_policy, dict):
		for field in ("executor", "generation"):
			if source_policy.get(field) != target_policy.get(field):
				raise ValueError(
					f"source initial-generator {field} differs from target policy: "
					f"source={source_policy.get(field)!r} target={target_policy.get(field)!r}"
				)
		return config, {
			"mode": "strict",
			"status": "exact_policy_match",
			"source_policy_present": True,
		}
	if policy_mode != "legacy-attested":
		raise ValueError("source run does not contain an L3/initial-generator policy contract")
	source_family = str(config.get("initial_generator") or "").strip().lower()
	target_family = str((target_policy.get("executor") or {}).get("family") or "").strip().lower()
	family_aliases = {"flux": "flux2", "flux2": "flux2", "bagel": "bagel", "sd35": "sd35"}
	if family_aliases.get(source_family) != target_family:
		raise ValueError(
			"legacy source initial-generator family differs from the target policy: "
			f"source={source_family!r} target={target_family!r}"
		)
	return config, {
		"mode": "legacy-attested",
		"status": "family_match_without_historical_immutable_policy",
		"source_policy_present": False,
		"source_initial_generator": source_family,
		"target_initial_generator": target_family,
	}


def seed_initial_images(
	*,
	source_run: Path,
	target_run: Path,
	prompts: Iterable[tuple[int, str]],
	initial_generator_family: str,
	initial_generator_policy_digest: str,
	target_policy: Mapping[str, Any],
	source_image_name: str = "initial.png",
	source_policy_mode: str = "strict",
) -> dict[str, Any]:
	"""Materialize policy-verified initial images as resumable schema-11 states."""
	source_run = source_run.resolve()
	target_run = target_run.resolve()
	source_config, source_policy_validation = _validate_source_policy(
		source_run,
		target_policy,
		source_policy_mode,
	)
	source_images = _source_by_prompt(source_run, source_image_name)
	target_run.mkdir(parents=True, exist_ok=True)
	rows = []
	for index, prompt in prompts:
		source = source_images.get(prompt)
		if source is None:
			raise FileNotFoundError(f"source initial missing for prompt index={index}: {prompt!r}")
		sample_id = safe_prompt_id(index, prompt)
		sample_dir = target_run / sample_id
		sample_dir.mkdir(parents=True, exist_ok=True)
		initial = sample_dir / "initial.png"
		_link_or_copy(source, initial)
		(sample_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
		source_generation_prompt = source.parent / "generation_prompt.txt"
		generation_prompt = (
			source_generation_prompt.read_text(encoding="utf-8").strip()
			if source_generation_prompt.is_file()
			else prompt
		)
		(sample_dir / "generation_prompt.txt").write_text(generation_prompt + "\n", encoding="utf-8")
		generation_payload = {
			"generation_prompt": generation_prompt,
			"valid_json": True,
			"seeded": True,
			"source_run": str(source_run),
			"source_image": str(source),
		}
		(sample_dir / "generation_prompt.json").write_text(
			json.dumps(generation_payload, ensure_ascii=False, indent=2) + "\n",
			encoding="utf-8",
		)
		now = time.time()
		source_hash = _sha256(source)
		event = {
			"event": "initial_generate",
			"model": initial_generator_family.upper(),
			"image": str(initial),
			"seeded": True,
			"source_run": str(source_run),
			"source_image": str(source),
			"source_sha256": source_hash,
			"generation_prompt": generation_prompt,
			"initial_generator_policy_digest": initial_generator_policy_digest,
			"timestamp": now,
			"phase": "global",
		}
		status = {
			"state_schema_version": STATE_SCHEMA_VERSION,
			"index": index,
			"sample_id": sample_id,
			"prompt": prompt,
			"initial_generator_family": initial_generator_family,
			"initial_generator_policy_digest": initial_generator_policy_digest,
			"generation_prompt": generation_prompt,
			"phase": "global",
			"status": "pending",
			"done": False,
			"global_round": 0,
			"micro_edit_index": 0,
			"current_image": str(initial),
			"final_image": None,
			"final_reason": None,
			"needs_global_analysis": True,
			"pending_global_analysis_reason": "seeded_policy_verified_initial",
			"pending_global_compare": None,
			"pending_local_review": None,
			"micro_plan": [],
			"micro_box_review_done": False,
			"repair_memory": [],
			"best_round": None,
			"best_rank": 999,
			"error": None,
			"started_at": now,
			"updated_at": now,
		}
		status_path = sample_dir / "status.json"
		if status_path.exists():
			existing = json.loads(status_path.read_text(encoding="utf-8"))
			if int(existing.get("state_schema_version", -1)) != STATE_SCHEMA_VERSION:
				raise ValueError(f"cannot overwrite incompatible status: {status_path}")
			if (
				str(existing.get("initial_generator_policy_digest") or "")
				!= initial_generator_policy_digest
			):
				raise ValueError(f"existing seeded policy digest differs: {status_path}")
		else:
			status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
			(sample_dir / "timeline.jsonl").write_text(
				json.dumps(event, ensure_ascii=False) + "\n",
				encoding="utf-8",
			)
		rows.append(
			{
				"index": index,
				"sample_id": sample_id,
				"prompt": prompt,
				"source_image": str(source),
				"target_image": str(initial),
				"sha256": source_hash,
			}
		)
	manifest = {
		"schema": "vcot_self_route_policy_verified_initials_v1",
		"source_run": str(source_run),
		"source_run_config": str(source_run / "run_config.json"),
		"source_image_name": source_image_name,
		"source_policy_validation": source_policy_validation,
		"source_initial_generator": source_config.get("initial_generator"),
		"target_run": str(target_run),
		"initial_generator_family": initial_generator_family,
		"initial_generator_policy_digest": initial_generator_policy_digest,
		"seeded": len(rows),
		"rows": rows,
	}
	(target_run / "seeded_initials.json").write_text(
		json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	return manifest
