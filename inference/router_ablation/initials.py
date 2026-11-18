"""Exact-initial seeding for V10 route ablations.

Only V10 sources are accepted.  Treating a legacy V9 image as if it carried a
V10 immutable initial/L3 policy would create false provenance, so callers must
first produce one policy-bound V10 source run.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from inference.main_method.modules.flux_policy import flux2_policy_digest as compute_flux2_policy_digest
from inference.main_method.modules.flux_policy import validate_flux2_policy_contract
from inference.main_method.modules.io_utils import append_event, ensure_round_initial, save_status, select_prompts
from inference.main_method.modules.l3_policy import (
	l3_policy_digest,
	normalize_initial_generator_family,
	validate_l3_policy_contract,
)
from inference.main_method.modules.state import STATE_SCHEMA_VERSION, SampleState, safe_prompt_id


def _load_object(path: Path, label: str) -> dict[str, Any]:
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except Exception as exc:
		raise ValueError(f"invalid {label} at {path}: {exc}") from exc
	if not isinstance(payload, dict):
		raise ValueError(f"invalid {label} at {path}: expected a JSON object")
	return payload


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _materialize(source: Path, target: Path, mode: str) -> str:
	target.parent.mkdir(parents=True, exist_ok=True)
	if mode == "copy":
		shutil.copy2(source, target)
		return "copy"
	try:
		os.link(source, target)
		return "hardlink"
	except OSError as exc:
		if exc.errno != errno.EXDEV:
			raise
		shutil.copy2(source, target)
		return "copy_exdev_fallback"


def _validated_source_policy(
	source_run: Path,
	requested_family: str,
	requested_flux2_digest: str,
	*,
	allow_cross_policy_initial_assets: bool,
) -> tuple[dict[str, Any], str, str]:
	"""Validate the source run's immutable FLUX.2 and initial/L3 policies before image reuse."""
	config_path = source_run / "run_config.json"
	if not config_path.is_file():
		raise ValueError(f"source is not a V10 run: missing {config_path}")
	config = _load_object(config_path, "V10 source run_config")
	source_family = normalize_initial_generator_family(config.get("initial_generator"))
	if source_family != requested_family:
		raise ValueError(
			f"initial generator mismatch: source={source_family} requested={requested_family}"
		)
	policy_payload = config.get("l3_policy_contract")
	if not isinstance(policy_payload, dict):
		raise ValueError(
			"source V10 run has no immutable l3_policy_contract; it cannot establish exact-initial provenance"
		)
	policy = validate_l3_policy_contract(policy_payload)
	if normalize_initial_generator_family(policy["executor"]["family"]) != requested_family:
		raise ValueError("source V10 L3 policy family does not match its initial generator")
	digest = l3_policy_digest(policy)
	declared = str(config.get("l3_policy_digest") or "").strip().lower()
	if not declared or declared != digest:
		raise ValueError("source V10 run_config has a missing or inconsistent l3_policy_digest")
	flux2_payload = config.get("flux2_collector_policy")
	declared_flux2_digest = str(config.get("flux2_policy_digest") or "").strip().lower()
	if not isinstance(flux2_payload, dict):
		ablation_config_path = source_run / "route_ablation_config.json"
		if ablation_config_path.is_file():
			ablation_config = _load_object(ablation_config_path, "V10 source route-ablation config")
			flux2_payload = ablation_config.get("flux2_policy_contract")
			declared_flux2_digest = str(ablation_config.get("flux2_policy_digest") or "").strip().lower()
	if not isinstance(flux2_payload, dict):
		raise ValueError("source V10 run has no immutable flux2_collector_policy")
	flux2_policy = validate_flux2_policy_contract(flux2_payload, require_immutable_revision=True)
	actual_flux2_digest = compute_flux2_policy_digest(flux2_policy)
	if not declared_flux2_digest or declared_flux2_digest != actual_flux2_digest:
		raise ValueError("source V10 run_config has a missing or inconsistent flux2_policy_digest")
	if (
		actual_flux2_digest != str(requested_flux2_digest).strip().lower()
		and not allow_cross_policy_initial_assets
	):
		raise ValueError(
			"source and target FLUX.2 L1/L2 policy digests differ; all ablation arms must share one editor policy"
		)
	return policy, digest, actual_flux2_digest


def _validate_source_sample(
	*,
	source_run: Path,
	index: int,
	prompt: str,
	family: str,
	policy_digest: str,
) -> dict[str, Any]:
	"""Verify one source sample's prompt identity, status, timeline, and initial-image hash."""
	sample_id = safe_prompt_id(index, prompt)
	sample_dir = source_run / sample_id
	image_path = sample_dir / "initial.png"
	status_path = sample_dir / "status.json"
	if not image_path.is_file() or image_path.stat().st_size <= 0:
		raise ValueError(f"source sample has no non-empty initial.png: {sample_id}")
	if not status_path.is_file():
		raise ValueError(f"source sample has no V10 status.json: {sample_id}")
	status = _load_object(status_path, f"source status for {sample_id}")
	if int(status.get("state_schema_version", -1)) != STATE_SCHEMA_VERSION:
		raise ValueError(
			f"source sample {sample_id} uses state schema {status.get('state_schema_version')!r}, "
			f"expected V10 schema {STATE_SCHEMA_VERSION}"
		)
	if int(status.get("index", -1)) != index or str(status.get("prompt") or "") != prompt:
		raise ValueError(f"source sample identity/prompt mismatch: {sample_id}")
	status_family = normalize_initial_generator_family(status.get("initial_generator_family"))
	if status_family != family:
		raise ValueError(f"source sample initial family mismatch: {sample_id}")
	status_digest = str(status.get("initial_generator_policy_digest") or "").strip().lower()
	if status_digest != policy_digest:
		raise ValueError(f"source sample initial policy digest mismatch: {sample_id}")
	return {
		"sample_id": sample_id,
		"sample_dir": sample_dir,
		"image_path": image_path,
		"status": status,
		"sha256": _sha256(image_path),
	}


def _validate_existing_target(
	*,
	target_image: Path,
	target_status: Path,
	index: int,
	prompt: str,
	family: str,
	policy_digest: str,
	expected_sha256: str,
) -> None:
	"""Validate existing target and reject contract violations."""
	if not target_image.is_file() or not target_status.is_file():
		raise ValueError(
			f"target sample is only partially seeded: {target_image.parent}; use a clean directory"
		)
	if _sha256(target_image) != expected_sha256:
		raise ValueError(f"existing target initial differs from source: {target_image}")
	status = _load_object(target_status, f"target status for {target_image.parent.name}")
	if int(status.get("state_schema_version", -1)) != STATE_SCHEMA_VERSION:
		raise ValueError(f"existing target is not a V10 state: {target_image.parent}")
	if int(status.get("index", -1)) != index or str(status.get("prompt") or "") != prompt:
		raise ValueError(f"existing target sample identity/prompt mismatch: {target_image.parent}")
	if normalize_initial_generator_family(status.get("initial_generator_family")) != family:
		raise ValueError(f"existing target initial family mismatch: {target_image.parent}")
	if str(status.get("initial_generator_policy_digest") or "").strip().lower() != policy_digest:
		raise ValueError(f"existing target initial policy mismatch: {target_image.parent}")


def seed_v10_initials(
	*,
	source_run: Path,
	target_run: Path,
	data_path: Path,
	start_index: int,
	limit: int | None,
	initial_generator: str,
	flux2_policy_digest: str,
	requested_l3_policy_digest: str,
	copy_mode: str = "hardlink",
	allow_cross_policy_initial_assets: bool = False,
) -> dict[str, Any]:
	"""Seed exact V10 initial states and return a compact provenance report."""
	source_run = Path(source_run).resolve()
	target_run = Path(target_run).resolve()
	if source_run == target_run:
		raise ValueError("source and target run directories must differ")
	if copy_mode not in {"hardlink", "copy"}:
		raise ValueError(f"unsupported initial copy mode: {copy_mode!r}")
	family = normalize_initial_generator_family(initial_generator)
	_policy, source_policy_digest, source_flux2_policy_digest = _validated_source_policy(
		source_run,
		family,
		flux2_policy_digest,
		allow_cross_policy_initial_assets=allow_cross_policy_initial_assets,
	)
	target_policy_digest = str(requested_l3_policy_digest).strip().lower()
	if source_policy_digest != target_policy_digest and not allow_cross_policy_initial_assets:
		raise ValueError(
			"source and target initial/L3 policy digests differ; all ablation arms must share one initial-base policy"
		)
	selected = select_prompts(Path(data_path), int(start_index), limit)
	if not selected:
		raise ValueError("selected prompt range is empty")

	# Validate the complete source set before writing any target sample.
	source_samples = [
		_validate_source_sample(
			source_run=source_run,
			index=index,
			prompt=prompt,
			family=family,
			policy_digest=source_policy_digest,
		)
		for index, prompt in selected
	]

	seeded = 0
	already_seeded = 0
	mode_counts: dict[str, int] = {}
	hashes: dict[str, str] = {}
	for (index, prompt), source in zip(selected, source_samples):
		sample_id = source["sample_id"]
		target_dir = target_run / sample_id
		target_image = target_dir / "initial.png"
		target_status = target_dir / "status.json"
		hashes[sample_id] = source["sha256"]
		if target_image.exists() or target_status.exists():
			_validate_existing_target(
				target_image=target_image,
				target_status=target_status,
				index=index,
				prompt=prompt,
				family=family,
				policy_digest=target_policy_digest,
				expected_sha256=source["sha256"],
			)
			already_seeded += 1
			continue

		target_dir.mkdir(parents=True, exist_ok=True)
		(target_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
		actual_mode = _materialize(source["image_path"], target_image, copy_mode)
		mode_counts[actual_mode] = mode_counts.get(actual_mode, 0) + 1
		generation_prompt = source["status"].get("generation_prompt")
		state = SampleState(index=index, prompt=prompt, sample_dir=target_dir, sample_id=sample_id)
		state.initial_generator_family = family
		# This field binds any L3 action executed by the target run. The imported
		# image's actual source policy remains separately recorded below.
		state.initial_generator_policy_digest = target_policy_digest
		state.generation_prompt = str(generation_prompt) if generation_prompt else None
		state.current_image = target_image
		state.needs_global_analysis = True
		state.pending_global_analysis_reason = "seeded_route_ablation_initial"
		append_event(
			state,
			{
				"event": "initial_generate",
				"model": {"bagel": "BAGEL", "flux2": "FLUX2", "sd35": "SD35_TURBO"}[family],
				"image": str(target_image),
				"seeded": True,
				"source_run": str(source_run),
				"source_image": str(source["image_path"]),
				"source_sha256": source["sha256"],
				"policy_digest": target_policy_digest,
				"initial_asset_imported": source_policy_digest != target_policy_digest,
				"source_initial_policy_digest": source_policy_digest,
				"source_flux2_policy_digest": source_flux2_policy_digest,
				"target_l3_policy_digest": target_policy_digest,
				"target_flux2_policy_digest": str(flux2_policy_digest).strip().lower(),
				"generation_prompt": state.generation_prompt,
			},
		)
		ensure_round_initial(state)
		save_status(state)
		seeded += 1

	target_run.mkdir(parents=True, exist_ok=True)
	report = {
		"source_run": str(source_run),
		"target_run": str(target_run),
		"initial_generator_family": family,
		"initial_generator_policy_digest": target_policy_digest,
		"source_initial_generator_policy_digest": source_policy_digest,
		"source_flux2_policy_digest": source_flux2_policy_digest,
		"flux2_policy_digest": str(flux2_policy_digest).strip().lower(),
		"cross_policy_initial_asset_import": source_policy_digest != target_policy_digest
		or source_flux2_policy_digest != str(flux2_policy_digest).strip().lower(),
		"selected": len(selected),
		"seeded": seeded,
		"already_seeded": already_seeded,
		"copy_modes": mode_counts,
		"initial_hashes": hashes,
	}
	manifest_path = target_run / "route_ablation_initials.json"
	if manifest_path.is_file():
		existing = _load_object(manifest_path, "existing route-ablation initial manifest")
		for key in (
			"source_run",
			"initial_generator_family",
			"initial_generator_policy_digest",
			"source_initial_generator_policy_digest",
			"source_flux2_policy_digest",
			"flux2_policy_digest",
			"cross_policy_initial_asset_import",
			"selected",
			"initial_hashes",
		):
			if existing.get(key) != report.get(key):
				raise ValueError("existing route-ablation initial manifest is incompatible with this seed request")
	else:
		manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	return report
