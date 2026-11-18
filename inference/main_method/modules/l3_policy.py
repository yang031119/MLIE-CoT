"""Immutable L3 policy bound to the initial-image generator.

L1/L2 are deliberately controlled FLUX.2 edits.  L3 is different: it throws
away the current image and therefore must use the same text-to-image base that
created the initial state.  Router labels are only comparable when that L3
executor is frozen, so a route-head checkpoint owns exactly one canonical L3
policy and serving fails closed when ``--initial-generator`` or its model
identity/settings differ.

This module has no GPU imports and is shared by collection, training, and
online startup validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .contracts import ROUTER_ACTION_POLICY_VERSION


L3_POLICY_SCHEMA = "vcot_v10_initial_base_l3_policy_v1"
L3_BINDING = "exact_initial_generator_family_and_model_identity"
L3_PROMPT_VERSION = "vcot_v10_l3_initial_base_prompt_v2_positive"
SUPPORTED_L3_FAMILIES = ("bagel", "flux2", "sd35")
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$", flags=re.IGNORECASE)

_IMPLEMENTATIONS = {
	"bagel": "UniCoT.BAGEL.generate_image",
	"flux2": "diffusers.Flux2Pipeline",
	"sd35": "diffusers.StableDiffusion3Pipeline",
}
_LOADERS = {
	"bagel": "BAGEL.from_pretrained(local_files_only=True)",
	"flux2": "from_pretrained(local_files_only=True)",
	"sd35": "from_pretrained(local_files_only=True)",
}
_ACTION_SEMANTICS = {
	"operation": "generate_text_only",
	"candidate_count": 1,
	"successful_executor_call_count": 1,
	"semantic_retry_count": 0,
	"references": "none",
}


def normalize_initial_generator_family(value: Any) -> str:
	"""Map user-facing backend aliases to one canonical L3 family name."""
	name = str(value or "").strip().lower().replace("-", "").replace("_", "").replace(".", "")
	aliases = {
		"bagel": "bagel",
		"flux": "flux2",
		"flux2": "flux2",
		"flux2dev": "flux2",
		"sd35": "sd35",
		"sd35turbo": "sd35",
		"stablediffusion35": "sd35",
		"stablediffusion35turbo": "sd35",
	}
	if name not in aliases:
		raise ValueError(
			f"Unsupported initial generator for L3: {value!r}; expected bagel, flux/flux2, or sd35"
		)
	return aliases[name]


def _canonical_json(value: Any) -> str:
	"""Serialize policy data deterministically for digest computation."""
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def l3_policy_digest(contract: Mapping[str, Any]) -> str:
	"""Hash the canonical policy for compact status/resume comparisons."""
	return hashlib.sha256(_canonical_json(dict(contract)).encode("utf-8")).hexdigest()


def _text(value: Any, name: str) -> str:
	"""Require and return a non-empty textual policy field."""
	result = str(value or "").strip()
	if not result:
		raise ValueError(f"L3 {name} is required")
	return result


def _revision(value: Any, name: str, require_immutable: bool) -> str:
	"""Normalize a revision and optionally require a commit-like hex id."""
	revision = _text(value, name).lower()
	if require_immutable and IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
		raise ValueError(f"L3 {name} must be an immutable 40-64 character hexadecimal snapshot")
	return revision


def build_l3_policy_contract(
	*,
	family: str,
	model_id: str,
	model_revision: str,
	resolution: int,
	steps: int,
	guidance_scale: float,
	max_sequence_length: int | None,
	dtype: str,
	prompt_version: str,
	auxiliary_model_id: str = "",
	auxiliary_model_revision: str = "",
	max_latent_size: int | None = None,
	require_immutable_revision: bool = True,
) -> dict[str, Any]:
	"""Build one canonical prompt-only executor policy.

	For BAGEL, ``auxiliary_model_*`` identifies the fine-tuned checkpoint and is
	required.  For FLUX.2/SD3.5 those fields must be empty.  Fields that are not
	used by an executor are represented by ``None`` so every serialized policy
	has one strict shape.
	"""

	family = normalize_initial_generator_family(family)
	model_id = _text(model_id, "model_id")
	model_revision = _revision(model_revision, "model_revision", require_immutable_revision)
	auxiliary_model_id = str(auxiliary_model_id or "").strip()
	auxiliary_model_revision = str(auxiliary_model_revision or "").strip().lower()
	if family == "bagel":
		auxiliary_model_id = _text(auxiliary_model_id, "auxiliary_model_id")
		auxiliary_model_revision = _revision(
			auxiliary_model_revision,
			"auxiliary_model_revision",
			require_immutable_revision,
		)
	else:
		if auxiliary_model_id or auxiliary_model_revision:
			raise ValueError(f"L3 {family} must not declare an auxiliary model")
	resolution = int(resolution)
	steps = int(steps)
	guidance_scale = float(guidance_scale)
	if resolution <= 0 or steps <= 0:
		raise ValueError("L3 resolution and steps must be positive")
	if not math.isfinite(guidance_scale) or guidance_scale < 0:
		raise ValueError("L3 guidance_scale must be finite and non-negative")
	if max_sequence_length is not None:
		max_sequence_length = int(max_sequence_length)
		if max_sequence_length <= 0:
			raise ValueError("L3 max_sequence_length must be positive when present")
	if family == "bagel":
		if max_sequence_length is not None:
			raise ValueError("BAGEL L3 does not use max_sequence_length")
		max_latent_size = int(max_latent_size or 0)
		if max_latent_size <= 0:
			raise ValueError("BAGEL L3 max_latent_size must be positive")
	else:
		if max_sequence_length is None:
			raise ValueError(f"{family} L3 requires max_sequence_length")
		if max_latent_size is not None:
			raise ValueError(f"{family} L3 must not declare max_latent_size")
	contract = {
		"schema_version": L3_POLICY_SCHEMA,
		"action_policy_version": ROUTER_ACTION_POLICY_VERSION,
		"binding": L3_BINDING,
		"executor": {
			"family": family,
			"implementation": _IMPLEMENTATIONS[family],
			"loader": _LOADERS[family],
			"model_id": model_id,
			"model_revision": model_revision,
			"auxiliary_model_id": auxiliary_model_id,
			"auxiliary_model_revision": auxiliary_model_revision,
			"dtype": str(dtype or "").replace("torch.", "").strip().lower(),
		},
		"generation": {
			"resolution": resolution,
			"steps": steps,
			"guidance_scale": guidance_scale,
			"max_sequence_length": max_sequence_length,
			"max_latent_size": max_latent_size,
		},
		"action": dict(_ACTION_SEMANTICS),
		"prompt_version": _text(prompt_version, "prompt_version"),
		"infrastructure_retry_policy": "exact_same_request_only; no extra semantic candidate",
	}
	if contract["executor"]["dtype"] not in {"bfloat16", "float16", "float32", "runtime_default"}:
		raise ValueError(f"Unsupported L3 dtype: {dtype!r}")
	return contract


def validate_l3_policy_contract(
	value: Mapping[str, Any],
	*,
	require_immutable_revision: bool = True,
) -> dict[str, Any]:
	"""Validate strict shape, executor binding, settings, and one-call semantics."""
	if not isinstance(value, Mapping):
		raise ValueError("L3 policy must be an object")
	if set(value) != {
		"schema_version", "action_policy_version", "binding", "executor",
		"generation", "action", "prompt_version", "infrastructure_retry_policy",
	}:
		raise ValueError("L3 policy fields are incomplete or contain unknown values")
	if value.get("schema_version") != L3_POLICY_SCHEMA:
		raise ValueError("L3 policy schema mismatch")
	if value.get("action_policy_version") != ROUTER_ACTION_POLICY_VERSION:
		raise ValueError("L3 action-policy version mismatch")
	if value.get("binding") != L3_BINDING:
		raise ValueError("L3 policy is not bound to the initial generator")
	executor = value.get("executor")
	generation = value.get("generation")
	if not isinstance(executor, Mapping) or not isinstance(generation, Mapping):
		raise ValueError("L3 executor and generation must be objects")
	if set(executor) != {
		"family", "implementation", "loader", "model_id", "model_revision",
		"auxiliary_model_id", "auxiliary_model_revision", "dtype",
	}:
		raise ValueError("L3 executor fields mismatch")
	if set(generation) != {"resolution", "steps", "guidance_scale", "max_sequence_length", "max_latent_size"}:
		raise ValueError("L3 generation fields mismatch")
	family = normalize_initial_generator_family(executor.get("family"))
	if executor.get("implementation") != _IMPLEMENTATIONS[family] or executor.get("loader") != _LOADERS[family]:
		raise ValueError("L3 executor implementation/loader mismatch")
	if dict(value.get("action") or {}) != _ACTION_SEMANTICS:
		raise ValueError("L3 one-call/candidate semantics mismatch")
	if value.get("infrastructure_retry_policy") != "exact_same_request_only; no extra semantic candidate":
		raise ValueError("L3 retry semantics mismatch")
	rebuilt = build_l3_policy_contract(
		family=family,
		model_id=executor.get("model_id"),
		model_revision=executor.get("model_revision"),
		auxiliary_model_id=executor.get("auxiliary_model_id"),
		auxiliary_model_revision=executor.get("auxiliary_model_revision"),
		dtype=executor.get("dtype"),
		resolution=generation.get("resolution"),
		steps=generation.get("steps"),
		guidance_scale=generation.get("guidance_scale"),
		max_sequence_length=generation.get("max_sequence_length"),
		max_latent_size=generation.get("max_latent_size"),
		prompt_version=value.get("prompt_version"),
		require_immutable_revision=require_immutable_revision,
	)
	if dict(value) != rebuilt:
		raise ValueError("L3 policy is not canonical")
	return rebuilt


def validate_matching_l3_policy(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
	"""Reject a serving L3 policy that differs from router-training utilities."""
	expected_contract = validate_l3_policy_contract(expected)
	actual_contract = validate_l3_policy_contract(actual)
	if expected_contract != actual_contract:
		raise ValueError(
			"Online L3 initial-base policy differs from router-training counterfactuals: "
			f"expected={l3_policy_digest(expected_contract)} actual={l3_policy_digest(actual_contract)}"
		)


def build_online_l3_policy(
	args: Any,
	*,
	flux2_policy: Mapping[str, Any] | None = None,
	require_immutable_revision: bool = True,
) -> dict[str, Any]:
	"""Build the actual online L3 contract from the selected initial backend."""

	family = normalize_initial_generator_family(getattr(args, "initial_generator", ""))
	resolution = int(getattr(args, "resolution", 1024))
	if family == "flux2":
		if flux2_policy is None:
			raise ValueError("Flux.2 initial-base L3 requires the resolved online Flux.2 policy")
		executor = flux2_policy["executor"]
		generation = flux2_policy["generation"]
		return build_l3_policy_contract(
			family=family,
			model_id=executor["model_id"],
			model_revision=executor["model_revision"],
			resolution=resolution,
			steps=int(generation["steps"]),
			guidance_scale=float(generation["guidance_scale"]),
			max_sequence_length=int(generation["max_sequence_length"]),
			dtype=executor["dtype"],
			prompt_version=L3_PROMPT_VERSION,
			require_immutable_revision=require_immutable_revision,
		)
	if family == "bagel":
		return build_l3_policy_contract(
			family=family,
			model_id=str(getattr(args, "bagel_model_id", "") or getattr(args, "bagel_model_path", "")),
			model_revision=str(getattr(args, "bagel_model_revision", "") or ""),
			auxiliary_model_id=str(
				getattr(args, "bagel_fine_tuned_model_id", "")
				or getattr(args, "bagel_fine_tuned_path", "")
			),
			auxiliary_model_revision=str(getattr(args, "bagel_fine_tuned_model_revision", "") or ""),
			resolution=resolution,
			steps=int(getattr(args, "bagel_steps", 50)),
			guidance_scale=float(getattr(args, "bagel_cfg_scale", 4.0)),
			max_sequence_length=None,
			max_latent_size=int(getattr(args, "max_latent_size", 64)),
			dtype="bfloat16",
			prompt_version=L3_PROMPT_VERSION,
			require_immutable_revision=require_immutable_revision,
		)
	return build_l3_policy_contract(
		family=family,
		model_id=str(getattr(args, "sd35_model_id", "") or getattr(args, "sd35_model", "")),
		model_revision=str(getattr(args, "sd35_model_revision", "") or ""),
		resolution=resolution,
		steps=int(getattr(args, "sd35_steps", 4)),
		guidance_scale=float(getattr(args, "sd35_guidance_scale", 0.0)),
		max_sequence_length=int(getattr(args, "sd35_max_sequence_length", 512)),
		dtype=str(getattr(args, "sd35_dtype", "bfloat16")),
		prompt_version=L3_PROMPT_VERSION,
		require_immutable_revision=require_immutable_revision,
	)


def _snapshot_repo_id(path: Path) -> str | None:
	"""Recover a Hugging Face repository id encoded in a cache path."""
	for part in path.parts:
		if part.startswith("models--"):
			segments = part[len("models--") :].split("--")
			if len(segments) >= 2 and all(segments):
				return "/".join(segments)
	return None


def resolve_initial_base_snapshot(
	model: str,
	*,
	model_revision: str,
	model_id: str = "",
) -> tuple[Path, str, str]:
	"""Resolve actual model bytes and reject mutable/declared-only identities."""

	raw_model = str(model or "").strip()
	revision = str(model_revision or "").strip().lower()
	if not raw_model:
		raise ValueError("initial-base model path/repo is required")
	if IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
		raise ValueError("formal routed L3 requires an immutable 40-64 hex model revision")
	path = Path(raw_model).expanduser()
	if path.exists():
		resolved = path.resolve()
		if not resolved.is_dir() or resolved.parent.name != "snapshots" or resolved.name.lower() != revision:
			raise ValueError(
				"initial-base model must point directly to snapshots/<revision>; "
				f"resolved path={str(resolved)!r} declared revision={revision!r}"
			)
		declared_identity = str(model_id or "").strip()
		path_identity = _snapshot_repo_id(resolved)
		if declared_identity and path_identity and declared_identity != path_identity:
			raise ValueError(
				"initial-base model_id differs from the repository encoded by the snapshot path: "
				f"declared={declared_identity!r} path={path_identity!r}"
			)
		identity = declared_identity or path_identity or str(resolved)
		return resolved, identity, revision
	identity = str(model_id or "").strip() or raw_model
	roots = [
		Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub",
	]
	cache_name = "models--" + identity.replace("/", "--")
	for root in roots:
		candidate = root / cache_name / "snapshots" / revision
		if candidate.is_dir():
			return candidate.resolve(), identity, revision
	raise FileNotFoundError(f"Pinned initial-base snapshot not found: {identity}@{revision}")


def resolve_online_l3_policy(
	args: Any,
	*,
	flux2_policy: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
	"""Resolve the actual online initial/L3 bytes, then build their policy."""

	family = normalize_initial_generator_family(getattr(args, "initial_generator", ""))
	if family == "flux2":
		return {}, build_online_l3_policy(
			args,
			flux2_policy=flux2_policy,
			require_immutable_revision=True,
		)
	if family == "bagel":
		base_path, base_id, base_revision = resolve_initial_base_snapshot(
			str(getattr(args, "bagel_model_path", "") or ""),
			model_id=str(getattr(args, "bagel_model_id", "") or ""),
			model_revision=str(getattr(args, "bagel_model_revision", "") or ""),
		)
		ft_path, ft_id, ft_revision = resolve_initial_base_snapshot(
			str(getattr(args, "bagel_fine_tuned_path", "") or ""),
			model_id=str(getattr(args, "bagel_fine_tuned_model_id", "") or ""),
			model_revision=str(getattr(args, "bagel_fine_tuned_model_revision", "") or ""),
		)
		contract = build_l3_policy_contract(
			family=family,
			model_id=base_id,
			model_revision=base_revision,
			auxiliary_model_id=ft_id,
			auxiliary_model_revision=ft_revision,
			resolution=int(getattr(args, "resolution", 1024)),
			steps=int(getattr(args, "bagel_steps", 50)),
			guidance_scale=float(getattr(args, "bagel_cfg_scale", 4.0)),
			max_sequence_length=None,
			max_latent_size=int(getattr(args, "max_latent_size", 64)),
			dtype="bfloat16",
			prompt_version=L3_PROMPT_VERSION,
		)
		return {"model": base_path, "auxiliary_model": ft_path}, contract
	model_path, model_id, revision = resolve_initial_base_snapshot(
		str(getattr(args, "sd35_model", "") or ""),
		model_id=str(getattr(args, "sd35_model_id", "") or ""),
		model_revision=str(getattr(args, "sd35_model_revision", "") or ""),
	)
	contract = build_l3_policy_contract(
		family=family,
		model_id=model_id,
		model_revision=revision,
		resolution=int(getattr(args, "resolution", 1024)),
		steps=int(getattr(args, "sd35_steps", 4)),
		guidance_scale=float(getattr(args, "sd35_guidance_scale", 0.0)),
		max_sequence_length=int(getattr(args, "sd35_max_sequence_length", 512)),
		dtype=str(getattr(args, "sd35_dtype", "bfloat16")),
		prompt_version=L3_PROMPT_VERSION,
	)
	return {"model": model_path}, contract


__all__ = [
	"L3_BINDING",
	"L3_POLICY_SCHEMA",
	"L3_PROMPT_VERSION",
	"SUPPORTED_L3_FAMILIES",
	"build_l3_policy_contract",
	"build_online_l3_policy",
	"l3_policy_digest",
	"normalize_initial_generator_family",
	"resolve_initial_base_snapshot",
	"resolve_online_l3_policy",
	"validate_l3_policy_contract",
	"validate_matching_l3_policy",
]
