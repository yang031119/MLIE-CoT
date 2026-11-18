"""Export route-time states visited by the V10 online pipeline.

This module is intentionally CPU-only.  It ingests observations, not rollout
returns: the image and canonical diagnosis can become another atomic router
state, while prior actions and retain/rollback decisions remain audit-only
provenance and never enter ``diagnosis``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from inference.main_method.modules.contracts import (
	CONSTRAINT_AUDIT_KEYS,
	DIAGNOSIS_SCHEMA,
	ISSUE_TYPES,
	diagnosis_payload_is_valid,
	validate_qwen_analyzer_l1_distribution_contract,
)
from inference.main_method.modules.l3_policy import l3_policy_digest, validate_l3_policy_contract

from .canonical import (
	canonical_json,
	iter_jsonl,
	read_json,
	sha256_file,
	sha256_json,
	stable_id,
	write_if_identical_or_absent,
)
from .schema import (
	DataValidationError,
	canonicalize_diagnosis,
	image_descriptor,
	normalize_prompt_for_group,
	normalize_text,
)


EXPORT_SCHEMA_VERSION = "vcot_v10_online_visited_states_v1"
EXPORT_FILE_NAME = "states.jsonl"
MANIFEST_FILE_NAME = "manifest.json"
ROUND_DIR_RE = re.compile(r"^round(?P<round>[0-9]+)$")
ROUND_TAG_RE = re.compile(r"^round_(?P<round>[0-9]+)_triage$")

DIAGNOSIS_KEYS = {
	"schema_version",
	"fully_satisfied",
	"reason",
	"constraint_audit",
	"issues",
	"preservation_constraints",
}
ISSUE_KEYS = {
	"constraint_type",
	"target_entity",
	"observed",
	"expected",
	"evidence",
	"confidence",
}

# One entry per generated action.  Output/cache events are deliberately not
# counted as additional actions.
ACTION_EVENT_LEVEL = {
	"l1_joint_generation_start": 1,
	"global_flux2_edit": 2,
	"prompt_only_regenerate_candidate": 3,
}
RETENTION_EVENTS = {"l1_joint_review", "global_compare"}


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise DataValidationError(message)


def _read_object(path: Path, label: str) -> dict[str, Any]:
	try:
		value = read_json(path)
	except Exception as exc:
		raise DataValidationError(f"cannot read {label} {path}: {exc}") from exc
	_require(isinstance(value, Mapping), f"{label} must be a JSON object: {path}")
	return dict(value)


def _strict_canonical_diagnosis(value: Any, source: Path) -> dict[str, Any]:
	"""Accept only the exact route-visible diagnosis schema.

	The V10 ``error_analysis.json`` wrapper may contain ``raw_output`` for audit,
	but only its nested ``canonical_diagnosis`` reaches this function.  A direct
	diagnosis document, by contrast, may not contain wrapper/action/route keys.
	"""

	_require(isinstance(value, Mapping), f"canonical_diagnosis must be an object: {source}")
	payload = dict(value)
	_require(set(payload) == DIAGNOSIS_KEYS, f"diagnosis has non-canonical or missing keys: {source}")
	_require(payload.get("schema_version") == DIAGNOSIS_SCHEMA, f"diagnosis schema mismatch: {source}")
	audit = payload.get("constraint_audit")
	_require(isinstance(audit, Mapping), f"diagnosis constraint_audit must be an object: {source}")
	_require(set(audit) == set(CONSTRAINT_AUDIT_KEYS), f"diagnosis audit has non-canonical keys: {source}")
	issues = payload.get("issues")
	_require(isinstance(issues, list), f"diagnosis issues must be a list: {source}")
	for index, issue in enumerate(issues):
		_require(isinstance(issue, Mapping), f"diagnosis issue {index} must be an object: {source}")
		_require(set(issue) == ISSUE_KEYS, f"diagnosis issue {index} has non-canonical keys: {source}")
		_require(
			str(issue.get("constraint_type") or "").strip().lower() in ISSUE_TYPES,
			f"diagnosis issue {index} has an invalid constraint_type: {source}",
		)
	_require(diagnosis_payload_is_valid(payload), f"diagnosis violates the V10 contract: {source}")
	canonical = canonicalize_diagnosis(payload)
	_require(
		canonical_json(canonical) == canonical_json(payload),
		f"diagnosis is valid but not canonicalized: {source}",
	)
	return canonical


def _diagnosis_from_error_analysis(payload: Mapping[str, Any], source: Path) -> dict[str, Any]:
	if "canonical_diagnosis" in payload:
		_require(payload.get("valid_json") is not False, f"online wrapper marks diagnosis invalid: {source}")
		if payload.get("reflection_schema_version") is not None:
			_require(
				payload["reflection_schema_version"] == 10,
				f"online reflection wrapper is not V10: {source}",
			)
		return _strict_canonical_diagnosis(payload.get("canonical_diagnosis"), source)
	# A future online writer may persist the canonical object directly.  Exact
	# key equality ensures raw/action/route wrapper fields cannot be projected
	# away silently in this mode.
	return _strict_canonical_diagnosis(payload, source)


def _discover_analysis_files(inputs: Sequence[Path]) -> tuple[list[Path], list[Path]]:
	_require(bool(inputs), "at least one V10 output root/sample directory is required")
	roots: list[Path] = []
	paths: set[Path] = set()
	for supplied in inputs:
		root = supplied.resolve()
		_require(root.exists(), f"V10 input does not exist: {root}")
		roots.append(root)
		if root.is_file():
			_require(root.name == "error_analysis.json", f"input file is not error_analysis.json: {root}")
			paths.add(root)
			continue
		if ROUND_DIR_RE.fullmatch(root.name) and (root / "error_analysis.json").is_file():
			paths.add((root / "error_analysis.json").resolve())
		if (root / "error_analysis.json").is_file() and ROUND_DIR_RE.fullmatch(root.parent.name):
			paths.add((root / "error_analysis.json").resolve())
		for candidate in root.rglob("error_analysis.json"):
			if candidate.is_file() and ROUND_DIR_RE.fullmatch(candidate.parent.name):
				paths.add(candidate.resolve())
	return sorted(set(roots), key=str), sorted(paths, key=str)


def _source_inventory(paths: Iterable[Path]) -> list[dict[str, str]]:
	return [{"path": str(path), "sha256": sha256_file(path)} for path in sorted(paths, key=str)]


def _sample_prompt(sample_dir: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
	"""Resolve the original target prompt from a sample's durable artifacts."""
	status_path = sample_dir / "status.json"
	prompt_path = sample_dir / "prompt.txt"
	status: dict[str, Any] = {}
	status_prompt = ""
	if status_path.is_file():
		status = _read_object(status_path, "V10 status")
		if status.get("state_schema_version") is not None:
			_require(status["state_schema_version"] == 10, f"status is not V10: {status_path}")
		status_prompt = normalize_text(status.get("prompt") or status.get("target_prompt"))
	file_prompt = ""
	if prompt_path.is_file():
		try:
			file_prompt = normalize_text(prompt_path.read_text(encoding="utf-8"))
		except Exception as exc:
			raise DataValidationError(f"cannot read target prompt {prompt_path}: {exc}") from exc
	_require(bool(status_prompt or file_prompt), f"target prompt is absent from status.json/prompt.txt: {sample_dir}")
	if status_prompt and file_prompt:
		_require(
			normalize_prompt_for_group(status_prompt) == normalize_prompt_for_group(file_prompt),
			f"status.json and prompt.txt disagree: {sample_dir}",
		)
	prompt = status_prompt or file_prompt
	metadata = {
		"status": {
			"path": str(status_path.resolve()),
			"sha256": sha256_file(status_path),
		} if status_path.is_file() else None,
		"prompt": {
			"path": str(prompt_path.resolve()),
			"sha256": sha256_file(prompt_path),
		} if prompt_path.is_file() else None,
	}
	return prompt, status, metadata


def _source_qwen_distribution(sample_dir: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
	"""Lock a visited state to the online run that produced its diagnosis."""

	config_path: Path | None = None
	for directory in (sample_dir, *sample_dir.parents):
		candidate = directory / "run_config.json"
		if candidate.is_file():
			config_path = candidate.resolve()
			break
	_require(config_path is not None, f"source V10 run_config.json is missing for {sample_dir}")
	config = _read_object(config_path, "V10 run config")
	value = config.get("qwen_analyzer_l1_distribution")
	_require(isinstance(value, Mapping), f"run config has no Qwen analyzer/L1 distribution: {config_path}")
	try:
		distribution = validate_qwen_analyzer_l1_distribution_contract(value)
	except ValueError as exc:
		raise DataValidationError(f"invalid Qwen distribution in {config_path}: {exc}") from exc
	return distribution, {"path": str(config_path), "sha256": sha256_file(config_path)}, config


def _declared_image_path(value: Any, analysis_path: Path, project_root: Path) -> Path | None:
	text = str(value or "").strip()
	if not text:
		return None
	path = Path(text).expanduser()
	if path.is_absolute():
		return path.resolve() if path.is_file() else None
	for base in (project_root, analysis_path.parent, analysis_path.parent.parent):
		candidate = (base / path).resolve()
		if candidate.is_file():
			return candidate
	return None


def _triage_image(
	analysis_path: Path,
	payload: Mapping[str, Any],
	project_root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
	round_initial = analysis_path.parent / "initial.png"
	declared = _declared_image_path(payload.get("image"), analysis_path, project_root)
	selected: Path | None = round_initial.resolve() if round_initial.is_file() else declared
	if selected is None:
		return None, "missing_round_initial_and_declared_image"
	descriptor = image_descriptor(selected)
	if declared is not None and declared.resolve() != selected.resolve():
		declared_hash = sha256_file(declared)
		_require(
			declared_hash == descriptor["sha256"],
			f"round initial image and error_analysis.image disagree: {analysis_path}",
		)
	return descriptor, None


def _timeline_rows(path: Path) -> list[dict[str, Any]]:
	if not path.is_file():
		return []
	try:
		return list(iter_jsonl(path))
	except Exception as exc:
		raise DataValidationError(f"cannot read V10 timeline {path}: {exc}") from exc


def _round_from_event(event: Mapping[str, Any]) -> int | None:
	for key in ("round_before_action", "round"):
		value = event.get(key)
		if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
			return value
	match = ROUND_TAG_RE.fullmatch(str(event.get("tag") or ""))
	return int(match.group("round")) if match else None


def _prior_trajectory_summary(sample_dir: Path, source_round: int) -> dict[str, Any]:
	"""Summarize prior repairs using provenance only, excluding hidden rationale."""
	timeline_path = sample_dir / "timeline.jsonl"
	rows = _timeline_rows(timeline_path)
	target_tag = f"round_{source_round:02d}_triage"
	prior: list[dict[str, Any]] = []
	matched = False
	for row in rows:
		if row.get("event") == "qwen_diagnosis" and row.get("tag") == target_tag:
			matched = True
			break
		prior.append(row)

	routes: list[dict[str, Any]] = []
	actions: list[dict[str, Any]] = []
	retentions: list[dict[str, Any]] = []
	last_route: tuple[int | None, int | None] = (None, None)
	for event in prior if matched else []:
		name = str(event.get("event") or "")
		if name == "route_head_decision":
			decision = event.get("decision") if isinstance(event.get("decision"), Mapping) else {}
			level_value = decision.get("level")
			level = int(level_value) if isinstance(level_value, int) and not isinstance(level_value, bool) else None
			round_index = _round_from_event(event)
			last_route = (round_index, level)
			if level in {1, 2, 3}:
				routes.append({"source_round": round_index, "level": level})
		if name in ACTION_EVENT_LEVEL:
			reported_round = _round_from_event(event)
			action_round = last_route[0] if last_route[0] is not None else reported_round
			actions.append(
				{
					"event": name,
					"source_round": action_round,
					"reported_round": reported_round,
					"action_level": ACTION_EVENT_LEVEL[name],
				}
			)
		if name in RETENTION_EVENTS:
			review = event.get("review") if isinstance(event.get("review"), Mapping) else {}
			winner_role = normalize_text(review.get("winner_role")) or "unknown"
			retained_value = review.get("retained_repair")
			retained = (
				bool(retained_value)
				if isinstance(retained_value, bool)
				else winner_role not in {"", "unknown", "original"}
			)
			reported_round = _round_from_event(event)
			action_round = last_route[0] if last_route[0] is not None else reported_round
			retentions.append(
				{
					"event": name,
					"source_round": action_round,
					"reported_round": reported_round,
					"outcome": "retain_candidate" if retained else "rollback_original",
					"winner_role": winner_role,
				}
			)

	level_counts = Counter(int(item["action_level"]) for item in actions)
	return {
		"timeline": {
			"path": str(timeline_path.resolve()),
			"sha256": sha256_file(timeline_path),
		} if timeline_path.is_file() else None,
		"matched_current_diagnosis_event": matched,
		"prior_event_count": len(prior) if matched else 0,
		"route_decisions": routes,
		"action_events": actions,
		"retention_events": retentions,
		"counts": {
			"actions": len(actions),
			"l1": level_counts[1],
			"l2": level_counts[2],
			"l3": level_counts[3],
			"retained_candidates": sum(item["outcome"] == "retain_candidate" for item in retentions),
			"rollbacks": sum(item["outcome"] == "rollback_original" for item in retentions),
		},
	}


def _analysis_round(path: Path) -> int:
	match = ROUND_DIR_RE.fullmatch(path.parent.name)
	_require(match is not None, f"error_analysis.json is not in roundN/: {path}")
	return int(match.group("round"))


class _UnionFind:
	def __init__(self) -> None:
		self.parent: dict[str, str] = {}

	def find(self, value: str) -> str:
		"""Return the canonical component root with path compression."""
		self.parent.setdefault(value, value)
		if self.parent[value] != value:
			self.parent[value] = self.find(self.parent[value])
		return self.parent[value]

	def union(self, left: str, right: str) -> None:
		"""Merge two components using a deterministic lexical root."""
		left_root = self.find(left)
		right_root = self.find(right)
		if left_root != right_root:
			first, second = sorted((left_root, right_root))
			self.parent[second] = first


def _assign_prompt_families(candidates: list[dict[str, Any]]) -> None:
	"""Assign deterministic prompt-family labels used for balancing and audits."""
	union = _UnionFind()
	for candidate in candidates:
		target_node = "target:" + candidate["normalized_target_prompt"]
		union.find(target_node)
		declared = candidate["declared_prompt_family_id"]
		if declared:
			union.union(target_node, "declared:" + declared)
	components: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"targets": set(), "declared": set()})
	for candidate in candidates:
		target_node = "target:" + candidate["normalized_target_prompt"]
		component = components[union.find(target_node)]
		component["targets"].add(candidate["normalized_target_prompt"])
		if candidate["declared_prompt_family_id"]:
			component["declared"].add(candidate["declared_prompt_family_id"])
	family_ids = {
		root: stable_id(
			"online_prompt_family",
			{
				"targets": sorted(component["targets"]),
				"declared_prompt_family_ids": sorted(component["declared"]),
			},
			length=32,
		)
		for root, component in components.items()
	}
	for candidate in candidates:
		root = union.find("target:" + candidate["normalized_target_prompt"])
		candidate["prompt_family_id"] = family_ids[root]


def _state_identity(target_prompt: str, image_sha256: str, diagnosis: Mapping[str, Any]) -> str:
	return stable_id(
		"online_state",
		{
			"target_prompt": normalize_prompt_for_group(target_prompt),
			"image_sha256": image_sha256,
			"diagnosis_sha256": sha256_json(diagnosis),
		},
		length=32,
	)


def _candidate_from_analysis(
	analysis_path: Path,
	project_root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
	"""Convert one canonical analysis artifact into an atomic training candidate."""
	payload = _read_object(analysis_path, "V10 error analysis")
	diagnosis = _diagnosis_from_error_analysis(payload, analysis_path)
	if diagnosis["fully_satisfied"]:
		return None, "fully_satisfied_l0"
	round_index = _analysis_round(analysis_path)
	sample_dir = analysis_path.parent.parent
	target_prompt, status, prompt_sources = _sample_prompt(sample_dir)
	qwen_distribution, run_config_source, run_config = _source_qwen_distribution(sample_dir)
	try:
		l3_policy = validate_l3_policy_contract(run_config.get("l3_policy_contract"))
	except ValueError as exc:
		raise DataValidationError(f"source V10 run has no valid initial-base L3 policy: {exc}") from exc
	l3_digest = l3_policy_digest(l3_policy)
	declared_l3_digest = normalize_text(run_config.get("l3_policy_digest")).lower()
	if declared_l3_digest:
		_require(declared_l3_digest == l3_digest, f"source V10 run_config L3 policy digest mismatch: {sample_dir}")
	status_l3_digest = normalize_text(status.get("initial_generator_policy_digest")).lower()
	if status_l3_digest:
		_require(
			status_l3_digest == l3_digest,
			f"source V10 status/run_config L3 policy mismatch: {sample_dir}",
		)
	descriptor, image_exclusion = _triage_image(analysis_path, payload, project_root)
	if descriptor is None:
		return None, image_exclusion
	declared_family = normalize_text(status.get("prompt_family_id") or status.get("base_prompt_id"))
	status_sample_id = normalize_text(status.get("sample_id")) or sample_dir.name
	return {
		"analysis_path": analysis_path,
		"analysis_sha256": sha256_file(analysis_path),
		"sample_dir": sample_dir.resolve(),
		"status_sample_id": status_sample_id,
		"source_round": round_index,
		"target_prompt": target_prompt,
		"normalized_target_prompt": normalize_prompt_for_group(target_prompt),
		"declared_prompt_family_id": declared_family,
		"image": descriptor,
		"diagnosis": diagnosis,
		"prompt_sources": prompt_sources,
		"qwen_distribution_signature": qwen_distribution["distribution_signature"],
		"run_config_source": run_config_source,
		"l3_policy": l3_policy,
		"l3_policy_digest": l3_digest,
		"status_l3_policy_digest_verified": bool(status_l3_digest),
		"trajectory": _prior_trajectory_summary(sample_dir, round_index),
	}, None


def _materialize_rows(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
	"""Convert visited online states into grouped atomic counterfactual collection rows."""
	_assign_prompt_families(candidates)
	rows_by_state: dict[str, dict[str, Any]] = {}
	duplicate_count = 0
	for candidate in sorted(candidates, key=lambda item: str(item["analysis_path"])):
		state_id = _state_identity(
			candidate["target_prompt"], candidate["image"]["sha256"], candidate["diagnosis"]
		)
		if state_id in rows_by_state:
			duplicate_count += 1
			rows_by_state[state_id]["provenance"]["duplicate_source_count"] += 1
			continue
		family_id = candidate["prompt_family_id"]
		lineage_id = stable_id(
			"online_lineage",
			{
				"prompt_family_id": family_id,
				"source_sample_id": candidate["status_sample_id"],
				"source_sample_dir": str(candidate["sample_dir"]),
			},
			length=32,
		)
		group_id = stable_id("group", {"prompt_family_id": family_id}, length=32)
		rows_by_state[state_id] = {
			"state_id": state_id,
			"sample_id": state_id,
			"lineage_id": lineage_id,
			"group_id": group_id,
			"prompt_family_id": family_id,
			"dataset": "online_v10_visited_states",
			"source_prompt_id": stable_id(
				"online_prompt", candidate["normalized_target_prompt"], length=32
			),
			"variant_id": f"visited_round_{candidate['source_round']:02d}",
			"error_family": "online_v10_visited_state",
			"source_round": candidate["source_round"],
			"target_prompt": candidate["target_prompt"],
			"current_image": candidate["image"]["path"],
			"initial_generator": {
				"family": candidate["l3_policy"]["executor"]["family"],
				"model_id": candidate["l3_policy"]["executor"]["model_id"],
				"model_revision": candidate["l3_policy"]["executor"]["model_revision"],
				"auxiliary_model_id": candidate["l3_policy"]["executor"]["auxiliary_model_id"],
				"auxiliary_model_revision": candidate["l3_policy"]["executor"]["auxiliary_model_revision"],
				"seed": None,
			},
			"initial_record": {},
			"diagnosis": candidate["diagnosis"],
			"provenance": {
				"kind": "online_v10_visited_state",
				"source_sample_id": candidate["status_sample_id"],
				"source_sample_dir": str(candidate["sample_dir"]),
				"source_round": candidate["source_round"],
				"source_error_analysis": {
					"path": str(candidate["analysis_path"]),
					"sha256": candidate["analysis_sha256"],
				},
				"source_image": candidate["image"],
				"prompt_sources": candidate["prompt_sources"],
				"source_run_config": candidate["run_config_source"],
				"l3_policy_digest": candidate["l3_policy_digest"],
				"status_l3_policy_digest_verified": candidate["status_l3_policy_digest_verified"],
				"qwen_distribution_signature": candidate["qwen_distribution_signature"],
				"declared_prompt_family_id": candidate["declared_prompt_family_id"],
				"prior_trajectory_summary": candidate["trajectory"],
				"history_encoded_in_diagnosis": False,
				"multistep_return_computed": False,
				"duplicate_source_count": 1,
			},
		}
	rows = sorted(rows_by_state.values(), key=lambda row: row["state_id"])
	return rows, duplicate_count


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
	return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def export_online_states(
	inputs: Sequence[Path],
	output_dir: Path,
	*,
	project_root: Path,
) -> dict[str, Any]:
	"""Scan V10 roots and write deterministic planner-input states."""

	project_root = project_root.resolve()
	output_dir = output_dir.resolve()
	roots, analysis_paths = _discover_analysis_files(inputs)
	inventory = _source_inventory(analysis_paths)
	candidates: list[dict[str, Any]] = []
	exclusions: Counter[str] = Counter()
	for analysis_path in analysis_paths:
		candidate, excluded_reason = _candidate_from_analysis(analysis_path, project_root)
		if candidate is None:
			exclusions[str(excluded_reason or "unknown")] += 1
			continue
		candidates.append(candidate)
	signatures = {str(candidate["qwen_distribution_signature"]) for candidate in candidates}
	_require(len(signatures) == 1, "visited states are missing or mix Qwen analyzer/L1 distributions")
	qwen_distribution_signature = next(iter(signatures))
	l3_digests = {str(candidate["l3_policy_digest"]) for candidate in candidates}
	_require(len(l3_digests) == 1, "visited states mix initial-base L3 policies; export one model/revision per dataset")
	l3_policy_contract = dict(candidates[0]["l3_policy"])
	l3_contract_digest = next(iter(l3_digests))
	rows, duplicate_count = _materialize_rows(candidates)
	states_text = _jsonl_text(rows)
	states_path = output_dir / EXPORT_FILE_NAME
	write_if_identical_or_absent(states_path, states_text)
	manifest = {
		"schema_version": EXPORT_SCHEMA_VERSION,
		"status": "online_states_exported",
		"semantics": {
			"kind": "visited_state_ingestion_only",
			"history_feature": False,
			"multistep_return": False,
			"downstream_labeling": "atomic L1/L2/L3 counterfactual evaluation",
		},
		"source_roots": [str(root) for root in roots],
		"source_analysis_count": len(analysis_paths),
		"source_inventory_sha256": sha256_json(inventory),
		"eligible_before_dedup": len(candidates),
		"state_count": len(rows),
		"qwen_distribution_signature": qwen_distribution_signature,
		"l3_policy_contract": l3_policy_contract,
		"l3_policy_digest": l3_contract_digest,
		"status_l3_policy_digest_verified_count": sum(
			bool(candidate["status_l3_policy_digest_verified"]) for candidate in candidates
		),
		"duplicate_count": duplicate_count,
		"excluded_counts": dict(sorted(exclusions.items())),
		"dedup_policy": "normalized_target_prompt+image_sha256+canonical_diagnosis_sha256",
		"family_policy": "transitive_closure_of_normalized_target_prompt_or_declared_prompt_family_id",
		"states_content_sha256": sha256_json(rows),
		"states_file_sha256": sha256_file(states_path),
		"files": {"states": EXPORT_FILE_NAME},
	}
	write_if_identical_or_absent(
		output_dir / MANIFEST_FILE_NAME,
		json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
	)
	return manifest


def _validate_exported_row(
	row: Mapping[str, Any],
	index: int,
	*,
	expected_l3_policy: Mapping[str, Any],
	expected_l3_digest: str,
) -> None:
	"""Revalidate an exported state, its source hashes, prompt family, and immutable model policies."""
	label = f"exported state row {index}"
	diagnosis = _strict_canonical_diagnosis(row.get("diagnosis"), Path(label))
	_require(not diagnosis["fully_satisfied"], f"{label} is L0")
	image = image_descriptor(Path(str(row.get("current_image") or "")))
	provenance = row.get("provenance")
	_require(isinstance(provenance, Mapping), f"{label} provenance must be an object")
	_require(provenance.get("history_encoded_in_diagnosis") is False, f"{label} history flag mismatch")
	_require(provenance.get("multistep_return_computed") is False, f"{label} return flag mismatch")
	_require(provenance.get("source_image") == image, f"{label} source image descriptor mismatch")
	source = provenance.get("source_error_analysis")
	_require(isinstance(source, Mapping), f"{label} source analysis provenance missing")
	source_path = Path(str(source.get("path") or ""))
	_require(source_path.is_file(), f"{label} source analysis is missing: {source_path}")
	_require(source.get("sha256") == sha256_file(source_path), f"{label} source analysis hash mismatch")
	prompt_sources = provenance.get("prompt_sources")
	_require(isinstance(prompt_sources, Mapping), f"{label} prompt source provenance is invalid")
	for prompt_source_name, prompt_source in prompt_sources.items():
		if prompt_source is None:
			continue
		_require(isinstance(prompt_source, Mapping), f"{label} {prompt_source_name} provenance is invalid")
		prompt_source_path = Path(str(prompt_source.get("path") or ""))
		_require(prompt_source_path.is_file(), f"{label} prompt source is missing: {prompt_source_path}")
		_require(
			prompt_source.get("sha256") == sha256_file(prompt_source_path),
			f"{label} prompt source hash mismatch: {prompt_source_path}",
		)
	run_config_source = provenance.get("source_run_config")
	_require(isinstance(run_config_source, Mapping), f"{label} source run config is missing")
	run_config_path = Path(str(run_config_source.get("path") or ""))
	_require(run_config_path.is_file(), f"{label} source run config is missing: {run_config_path}")
	_require(
		run_config_source.get("sha256") == sha256_file(run_config_path),
		f"{label} source run config hash mismatch",
	)
	run_config = _read_object(run_config_path, "V10 run config")
	try:
		l3_policy = validate_l3_policy_contract(run_config.get("l3_policy_contract"))
	except ValueError as exc:
		raise DataValidationError(f"{label} source L3 policy is invalid: {exc}") from exc
	_require(l3_policy == dict(expected_l3_policy), f"{label} source L3 policy differs from export manifest")
	_require(l3_policy_digest(l3_policy) == expected_l3_digest, f"{label} source L3 policy digest mismatch")
	declared_l3_digest = normalize_text(run_config.get("l3_policy_digest")).lower()
	if declared_l3_digest:
		_require(declared_l3_digest == expected_l3_digest, f"{label} run config L3 policy digest mismatch")
	_require(provenance.get("l3_policy_digest") == expected_l3_digest, f"{label} provenance L3 policy digest mismatch")
	status_source = prompt_sources.get("status")
	status_l3_digest = ""
	if status_source is not None:
		status = _read_object(Path(str(status_source.get("path") or "")), "V10 status")
		status_l3_digest = normalize_text(status.get("initial_generator_policy_digest")).lower()
	_require(
		bool(provenance.get("status_l3_policy_digest_verified")) == bool(status_l3_digest),
		f"{label} status L3 verification flag mismatch",
	)
	if status_l3_digest:
		_require(status_l3_digest == expected_l3_digest, f"{label} status L3 policy digest mismatch")
	executor = l3_policy["executor"]
	_require(row.get("initial_generator") == {
		"family": executor["family"],
		"model_id": executor["model_id"],
		"model_revision": executor["model_revision"],
		"auxiliary_model_id": executor["auxiliary_model_id"],
		"auxiliary_model_revision": executor["auxiliary_model_revision"],
		"seed": None,
	}, f"{label} initial_generator mismatch")
	try:
		distribution = validate_qwen_analyzer_l1_distribution_contract(
			run_config.get("qwen_analyzer_l1_distribution")
		)
	except ValueError as exc:
		raise DataValidationError(f"{label} source Qwen distribution is invalid: {exc}") from exc
	_require(
		provenance.get("qwen_distribution_signature") == distribution["distribution_signature"],
		f"{label} Qwen distribution signature mismatch",
	)
	trajectory = provenance.get("prior_trajectory_summary")
	_require(isinstance(trajectory, Mapping), f"{label} prior trajectory summary is missing")
	timeline = trajectory.get("timeline")
	if timeline is not None:
		_require(isinstance(timeline, Mapping), f"{label} timeline provenance is invalid")
		timeline_path = Path(str(timeline.get("path") or ""))
		_require(timeline_path.is_file(), f"{label} timeline is missing: {timeline_path}")
		_require(timeline.get("sha256") == sha256_file(timeline_path), f"{label} timeline hash mismatch")
	expected_id = _state_identity(row.get("target_prompt") or "", image["sha256"], diagnosis)
	_require(row.get("state_id") == expected_id, f"{label} state_id mismatch")
	_require(bool(normalize_text(row.get("prompt_family_id"))), f"{label} prompt_family_id missing")
	_require(bool(normalize_text(row.get("lineage_id"))), f"{label} lineage_id missing")


def verify_online_states(output_dir: Path, *, inputs: Sequence[Path] = ()) -> dict[str, Any]:
	"""Verify content/source hashes, strict diagnosis isolation and grouping."""

	output_dir = output_dir.resolve()
	manifest = _read_object(output_dir / MANIFEST_FILE_NAME, "online-state manifest")
	_require(manifest.get("schema_version") == EXPORT_SCHEMA_VERSION, "online-state manifest schema mismatch")
	files = manifest.get("files")
	_require(isinstance(files, Mapping), "online-state manifest files must be an object")
	states_path = output_dir / str(files.get("states") or EXPORT_FILE_NAME)
	_require(states_path.is_file(), f"online-state JSONL is missing: {states_path}")
	rows = list(iter_jsonl(states_path))
	_require(len(rows) == int(manifest.get("state_count", -1)), "online-state count mismatch")
	_require(sha256_json(rows) == manifest.get("states_content_sha256"), "online-state content hash mismatch")
	_require(sha256_file(states_path) == manifest.get("states_file_sha256"), "online-state file hash mismatch")
	try:
		manifest_l3_policy = validate_l3_policy_contract(manifest.get("l3_policy_contract"))
	except ValueError as exc:
		raise DataValidationError(f"online-state manifest L3 policy is invalid: {exc}") from exc
	manifest_l3_digest = l3_policy_digest(manifest_l3_policy)
	_require(manifest.get("l3_policy_digest") == manifest_l3_digest, "online-state manifest L3 policy digest mismatch")
	seen: set[str] = set()
	target_families: dict[str, set[str]] = defaultdict(set)
	declared_families: dict[str, set[str]] = defaultdict(set)
	lineage_families: dict[str, set[str]] = defaultdict(set)
	for index, row in enumerate(rows):
		_validate_exported_row(
			row,
			index,
			expected_l3_policy=manifest_l3_policy,
			expected_l3_digest=manifest_l3_digest,
		)
		state_id = str(row["state_id"])
		_require(state_id not in seen, f"duplicate exported state_id: {state_id}")
		seen.add(state_id)
		family_id = str(row["prompt_family_id"])
		target_families[normalize_prompt_for_group(row["target_prompt"])].add(family_id)
		lineage_families[str(row["lineage_id"])].add(family_id)
		_require(
			(row.get("provenance") or {}).get("qwen_distribution_signature")
			== manifest.get("qwen_distribution_signature"),
			f"exported state row {index} differs from manifest Qwen distribution",
		)
		_require(
			(row.get("provenance") or {}).get("l3_policy_digest") == manifest_l3_digest,
			f"exported state row {index} differs from manifest L3 policy",
		)
		declared = normalize_text((row.get("provenance") or {}).get("declared_prompt_family_id"))
		if declared:
			declared_families[declared].add(family_id)
	for target, families in target_families.items():
		_require(len(families) == 1, f"target prompt crosses prompt families: {target}")
	for declared, families in declared_families.items():
		_require(len(families) == 1, f"declared prompt family was split: {declared}")
	for lineage, families in lineage_families.items():
		_require(len(families) == 1, f"lineage crosses prompt families: {lineage}")

	root_values = list(inputs) if inputs else [Path(item) for item in manifest.get("source_roots") or []]
	roots, analysis_paths = _discover_analysis_files(root_values)
	if inputs:
		_require(
			[str(root) for root in roots] == manifest.get("source_roots"),
			"provided verify inputs do not match frozen source roots",
		)
	inventory = _source_inventory(analysis_paths)
	_require(
		sha256_json(inventory) == manifest.get("source_inventory_sha256"),
		"V10 source inventory changed after export",
	)
	_require(len(analysis_paths) == int(manifest.get("source_analysis_count", -1)), "source analysis count mismatch")
	return {
		"schema_version": EXPORT_SCHEMA_VERSION,
		"status": "verified",
		"state_count": len(rows),
		"source_analysis_count": len(analysis_paths),
		"states_content_sha256": manifest["states_content_sha256"],
	}


__all__ = [
	"EXPORT_SCHEMA_VERSION",
	"export_online_states",
	"verify_online_states",
]
