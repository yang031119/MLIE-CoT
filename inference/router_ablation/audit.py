"""Post-run integrity audit for V10 route-ablation arms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from inference.main_method.modules.io_utils import select_prompts
from inference.main_method.modules.state import safe_prompt_id

from .protocol import FIXED_LEVELS, normalize_strategy


def _load_object(path: Path) -> dict[str, Any]:
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except Exception as exc:
		raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
	if not isinstance(payload, dict):
		raise ValueError(f"expected JSON object in {path}")
	return payload


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def audit_ablation_run(
	*,
	output_dir: Path,
	strategy: str,
	data_path: Path,
	start_index: int,
	limit: int | None,
	seed_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
	"""Check completeness, same-initial bytes, and fixed-level route purity."""
	output_dir = Path(output_dir)
	strategy = normalize_strategy(strategy)
	expected_level = FIXED_LEVELS.get(strategy)
	selected = select_prompts(Path(data_path), int(start_index), limit)
	errors: list[str] = []
	cross_level: list[dict[str, Any]] = []
	classification_counts: dict[str, int] = {}
	action_plan_counts: dict[str, int] = {}
	status_counts: dict[str, int] = {}
	completed = 0
	initial_hashes = dict(seed_report.get("initial_hashes") or {}) if seed_report else {}
	if not selected:
		errors.append("selected prompt range is empty")

	for index, prompt in selected:
		sample_id = safe_prompt_id(index, prompt)
		sample_dir = output_dir / sample_id
		status_path = sample_dir / "status.json"
		summary_path = sample_dir / "summary.json"
		final_path = sample_dir / "final.png"
		initial_path = sample_dir / "initial.png"
		if not status_path.is_file():
			errors.append(f"{sample_id}: missing status.json")
			continue
		status = _load_object(status_path)
		status_name = str(status.get("status") or "unknown")
		status_counts[status_name] = status_counts.get(status_name, 0) + 1
		if status.get("error") or status_name == "error":
			errors.append(f"{sample_id}: pipeline status is error")
		if not summary_path.is_file() or not final_path.is_file() or not bool(status.get("done")):
			errors.append(f"{sample_id}: run is incomplete")
		else:
			completed += 1
		if sample_id in initial_hashes:
			if not initial_path.is_file():
				errors.append(f"{sample_id}: missing seeded initial.png")
			elif _sha256(initial_path) != str(initial_hashes[sample_id]):
				errors.append(f"{sample_id}: initial.png changed after seeding")

		for classification_path in sorted(sample_dir.glob("round*/level_classification.json")):
			payload = _load_object(classification_path)
			decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
			source = str(decision.get("source") or "unknown")
			level = decision.get("level")
			key = f"{source}:L{level if level is not None else 'none'}"
			classification_counts[key] = classification_counts.get(key, 0) + 1
			if expected_level is None or bool(payload.get("fully_satisfied")):
				continue
			if source == "action_budget_gate":
				continue
			if source != "fixed_repair_level_ablation" or int(level or 0) != expected_level:
				cross_level.append(
					{
						"sample_id": sample_id,
						"artifact": str(classification_path),
						"expected_level": expected_level,
						"actual_level": level,
						"decision_source": source,
						"original_level": decision.get("original_level"),
						"planning_failure": decision.get("planning_failure"),
					}
				)

		for plan_path in sorted(sample_dir.glob("round*/repair_plan.json")):
			payload = _load_object(plan_path)
			plan = payload.get("action_plan") if isinstance(payload.get("action_plan"), dict) else {}
			level = plan.get("level")
			key = f"L{level if level is not None else 'none'}"
			action_plan_counts[key] = action_plan_counts.get(key, 0) + 1
			if expected_level is not None and int(level or 0) != expected_level:
				cross_level.append(
					{
						"sample_id": sample_id,
						"artifact": str(plan_path),
						"expected_level": expected_level,
						"actual_level": level,
						"decision_source": "action_plan",
					}
				)

	if cross_level:
		errors.append(
			f"{len(cross_level)} fixed-level purity violation(s); fixed-L1 planner failures must not be reported as pure L1"
		)
	report = {
		"strategy": strategy,
		"expected_samples": len(selected),
		"completed_samples": completed,
		"status_counts": status_counts,
		"classification_counts": classification_counts,
		"action_plan_counts": action_plan_counts,
		"same_initial_checked": bool(initial_hashes),
		"cross_level_violations": cross_level,
		"errors": errors,
		"valid": not errors and completed == len(selected),
	}
	path = output_dir / "route_ablation_audit.json"
	path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	return report
