"""Build deterministic, auditable prompt pairs for harder repair states.

The generator operates on structured V7 metadata and uses no model.  It adds
objective relation, composition, scene, and multi-constraint failures instead
of assigning a desired L1/L2/L3 label.  Repair levels remain outcome-derived.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .canonical import canonical_json, sha256_json, sha256_text, stable_id, stable_seed, write_json_atomic


SCHEMA_VERSION = "v10_advanced_prompt_pair_v1"
PROMPT_POLICY_VERSION = "objective_constraint_graph_perturbations_v1"
DEFAULT_INPUT = Path("results/router_data/samples.jsonl")
DEFAULT_OUTPUT_ROOT = Path("results/router_data")
DEFAULT_SEED = 20260715

COLORS = ("red", "blue", "green", "yellow", "purple", "orange", "black", "white", "brown", "pink")
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
RELATION_PAIRS = (
	("to the left of", "to the right of"),
	("above", "below"),
	("in front of", "behind"),
)
SCENE_PAIRS = (
	("against a plain light studio background", "inside a dark cluttered room"),
	("in a clean, bright gallery", "in a crowded outdoor market at night"),
	("on an uncluttered indoor set in daylight", "in a cramped, dim storage area"),
	("in an open sunlit plaza", "inside a busy narrow hallway at night"),
)
IRREGULAR_PLURALS = {
	"person": "people",
	"knife": "knives",
	"sheep": "sheep",
	"scissors": "scissors",
	"mouse": "mice",
	"computer mouse": "computer mice",
	"tooth": "teeth",
	"wine glass": "wine glasses",
}
PAIR_OBJECTS = {"scissors", "skis"}


class AdvancedPromptError(RuntimeError):
	"""Raised when a deterministic prompt contract cannot be satisfied."""


def _normalize(value: Any) -> str:
	return " ".join(str(value or "").strip().split())


def _choice(options: Sequence[Any], *parts: Any) -> Any:
	if not options:
		raise AdvancedPromptError("cannot choose from an empty option sequence")
	index = int(sha256_json(parts)[:16], 16) % len(options)
	return options[index]


def _ranked(options: Iterable[str], *parts: Any) -> list[str]:
	return sorted(set(options), key=lambda item: sha256_json([*parts, item]))


def _plural(noun: str) -> str:
	noun = _normalize(noun)
	if noun in IRREGULAR_PLURALS:
		return IRREGULAR_PLURALS[noun]
	if noun.endswith(("s", "x", "z", "ch", "sh")):
		return noun + "es"
	if noun.endswith("y") and len(noun) > 1 and noun[-2].lower() not in "aeiou":
		return noun[:-1] + "ies"
	if noun.endswith("fe"):
		return noun[:-2] + "ves"
	if noun.endswith("f"):
		return noun[:-1] + "ves"
	return noun + "s"


def _article(phrase: str) -> str:
	return "an" if phrase[:1].lower() in "aeiou" else "a"


def _entity_phrase(entity: Mapping[str, Any], *, force_count: bool = False) -> str:
	count = int(entity.get("count", 1))
	color = _normalize(entity.get("color"))
	noun = _normalize(entity.get("class"))
	prefix = f"{color} " if color else ""
	if noun in PAIR_OBJECTS:
		if count == 1:
			return f"a pair of {prefix}{noun}"
		return f"{NUMBER_WORDS.get(count, str(count))} pairs of {prefix}{noun}"
	if count == 1 and not force_count:
		body = f"{prefix}{noun}"
		return f"{_article(body)} {body}"
	return f"{NUMBER_WORDS.get(count, str(count))} {prefix}{_plural(noun)}"


def _entities_from_row(row: Mapping[str, Any]) -> list[dict[str, Any]]:
	metadata = row.get("source_metadata") if isinstance(row.get("source_metadata"), Mapping) else {}
	include = metadata.get("include") if isinstance(metadata.get("include"), list) else []
	entities: list[dict[str, Any]] = []
	seen: set[str] = set()
	for raw in include:
		if not isinstance(raw, Mapping):
			continue
		name = _normalize(raw.get("class"))
		if not name or name in seen:
			continue
		seen.add(name)
		entities.append(
			{
				"id": f"entity_{len(entities) + 1}",
				"class": name,
				"count": int(raw.get("count", 1) or 1),
				"color": _normalize(raw.get("color")) or None,
			}
		)
	return entities


def _object_vocabulary(rows: Sequence[Mapping[str, Any]]) -> list[str]:
	values: set[str] = set()
	for row in rows:
		for entity in _entities_from_row(row):
			values.add(entity["class"])
	if len(values) < 8:
		raise AdvancedPromptError("source metadata has too few distinct object classes")
	return sorted(values)


def _ensure_entities(
	row: Mapping[str, Any], vocabulary: Sequence[str], count: int, recipe_id: str
) -> list[dict[str, Any]]:
	"""Choose distinct deterministic entities required by an advanced recipe."""
	entities = [dict(entity) for entity in _entities_from_row(row)]
	used = {entity["class"] for entity in entities}
	candidates = _ranked(
		(name for name in vocabulary if name not in used),
		PROMPT_POLICY_VERSION,
		recipe_id,
		row["lineage_id"],
		"supplemental_objects",
	)
	for name in candidates:
		if len(entities) >= count:
			break
		entities.append(
			{
				"id": f"entity_{len(entities) + 1}",
				"class": name,
				"count": 1,
				"color": None,
			}
		)
	if len(entities) < count:
		raise AdvancedPromptError(f"cannot obtain {count} distinct entities for {row['lineage_id']}")
	return entities[:count]


def _colorize(entities: Sequence[Mapping[str, Any]], row: Mapping[str, Any], recipe_id: str) -> list[dict[str, Any]]:
	result: list[dict[str, Any]] = []
	used: set[str] = set()
	available = _ranked(COLORS, PROMPT_POLICY_VERSION, recipe_id, row["lineage_id"], "colors")
	for index, entity in enumerate(entities):
		value = dict(entity)
		color = _normalize(value.get("color"))
		if not color or color in used:
			color = next(candidate for candidate in available if candidate not in used)
		value["color"] = color
		used.add(color)
		result.append(value)
	return result


def _graph(
	*,
	entities: Sequence[Mapping[str, Any]],
	relations: Sequence[Mapping[str, str]],
	scene: str,
	framing: str,
	composition: str,
) -> dict[str, Any]:
	return {
		"entities": [dict(entity) for entity in entities],
		"relations": [dict(relation) for relation in relations],
		"scene": scene,
		"framing": framing,
		"composition": composition,
	}


def _coupled_relation_binding(
	row: Mapping[str, Any], vocabulary: Sequence[str]
) -> dict[str, Any]:
	"""Build a prompt whose entities participate in coupled spatial relations."""
	recipe_id = "coupled_relation_binding_v1"
	left, right = _colorize(_ensure_entities(row, vocabulary, 2, recipe_id), row, recipe_id)
	target_relation, corrupt_relation = _choice(
		RELATION_PAIRS, PROMPT_POLICY_VERSION, recipe_id, row["lineage_id"], "relation"
	)
	target_scene, corrupt_scene = _choice(
		SCENE_PAIRS, PROMPT_POLICY_VERSION, recipe_id, row["lineage_id"], "scene"
	)
	corrupt_left = {**left, "color": right["color"]}
	corrupt_right = {**right, "color": left["color"]}
	target_prompt = (
		f"A wide, eye-level photograph of {_entity_phrase(left)} {target_relation} {_entity_phrase(right)}, "
		f"with both objects fully visible and separated by clear empty space, {target_scene}."
	)
	initial_prompt = (
		f"A tight close-up photograph of {_entity_phrase(corrupt_left)} {corrupt_relation} "
		f"{_entity_phrase(corrupt_right)}, overlapping near the center with the second object partly occluded "
		f"and both edges cropped, {corrupt_scene}."
	)
	return {
		"recipe_id": recipe_id,
		"collection_stratum": "coupled_relation_and_attribute",
		"target_prompt": target_prompt,
		"initial_generation_prompt": initial_prompt,
		"target_constraint_graph": _graph(
			entities=[left, right],
			relations=[{"subject": left["id"], "predicate": target_relation, "object": right["id"]}],
			scene=target_scene,
			framing="wide eye-level; all objects fully visible",
			composition="separated by clear empty space",
		),
		"initial_constraint_graph": _graph(
			entities=[corrupt_left, corrupt_right],
			relations=[{"subject": left["id"], "predicate": corrupt_relation, "object": right["id"]}],
			scene=corrupt_scene,
			framing="tight close-up; cropped edges",
			composition="central overlap; second entity partly occluded",
		),
		"corruption_operations": [
			"swap_color_bindings",
			"invert_spatial_relation",
			"collapse_separated_layout_into_overlap",
			"change_scene",
			"change_framing_and_occlude",
		],
	}


def _global_composition_scene(
	row: Mapping[str, Any], vocabulary: Sequence[str]
) -> dict[str, Any]:
	"""Build a scene whose correctness depends on global layout and composition."""
	recipe_id = "global_composition_scene_v1"
	first, second = _ensure_entities(row, vocabulary, 2, recipe_id)
	target_scene, corrupt_scene = _choice(
		SCENE_PAIRS, PROMPT_POLICY_VERSION, recipe_id, row["lineage_id"], "scene"
	)
	target_prompt = (
		f"A wide daylight photograph showing {_entity_phrase(first)} on the left and {_entity_phrase(second)} "
		f"on the right, with all depicted items entirely inside the frame and the two groups clearly separated, "
		f"{target_scene}."
	)
	initial_prompt = (
		f"An overhead night-time close-up of {_entity_phrase(first)} and {_entity_phrase(second)} clustered "
		f"in the lower-right corner, with the first object mostly hidden behind the second and both partly outside "
		f"the frame, {corrupt_scene}."
	)
	return {
		"recipe_id": recipe_id,
		"collection_stratum": "global_composition_and_scene",
		"target_prompt": target_prompt,
		"initial_generation_prompt": initial_prompt,
		"target_constraint_graph": _graph(
			entities=[first, second],
			relations=[{"subject": first["id"], "predicate": "left of", "object": second["id"]}],
			scene=target_scene,
			framing="wide daylight eye-level; full objects",
			composition="balanced left/right separation",
		),
		"initial_constraint_graph": _graph(
			entities=[first, second],
			relations=[{"subject": first["id"], "predicate": "behind", "object": second["id"]}],
			scene=corrupt_scene,
			framing="overhead night close-up; both cropped",
			composition="lower-right cluster; first mostly occluded",
		),
		"corruption_operations": [
			"move_balanced_layout_to_corner",
			"introduce_major_occlusion",
			"change_viewpoint_and_framing",
			"change_scene_and_lighting",
		],
	}


def _multi_constraint_bundle(
	row: Mapping[str, Any], vocabulary: Sequence[str]
) -> dict[str, Any]:
	"""Combine several independently auditable constraints into one prompt."""
	recipe_id = "multi_constraint_bundle_v1"
	first, second, third = _colorize(
		_ensure_entities(row, vocabulary, 3, recipe_id), row, recipe_id
	)
	first = {**first, "count": 2}
	target_scene, corrupt_scene = _choice(
		SCENE_PAIRS, PROMPT_POLICY_VERSION, recipe_id, row["lineage_id"], "scene"
	)
	corrupt_first = {**first, "count": 4, "color": second["color"]}
	corrupt_second = {**second, "color": first["color"]}
	target_prompt = (
		f"A wide photograph with exactly {_entity_phrase(first, force_count=True)} on the left, "
		f"{_entity_phrase(second)} in the center, and {_entity_phrase(third)} on the right. "
		f"Every object is fully visible, evenly spaced, and distinct, {target_scene}."
	)
	initial_prompt = (
		f"A tight photograph of {_entity_phrase(corrupt_second)} on the left and exactly "
		f"{_entity_phrase(corrupt_first, force_count=True)} crowded on the right, with the objects overlapping "
		f"and partly cropped, {corrupt_scene}."
	)
	return {
		"recipe_id": recipe_id,
		"collection_stratum": "multi_constraint_severe",
		"target_prompt": target_prompt,
		"initial_generation_prompt": initial_prompt,
		"target_constraint_graph": _graph(
			entities=[first, second, third],
			relations=[
				{"subject": first["id"], "predicate": "left of", "object": second["id"]},
				{"subject": second["id"], "predicate": "left of", "object": third["id"]},
			],
			scene=target_scene,
			framing="wide; all entities fully visible",
			composition="ordered left/center/right with even spacing",
		),
		"initial_constraint_graph": _graph(
			entities=[corrupt_first, corrupt_second],
			relations=[{"subject": second["id"], "predicate": "left of", "object": first["id"]}],
			scene=corrupt_scene,
			framing="tight; partially cropped",
			composition="reversed order and overlap; third entity absent",
		),
		"corruption_operations": [
			"change_exact_count",
			"swap_color_bindings",
			"reverse_entity_order",
			"drop_one_entity_class",
			"change_scene",
			"collapse_layout_and_crop",
		],
	}


def _semantic_scene_reset(
	row: Mapping[str, Any], vocabulary: Sequence[str]
) -> dict[str, Any]:
	"""Build a reconstruction case whose target semantics invalidate the old scene."""
	recipe_id = "semantic_scene_reset_v1"
	target_entities = _colorize(_ensure_entities(row, vocabulary, 3, recipe_id), row, recipe_id)
	target_names = {entity["class"] for entity in target_entities}
	replacements = _ranked(
		(name for name in vocabulary if name not in target_names),
		PROMPT_POLICY_VERSION,
		recipe_id,
		row["lineage_id"],
		"replacement_objects",
	)[:2]
	if len(replacements) < 2:
		raise AdvancedPromptError("semantic reset needs two disjoint replacement classes")
	initial_entities = _colorize(
		[
			{"id": "replacement_1", "class": replacements[0], "count": 1, "color": None},
			{"id": "replacement_2", "class": replacements[1], "count": 1, "color": None},
		],
		row,
		recipe_id + "_initial",
	)
	target_scene, corrupt_scene = _choice(
		SCENE_PAIRS, PROMPT_POLICY_VERSION, recipe_id, row["lineage_id"], "scene"
	)
	first, second, third = target_entities
	replacement_first, replacement_second = initial_entities
	target_prompt = (
		f"A wide, eye-level photograph of {_entity_phrase(first)} on the left, {_entity_phrase(second)} in the "
		f"center, and {_entity_phrase(third)} on the right, all fully visible and evenly spaced, {target_scene}."
	)
	initial_prompt = (
		f"A tight overhead close-up of {_entity_phrase(replacement_first)} behind "
		f"{_entity_phrase(replacement_second)}, both filling most of the frame, {corrupt_scene}."
	)
	return {
		"recipe_id": recipe_id,
		"collection_stratum": "semantic_and_scene_reset",
		"target_prompt": target_prompt,
		"initial_generation_prompt": initial_prompt,
		"target_constraint_graph": _graph(
			entities=target_entities,
			relations=[
				{"subject": first["id"], "predicate": "left of", "object": second["id"]},
				{"subject": second["id"], "predicate": "left of", "object": third["id"]},
			],
			scene=target_scene,
			framing="wide eye-level; all entities fully visible",
			composition="ordered left/center/right",
		),
		"initial_constraint_graph": _graph(
			entities=initial_entities,
			relations=[
				{
					"subject": replacement_first["id"],
					"predicate": "behind",
					"object": replacement_second["id"],
				}
			],
			scene=corrupt_scene,
			framing="tight overhead close-up",
			composition="two unrelated replacement entities fill the frame",
		),
		"corruption_operations": [
			"replace_all_target_entity_classes",
			"drop_target_relation_chain",
			"change_scene",
			"change_viewpoint_and_framing",
			"change_composition",
		],
	}


RECIPES: tuple[tuple[str, Callable[[Mapping[str, Any], Sequence[str]], dict[str, Any]]], ...] = (
	("coupled_relation_binding_v1", _coupled_relation_binding),
	("global_composition_scene_v1", _global_composition_scene),
	("multi_constraint_bundle_v1", _multi_constraint_bundle),
	("semantic_scene_reset_v1", _semantic_scene_reset),
)


def _load_core_rows(path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
		if not line.strip():
			continue
		try:
			row = json.loads(line)
		except json.JSONDecodeError as exc:
			raise AdvancedPromptError(f"invalid input JSONL at {path}:{line_number}: {exc}") from exc
		if isinstance(row, dict) and row.get("partition") == "core":
			rows.append(row)
	if not rows:
		raise AdvancedPromptError(f"no core rows found in {path}")
	return rows


def build_advanced_prompt_rows(
	core_rows: Sequence[Mapping[str, Any]], *, per_recipe: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
	"""Materialize deterministic multi-constraint prompt families without embedding route labels."""
	if per_recipe <= 0:
		raise AdvancedPromptError("per_recipe must be positive")
	by_lineage: dict[str, Mapping[str, Any]] = {}
	for row in core_rows:
		lineage_id = _normalize(row.get("lineage_id"))
		if not lineage_id:
			raise AdvancedPromptError("core row has no lineage_id")
		current = by_lineage.get(lineage_id)
		if current is None or int(row.get("source_index", 10**12)) < int(current.get("source_index", 10**12)):
			by_lineage[lineage_id] = row
	vocabulary = _object_vocabulary(list(by_lineage.values()))
	available = set(by_lineage)
	result: list[dict[str, Any]] = []
	seen_target_prompts: set[str] = set()

	for recipe_id, recipe in RECIPES:
		ordered = sorted(
			available,
			key=lambda lineage: sha256_json(
				[PROMPT_POLICY_VERSION, seed, recipe_id, lineage, "selection_order"]
			),
		)
		created = 0
		for lineage_id in ordered:
			parent = by_lineage[lineage_id]
			payload = recipe(parent, vocabulary)
			target_prompt = _normalize(payload["target_prompt"])
			generation_prompt = _normalize(payload["initial_generation_prompt"])
			if not target_prompt or not generation_prompt or target_prompt == generation_prompt:
				raise AdvancedPromptError(f"recipe produced an invalid pair: {recipe_id}/{lineage_id}")
			if target_prompt in seen_target_prompts:
				continue
			pair_id = stable_id(
				"advanced",
				{
					"policy": PROMPT_POLICY_VERSION,
					"seed": seed,
					"recipe_id": recipe_id,
					"parent_lineage_id": lineage_id,
					"target_prompt": target_prompt,
					"initial_generation_prompt": generation_prompt,
				},
				length=32,
			)
			row = {
				"schema_version": SCHEMA_VERSION,
				"sample_id": pair_id,
				"source_prompt_id": pair_id,
				"dataset": "v10_advanced_prompt_pairs",
				"lineage_id": lineage_id,
				"parent_sample_id": parent.get("sample_id"),
				"parent_source_index": parent.get("source_index"),
				"split": parent.get("split"),
				"recipe_id": recipe_id,
				"prompt_policy_version": PROMPT_POLICY_VERSION,
				"collection_stratum": payload["collection_stratum"],
				"target_prompt": target_prompt,
				"initial_generation_prompt": generation_prompt,
				"initial_seed": stable_seed(seed, PROMPT_POLICY_VERSION, recipe_id, lineage_id),
				"error_family": f"advanced_{payload['collection_stratum']}",
				"corruption": {
					"type": "structured_multi_constraint_perturbation",
					"recipe_id": recipe_id,
					"operations": payload["corruption_operations"],
				},
				"target_constraint_graph": payload["target_constraint_graph"],
				"initial_constraint_graph": payload["initial_constraint_graph"],
				"source_metadata": {
					"parent_error_family": parent.get("error_family"),
					"parent_target_prompt": parent.get("target_prompt"),
				},
				"requires_initial_generation": True,
				"sampling_weight": 1.0,
			}
			result.append(row)
			seen_target_prompts.add(target_prompt)
			available.remove(lineage_id)
			created += 1
			if created == per_recipe:
				break
		if created != per_recipe:
			raise AdvancedPromptError(
				f"recipe {recipe_id} produced {created} unique rows, expected {per_recipe}"
			)

	result.sort(key=lambda row: (row["recipe_id"], row["sample_id"]))
	text = "".join(canonical_json(row) + "\n" for row in result)
	manifest = {
		"schema_version": SCHEMA_VERSION,
		"prompt_policy_version": PROMPT_POLICY_VERSION,
		"seed": seed,
		"per_recipe": per_recipe,
		"rows": len(result),
		"recipes": dict(sorted(Counter(row["recipe_id"] for row in result).items())),
		"collection_strata": dict(
			sorted(Counter(row["collection_stratum"] for row in result).items())
		),
		"splits": dict(sorted(Counter(str(row["split"]) for row in result).items())),
		"unique_parent_lineages": len({row["lineage_id"] for row in result}),
		"unique_target_prompts": len({row["target_prompt"] for row in result}),
		"route_labels_assigned": False,
		"generation_or_qwen_used": False,
		"rows_sha256": sha256_text(text),
	}
	return result, manifest


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
	temporary.write_text("".join(canonical_json(dict(row)) + "\n" for row in rows), encoding="utf-8")
	os.replace(temporary, path)


def build_advanced_prompt_file(
	*, input_path: Path, output_root: Path, per_recipe: int = 500, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
	"""Generate every advanced recipe and atomically write its audited manifest."""
	core_rows = _load_core_rows(input_path)
	rows, manifest = build_advanced_prompt_rows(core_rows, per_recipe=per_recipe, seed=seed)
	output_root = output_root.resolve()
	rows_path = output_root / "advanced_prompt_pairs.jsonl"
	manifest_path = output_root / "advanced_prompt_manifest.json"
	_write_jsonl_atomic(rows_path, rows)
	write_json_atomic(manifest_path, {**manifest, "rows_path": rows_path.name})
	return {**manifest, "rows_path": str(rows_path), "manifest_path": str(manifest_path)}


def verify_advanced_prompt_file(output_root: Path) -> dict[str, Any]:
	"""Verify advanced prompt file and reject contract violations."""
	output_root = output_root.resolve()
	manifest = json.loads((output_root / "advanced_prompt_manifest.json").read_text(encoding="utf-8"))
	rows_path = output_root / str(manifest["rows_path"])
	text = rows_path.read_text(encoding="utf-8")
	rows = [json.loads(line) for line in text.splitlines() if line.strip()]
	if sha256_text("".join(canonical_json(row) + "\n" for row in rows)) != manifest["rows_sha256"]:
		raise AdvancedPromptError("advanced prompt rows digest mismatch")
	if any(row["target_prompt"] == row["initial_generation_prompt"] for row in rows):
		raise AdvancedPromptError("advanced prompt file contains an uncorrupted pair")
	if len({row["lineage_id"] for row in rows}) != len(rows):
		raise AdvancedPromptError("advanced prompt file reuses a parent lineage")
	return {"status": "ok", "rows": len(rows), "rows_sha256": manifest["rows_sha256"]}


def build_parser() -> argparse.ArgumentParser:
	"""Construct the CLI parser for advanced-prompt generation."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
	parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
	parser.add_argument("--per-recipe", type=int, default=500)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	parser.add_argument("--verify-only", action="store_true")
	return parser


def main(argv: list[str] | None = None) -> None:
	"""Run this module's command-line workflow and return its process status."""
	args = build_parser().parse_args(argv)
	if args.verify_only:
		result = verify_advanced_prompt_file(args.output_root)
	else:
		result = build_advanced_prompt_file(
			input_path=args.input,
			output_root=args.output_root,
			per_recipe=args.per_recipe,
			seed=args.seed,
		)
	print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
	main()
