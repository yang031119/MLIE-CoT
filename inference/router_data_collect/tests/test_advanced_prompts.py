"""Regression tests for deterministic, route-free advanced prompt recipes."""
from __future__ import annotations

from inference.router_data_collect.advanced_prompts import RECIPES, build_advanced_prompt_rows


def _row(index: int, first: str, second: str) -> dict:
	return {
		"sample_id": f"core_{index}",
		"lineage_id": f"lineage_{index}",
		"source_index": index,
		"partition": "core",
		"split": ("train", "validation", "test", "train")[index],
		"target_prompt": f"a photo of a {first} and a {second}",
		"error_family": "two_object",
		"source_metadata": {
			"tag": "two_object",
			"include": [
				{"class": first, "count": 1},
				{"class": second, "count": 1},
			],
		},
	}


def test_advanced_prompt_recipes_are_deterministic_auditable_and_route_free() -> None:
	"""Verify advanced prompt recipes are deterministic auditable and route free."""
	core = [
		_row(0, "cat", "cake"),
		_row(1, "dog", "book"),
		_row(2, "bus", "chair"),
		_row(3, "bottle", "horse"),
	]
	rows_a, manifest_a = build_advanced_prompt_rows(core, per_recipe=1, seed=123)
	rows_b, manifest_b = build_advanced_prompt_rows(core, per_recipe=1, seed=123)
	assert rows_a == rows_b
	assert manifest_a == manifest_b
	assert len(rows_a) == len(RECIPES) == 4
	assert len({row["lineage_id"] for row in rows_a}) == 4
	assert len({row["target_prompt"] for row in rows_a}) == 4
	assert {row["recipe_id"] for row in rows_a} == {name for name, _ in RECIPES}
	for row in rows_a:
		assert row["target_prompt"] != row["initial_generation_prompt"]
		assert len(row["corruption"]["operations"]) >= 4
		assert row["target_constraint_graph"] != row["initial_constraint_graph"]
		assert row["split"] in {"train", "validation", "test"}
		assert "route" not in row
		assert "level" not in row
	assert manifest_a["route_labels_assigned"] is False
	assert manifest_a["generation_or_qwen_used"] is False
