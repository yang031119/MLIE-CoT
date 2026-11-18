"""Regression tests for balanced prompt-set normalization, disjointness, and tamper detection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference.router_data_collect.balanced_prompt_sets import (
	BalancedPromptSetError,
	COCO_80_CLASS_INFLECTIONS,
	EASY_RECIPES,
	HARD_RECIPES,
	MEDIUM_RECIPES,
	PROMPT_NORMALIZATION_POLICY_VERSION,
	SET_FILES,
	_recipe_split_quotas,
	_source_candidate,
	build_balanced_prompt_sets,
	normalize_prompt_noun_phrases,
	prompt_grammar_violations,
	router_split_for_lineage,
	verify_balanced_prompt_sets,
)
from inference.router_data_collect.canonical import iter_jsonl, write_jsonl_atomic


def _required_counts() -> dict[tuple[str, str], int]:
	result: dict[tuple[str, str], int] = {}
	for set_id in SET_FILES:
		for key, value in _recipe_split_quotas(set_id).items():
			result[key] = result.get(key, 0) + value
	return result


def test_coco_nouns_are_normalized_without_changing_pair_semantics(
	tmp_path: Path,
) -> None:
	"""Verify coco nouns are normalized without changing pair semantics."""
	normalized, operations = normalize_prompt_noun_phrases(
		"a photo of a skis and a scissors",
		field="target_prompt",
	)
	assert normalized == "a photo of a pair of skis and a pair of scissors"
	assert [operation["noun"] for operation in operations] == ["skis", "scissors"]
	assert not prompt_grammar_violations(normalized, field="target_prompt")
	articles, article_operations = normalize_prompt_noun_phrases(
		"a orange beside an red chair", field="target_prompt"
	)
	assert articles == "an orange beside a red chair"
	assert len(article_operations) == 2
	rules = {item["rule"] for item in prompt_grammar_violations(
		"a people beside one scissors, one mice and three person", field="target_prompt"
	)}
	assert "noncanonical_indefinite_coco_noun_phrase" in rules
	assert "noncanonical_counted_coco_noun_phrase" in rules

	row = {
		"schema_version": "v10_legacy_initial_assets_v1",
		"sample_id": "legacy_pair_nouns",
		"lineage_id": "lineage_pair_nouns",
		"partition": "core",
		"split": "train",
		"source_index": 1,
		"target_prompt": "a photo of a skis and a scissors",
		"initial_generation_prompt": "a photo of a skis",
		"corruption": {
			"type": "drop_one_object",
			"target_prompt": "a photo of a skis and a scissors",
		},
	}
	candidate = _source_candidate(
		row,
		source_kind="legacy_v7_single_constraint",
		source_path=tmp_path / "legacy.jsonl",
	)
	assert candidate["target_prompt"] == "a photo of a pair of skis and a pair of scissors"
	assert candidate["initial_generation_prompt"] == "a photo of a pair of skis"
	assert candidate["corruption"]["target_prompt"] == candidate["target_prompt"]
	assert candidate["prompt_normalization"]["policy_version"] == PROMPT_NORMALIZATION_POLICY_VERSION
	assert candidate["prompt_normalization"]["applied"] is True
	assert len(candidate["prompt_normalization"]["operations"]) == 5


@pytest.mark.parametrize(
	("source", "expected"),
	[
		("one toilets and four toilet", "one toilet and four toilets"),
		("one red buses and two blue bus", "one red bus and two blue buses"),
		("one skises and four black scissorses", "one pair of skis and four pairs of black scissors"),
		("one pairs of skis and two pair of scissors", "one pair of skis and two pairs of scissors"),
		("two knifes beside one persons", "two knives beside one person"),
		("three sheeps beside one sheeps", "three sheep beside one sheep"),
		("two mouses beside one mice", "two mice beside one mouse"),
		("two computer mouses beside one computer mice", "two computer mice beside one computer mouse"),
		("four sandwichs and one sandwiches", "four sandwiches and one sandwich"),
		("four couchs and four benchs", "four couches and four benches"),
		("four girafves and four toothbrushs", "four giraffes and four toothbrushes"),
		("four broccolis and one broccolis", "four heads of broccoli and one head of broccoli"),
		("two blue broccoli and one red broccolis", "two blue heads of broccoli and one red head of broccoli"),
		("two blue head of broccoli and one red heads of broccoli", "two blue heads of broccoli and one red head of broccoli"),
	],
)
def test_counted_coco_forms_use_canonical_inflection(source: str, expected: str) -> None:
	"""Verify counted coco forms use canonical inflection."""
	normalized, operations = normalize_prompt_noun_phrases(source, field="target_prompt")
	assert normalized == expected
	assert operations
	assert not prompt_grammar_violations(normalized, field="target_prompt")


def test_explicit_coco_80_inflection_table_is_exhaustively_enforced() -> None:
	"""Verify explicit coco 80 inflection table is exhaustively enforced."""
	assert len(COCO_80_CLASS_INFLECTIONS) == 80
	for class_name, (singular, plural) in COCO_80_CLASS_INFLECTIONS.items():
		one, _one_operations = normalize_prompt_noun_phrases(
			f"one {plural}", field=f"one.{class_name}"
		)
		two, _two_operations = normalize_prompt_noun_phrases(
			f"two {singular}", field=f"two.{class_name}"
		)
		if class_name in {"scissors", "skis"}:
			assert one == f"one pair of {plural}"
			assert two == f"two pairs of {plural}"
		elif class_name == "broccoli":
			assert one == "one head of broccoli"
			assert two == "two heads of broccoli"
		else:
			assert one == f"one {singular}"
			assert two == f"two {plural}"
		assert not prompt_grammar_violations(one, field=f"one.{class_name}")
		assert not prompt_grammar_violations(two, field=f"two.{class_name}")


def test_count_audit_rejects_known_bad_and_unrecognized_phrases() -> None:
	"""Verify count audit rejects known bad and unrecognized phrases."""
	known = prompt_grammar_violations(
		"one toilets, four toilet, two knifes and three girafves", field="target_prompt"
	)
	assert "noncanonical_counted_coco_noun_phrase" in {
		item["rule"] for item in known
	}
	assert "nonstandard_coco_noun_form" in {item["rule"] for item in known}
	assert prompt_grammar_violations(
		"two groups and one object", field="target_prompt"
	) == []
	assert {item["rule"] for item in prompt_grammar_violations(
		"two widgets", field="target_prompt"
	)} == {"unrecognized_number_word_count_phrase"}
	article, article_operations = normalize_prompt_noun_phrases(
		"a broccoli beside a blue broccoli", field="target_prompt"
	)
	assert article == "a head of broccoli beside a blue head of broccoli"
	assert len(article_operations) == 2
	assert not prompt_grammar_violations(article, field="target_prompt")
	orange_article, _ = normalize_prompt_noun_phrases(
		"a orange broccoli", field="target_prompt"
	)
	assert orange_article == "an orange head of broccoli"
	assert "noncanonical_counted_measure_noun_phrase" in {
		item["rule"] for item in prompt_grammar_violations(
			"two blue head of broccoli", field="target_prompt"
		)
	}


def _write_sources(root: Path) -> tuple[Path, Path]:
	"""Write minimal legacy and advanced source manifests for selection tests."""
	required = _required_counts()
	legacy: list[dict] = []
	advanced: list[dict] = []
	serial = 0
	for recipe in (*EASY_RECIPES, *MEDIUM_RECIPES, *HARD_RECIPES):
		for split in ("train", "validation", "test"):
			for index in range(required[(recipe, split)] + 2):
				while True:
					serial += 1
					lineage = f"lineage_{serial:06d}"
					if router_split_for_lineage(lineage) == split:
						break
				base = {
					"sample_id": f"source_{serial:06d}",
					"lineage_id": lineage,
					"split": split,
					"target_prompt": f"target {recipe} {split} {index}",
					"initial_generation_prompt": f"corrupted {recipe} {split} {index}",
					"corruption": {"type": recipe, "operations": ["unit_test_change"]},
				}
				if recipe in EASY_RECIPES:
					legacy.append(
						{
							**base,
							"schema_version": "v10_legacy_initial_assets_v1",
							"partition": "core",
							"source_index": serial,
						}
					)
				else:
					advanced.append(
						{
							**base,
							"schema_version": "v10_advanced_prompt_pair_v1",
							"recipe_id": recipe,
							"parent_sample_id": f"parent_{serial:06d}",
							"parent_source_index": serial,
						}
					)
	legacy_path = root / "legacy.jsonl"
	advanced_path = root / "advanced.jsonl"
	write_jsonl_atomic(legacy_path, legacy)
	write_jsonl_atomic(advanced_path, advanced)
	return legacy_path, advanced_path


def test_two_balanced_prompt_sets_are_deterministic_disjoint_and_verifiable(
	tmp_path: Path,
) -> None:
	"""Verify two balanced prompt sets are deterministic disjoint and verifiable."""
	legacy, advanced = _write_sources(tmp_path)
	output = tmp_path / "balanced"
	manifest = build_balanced_prompt_sets(
		legacy_input=legacy,
		advanced_input=advanced,
		output_root=output,
		seed=123,
	)
	verified = verify_balanced_prompt_sets(output)
	assert verified["status"] == "ok"
	assert manifest["rows"] == 2000
	assert manifest["cross_set_lineage_overlap"] == 0
	assert manifest["route_labels_assigned"] is False
	assert manifest["generation_or_qwen_used"] is False
	assert manifest["prompt_normalization"] == {
		"policy_version": PROMPT_NORMALIZATION_POLICY_VERSION,
		"normalized_rows": 0,
		"operation_count": 0,
		"operations": {},
		"nouns": {},
		"fields": {},
		"remaining_violations": 0,
	}

	rows_by_set = {
		set_id: list(iter_jsonl(output / filename)) for set_id, filename in SET_FILES.items()
	}
	assert all(len(rows) == 1000 for rows in rows_by_set.values())
	assert not (
		{row["lineage_id"] for row in rows_by_set["set_a"]}
		& {row["lineage_id"] for row in rows_by_set["set_b"]}
	)
	for set_id, rows in rows_by_set.items():
		assert {row["difficulty"] for row in rows} == {"easy", "medium", "hard"}
		assert all(row["original_prompt"] == row["target_prompt"] for row in rows)
		assert all(
			row["believed_corrupted_prompt"] == row["initial_generation_prompt"]
			for row in rows
		)
		assert all(row["target_prompt"] != row["initial_generation_prompt"] for row in rows)

	first_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
	build_balanced_prompt_sets(
		legacy_input=legacy,
		advanced_input=advanced,
		output_root=output,
		seed=123,
	)
	assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == first_manifest


def test_balanced_prompt_verifier_rejects_row_tampering(tmp_path: Path) -> None:
	"""Verify balanced prompt verifier rejects row tampering."""
	legacy, advanced = _write_sources(tmp_path)
	output = tmp_path / "balanced"
	build_balanced_prompt_sets(
		legacy_input=legacy,
		advanced_input=advanced,
		output_root=output,
		seed=321,
	)
	path = output / SET_FILES["set_a"]
	rows = list(iter_jsonl(path))
	rows[0]["believed_corrupted_prompt"] = "tampered"
	write_jsonl_atomic(path, rows)
	with pytest.raises(BalancedPromptSetError, match="alias mismatch|digest"):
		verify_balanced_prompt_sets(output)


def test_balanced_prompt_verifier_rejects_noncanonical_count_tampering(
	tmp_path: Path,
) -> None:
	"""Verify balanced prompt verifier rejects noncanonical count tampering."""
	legacy, advanced = _write_sources(tmp_path)
	output = tmp_path / "balanced"
	build_balanced_prompt_sets(
		legacy_input=legacy,
		advanced_input=advanced,
		output_root=output,
		seed=654,
	)
	path = output / SET_FILES["set_a"]
	rows = list(iter_jsonl(path))
	rows[0]["target_prompt"] = "A photo of one toilets."
	rows[0]["original_prompt"] = rows[0]["target_prompt"]
	write_jsonl_atomic(path, rows)
	with pytest.raises(BalancedPromptSetError, match="prompt grammar violations"):
		verify_balanced_prompt_sets(output)
