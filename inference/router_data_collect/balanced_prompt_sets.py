"""Build two disjoint, difficulty-balanced 1K prompt-pair sets.

This is a CPU-only curation step.  It combines simple one-constraint V7
counterfactual prompts with the structured V10 prompt pairs, without assigning
router labels or generating images.  Parent lineage is the leakage boundary:
one lineage may occur in at most one row across both output files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .actions import assign_split
from .canonical import (
	canonical_json,
	iter_jsonl,
	sha256_file,
	sha256_json,
	sha256_text,
	stable_id,
	stable_seed,
	write_json_atomic,
	write_jsonl_atomic,
)


SCHEMA_VERSION = "vcot_v10_balanced_prompt_pair_v4"
MANIFEST_SCHEMA_VERSION = "vcot_v10_balanced_prompt_sets_manifest_v4"
POLICY_VERSION = "two_disjoint_1k_easy_medium_hard_v1"
PROMPT_NORMALIZATION_POLICY_VERSION = "auditable_coco_noun_phrases_v3"
PROMPT_GRAMMAR_AUDIT_VERSION = "coco_count_article_inflection_scan_v3"
DEFAULT_SEED = 20260716
DEFAULT_LEGACY_INPUT = Path("results/router_data/samples.jsonl")
DEFAULT_ADVANCED_INPUT = Path("results/router_data/advanced_prompt_pairs.jsonl")
DEFAULT_OUTPUT_ROOT = Path("results/router_data/prompt_sets_2x1k")

SET_FILES = {
	"set_a": "balanced_prompt_set_a_1k.jsonl",
	"set_b": "balanced_prompt_set_b_1k.jsonl",
}
MANIFEST_FILE = "manifest.json"
CHECKSUM_FILE = "checksums.sha256"

SPLITS = ("train", "validation", "test")
ROUTER_SPLIT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}

# Each set is exactly 1,000 rows, exactly 80/10/10 by split, and as close to
# one-third per difficulty as integers permit.
DIFFICULTY_SPLIT_QUOTAS: dict[str, dict[str, int]] = {
	"easy": {"train": 267, "validation": 34, "test": 33},
	"medium": {"train": 266, "validation": 33, "test": 34},
	"hard": {"train": 267, "validation": 33, "test": 33},
}

EASY_RECIPES = (
	"drop_one_object",
	"reversed_position",
	"swapped_color_attribution",
	"wrong_color",
	"wrong_count",
)
MEDIUM_RECIPES = ("coupled_relation_binding_v1", "global_composition_scene_v1")
HARD_RECIPES = ("multi_constraint_bundle_v1", "semantic_scene_reset_v1")

# Explicit inflection contract for all 80 COCO classes.  The V7 source is a
# frozen audit input and contains several outputs of old heuristic pluralizers
# (for example ``sandwichs``, ``girafves`` and ``persons``).  The balanced-set
# materializer accepts those source spellings but always emits the canonical
# forms below.  Dataset aliases retain the less ambiguous names used by V7.
COCO_80_CLASS_INFLECTIONS: dict[str, tuple[str, str]] = {
	"person": ("person", "people"),
	"bicycle": ("bicycle", "bicycles"),
	"car": ("car", "cars"),
	"motorcycle": ("motorcycle", "motorcycles"),
	"airplane": ("airplane", "airplanes"),
	"bus": ("bus", "buses"),
	"train": ("train", "trains"),
	"truck": ("truck", "trucks"),
	"boat": ("boat", "boats"),
	"traffic light": ("traffic light", "traffic lights"),
	"fire hydrant": ("fire hydrant", "fire hydrants"),
	"stop sign": ("stop sign", "stop signs"),
	"parking meter": ("parking meter", "parking meters"),
	"bench": ("bench", "benches"),
	"bird": ("bird", "birds"),
	"cat": ("cat", "cats"),
	"dog": ("dog", "dogs"),
	"horse": ("horse", "horses"),
	"sheep": ("sheep", "sheep"),
	"cow": ("cow", "cows"),
	"elephant": ("elephant", "elephants"),
	"bear": ("bear", "bears"),
	"zebra": ("zebra", "zebras"),
	"giraffe": ("giraffe", "giraffes"),
	"backpack": ("backpack", "backpacks"),
	"umbrella": ("umbrella", "umbrellas"),
	"handbag": ("handbag", "handbags"),
	"tie": ("tie", "ties"),
	"suitcase": ("suitcase", "suitcases"),
	"frisbee": ("frisbee", "frisbees"),
	"skis": ("ski", "skis"),
	"snowboard": ("snowboard", "snowboards"),
	"sports ball": ("sports ball", "sports balls"),
	"kite": ("kite", "kites"),
	"baseball bat": ("baseball bat", "baseball bats"),
	"baseball glove": ("baseball glove", "baseball gloves"),
	"skateboard": ("skateboard", "skateboards"),
	"surfboard": ("surfboard", "surfboards"),
	"tennis racket": ("tennis racket", "tennis rackets"),
	"bottle": ("bottle", "bottles"),
	"wine glass": ("wine glass", "wine glasses"),
	"cup": ("cup", "cups"),
	"fork": ("fork", "forks"),
	"knife": ("knife", "knives"),
	"spoon": ("spoon", "spoons"),
	"bowl": ("bowl", "bowls"),
	"banana": ("banana", "bananas"),
	"apple": ("apple", "apples"),
	"sandwich": ("sandwich", "sandwiches"),
	"orange": ("orange", "oranges"),
	"broccoli": ("broccoli", "broccoli"),
	"carrot": ("carrot", "carrots"),
	"hot dog": ("hot dog", "hot dogs"),
	"pizza": ("pizza", "pizzas"),
	"donut": ("donut", "donuts"),
	"cake": ("cake", "cakes"),
	"chair": ("chair", "chairs"),
	"couch": ("couch", "couches"),
	"potted plant": ("potted plant", "potted plants"),
	"bed": ("bed", "beds"),
	"dining table": ("dining table", "dining tables"),
	"toilet": ("toilet", "toilets"),
	"tv": ("tv", "tvs"),
	"laptop": ("laptop", "laptops"),
	"mouse": ("mouse", "mice"),
	"remote": ("remote", "remotes"),
	"keyboard": ("keyboard", "keyboards"),
	"cell phone": ("cell phone", "cell phones"),
	"microwave": ("microwave", "microwaves"),
	"oven": ("oven", "ovens"),
	"toaster": ("toaster", "toasters"),
	"sink": ("sink", "sinks"),
	"refrigerator": ("refrigerator", "refrigerators"),
	"book": ("book", "books"),
	"clock": ("clock", "clocks"),
	"vase": ("vase", "vases"),
	"scissors": ("scissor", "scissors"),
	"teddy bear": ("teddy bear", "teddy bears"),
	"hair drier": ("hair drier", "hair driers"),
	"toothbrush": ("toothbrush", "toothbrushes"),
}
if len(COCO_80_CLASS_INFLECTIONS) != 80:
	raise RuntimeError("the explicit COCO inflection table must contain exactly 80 classes")

COCO_DATASET_CLASS_ALIASES: dict[str, tuple[str, str]] = {
	"computer mouse": ("computer mouse", "computer mice"),
	"tv remote": ("tv remote", "tv remotes"),
	"computer keyboard": ("computer keyboard", "computer keyboards"),
}
COCO_CLASS_INFLECTIONS = {
	**COCO_80_CLASS_INFLECTIONS,
	**COCO_DATASET_CLASS_ALIASES,
}
PAIR_ONLY_PLURAL_NOUNS = ("scissors", "skis")
MEASURE_NOUN_CLASSES = {"broccoli": ("head", "heads")}
NUMBER_WORD_TO_COUNT = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
COUNT_TO_NUMBER_WORD = {value: key for key, value in NUMBER_WORD_TO_COUNT.items()}
COCO_COLORS = (
	"black", "blue", "brown", "green", "orange", "pink", "purple", "red", "white", "yellow",
)
GENERIC_COUNT_INFLECTIONS = {
	"group": ("group", "groups"),
	"object": ("object", "objects"),
	"item": ("item", "items"),
	"entity": ("entity", "entities"),
}
VOWEL_ARTICLE_WORDS = ("airplane", "apple", "elephant", "orange", "oven", "umbrella")
CONSONANT_ARTICLE_WORDS = (
	"backpack", "banana", "baseball", "bear", "bed", "bench", "bicycle", "bird",
	"black", "blue", "boat", "book", "bottle", "bowl", "broccoli", "brown", "bus",
	"cake", "car", "carrot", "cat", "cell", "chair", "clock", "computer", "couch",
	"cow", "cup", "dining", "dog", "donut", "fire", "fork", "frisbee", "giraffe",
	"green", "hair", "handbag", "horse", "hot", "kite", "knife", "laptop",
	"microwave", "motorcycle", "parking", "person", "pink", "pizza", "potted",
	"purple", "red", "refrigerator", "sandwich", "sheep", "sink", "skateboard",
	"snowboard", "spoon", "sports", "stop", "suitcase", "surfboard", "teddy",
	"tennis", "tie", "toaster", "toilet", "toothbrush", "traffic", "train", "truck",
	"tv", "vase", "white", "wine", "yellow", "zebra",
)


def _heuristic_plural_variant(noun: str) -> str:
	"""Reproduce historical generator spellings so they can be normalized."""

	if noun.endswith(("s", "x", "z", "ch", "sh")):
		return noun + "es"
	if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
		return noun[:-1] + "ies"
	if noun.endswith("fe"):
		return noun[:-2] + "ves"
	if noun.endswith("f"):
		return noun[:-1] + "ves"
	return noun + "s"


def _accepted_source_forms(
	class_name: str, singular: str, plural: str
) -> set[str]:
	# V7 mostly appended ``s`` (or ``es`` after an existing ``s``), while the
	# structured generator used a broad f/fe heuristic.  Both are provenance,
	# not acceptable final spellings.
	forms = {class_name, singular, plural}
	for base in (class_name, singular):
		forms.add(base + "s")
		forms.add(_heuristic_plural_variant(base))
	return forms


COCO_FORM_TO_CLASS: dict[str, str] = {}
COCO_ACCEPTED_SOURCE_FORMS: set[str] = set()
for _class_name, (_singular, _plural) in COCO_CLASS_INFLECTIONS.items():
	for _form in _accepted_source_forms(_class_name, _singular, _plural):
		_existing = COCO_FORM_TO_CLASS.get(_form)
		if _existing is not None and _existing != _class_name:
			raise RuntimeError(
				f"ambiguous COCO noun form {_form!r}: {_existing!r}/{_class_name!r}"
			)
		COCO_FORM_TO_CLASS[_form] = _class_name
		COCO_ACCEPTED_SOURCE_FORMS.add(_form)

COCO_CANONICAL_FORMS = {
	form
	for class_name, (singular, plural) in COCO_CLASS_INFLECTIONS.items()
	for form in (class_name, singular, plural)
}
COCO_NONSTANDARD_SOURCE_FORMS = tuple(
	sorted(COCO_ACCEPTED_SOURCE_FORMS - COCO_CANONICAL_FORMS, key=lambda item: (-len(item), item))
)
_COCO_FORM_PATTERN = "|".join(
	re.escape(form)
	for form in sorted(COCO_ACCEPTED_SOURCE_FORMS, key=lambda item: (-len(item), item))
)
_COLOR_PATTERN = "|".join(COCO_COLORS)
_COUNT_WORD_PATTERN = "|".join(NUMBER_WORD_TO_COUNT)

_ARTICLE_COCO_NOUN_RE = re.compile(
	r"\b(?P<article>a|an)\s+(?:(?P<color>" + _COLOR_PATTERN + r")\s+)?"
	r"(?P<form>" + _COCO_FORM_PATTERN + r")\b",
	flags=re.IGNORECASE,
)
_COUNTED_COCO_NOUN_RE = re.compile(
	r"\b(?P<count>" + _COUNT_WORD_PATTERN + r")\s+"
	r"(?:(?P<color>" + _COLOR_PATTERN + r")\s+)?(?P<form>" + _COCO_FORM_PATTERN + r")\b"
	r"(?!\s+heads?\s+of\s+broccoli\b)",
	flags=re.IGNORECASE,
)
_COUNTED_PAIR_PHRASE_RE = re.compile(
	r"\b(?P<count>" + _COUNT_WORD_PATTERN + r")\s+(?P<pair_word>pair|pairs)\s+of\s+"
	r"(?:(?P<color>" + _COLOR_PATTERN + r")\s+)?"
	r"(?P<noun>" + "|".join(PAIR_ONLY_PLURAL_NOUNS) + r")\b",
	flags=re.IGNORECASE,
)
_ARTICLE_MEASURE_PHRASE_RE = re.compile(
	r"\b(?P<article>a|an)\s+(?:(?P<color>" + _COLOR_PATTERN + r")\s+)?"
	r"(?P<measure>head|heads)\s+of\s+(?P<noun>broccoli)\b",
	flags=re.IGNORECASE,
)
_COUNTED_MEASURE_PHRASE_RE = re.compile(
	r"\b(?P<count>" + _COUNT_WORD_PATTERN + r")\s+"
	r"(?:(?P<color>" + _COLOR_PATTERN + r")\s+)?"
	r"(?P<measure>head|heads)\s+of\s+(?P<noun>broccoli)\b",
	flags=re.IGNORECASE,
)
_GENERIC_COUNT_FORM_TO_NOUN = {
	form: noun
	for noun, forms in GENERIC_COUNT_INFLECTIONS.items()
	for form in forms
}
_COUNTED_GENERIC_NOUN_RE = re.compile(
	r"\b(?P<count>" + _COUNT_WORD_PATTERN + r")\s+(?P<form>"
	+ "|".join(
		re.escape(form)
		for form in sorted(_GENERIC_COUNT_FORM_TO_NOUN, key=lambda item: (-len(item), item))
	)
	+ r")\b",
	flags=re.IGNORECASE,
)
_COUNT_WORD_RE = re.compile(r"\b(?:" + _COUNT_WORD_PATTERN + r")\b", flags=re.IGNORECASE)
_NONSTANDARD_COCO_FORM_RE = re.compile(
	r"\b(?:" + "|".join(re.escape(form) for form in COCO_NONSTANDARD_SOURCE_FORMS) + r")\b",
	flags=re.IGNORECASE,
)
_A_BEFORE_VOWEL_RE = re.compile(
	r"\ba\s+(?P<word>" + "|".join(VOWEL_ARTICLE_WORDS) + r")\b",
	flags=re.IGNORECASE,
)
_AN_BEFORE_CONSONANT_RE = re.compile(
	r"\ban\s+(?P<word>" + "|".join(CONSONANT_ARTICLE_WORDS) + r")\b",
	flags=re.IGNORECASE,
)
GRAMMAR_AUDIT_RULES = (
	("a_before_known_vowel_sound", _A_BEFORE_VOWEL_RE),
	("an_before_known_consonant_sound", _AN_BEFORE_CONSONANT_RE),
	("nonstandard_coco_noun_form", _NONSTANDARD_COCO_FORM_RE),
)
DYNAMIC_GRAMMAR_AUDIT_RULE_NAMES = (
	"noncanonical_indefinite_coco_noun_phrase",
	"noncanonical_counted_coco_noun_phrase",
	"noncanonical_counted_pair_phrase",
	"noncanonical_indefinite_measure_noun_phrase",
	"noncanonical_counted_measure_noun_phrase",
	"noncanonical_counted_generic_noun_phrase",
	"unrecognized_number_word_count_phrase",
)
PROMPT_GRAMMAR_AUDIT_RULE_NAMES = (
	*(name for name, _pattern in GRAMMAR_AUDIT_RULES),
	*DYNAMIC_GRAMMAR_AUDIT_RULE_NAMES,
)


class BalancedPromptSetError(RuntimeError):
	"""Raised when a source or output violates the curation contract."""


def _normalize(value: Any) -> str:
	return " ".join(str(value or "").strip().split())


def _capitalized_replacement(match: re.Match[str], replacement: str) -> str:
	return replacement[:1].upper() + replacement[1:] if match.group(0)[:1].isupper() else replacement


def _canonical_indefinite_coco_phrase(match: re.Match[str]) -> tuple[str, str]:
	form = match.group("form").lower()
	class_name = COCO_FORM_TO_CLASS[form]
	singular, plural = COCO_CLASS_INFLECTIONS[class_name]
	color = (match.group("color") or "").lower()
	prefix = f"{color} " if color else ""
	if class_name in PAIR_ONLY_PLURAL_NOUNS:
		return f"a pair of {prefix}{plural}", class_name
	if class_name in MEASURE_NOUN_CLASSES:
		measure, _plural_measure = MEASURE_NOUN_CLASSES[class_name]
		body = f"{prefix}{measure} of {singular}"
		article = "an" if body[:1] in "aeiou" else "a"
		return f"{article} {body}", class_name
	body = f"{prefix}{singular}"
	article = "an" if body[:1] in "aeiou" else "a"
	return f"{article} {body}", class_name


def _canonical_counted_coco_phrase(match: re.Match[str]) -> tuple[str, str]:
	count_word = match.group("count").lower()
	count = NUMBER_WORD_TO_COUNT[count_word]
	form = match.group("form").lower()
	class_name = COCO_FORM_TO_CLASS[form]
	singular, plural = COCO_CLASS_INFLECTIONS[class_name]
	color = (match.group("color") or "").lower()
	prefix = f"{color} " if color else ""
	if class_name in PAIR_ONLY_PLURAL_NOUNS:
		pair_word = "pair" if count == 1 else "pairs"
		return f"{count_word} {pair_word} of {prefix}{plural}", class_name
	if class_name in MEASURE_NOUN_CLASSES:
		measure, plural_measure = MEASURE_NOUN_CLASSES[class_name]
		measure_form = measure if count == 1 else plural_measure
		return f"{count_word} {prefix}{measure_form} of {singular}", class_name
	noun = singular if count == 1 else plural
	return f"{count_word} {prefix}{noun}", class_name


def _canonical_counted_pair_phrase(match: re.Match[str]) -> str:
	count_word = match.group("count").lower()
	pair_word = "pair" if NUMBER_WORD_TO_COUNT[count_word] == 1 else "pairs"
	color = (match.group("color") or "").lower()
	prefix = f"{color} " if color else ""
	return f"{count_word} {pair_word} of {prefix}{match.group('noun').lower()}"


def _canonical_indefinite_measure_phrase(match: re.Match[str]) -> str:
	color = (match.group("color") or "").lower()
	prefix = f"{color} " if color else ""
	noun = match.group("noun").lower()
	measure, _plural_measure = MEASURE_NOUN_CLASSES[noun]
	body = f"{prefix}{measure} of {noun}"
	article = "an" if body[:1] in "aeiou" else "a"
	return f"{article} {body}"


def _canonical_counted_measure_phrase(match: re.Match[str]) -> str:
	count_word = match.group("count").lower()
	color = (match.group("color") or "").lower()
	prefix = f"{color} " if color else ""
	noun = match.group("noun").lower()
	measure, plural_measure = MEASURE_NOUN_CLASSES[noun]
	measure_form = measure if NUMBER_WORD_TO_COUNT[count_word] == 1 else plural_measure
	return f"{count_word} {prefix}{measure_form} of {noun}"


def _canonical_counted_generic_phrase(match: re.Match[str]) -> str:
	count_word = match.group("count").lower()
	noun = _GENERIC_COUNT_FORM_TO_NOUN[match.group("form").lower()]
	singular, plural = GENERIC_COUNT_INFLECTIONS[noun]
	return f"{count_word} {singular if NUMBER_WORD_TO_COUNT[count_word] == 1 else plural}"


def normalize_prompt_noun_phrases(text: Any, *, field: str) -> tuple[str, list[dict[str, Any]]]:
	"""Canonicalize COCO count/article noun phrases without changing facts."""

	value = _normalize(text)
	operations: list[dict[str, Any]] = []

	def replace_counted_pair(match: re.Match[str]) -> str:
		"""Normalize counted pair-only nouns and record any textual change."""
		canonical = _canonical_counted_pair_phrase(match)
		replacement = _capitalized_replacement(match, canonical)
		if replacement == match.group(0):
			return replacement
		operations.append(
			{
				"field": field,
				"operation": "counted_pair_phrase_number_agreement",
				"noun": match.group("noun").lower(),
				"count_word": match.group("count").lower(),
				"input_form": match.group(0),
				"canonical_form": canonical,
			}
		)
		return replacement

	def replace_counted_measure(match: re.Match[str]) -> str:
		"""Normalize counted measure nouns and record any textual change."""
		canonical = _canonical_counted_measure_phrase(match)
		replacement = _capitalized_replacement(match, canonical)
		if replacement == match.group(0):
			return replacement
		operations.append(
			{
				"field": field,
				"operation": "counted_measure_phrase_number_agreement",
				"noun": match.group("noun").lower(),
				"count_word": match.group("count").lower(),
				"input_form": match.group(0),
				"canonical_form": canonical,
			}
		)
		return replacement

	def replace_counted_coco(match: re.Match[str]) -> str:
		"""Replace a counted COCO noun while preserving number and grammar."""
		canonical, class_name = _canonical_counted_coco_phrase(match)
		replacement = _capitalized_replacement(match, canonical)
		if replacement == match.group(0):
			return replacement
		operation = (
			"counted_pair_only_noun_to_pair_phrase"
			if class_name in PAIR_ONLY_PLURAL_NOUNS
			else (
				"counted_measure_noun_to_head_phrase"
				if class_name in MEASURE_NOUN_CLASSES
				else "counted_coco_noun_to_canonical_inflection"
			)
		)
		operations.append(
			{
				"field": field,
				"operation": operation,
				"noun": class_name,
				"count_word": match.group("count").lower(),
				"input_form": match.group("form").lower(),
				"canonical_form": canonical,
			}
		)
		return replacement

	def replace_counted_generic(match: re.Match[str]) -> str:
		"""Normalize counted generic nouns and record any textual change."""
		canonical = _canonical_counted_generic_phrase(match)
		replacement = _capitalized_replacement(match, canonical)
		if replacement == match.group(0):
			return replacement
		operations.append(
			{
				"field": field,
				"operation": "counted_generic_noun_to_canonical_inflection",
				"noun": _GENERIC_COUNT_FORM_TO_NOUN[match.group("form").lower()],
				"count_word": match.group("count").lower(),
				"input_form": match.group("form").lower(),
				"canonical_form": canonical,
			}
		)
		return replacement

	def replace_indefinite_coco(match: re.Match[str]) -> str:
		"""Normalize an indefinite COCO noun phrase and audit the replacement."""
		canonical, class_name = _canonical_indefinite_coco_phrase(match)
		replacement = _capitalized_replacement(match, canonical)
		if replacement == match.group(0):
			return replacement
		operation = (
			"pair_only_plural_indefinite_article_to_pair_phrase"
			if class_name in PAIR_ONLY_PLURAL_NOUNS
			else (
				"indefinite_measure_noun_to_head_phrase"
				if class_name in MEASURE_NOUN_CLASSES
				else "indefinite_coco_noun_to_canonical_form"
			)
		)
		operations.append(
			{
				"field": field,
				"operation": operation,
				"noun": class_name,
				"input_form": match.group("form").lower(),
				"canonical_form": canonical,
			}
		)
		return replacement

	def replace_indefinite_measure(match: re.Match[str]) -> str:
		"""Normalize an indefinite measure phrase and audit the replacement."""
		canonical = _canonical_indefinite_measure_phrase(match)
		replacement = _capitalized_replacement(match, canonical)
		if replacement == match.group(0):
			return replacement
		operations.append(
			{
				"field": field,
				"operation": "indefinite_measure_phrase_number_agreement",
				"noun": match.group("noun").lower(),
				"input_form": match.group(0),
				"canonical_form": canonical,
			}
		)
		return replacement

	def replace_vowel_article(match: re.Match[str]) -> str:
		"""Replace ``a`` before a vowel sound and record the correction."""
		word = match.group("word").lower()
		operations.append(
			{
				"field": field,
				"operation": "indefinite_article_a_to_an",
				"noun_head": word,
			}
		)
		return _capitalized_replacement(match, f"an {word}")

	def replace_consonant_article(match: re.Match[str]) -> str:
		"""Replace ``an`` before a consonant sound and record the correction."""
		word = match.group("word").lower()
		operations.append(
			{
				"field": field,
				"operation": "indefinite_article_an_to_a",
				"noun_head": word,
			}
		)
		return _capitalized_replacement(match, f"a {word}")

	value = _COUNTED_PAIR_PHRASE_RE.sub(replace_counted_pair, value)
	value = _COUNTED_MEASURE_PHRASE_RE.sub(replace_counted_measure, value)
	value = _COUNTED_COCO_NOUN_RE.sub(replace_counted_coco, value)
	value = _COUNTED_GENERIC_NOUN_RE.sub(replace_counted_generic, value)
	value = _ARTICLE_MEASURE_PHRASE_RE.sub(replace_indefinite_measure, value)
	value = _ARTICLE_COCO_NOUN_RE.sub(replace_indefinite_coco, value)
	value = _A_BEFORE_VOWEL_RE.sub(replace_vowel_article, value)
	value = _AN_BEFORE_CONSONANT_RE.sub(replace_consonant_article, value)
	return value, operations


def _normalize_nested_prompt_phrases(
	value: Any, *, field: str
) -> tuple[Any, list[dict[str, Any]]]:
	"""Normalize prompt-bearing fields recursively and return a change audit."""
	if isinstance(value, Mapping):
		result: dict[str, Any] = {}
		operations: list[dict[str, Any]] = []
		for key, child in value.items():
			normalized, child_operations = _normalize_nested_prompt_phrases(
				child, field=f"{field}.{key}"
			)
			result[str(key)] = normalized
			operations.extend(child_operations)
		return result, operations
	if isinstance(value, list):
		result_list: list[Any] = []
		operations = []
		for index, child in enumerate(value):
			normalized, child_operations = _normalize_nested_prompt_phrases(
				child, field=f"{field}[{index}]"
			)
			result_list.append(normalized)
			operations.extend(child_operations)
		return result_list, operations
	if isinstance(value, str):
		return normalize_prompt_noun_phrases(value, field=field)
	return value, []


def prompt_grammar_violations(text: Any, *, field: str) -> list[dict[str, str]]:
	"""Audit every recognized COCO count/article phrase and count-word use."""

	value = _normalize(text)
	violations = [
		{"field": field, "rule": name, "match": match.group(0).lower()}
		for name, pattern in GRAMMAR_AUDIT_RULES
		for match in pattern.finditer(value)
	]
	article_matches = list(_ARTICLE_COCO_NOUN_RE.finditer(value))
	count_matches = list(_COUNTED_COCO_NOUN_RE.finditer(value))
	pair_matches = list(_COUNTED_PAIR_PHRASE_RE.finditer(value))
	measure_article_matches = list(_ARTICLE_MEASURE_PHRASE_RE.finditer(value))
	measure_count_matches = list(_COUNTED_MEASURE_PHRASE_RE.finditer(value))
	generic_count_matches = list(_COUNTED_GENERIC_NOUN_RE.finditer(value))
	for match in article_matches:
		canonical, _class_name = _canonical_indefinite_coco_phrase(match)
		if match.group(0).lower() != canonical:
			violations.append(
				{
					"field": field,
					"rule": "noncanonical_indefinite_coco_noun_phrase",
					"match": match.group(0).lower(),
				}
			)
	for match in count_matches:
		canonical, _class_name = _canonical_counted_coco_phrase(match)
		if match.group(0).lower() != canonical:
			violations.append(
				{
					"field": field,
					"rule": "noncanonical_counted_coco_noun_phrase",
					"match": match.group(0).lower(),
				}
			)
	for match in pair_matches:
		canonical = _canonical_counted_pair_phrase(match)
		if match.group(0).lower() != canonical:
			violations.append(
				{
					"field": field,
					"rule": "noncanonical_counted_pair_phrase",
					"match": match.group(0).lower(),
				}
			)
	for match in measure_article_matches:
		canonical = _canonical_indefinite_measure_phrase(match)
		if match.group(0).lower() != canonical:
			violations.append(
				{
					"field": field,
					"rule": "noncanonical_indefinite_measure_noun_phrase",
					"match": match.group(0).lower(),
				}
			)
	for match in measure_count_matches:
		canonical = _canonical_counted_measure_phrase(match)
		if match.group(0).lower() != canonical:
			violations.append(
				{
					"field": field,
					"rule": "noncanonical_counted_measure_noun_phrase",
					"match": match.group(0).lower(),
				}
			)
	for match in generic_count_matches:
		canonical = _canonical_counted_generic_phrase(match)
		if match.group(0).lower() != canonical:
			violations.append(
				{
					"field": field,
					"rule": "noncanonical_counted_generic_noun_phrase",
					"match": match.group(0).lower(),
				}
			)
	recognized_count_starts = {
		match.start()
		for match in (*count_matches, *pair_matches, *measure_count_matches, *generic_count_matches)
	}
	for match in _COUNT_WORD_RE.finditer(value):
		if match.start() not in recognized_count_starts:
			violations.append(
				{
					"field": field,
					"rule": "unrecognized_number_word_count_phrase",
					"match": match.group(0).lower(),
				}
			)
	return violations


def _nested_prompt_grammar_violations(value: Any, *, field: str) -> list[dict[str, str]]:
	if isinstance(value, Mapping):
		return [
			violation
			for key, child in value.items()
			for violation in _nested_prompt_grammar_violations(child, field=f"{field}.{key}")
		]
	if isinstance(value, list):
		return [
			violation
			for index, child in enumerate(value)
			for violation in _nested_prompt_grammar_violations(child, field=f"{field}[{index}]")
		]
	if isinstance(value, str):
		return prompt_grammar_violations(value, field=field)
	return []


def _rank(value: Mapping[str, Any], *, seed: int, set_id: str) -> str:
	return sha256_json(
		[
			POLICY_VERSION,
			seed,
			set_id,
			value["difficulty"],
			value["recipe_id"],
			value["split"],
			value["lineage_id"],
			value["source_sample_id"],
		]
	)


def _difficulty_for_advanced(recipe_id: str) -> str:
	if recipe_id in MEDIUM_RECIPES:
		return "medium"
	if recipe_id in HARD_RECIPES:
		return "hard"
	raise BalancedPromptSetError(f"unsupported advanced recipe: {recipe_id!r}")


def router_split_for_lineage(lineage_id: str) -> str:
	"""Use the exact canonicalize_state -> planner split derivation."""

	lineage_id = _normalize(lineage_id)
	if not lineage_id:
		raise BalancedPromptSetError("cannot split an empty lineage")
	group_id = stable_id("group", {"prompt_family_id": lineage_id}, length=32)
	return assign_split(group_id, ROUTER_SPLIT_FRACTIONS)


def _source_candidate(
	row: Mapping[str, Any],
	*,
	source_kind: str,
	source_path: Path,
) -> dict[str, Any]:
	"""Convert one source row into a normalized, provenance-carrying candidate."""
	lineage_id = _normalize(row.get("lineage_id"))
	source_split = _normalize(row.get("split"))
	split = router_split_for_lineage(lineage_id)
	source_target = _normalize(row.get("target_prompt"))
	source_initial = _normalize(row.get("initial_generation_prompt"))
	if not lineage_id or source_split not in SPLITS or split not in SPLITS or not source_target or not source_initial or source_target == source_initial:
		raise BalancedPromptSetError(
			f"invalid {source_kind} prompt pair: lineage={lineage_id!r}, split={split!r}"
		)
	if source_kind == "legacy_v7_single_constraint":
		corruption = row.get("corruption") if isinstance(row.get("corruption"), Mapping) else {}
		recipe_id = _normalize(corruption.get("type"))
		if recipe_id not in EASY_RECIPES:
			raise BalancedPromptSetError(f"unsupported legacy recipe: {recipe_id!r}")
		difficulty = "easy"
		source_sample_id = _normalize(row.get("sample_id"))
		source_schema = _normalize(row.get("schema_version"))
		difficulty_rationale = "single_explicit_constraint_perturbation"
	else:
		recipe_id = _normalize(row.get("recipe_id"))
		difficulty = _difficulty_for_advanced(recipe_id)
		source_sample_id = _normalize(row.get("sample_id"))
		source_schema = _normalize(row.get("schema_version"))
		corruption = row.get("corruption") if isinstance(row.get("corruption"), Mapping) else {}
		difficulty_rationale = (
			"coupled_relation_composition_or_scene_perturbation"
			if difficulty == "medium"
			else "multi_constraint_or_semantic_scene_reset"
		)
	if not source_sample_id:
		raise BalancedPromptSetError(f"source sample id is empty for lineage {lineage_id}")
	target, target_operations = normalize_prompt_noun_phrases(
		source_target, field="target_prompt"
	)
	initial, initial_operations = normalize_prompt_noun_phrases(
		source_initial, field="initial_generation_prompt"
	)
	normalized_corruption, corruption_operations = _normalize_nested_prompt_phrases(
		dict(corruption), field="corruption"
	)
	operations = [*target_operations, *initial_operations, *corruption_operations]
	violations = [
		*prompt_grammar_violations(target, field="target_prompt"),
		*prompt_grammar_violations(initial, field="initial_generation_prompt"),
		*_nested_prompt_grammar_violations(normalized_corruption, field="corruption"),
	]
	if violations:
		raise BalancedPromptSetError(
			f"prompt grammar audit failed after normalization: {source_sample_id}: {violations[:3]}"
		)
	if target == initial:
		raise BalancedPromptSetError(
			f"prompt normalization collapsed target/initial semantics: {source_sample_id}"
		)
	return {
		"lineage_id": lineage_id,
		"split": split,
		"difficulty": difficulty,
		"difficulty_rationale": difficulty_rationale,
		"recipe_id": recipe_id,
		"target_prompt": target,
		"initial_generation_prompt": initial,
		"corruption": normalized_corruption,
		"prompt_normalization": {
			"policy_version": PROMPT_NORMALIZATION_POLICY_VERSION,
			"applied": bool(operations),
			"operations": operations,
		},
		"source_kind": source_kind,
		"source_path": source_path.as_posix(),
		"source_schema_version": source_schema,
		"source_sample_id": source_sample_id,
		"source_row_sha256": sha256_json(dict(row)),
		"source_metadata": {
			"source_split": source_split,
			"source_index": row.get("source_index"),
			"parent_sample_id": row.get("parent_sample_id"),
			"parent_source_index": row.get("parent_source_index"),
			"parent_error_family": (
				(row.get("source_metadata") or {}).get("parent_error_family")
				if isinstance(row.get("source_metadata"), Mapping)
				else None
			),
		},
	}


def _load_candidates(legacy_input: Path, advanced_input: Path) -> list[dict[str, Any]]:
	"""Load legacy and advanced candidates under one validated schema."""
	legacy_input = legacy_input.resolve()
	advanced_input = advanced_input.resolve()
	if not legacy_input.is_file() or not advanced_input.is_file():
		raise BalancedPromptSetError("legacy and advanced prompt inputs must both exist")
	advanced_rows = list(iter_jsonl(advanced_input))
	advanced_lineages = {_normalize(row.get("lineage_id")) for row in advanced_rows}
	candidates: list[dict[str, Any]] = []
	# A lineage used by an advanced row is excluded from the easy pool.  This
	# makes leakage impossible regardless of which difficulty is selected first.
	legacy_seen: set[str] = set()
	for row in iter_jsonl(legacy_input):
		if row.get("partition") != "core":
			continue
		lineage = _normalize(row.get("lineage_id"))
		corruption = row.get("corruption") if isinstance(row.get("corruption"), Mapping) else {}
		if lineage in advanced_lineages or lineage in legacy_seen or _normalize(corruption.get("type")) not in EASY_RECIPES:
			continue
		legacy_seen.add(lineage)
		candidates.append(
			_source_candidate(
				row,
				source_kind="legacy_v7_single_constraint",
				source_path=legacy_input,
			)
		)
	advanced_seen: set[str] = set()
	for row in advanced_rows:
		lineage = _normalize(row.get("lineage_id"))
		if lineage in advanced_seen:
			raise BalancedPromptSetError(f"advanced input repeats lineage: {lineage}")
		advanced_seen.add(lineage)
		candidates.append(
			_source_candidate(
				row,
				source_kind="v10_advanced_structured",
				source_path=advanced_input,
			)
		)
	return candidates


def _recipe_totals(set_id: str) -> dict[str, int]:
	if set_id not in SET_FILES:
		raise BalancedPromptSetError(f"unknown set id: {set_id}")
	# Rotate the unavoidable one-row remainders so A and B do not privilege the
	# same recipe.  Totals remain easy=334, medium=333, hard=333.
	easy = [67, 67, 67, 67, 66] if set_id == "set_a" else [67, 67, 67, 66, 67]
	medium = [167, 166] if set_id == "set_a" else [166, 167]
	hard = [167, 166] if set_id == "set_a" else [166, 167]
	return dict(zip((*EASY_RECIPES, *MEDIUM_RECIPES, *HARD_RECIPES), (*easy, *medium, *hard)))


def _recipe_split_quotas(set_id: str) -> dict[tuple[str, str], int]:
	"""Return the frozen recipe/split allocation for one 1K set.

	The small integer remainders are rotated between A/B.  Values are explicit
	so the exact 80/10/10 split remains reproducible even for the rare legacy
	``wrong_count`` validation lineages.
	"""

	if set_id == "set_a":
		allocations = {
			"drop_one_object": (54, 7, 6),
			"reversed_position": (54, 7, 6),
			"swapped_color_attribution": (54, 7, 6),
			"wrong_color": (53, 6, 8),
			"wrong_count": (52, 7, 7),
			"coupled_relation_binding_v1": (134, 16, 17),
			"global_composition_scene_v1": (132, 17, 17),
			"multi_constraint_bundle_v1": (134, 17, 16),
			"semantic_scene_reset_v1": (133, 16, 17),
		}
	elif set_id == "set_b":
		allocations = {
			"drop_one_object": (54, 7, 6),
			"reversed_position": (54, 7, 6),
			"swapped_color_attribution": (54, 7, 6),
			"wrong_color": (52, 4, 10),
			"wrong_count": (53, 9, 5),
			"coupled_relation_binding_v1": (133, 16, 17),
			"global_composition_scene_v1": (133, 17, 17),
			"multi_constraint_bundle_v1": (133, 17, 16),
			"semantic_scene_reset_v1": (134, 16, 17),
		}
	else:
		raise BalancedPromptSetError(f"unknown set id: {set_id}")
	result = {
		(recipe, split): count
		for recipe, counts in allocations.items()
		for split, count in zip(SPLITS, counts)
	}
	for difficulty, recipes in (
		("easy", EASY_RECIPES),
		("medium", MEDIUM_RECIPES),
		("hard", HARD_RECIPES),
	):
		actual = {
			split: sum(result[(recipe, split)] for recipe in recipes) for split in SPLITS
		}
		if actual != DIFFICULTY_SPLIT_QUOTAS[difficulty]:
			raise BalancedPromptSetError(f"internal quota mismatch: {set_id}/{difficulty}")
	if {recipe: sum(result[(recipe, split)] for split in SPLITS) for recipe in allocations} != _recipe_totals(set_id):
		raise BalancedPromptSetError(f"internal recipe total mismatch: {set_id}")
	return result


def _materialize_row(candidate: Mapping[str, Any], *, set_id: str, seed: int) -> dict[str, Any]:
	"""Freeze a selected candidate into a stable set-specific manifest row."""
	pair_id = stable_id(
		"balanced_pair",
		{
			"policy": POLICY_VERSION,
			"set_id": set_id,
			"lineage_id": candidate["lineage_id"],
			"source_row_sha256": candidate["source_row_sha256"],
		},
		length=32,
	)
	return {
		"schema_version": SCHEMA_VERSION,
		"sample_id": pair_id,
		"source_prompt_id": pair_id,
		"dataset": f"v10_balanced_prompts_{set_id}_1k",
		"set_id": set_id,
		"lineage_id": candidate["lineage_id"],
		"prompt_family_id": candidate["lineage_id"],
		"split": candidate["split"],
		"difficulty": candidate["difficulty"],
		"difficulty_rationale": candidate["difficulty_rationale"],
		"recipe_id": candidate["recipe_id"],
		# Both explicit names requested for audit and the standard builder names
		# are retained.  Verification below requires the aliases to be identical.
		"original_prompt": candidate["target_prompt"],
		"believed_corrupted_prompt": candidate["initial_generation_prompt"],
		"target_prompt": candidate["target_prompt"],
		"initial_generation_prompt": candidate["initial_generation_prompt"],
		"initial_prompt_mode": "designed_prompt_counterfactual",
		"prompt_corruption_applied": True,
		"initial_seed": stable_seed(seed, POLICY_VERSION, set_id, candidate["lineage_id"]),
		"error_family": f"balanced_{candidate['difficulty']}_{candidate['recipe_id']}",
		"corruption": dict(candidate["corruption"]),
		"prompt_normalization": dict(candidate["prompt_normalization"]),
		"provenance": {
			"source_kind": candidate["source_kind"],
			"source_path": candidate["source_path"],
			"source_schema_version": candidate["source_schema_version"],
			"source_sample_id": candidate["source_sample_id"],
			"source_row_sha256": candidate["source_row_sha256"],
			"source_metadata": dict(candidate["source_metadata"]),
		},
		"requires_initial_generation": True,
		"route_label_assigned": False,
	}


def _select_set(
	candidates: Sequence[Mapping[str, Any]],
	*,
	set_id: str,
	seed: int,
	used_lineages: set[str],
) -> list[dict[str, Any]]:
	"""Select one balanced set without reusing a lineage chosen by earlier sets."""
	quotas = _recipe_split_quotas(set_id)
	rows: list[dict[str, Any]] = []
	for (recipe, split), count in sorted(quotas.items()):
		pool = [
			candidate
			for candidate in candidates
			if candidate["recipe_id"] == recipe
			and candidate["split"] == split
			and candidate["lineage_id"] not in used_lineages
		]
		pool.sort(key=lambda row: _rank(row, seed=seed, set_id=set_id))
		if len(pool) < count:
			raise BalancedPromptSetError(
				f"insufficient candidates for {set_id}/{recipe}/{split}: need={count}, have={len(pool)}"
			)
		for candidate in pool[:count]:
			used_lineages.add(str(candidate["lineage_id"]))
			rows.append(_materialize_row(candidate, set_id=set_id, seed=seed))
	rows.sort(key=lambda row: (row["difficulty"], row["recipe_id"], row["split"], row["sample_id"]))
	_validate_set_rows(rows, set_id=set_id)
	return rows


def _counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
	return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def _normalization_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
	"""Summarize which deterministic prompt normalizations changed the corpus."""
	operations = [
		operation
		for row in rows
		for operation in row["prompt_normalization"]["operations"]
	]
	return {
		"policy_version": PROMPT_NORMALIZATION_POLICY_VERSION,
		"normalized_rows": sum(bool(row["prompt_normalization"]["applied"]) for row in rows),
		"operation_count": len(operations),
		"operations": dict(
			sorted(Counter(str(operation["operation"]) for operation in operations).items())
		),
		"nouns": dict(
			sorted(
				Counter(
					str(operation.get("noun") or operation.get("noun_head"))
					for operation in operations
				).items()
			)
		),
		"fields": dict(
			sorted(Counter(str(operation["field"]) for operation in operations).items())
		),
		"remaining_violations": 0,
	}


def _nested_string_count(value: Any) -> int:
	if isinstance(value, Mapping):
		return sum(_nested_string_count(child) for child in value.values())
	if isinstance(value, list):
		return sum(_nested_string_count(child) for child in value)
	return int(isinstance(value, str))


def _iter_nested_strings(value: Any) -> Iterable[str]:
	if isinstance(value, Mapping):
		for child in value.values():
			yield from _iter_nested_strings(child)
	elif isinstance(value, list):
		for child in value:
			yield from _iter_nested_strings(child)
	elif isinstance(value, str):
		yield value


def _grammar_audit_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
	"""Aggregate grammar-audit failures without discarding row-level evidence."""
	strings = [
		text
		for row in rows
		for text in (
			str(row["target_prompt"]),
			str(row["initial_generation_prompt"]),
			*_iter_nested_strings(row["corruption"]),
		)
	]
	violations = [
		violation
		for row in rows
		for violation in (
			*prompt_grammar_violations(row["target_prompt"], field="target_prompt"),
			*prompt_grammar_violations(
				row["initial_generation_prompt"], field="initial_generation_prompt"
			),
			*_nested_prompt_grammar_violations(row["corruption"], field="corruption"),
		)
	]
	return {
		"policy_version": PROMPT_GRAMMAR_AUDIT_VERSION,
		"rows_scanned": len(rows),
		"canonical_prompt_fields_scanned": len(rows) * 2,
		"corruption_string_fields_scanned": sum(
			_nested_string_count(row["corruption"]) for row in rows
		),
		"rules": list(PROMPT_GRAMMAR_AUDIT_RULE_NAMES),
		"number_word_count_phrases_scanned": sum(
			len(list(_COUNT_WORD_RE.finditer(text))) for text in strings
		),
		"recognized_coco_count_phrases": sum(
			len(list(_COUNTED_COCO_NOUN_RE.finditer(text))) for text in strings
		),
		"recognized_pair_count_phrases": sum(
			len(list(_COUNTED_PAIR_PHRASE_RE.finditer(text))) for text in strings
		),
		"recognized_measure_count_phrases": sum(
			len(list(_COUNTED_MEASURE_PHRASE_RE.finditer(text))) for text in strings
		),
		"recognized_generic_count_phrases": sum(
			len(list(_COUNTED_GENERIC_NOUN_RE.finditer(text))) for text in strings
		),
		"violation_count": len(violations),
	}


def _validate_set_rows(rows: Sequence[Mapping[str, Any]], *, set_id: str) -> None:
	"""Validate set rows and reject contract violations."""
	if len(rows) != 1000:
		raise BalancedPromptSetError(f"{set_id} must contain exactly 1000 rows")
	if len({row.get("sample_id") for row in rows}) != len(rows):
		raise BalancedPromptSetError(f"{set_id} repeats sample ids")
	if len({row.get("lineage_id") for row in rows}) != len(rows):
		raise BalancedPromptSetError(f"{set_id} repeats parent lineages")
	if _counts(rows, "difficulty") != {"easy": 334, "hard": 333, "medium": 333}:
		raise BalancedPromptSetError(f"{set_id} difficulty balance mismatch")
	if _counts(rows, "split") != {"test": 100, "train": 800, "validation": 100}:
		raise BalancedPromptSetError(f"{set_id} split balance mismatch")
	for row in rows:
		if row.get("schema_version") != SCHEMA_VERSION or row.get("set_id") != set_id:
			raise BalancedPromptSetError(f"{set_id} row schema/identity mismatch")
		if row.get("split") != router_split_for_lineage(str(row.get("lineage_id") or "")):
			raise BalancedPromptSetError(f"{set_id} row does not match downstream planner split")
		if row.get("original_prompt") != row.get("target_prompt"):
			raise BalancedPromptSetError(f"{set_id} original prompt alias mismatch")
		if row.get("believed_corrupted_prompt") != row.get("initial_generation_prompt"):
			raise BalancedPromptSetError(f"{set_id} corrupted prompt alias mismatch")
		if not _normalize(row.get("target_prompt")) or row.get("target_prompt") == row.get("initial_generation_prompt"):
			raise BalancedPromptSetError(f"{set_id} contains an invalid/unmodified pair")
		normalization = row.get("prompt_normalization")
		if not isinstance(normalization, Mapping) or normalization.get("policy_version") != PROMPT_NORMALIZATION_POLICY_VERSION:
			raise BalancedPromptSetError(f"{set_id} prompt normalization contract mismatch")
		operations = normalization.get("operations")
		if not isinstance(operations, list) or normalization.get("applied") is not bool(operations):
			raise BalancedPromptSetError(f"{set_id} prompt normalization audit mismatch")
		violations = [
			*prompt_grammar_violations(row.get("target_prompt"), field="target_prompt"),
			*prompt_grammar_violations(
				row.get("initial_generation_prompt"), field="initial_generation_prompt"
			),
			*_nested_prompt_grammar_violations(row.get("corruption"), field="corruption"),
		]
		if violations:
			raise BalancedPromptSetError(
				f"{set_id} contains prompt grammar violations: {violations[:3]}"
			)
		if row.get("route_label_assigned") is not False or row.get("requires_initial_generation") is not True:
			raise BalancedPromptSetError(f"{set_id} contains premature labels or generation status")
		provenance = row.get("provenance")
		if (
			not isinstance(provenance, Mapping)
			or len(_normalize(provenance.get("source_row_sha256"))) != 64
		):
			raise BalancedPromptSetError(f"{set_id} row provenance is incomplete")


def _canonical_text(rows: Iterable[Mapping[str, Any]]) -> str:
	return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def build_balanced_prompt_sets(
	*,
	legacy_input: Path = DEFAULT_LEGACY_INPUT,
	advanced_input: Path = DEFAULT_ADVANCED_INPUT,
	output_root: Path = DEFAULT_OUTPUT_ROOT,
	seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
	"""Build two deterministic, family-balanced, mutually disjoint prompt sets with full provenance."""
	if not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0:
		raise BalancedPromptSetError("seed must be a positive integer")
	legacy_input = legacy_input.resolve()
	advanced_input = advanced_input.resolve()
	output_root = output_root.resolve()
	candidates = _load_candidates(legacy_input, advanced_input)
	used_lineages: set[str] = set()
	sets = {
		set_id: _select_set(candidates, set_id=set_id, seed=seed, used_lineages=used_lineages)
		for set_id in SET_FILES
	}
	if len(used_lineages) != 2000:
		raise BalancedPromptSetError("the two sets are not globally lineage-disjoint")
	output_root.mkdir(parents=True, exist_ok=True)
	set_manifests: dict[str, Any] = {}
	for set_id, rows in sets.items():
		path = output_root / SET_FILES[set_id]
		write_jsonl_atomic(path, rows)
		text = _canonical_text(rows)
		set_manifests[set_id] = {
			"path": path.name,
			"rows": len(rows),
			"rows_sha256": sha256_text(text),
			"unique_lineages": len({row["lineage_id"] for row in rows}),
			"difficulties": _counts(rows, "difficulty"),
			"recipes": _counts(rows, "recipe_id"),
			"splits": _counts(rows, "split"),
			"source_kinds": dict(
				sorted(Counter(row["provenance"]["source_kind"] for row in rows).items())
			),
			"prompt_normalization": _normalization_summary(rows),
			"prompt_grammar_audit": _grammar_audit_summary(rows),
		}
	union_rows = [row for set_id in SET_FILES for row in sets[set_id]]
	manifest = {
		"schema_version": MANIFEST_SCHEMA_VERSION,
		"policy_version": POLICY_VERSION,
		"seed": seed,
		"rows": len(union_rows),
		"sets": set_manifests,
		"inputs": {
			"legacy": {"path": str(legacy_input), "sha256": sha256_file(legacy_input)},
			"advanced": {"path": str(advanced_input), "sha256": sha256_file(advanced_input)},
		},
		"global_unique_lineages": len(used_lineages),
		"cross_set_lineage_overlap": 0,
		"split_policy": {
			"group_id": "stable_id('group', {'prompt_family_id': lineage_id}, length=32)",
			"assignment": "planner.actions.assign_split_sha256_v1",
			"fractions": ROUTER_SPLIT_FRACTIONS,
		},
		"difficulty_definition": {
			"easy": "one explicit target constraint is changed or removed",
			"medium": "relation/attribute or global composition/scene constraints are changed together",
			"hard": "multiple count/attribute/relation constraints or the semantic scene are reset",
		},
		"prompt_normalization": _normalization_summary(union_rows),
		"prompt_grammar_audit": _grammar_audit_summary(union_rows),
		"route_labels_assigned": False,
		"generation_or_qwen_used": False,
		"union_rows_sha256": sha256_json(union_rows),
	}
	manifest_path = output_root / MANIFEST_FILE
	write_json_atomic(manifest_path, manifest)
	checksum_paths = [output_root / name for name in SET_FILES.values()] + [manifest_path]
	checksum_text = "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths)
	(output_root / CHECKSUM_FILE).write_text(checksum_text, encoding="utf-8")
	return {**manifest, "output_root": str(output_root), "checksums_path": CHECKSUM_FILE}


def verify_balanced_prompt_sets(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
	"""Recompute source provenance and audits to reject stale or tampered balanced prompt sets."""
	output_root = output_root.resolve()
	manifest_path = output_root / MANIFEST_FILE
	if not manifest_path.is_file():
		raise BalancedPromptSetError(f"manifest is missing: {manifest_path}")
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or manifest.get("policy_version") != POLICY_VERSION:
		raise BalancedPromptSetError("unsupported balanced prompt manifest")
	all_rows: list[dict[str, Any]] = []
	lineages_by_set: dict[str, set[str]] = {}
	for set_id, expected_name in SET_FILES.items():
		entry = (manifest.get("sets") or {}).get(set_id)
		if not isinstance(entry, Mapping) or entry.get("path") != expected_name:
			raise BalancedPromptSetError(f"manifest entry mismatch: {set_id}")
		path = output_root / expected_name
		rows = list(iter_jsonl(path))
		_validate_set_rows(rows, set_id=set_id)
		text = _canonical_text(rows)
		if sha256_text(text) != entry.get("rows_sha256") or len(rows) != entry.get("rows"):
			raise BalancedPromptSetError(f"row digest/count mismatch: {set_id}")
		if _counts(rows, "difficulty") != entry.get("difficulties"):
			raise BalancedPromptSetError(f"difficulty manifest mismatch: {set_id}")
		if _counts(rows, "recipe_id") != entry.get("recipes") or _counts(rows, "split") != entry.get("splits"):
			raise BalancedPromptSetError(f"recipe/split manifest mismatch: {set_id}")
		source_kinds = dict(
			sorted(Counter(row["provenance"]["source_kind"] for row in rows).items())
		)
		if source_kinds != entry.get("source_kinds"):
			raise BalancedPromptSetError(f"source-kind manifest mismatch: {set_id}")
		if _normalization_summary(rows) != entry.get("prompt_normalization"):
			raise BalancedPromptSetError(f"prompt-normalization manifest mismatch: {set_id}")
		if _grammar_audit_summary(rows) != entry.get("prompt_grammar_audit"):
			raise BalancedPromptSetError(f"prompt-grammar audit manifest mismatch: {set_id}")
		lineages_by_set[set_id] = {str(row["lineage_id"]) for row in rows}
		all_rows.extend(rows)
	if lineages_by_set["set_a"] & lineages_by_set["set_b"]:
		raise BalancedPromptSetError("parent lineage leaks between set_a and set_b")
	if manifest.get("global_unique_lineages") != len({row["lineage_id"] for row in all_rows}):
		raise BalancedPromptSetError("global lineage count mismatch")
	if manifest.get("union_rows_sha256") != sha256_json(all_rows):
		raise BalancedPromptSetError("union digest mismatch")
	if manifest.get("prompt_normalization") != _normalization_summary(all_rows):
		raise BalancedPromptSetError("global prompt-normalization summary mismatch")
	if manifest.get("prompt_grammar_audit") != _grammar_audit_summary(all_rows):
		raise BalancedPromptSetError("global prompt-grammar audit summary mismatch")
	source_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
	source_paths: dict[str, str] = {}
	for source_name, source in (manifest.get("inputs") or {}).items():
		path = Path(str(source.get("path") or ""))
		if not path.is_file() or sha256_file(path) != source.get("sha256"):
			raise BalancedPromptSetError(f"source input changed: {path}")
		source_paths[source_name] = path.as_posix()
		source_rows[source_name] = {sha256_json(row): row for row in iter_jsonl(path)}
	for row in all_rows:
		provenance = row["provenance"]
		source_name = (
			"legacy"
			if provenance["source_kind"] == "legacy_v7_single_constraint"
			else "advanced"
		)
		if provenance["source_path"] != source_paths[source_name]:
			raise BalancedPromptSetError(f"source path provenance mismatch: {row['sample_id']}")
		source_row = source_rows[source_name].get(provenance["source_row_sha256"])
		if source_row is None:
			raise BalancedPromptSetError(f"source row provenance mismatch: {row['sample_id']}")
		expected = _source_candidate(
			source_row,
			source_kind=provenance["source_kind"],
			source_path=Path(provenance["source_path"]),
		)
		for field in (
			"lineage_id",
			"split",
			"difficulty",
			"recipe_id",
			"target_prompt",
			"initial_generation_prompt",
			"corruption",
			"prompt_normalization",
		):
			if row.get(field) != expected.get(field):
				raise BalancedPromptSetError(
					f"normalized source projection mismatch: {row['sample_id']}/{field}"
				)
	checksum_path = output_root / CHECKSUM_FILE
	expected_checksums = "".join(
		f"{sha256_file(path)}  {path.name}\n"
		for path in [*(output_root / name for name in SET_FILES.values()), manifest_path]
	)
	if not checksum_path.is_file() or checksum_path.read_text(encoding="utf-8") != expected_checksums:
		raise BalancedPromptSetError("checksums.sha256 is missing or stale")
	return {
		"status": "ok",
		"rows": len(all_rows),
		"rows_per_set": 1000,
		"global_unique_lineages": len({row["lineage_id"] for row in all_rows}),
		"cross_set_lineage_overlap": 0,
		"set_hashes": {
			set_id: manifest["sets"][set_id]["rows_sha256"] for set_id in SET_FILES
		},
	}


def build_parser() -> argparse.ArgumentParser:
	"""Construct the CLI parser for balanced prompt-set construction."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--legacy-input", type=Path, default=DEFAULT_LEGACY_INPUT)
	parser.add_argument("--advanced-input", type=Path, default=DEFAULT_ADVANCED_INPUT)
	parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	parser.add_argument("--verify-only", action="store_true")
	return parser


def main(argv: list[str] | None = None) -> None:
	"""Run this module's command-line workflow and return its process status."""
	args = build_parser().parse_args(argv)
	result = (
		verify_balanced_prompt_sets(args.output_root)
		if args.verify_only
		else build_balanced_prompt_sets(
			legacy_input=args.legacy_input,
			advanced_input=args.advanced_input,
			output_root=args.output_root,
			seed=args.seed,
		)
	)
	print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
	main()
