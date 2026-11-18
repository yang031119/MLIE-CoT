from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


def _quad(x0: int, y0: int, x1: int, y1: int):
	return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _edit(index: int, color: str, quad):
	return {
		"id": f"edit_{index:02d}",
		"target_object": f"object {index}",
		"issue": f"object {index} has the wrong color",
		"edit_prompt": f"make object {index} {color}",
		"whole_edit": f"Change object {index} to {color}",
		"quad_points": quad,
	}


def test_online_and_offline_share_exact_l1_crop_geometry_and_order() -> None:
	from inference.main_method.modules import local_edit_worker as online
	from inference.main_method.modules.contracts import l1_crop_roi, l1_crop_rois
	from inference.router_data_collect import execute_flux2 as offline

	# Both execution paths import the callable itself; neither keeps a second
	# padding/min-span/rounding implementation.
	assert online.l1_crop_rois is l1_crop_rois
	assert offline.l1_crop_rois is l1_crop_rois

	quads = [
		_quad(100, 100, 109, 109),
		_quad(0, 0, 2, 2),
		_quad(250, 248, 255, 255),
	]
	rois = l1_crop_rois(quads, 256, 256)
	assert rois == [
		[73, 73, 137, 137],
		[0, 0, 64, 64],
		[192, 192, 256, 256],
	]
	assert rois == [l1_crop_roi(quad, 256, 256) for quad in quads]
	# The 64-pixel minimum is the final crop span, not 64 * 1.35.
	assert rois[1][2] - rois[1][0] == 64


def test_prepare_joint_l1_task_has_all_edits_and_no_action_retry_fields(tmp_path) -> None:
	from inference.main_method.modules.local_benchmark import prepare_local_benchmark_retry_inputs

	input_path = tmp_path / "input.png"
	Image.new("RGB", (128, 64), "white").save(input_path)
	state = SimpleNamespace(
		current_image=input_path,
		sample_id="sample-7",
		prompt="three colored objects in a row",
		global_round=2,
		history=[
			{
				"analysis": {
					"canonical_diagnosis": {
						"fully_satisfied": False,
						"reason": "The three object colors are incorrect.",
						"issue": "incorrect object colors",
						"fix": "restore all requested colors",
					}
				}
			}
		],
		micro_plan=[
			_edit(1, "red", _quad(0, 0, 31, 31)),
			_edit(2, "blue", _quad(32, 0, 63, 31)),
			_edit(3, "green", _quad(64, 0, 95, 31)),
		],
	)
	first = prepare_local_benchmark_retry_inputs(
		state,
		tmp_path / "batch-a",
		tmp_path / "round-a",
		"repair_l1_joint",
		64,
		5,
		base_seed=123,
	)
	second = prepare_local_benchmark_retry_inputs(
		state,
		tmp_path / "batch-b",
		tmp_path / "round-b",
		"repair_l1_joint",
		64,
		1,
		base_seed=123,
	)
	payload = json.loads((tmp_path / "batch-a" / "edit.json").read_text(encoding="utf-8"))
	assert payload["action_schema"] == "l1_joint_multicrop_v1"
	assert payload["action"] == "joint_local_edit"
	assert payload["candidate_count"] == payload["editor_call_count"] == 1
	assert payload["target_prompt"] == "three colored objects in a row"
	assert payload["reference_roles"] == [
		"editable_full_image",
		"localization_crop_01",
		"localization_crop_02",
		"localization_crop_03",
	]
	assert payload["instruction"].startswith("Edit Image 1, which is the only editable full-resolution canvas.")
	assert len(payload["edits"]) == 3
	assert payload["action_seed"] == first["action_seed"] == second["action_seed"]
	assert payload["round_index"] == 2
	serialized = json.dumps(payload)
	assert "remaining_attempt" not in serialized
	assert "retry_feedback" not in serialized
	assert "micro_attempt" not in serialized
	assert payload["edits"][0]["quad_points"] == _quad(0, 0, 16, 31)


def test_flux2_joint_l1_uses_full_plus_ordered_crops_in_one_call(tmp_path, monkeypatch) -> None:
	from inference.main_method.modules.local_edit_worker import (
		LocalEditSpec,
		SampleSpec,
		run_flux2_edit,
	)
	import torch

	class FakeResult:
		images = [Image.new("RGB", (256, 256), "black")]

	class FakePipe:
		def __init__(self) -> None:
			self.calls = []

		def __call__(self, **kwargs):
			self.calls.append(kwargs)
			return FakeResult()

	class FakeGenerator:
		def __init__(self) -> None:
			self.seed = None

		def manual_seed(self, seed):
			self.seed = seed
			return self

	created_generators = []

	def make_generator(device=None):
		generator = FakeGenerator()
		created_generators.append((device, generator))
		return generator

	monkeypatch.setattr(torch, "Generator", make_generator)
	sample_dir = tmp_path / "sample"
	sample_dir.mkdir()
	input_path = sample_dir / "input.png"
	image = Image.new("RGB", (256, 256), "white")
	regions = [
		((16, 16, 80, 80), (255, 0, 0)),
		((96, 16, 160, 80), (0, 0, 255)),
		((176, 16, 240, 80), (0, 128, 0)),
	]
	for box, color in regions:
		for x in range(box[0], box[2]):
			for y in range(box[1], box[3]):
				image.putpixel((x, y), color)
	image.save(input_path)
	edits = [
		LocalEditSpec("edit_01", "first", "wrong", "make first cyan", "Make first cyan", [(16, 16), (79, 16), (79, 79), (16, 79)]),
		LocalEditSpec("edit_02", "second", "wrong", "make second magenta", "Make second magenta", [(96, 16), (159, 16), (159, 79), (96, 79)]),
		LocalEditSpec("edit_03", "third", "wrong", "make third yellow", "Make third yellow", [(176, 16), (239, 16), (239, 79), (176, 79)]),
	]
	sample = SampleSpec(
		sample_id="sample",
		sample_dir=sample_dir,
		input_image=input_path,
		edit_json=sample_dir / "edit.json",
		instruction="apply three edits",
		whole_edit="apply three edits",
		target_object="first",
		quad_points=edits[0].quad_points,
		edits=edits,
		action_seed=987654,
		round_index=4,
	)
	pipe = FakePipe()
	args = SimpleNamespace(
		model="flux2",
		resolution=256,
		seed=123,
		flux2_guidance_scale=4.0,
		flux2_steps=2,
		flux2_max_sequence_length=128,
	)
	record = run_flux2_edit(args, sample, tmp_path / "worker", 999, {"pipe": pipe, "resolved_model": "fake"})
	assert len(pipe.calls) == 1
	call = pipe.calls[0]
	assert len(call["image"]) == 4
	assert [image.getpixel((128, 128)) for image in call["image"][1:]] == [
		(255, 0, 0),
		(0, 0, 255),
		(0, 128, 0),
	]
	assert "Image 2" in call["prompt"] and "Image 3" in call["prompt"] and "Image 4" in call["prompt"]
	assert created_generators[0][1].seed == 987654
	assert record["action_schema"] == "l1_joint_multicrop_v1"
	assert record["flux_call_count"] == 1
	assert record["call_count"] == 1
	assert record["reference_roles"] == [
		"editable_full_image",
		"localization_crop_01",
		"localization_crop_02",
		"localization_crop_03",
	]
	assert record["seed"] == 987654
	assert len(record["rois_xyxy"]) == 3
	from inference.main_method.modules.contracts import l1_crop_rois

	assert record["rois_xyxy"] == l1_crop_rois(
		[[list(point) for point in edit.quad_points] for edit in edits],
		256,
		256,
	)
	assert [Path(path).name for path in record["reference_images"]] == [
		"input.png",
		"local_crop_01.png",
		"local_crop_02.png",
		"local_crop_03.png",
	]


def test_stable_l1_seed_is_independent_of_worker_sharding() -> None:
	from inference.main_method.modules.local_edit_worker import stable_l1_action_seed

	edits = [_edit(1, "blue", _quad(1, 2, 20, 21))]
	first = stable_l1_action_seed(42, "same-state", edits, round_index=3)
	second = stable_l1_action_seed(42, "same-state", edits, round_index=3)
	other_round = stable_l1_action_seed(42, "same-state", edits, round_index=4)
	assert first == second
	assert first != other_round


def test_joint_l1_prepared_manifest_round_trip_records_contract(tmp_path) -> None:
	from inference.main_method.modules.local_edit_worker import (
		LocalEditSpec,
		SampleSpec,
		load_prepare_manifest,
		save_prepare_manifest,
	)

	sample_dir = tmp_path / "sample"
	sample_dir.mkdir()
	input_path = sample_dir / "input.png"
	Image.new("RGB", (32, 32), "white").save(input_path)
	edits = [
		LocalEditSpec("edit_01", "left", "wrong", "fix left", "Fix left", [(0, 0), (15, 0), (15, 15), (0, 15)]),
		LocalEditSpec("edit_02", "right", "wrong", "fix right", "Fix right", [(16, 0), (31, 0), (31, 15), (16, 15)]),
	]
	sample = SampleSpec(
		sample_id="sample",
		sample_dir=sample_dir,
		input_image=input_path,
		edit_json=sample_dir / "edit.json",
		instruction="fix both",
		whole_edit="fix both",
		target_object="left",
		quad_points=edits[0].quad_points,
		edits=edits,
		action_seed=77,
		round_index=2,
		joint_prompt="shared joint prompt",
	)
	save_prepare_manifest(tmp_path / "worker", [sample])
	payload = json.loads((tmp_path / "worker" / "prepared_manifest.json").read_text(encoding="utf-8"))
	assert payload[0]["action_schema"] == "l1_joint_multicrop_v1"
	assert payload[0]["action_seed"] == 77
	assert len(payload[0]["edits"]) == 2
	loaded = load_prepare_manifest(tmp_path / "worker")
	assert loaded[0].action_seed == 77
	assert loaded[0].joint_prompt == "shared joint prompt"
	assert [edit.id for edit in loaded[0].edits] == ["edit_01", "edit_02"]
