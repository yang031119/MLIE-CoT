"""A failed localization must never turn a fixed L1 arm into L2."""
import importlib
import json
from types import SimpleNamespace

from PIL import Image
import pytest


@pytest.mark.parametrize("package", ["main_method", "self_route_version"])
@pytest.mark.parametrize("fixed", [True, False])
@pytest.mark.parametrize("valid_plan", [False, True])
def test_l1_plan_respects_fixed_level(tmp_path, monkeypatch, package, fixed, valid_plan):
    module = importlib.import_module(f"inference.{package}.modules.qwen_stages")
    state_type = importlib.import_module(f"inference.{package}.modules.state").SampleState
    io = importlib.import_module(f"inference.{package}.modules.io_utils")
    image = tmp_path / "initial.png"
    Image.new("RGB", (32, 32), "blue").save(image)
    original = image.read_bytes()
    state = state_type(index=0, prompt="a red car", sample_dir=tmp_path,
                       sample_id="sample", current_image=image, global_round=1,
                       needs_global_analysis=True, initial_generator_family="flux2")
    args = SimpleNamespace(fixed_repair_level=1 if fixed else None,
                           max_global_rounds=5, qwen_max_new_tokens=128,
                           qwen_batch_size=1, qwen_input_max_side=32,
                           language_backend="qwen", semantic_router="head")
    diagnosis = dict(valid_json=True, fully_satisfied=False, category="Unrouted",
                     recommended_mode="unrouted", reason="wrong color", issues=[],
                     preservation_constraints=[], raw_output="{}")
    plan = dict(valid_json=False, reason="L1 requires one to three edits.",
                edits=[], raw_output='{"edits": []}', json_retry_count=1)
    if valid_plan:
        plan = dict(valid_json=True, level=1, instruction="repaint the car",
                    edits=[dict(id="edit_1", whole_edit="repaint the car",
                                quad_points=[[0, 0], [32, 0], [32, 32], [0, 32]])])
    batches = iter([[diagnosis], [plan]])
    monkeypatch.setattr(module, "run_language_batch_with_retries", lambda *a, **kw: next(batches))
    monkeypatch.setattr(module, "_build_l1_plan_job", lambda item, analysis, args: item)
    def route(state, analysis):
        analysis.update(category="L1", recommended_mode="micro")
        return dict(enabled=True, level=1, source="test_adaptive_router")
    monkeypatch.setattr(module, "apply_route_head_decision", route)
    compiled = []
    def attach(args, state, analysis, level):
        compiled.append(level)
        analysis["action_plan"] = dict(level=level, instruction="repaint the car")
    monkeypatch.setattr(module, "_attach_deterministic_plan", attach)
    module.run_global_triage_batch(object(), args, [state])
    decision = json.loads((tmp_path / "round1/level_classification.json").read_text(encoding="utf-8"))
    assert state.global_round == 1
    assert image.read_bytes() == original
    if valid_plan:
        assert compiled == []
        assert decision["decision"]["level"] == 1
        assert state.phase == "micro" and state.status == "pending"
        assert state.final_image is None and len(state.micro_plan) == 1
        assert json.loads((tmp_path / "round1/repair_plan.json").read_text(encoding="utf-8"))["action_plan"]["level"] == 1
    elif fixed:
        assert compiled == []
        assert decision["category"] == "L1" and decision["decision"]["level"] == 1
        assert state.status == "partial"
        assert state.final_reason == "fixed_l1_planning_failed_after_json_retry"
        assert state.final_image == image and not state.needs_global_analysis
        assert not state.pending_global_compare and not state.pending_local_review
        assert not (tmp_path / "round1/repair_plan.json").exists()
        assert state.history[-1]["event"] == "fixed_level_planning_failed"
        io.finalize_state(state, tmp_path / "summary.jsonl")
        assert state.done and (tmp_path / "final.png").read_bytes() == original
    else:
        assert compiled == [2]
        assert decision["decision"]["level"] == 2
        assert state.status == "pending" and state.final_image is None


@pytest.mark.parametrize("package", ["main_method", "self_route_version"])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_fixed_levels_preserve_satisfied_gate(package, level):
    module = importlib.import_module(f"inference.{package}.modules.qwen_stages")
    analysis = dict(fully_satisfied=True, recommended_mode="done", category="L0")
    decision = module.apply_fixed_repair_level(
        SimpleNamespace(fixed_repair_level=level), SimpleNamespace(phase="global"), analysis)
    assert decision["level"] == 0 and analysis["fully_satisfied"]
