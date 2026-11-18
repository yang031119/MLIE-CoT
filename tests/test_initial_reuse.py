"""Initial-image reuse must keep bytes and reject a different model policy."""
import json
from pathlib import Path

from PIL import Image
import pytest

from inference.self_route_version.modules.initials import seed_initial_images
from inference.self_route_version.modules.state import STATE_SCHEMA_VERSION


def test_initial_reuse_preserves_image_and_policy(tmp_path):
    source = tmp_path / "source"
    sample = source / "sample"
    sample.mkdir(parents=True)
    prompt = "a red cube"
    (sample / "prompt.txt").write_text(prompt, encoding="utf-8")
    Image.new("RGB", (16, 16), "red").save(sample / "initial.png")
    policy = {"executor": {"family": "flux2", "revision": "a" * 40}, "generation": {"steps": 28}}
    (source / "run_config.json").write_text(json.dumps({"l3_policy_contract": policy}), encoding="utf-8")
    kwargs = dict(source_run=source, target_run=tmp_path / "target", prompts=[(0, prompt)],
                  initial_generator_family="flux2", initial_generator_policy_digest="digest",
                  target_policy=policy)
    result = seed_initial_images(**kwargs)
    image = Path(result["rows"][0]["target_image"])
    assert image.read_bytes() == (sample / "initial.png").read_bytes()
    state = json.loads((image.parent / "status.json").read_text(encoding="utf-8"))
    assert state["state_schema_version"] == STATE_SCHEMA_VERSION
    assert state["initial_generator_policy_digest"] == "digest"
    assert seed_initial_images(**kwargs)["seeded"] == 1
    kwargs["target_policy"] = {**policy, "generation": {"steps": 30}}
    with pytest.raises(ValueError, match="differs from target policy"):
        seed_initial_images(**kwargs)
