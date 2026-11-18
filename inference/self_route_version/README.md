# MLIE-CoT semantic self-routing baseline

`main.py` defaults to a Qwen semantic router with FLUX.2 generation/editing and
no learned route head. After factual diagnosis, Qwen chooses L1, L2, or L3.
The selected action generates one candidate, followed by a blind retain review.

```bash
python inference/self_route_version/main.py --help
python inference/self_route_version/single_infer.py --help
```

See the [root README](../../README.md) for runnable command templates, including
the bundled `data/router_heads/route_head.pt`. Use that file with
`--semantic-router head --qwen-route-head data/router_heads/route_head.pt`
and the pinned Qwen/FLUX.2 snapshots documented there. The default `main.py`
invocation remains the Qwen semantic baseline and does not load the head.
Use `--semantic-router` to choose a supported semantic controller or explicit
learned-head mode. In head mode, provide a checkpoint compatible with this
pipeline's encoder and action policy; the main-method checkpoint is not
automatically interchangeable. BAGEL mode requires the external UniCoT runtime
and BAGEL weights. Model paths, immutable revisions, and GPU resources remain
explicit CLI settings.

This package retains a separate implementation because its diagnosis/action
schemas, repair-memory behavior, and executor options differ from the main
method. Its local worker supports FLUX.2 and BAGEL; the old SD-inpaint local
adapter is excluded. SD3.5 remains available as an initial/L3 generator.

Each retained or rejected edit records the instruction and comparison feedback
in `status.json:repair_memory`. Later repair plans can use this history to avoid
repeating ineffective edits. See [SYSTEM_PROMPTS.md](SYSTEM_PROMPTS.md).
