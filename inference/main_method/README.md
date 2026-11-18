# MLIE-CoT learned-route inference

`main.py` runs factual diagnosis, frozen-Qwen MLP routing, a route-conditioned
plan, one generated candidate, and source-blind retain review. Native Qwen owns
the L0 satisfaction gate; the MLP has L1/L2/L3 logits only.

L1 supplies one full image plus one to three localization crops to FLUX.2 in a
single call. L2 uses one full-image FLUX.2 edit. L3 generates with the same
initial model family and immutable policy as the sample's initial image.
The action budget counts generated candidates, not JSON-format retries or
diagnosis/review calls. A terminal diagnosis follows the last allowed action.

```bash
python inference/main_method/main.py --help
python inference/main_method/single_infer.py --help
python inference/main_method/route_ablation.py --help
```

Follow the [root README](../../README.md) for model paths and an example command.
Use `--qwen-route-head` to select a trained checkpoint, or `--fixed-repair-level`
for a fixed L1/L2/L3 run. The default checkpoint path is
`data/router_heads/route_head.pt`, resolved from the repository root.
The bundled file currently targets the self-route package's learned-head mode;
the main-method loader still rejects its action-policy contract. Override the
path with a compatible checkpoint until that contract difference is resolved.
Checkpoint validation locks the Qwen encoder/processor,
diagnosis projection, prompt distribution, FLUX.2 L1/L2 executor policy, and
initial-base L3 policy. One route head belongs to one homogeneous initial base.
Legacy or incompatible checkpoint contracts are rejected.

`modules/local_edit_worker.py` is the FLUX.2 local-action subprocess.
`bagel_service.py`, `flux_service.py`, and `sd35_service.py` implement the current
initial/L3 backends. They are active dependencies, not obsolete comparison code.
The [prompt reference](SYSTEM_PROMPTS.md) documents the language stages.
