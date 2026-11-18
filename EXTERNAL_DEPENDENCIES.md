# External code, data, and weights

The project route head at `data/router_heads/route_head.pt` is bundled.
Users must obtain base-model weights, external baseline code/weights, and
benchmark datasets/evaluators themselves from the official repositories linked
in the [README](README.md#environment-and-model-assets). These external resources
are not bundled.

| Feature | Required asset | Configuration |
| --- | --- | --- |
| Qwen analysis/routing | Qwen3.5 model and processor snapshot | `--qwen-model`, `--qwen-model-revision`, `HF_HOME` |
| FLUX.2 initial/L1/L2/L3 | FLUX.2-dev snapshot | `--flux-model`, `--flux-model-revision` |
| Bundled learned router | `data/router_heads/route_head.pt`, compatible with the self-route package's FLUX.2 learned-head mode | `--semantic-router head --qwen-route-head data/router_heads/route_head.pt` |
| Main-method learned router | A matching checkpoint obtained separately or produced by `scripts/train_route_head.py` | `--qwen-route-head`; the bundled deployment-adapted head is incompatible |
| BAGEL generation/editing | UniCoT checkout exposing `inference_unicot_v0.2.py`, BAGEL configs and weights | `--unicot-root`, `--bagel-model-path`, `--bagel-fine-tuned-path`, corresponding revision flags |
| SD3.5 initial/L3 | Stable Diffusion 3.5 Large Turbo snapshot | `--sd35-model`, `--sd35-model-revision` |
| GenEval | External evaluator and matching metadata | `scripts/prepare_geneval_eval_input.py --help` |
| T2I-CompBench | External evaluator and benchmark metadata | `scripts/prepare_t2i_compbench_outputs.py --help` |
| External baselines, including Uni-CoT | Authors' implementations and baseline-specific weights | Follow the upstream repository instructions; these weights are separate from the MLIE-CoT route head |
| WISE | Matching release exposing `vllm_eval.py:build_evaluation_messages` and `data_verified/*.json` | `WISE_ROOT` (default `third_party/WISE`) |
| CLIP/PickScore/HPS | Their cached weights and optional evaluation dependencies | `requirements-evaluation.txt`, evaluator `--cache-dir` |

The WISE helper expects this exact interface; an arbitrary upstream checkout
may need adaptation. It is not available in the supplied source tree. The BAGEL
adapter likewise requires the stated UniCoT API. The historical filename in
that external API is retained so imports work.

Generation subprocesses default to the current Python interpreter. Separate
environments can be selected with `--vcot-python`, `--bagel-python`,
`--flux-python`, and `--sd35-python` where exposed by the selected entry point.
Grounded-Segment-Anything is not required by the current pipelines.

Check third-party licenses before redistributing their code, weights, or data.
The [project license](LICENSE) covers MLIE-CoT's own code and bundled route head;
it grants no rights to the external assets listed above.
