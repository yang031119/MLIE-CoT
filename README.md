# MLIE-CoT

**MLIE-CoT: Chain-of-Thought Reasoning via Multi-Level Image Editing**

MLIE-CoT (Multi-level Image Editing Chain-of-Thought) uses intermediate images
as reusable reasoning states. It diagnoses prompt-image mismatches and learns
how much of the current image to preserve through repair-level routing.
This source release includes the learned-router method, a semantic self-routing
baseline, router-data collection, training, controlled ablations, and evaluation utilities.

The repair loop is:

```text
target prompt → target reasoning / prompt optimization → initial image
current image + target prompt → factual diagnosis → L0 accept, or L1/L2/L3 route
→ route-conditioned plan → one candidate → blind retain review → next diagnosis
```

| Level | Paper terminology | Execution |
| --- | --- | --- |
| L0 | Accept current image | The diagnostic model accepts a satisfied image; no repair is generated. |
| L1 | Local / object-level repair | One full image followed by 1–3 localization crops, supplied together to the editor in one call. |
| L2 | Image-conditioned whole-image editing | Edit the current full image with a repair instruction. |
| L3 | Text-only regeneration | Generate from a new text condition using the initial model family, without the current image. |

The learned router uses a frozen Qwen3.5-9B encoder and a trainable two-layer
MLP with **three outputs (L1/L2/L3)**. L0 is a separate satisfaction gate.
Each online step executes only the selected repair, then compares the current
and candidate images to select the next state. In `main_method`, FLUX.2 executes
L1/L2 and the initial generator executes L3.

## Submission and release scope

The submitted paper evaluates **GenEval and WISE** with **BAGEL-7B-MoT and
FLUX.2-dev**. SD3.5 support, the T2I-CompBench adapter, and generic reward
scorers are additional code utilities; they are not reported main experiments
in this submission.

Router supervision uses controlled prompt perturbations, fixed plans, and
paired seeds across the three repair levels. The paper reports five repeats
per level (`K = 5`), 1K construction prompts, and 14.8K routing-training images.
The collection CLI's generic example defaults to `rollouts_per_action = 3`;
set it to **5** for the submitted collection protocol. The small files in
`data/` are illustrative inputs, not the complete training or benchmark data.
Training splits are isolated by target-prompt family. See the
[collection and training guide](inference/router_data_collect/README.md).

For the fixed-level comparison, reuse exactly the same initial images across
L1-only, L2-only, L3-only, and learned routing, as described in the
[ablation guide](inference/router_ablation/README.md). The semantic self-route
baseline is separate from learned routing, including when the learned head is
loaded through the `self_route_version` package.

## Layout

| Directory | Contents |
| --- | --- |
| `inference/main_method/` | Learned-route inference and fixed-level ablation entry point |
| `inference/self_route_version/` | Qwen/BAGEL semantic routing and optional learned-head execution |
| `inference/router_data_collect/` | Initial images, diagnosis, counterfactual actions, evaluation, and dataset validation |
| `inference/router_ablation/` | Shared-initial-image and ablation integrity checks |
| `scripts/` | Router training, static baseline/ensemble evaluation, and benchmark adapters |
| `evaluation/` | General image-directory CLIP/PickScore/HPS scoring |
| `data/` | Illustrative inputs and the bundled route-head checkpoint; benchmark data is external |
| `tests/` | CPU regression tests; data-collection tests also live beside that package |

## Environment and model assets

Use Python 3.10 or later. Model inference targets Linux with CUDA and
`nvidia-smi`; CPU tests do not generate images. Install a CUDA-compatible
PyTorch build for your machine, then the dependencies in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The requirements list records imported packages, not a verified GPU environment
lock. Your Transformers installation must expose `Qwen3_5ForConditionalGeneration`
and Diffusers must expose `Flux2Pipeline`. Quantized Qwen uses bitsandbytes.
Base-model weights are not included. Most loaders use local files only: download
the models before running and set `HF_HOME` if you use a custom cache.

**Users must obtain base models, third-party baseline implementations and their
weights, and evaluation datasets/evaluators from the corresponding official
repositories. These external resources are not redistributed in this release.**
Download the resources required for the experiment you want to reproduce and
follow each repository's installation and license instructions.

| Resource | Official repository / model source |
| --- | --- |
| Qwen analysis and routing backbone | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| FLUX.2 generation/editing backbone | [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) |
| Optional BAGEL backbone | [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) |
| Optional SD3.5 backbone | [Stable Diffusion 3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo) |
| External Uni-CoT baseline and BAGEL runtime | [UniCoT](https://github.com/Fr0zenCrane/UniCoT), including its linked baseline checkpoints |
| GenEval prompts, metadata, evaluator and evaluator weights | [GenEval](https://github.com/djghosh13/geneval) |
| T2I-CompBench data and evaluation tools | [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) |
| WISE data and evaluation tools | [WISE](https://github.com/PKU-YuanGroup/WISE) |

Other external baselines should likewise be obtained from their authors'
repositories. In-repository fixed-level, semantic-routing and traditional-router
baselines are included, but still require their respective external model assets.
Pin the benchmark release and evaluator protocol used by your experiment;
the WISE wrapper's expected interface is described in
[EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md).

## Bundled route-head checkpoint

`data/router_heads/route_head.pt` is included (2,186,077 bytes). It is a learned
three-way L1/L2/L3 router, not a base image-generation or Qwen model.
It targets **`inference/self_route_version/main.py --semantic-router head`
with FLUX.2 as the initial and repair executor**. This mode uses the learned
MLP, not Qwen's generative semantic-routing baseline.

| Required backbone | Immutable revision recorded in the checkpoint |
| --- | --- |
| Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| FLUX.2-dev | `d72c4754e67efcaca88832fc18ea694547a28bd2` |

After obtaining those snapshots, use:

```bash
python inference/self_route_version/main.py \
  --semantic-router head --initial-generator flux \
  --qwen-route-head data/router_heads/route_head.pt \
  --data-path data/prompts.txt --output-dir results/learned_router \
  --qwen-model /models/qwen/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --qwen-model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --flux-model /models/flux2/snapshots/d72c4754e67efcaca88832fc18ea694547a28bd2 \
  --flux-model-revision d72c4754e67efcaca88832fc18ea694547a28bd2
```

The bundled checkpoint passes CPU loading, a three-logit forward pass, the
current Qwen distribution-signature check, and the encoder-source fingerprint
check for this entry point. Full GPU inference is not yet verified.
See its [checkpoint record](data/router_heads/README.md) for the SHA-256.

The bundled file is **not compatible with `inference/main_method/main.py`**:
its action-policy contract differs, and that loader correctly rejects it.
To use the main-method entry point below, obtain a matching checkpoint or train
one with the [collection and training workflow](inference/router_data_collect/README.md).
The main-method default path is now `data/router_heads/route_head.pt`, matching
the release layout. This path correction does not change the loader's contract
checks: the currently bundled file is still rejected by that entry point.
The included head also does not establish compatibility with BAGEL/SD3.5 runs.

Qwen, FLUX.2, and the initial/L3 model must match the immutable snapshot
revisions and preprocessing policies stored in the checkpoint. Point model
arguments to `snapshots/<commit>` directories and pass the matching revision
arguments. Do not bypass these checks to force an incompatible checkpoint to load.

## Inference

The following main-method example requires its own compatible checkpoint.
Run commands from this repository root. Replace all model paths and revisions
with your local assets. The defaults allocate four workers across GPUs 0–3;
adjust the GPU/resource options shown by `--help` to match your machine.

```bash
python inference/main_method/main.py --help
python inference/main_method/main.py \
  --data-path data/prompts.txt --output-dir results/example \
  --qwen-model /models/qwen/snapshots/COMMIT \
  --qwen-model-revision COMMIT \
  --flux-model /models/flux2/snapshots/COMMIT \
  --flux-model-revision COMMIT \
  --qwen-route-head /path/to/main-method/route_head.pt
```

`COMMIT` denotes each model's own immutable hexadecimal commit, not a shared
value. `--max-global-rounds` controls generated repair actions (default 2).
Use `--resume` with the same output directory to continue an interrupted run.
For fixed-level inference, use `--fixed-repair-level 1`, `2`, or `3`.

The semantic Qwen router does not require a learned route-head checkpoint:

```bash
python inference/self_route_version/main.py --help
python inference/self_route_version/main.py \
  --data-path data/prompts.txt --output-dir results/self_route \
  --qwen-model /models/qwen/snapshots/COMMIT \
  --qwen-model-revision COMMIT \
  --flux-model /models/flux2/snapshots/COMMIT \
  --flux-model-revision COMMIT
```

See the [main method](inference/main_method/README.md),
[semantic routing](inference/self_route_version/README.md), and
[controlled ablations](inference/router_ablation/README.md) for their distinct
contracts. Each sample records `initial.png`, `status.json`, `timeline.jsonl`,
round artifacts, and `final.png`; run-level files include `run_config.json`
and `summary.jsonl`.

## Evaluation

```bash
python scripts/prepare_geneval_eval_input.py --help
python scripts/prepare_t2i_compbench_outputs.py --help
python scripts/eval_wise_transformers_direct.py --help
python -m pip install -r requirements-evaluation.txt
python -m evaluation.eval_image_dir_rewards --results-dir results/example --final-only
```

Benchmark adapters prepare inputs or drive an external evaluator; they do not
bundle benchmark datasets or official evaluation implementations. See
[external dependencies](EXTERNAL_DEPENDENCIES.md) for the required assets.

## Compatibility and release scope

The public method name is **MLIE-CoT** throughout documentation, code comments,
and CLI descriptions. Historical `vcot_*` schema values, `V10`/`V11` constants,
`vcot_branch.py` imports, `VCOT_*` environment variables, `--vcot-python`, and
the default `results/vcot` output directory remain compatibility identifiers.
They are not additional method names. Model-visible prompts remain task-based;
renaming the project does not change prompt bytes or checkpoint fingerprints.

Historical experiment launchers, server monitors, one-off repair/migration
scripts, fixed development-set artifacts, and generated reports are excluded.
Serialized schema identifiers, checkpoint format identifiers, model names,
immutable model revisions, and prompt-contract tags retain their existing
values because they participate in validation and hashing. These are data
compatibility contracts, not alternative experimental implementations.

## License

The [MLIE-CoT Non-Commercial Academic Research License, Version 1.0](LICENSE)
permits reproduction, modification, and sharing for non-commercial academic
research, including publishing results. Commercial use, including internal
product development or evaluation, requires separate written permission from
the relevant rights holders. The license covers the project's own materials,
including the bundled route head. [Third-party assets](EXTERNAL_DEPENDENCIES.md)
remain subject to their own terms.
