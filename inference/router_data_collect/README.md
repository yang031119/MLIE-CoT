# MLIE-CoT router data collection and training

Run the commands from the repository root. Each training example links an initial image and its diagnosis to the observed outcomes of all three repair levels.

## 1. Pin the models and set the rollout count

Create a configuration, then set each `model_revision` to the exact snapshot commit.

```bash
python -m inference.router_data_collect.cli init-config \
  --output work/router_flux2/config.json
```

To reproduce the submitted paper, change `rollouts_per_action` from the example default of `3` to `5` (`K = 5`). Each repairable state needs `3 × K` verifiable outcomes: five for each of L1, L2, and L3. Split training, validation, and test data by target-prompt family.

FLUX.2, Qwen, and the selected L3 base model require a 40–64 digit hexadecimal commit. Branch names such as `main`, `latest`, or `dev` and descriptive labels are rejected. Local model paths must point directly to the corresponding `snapshots/<commit>` directory.

## 2. Generate initial images

Generate verifiable FLUX.2 initial images from the prompt manifest. Prompt records can use the existing `target_prompt`, `initial_generation_prompt`, `index`, `tag`, and `corruption` fields.

```bash
python -m inference.router_data_collect.cli plan-initial \
  --config work/router_flux2/config.json \
  --prompts data/router_prompts.example.jsonl \
  --bootstrap-root work/router_flux2/bootstrap

# Run on the GPU
python -m inference.router_data_collect.execute_initial_flux2 \
  --bootstrap-root work/router_flux2/bootstrap \
  --device cuda:0
```

## 3. Diagnose each image and plan its L1 repair

Create `work/router_flux2/reflector.json`. Replace the model path and both occurrences of `COMMIT` with the path and revision of your Qwen snapshot.

```json
{
  "model": "/models/qwen/snapshots/COMMIT",
  "model_revision": "COMMIT",
  "dtype": "bfloat16",
  "quantization": "8bit",
  "allow_cpu_offload": false,
  "max_image_side": 1536,
  "max_new_tokens": 1536,
  "seed": 20260715
}
```

Lock the diagnosis to the initial image hash. This stage records facts about the image; the router and post-route planner make the routing and edit decisions later.

The example matches the online analyzer and L1 planner: `bfloat16`, 8-bit quantization, a maximum image side of 1536, and a budget of 1536 new tokens. `plan-reflections` writes the `qwen_distribution` signature to `reflector.lock.json`. It covers the model revision, dtype, quantization, dimensions, token budget, greedy decoding, diagnosis schema, both prompt versions and their bytes, and the model-visible diagnosis fields. It also distinguishes the analyzer's raw RGB input from the L1 planner's ruler overlay and disables image downsizing on out-of-memory retries.

The signature follows each job through its manifest, Qwen result, and final state or plan. `build-states` and `collect-l1-actions` reject mismatched settings or schemas. For collection, `model` must point to `snapshots/<commit>` matching `model_revision`. Online `--qwen-model` must match the revision frozen into the route-head checkpoint. Use `snapshots/main` only for preliminary debugging without a learned head. If you change the dtype, quantization, or image size, collect new data and retrain the head under the new signature.

```bash
python -m inference.router_data_collect.cli plan-reflections \
  --bootstrap-root work/router_flux2/bootstrap \
  --reflector-config work/router_flux2/reflector.json

# Run on the GPU
python -m inference.router_data_collect.execute_reflections_qwen \
  --bootstrap-root work/router_flux2/bootstrap \
  --batch-size 8

python -m inference.router_data_collect.cli build-states \
  --bootstrap-root work/router_flux2/bootstrap \
  --output work/router_flux2/input_states.jsonl \
  --project-root "$PWD"
```

`build-states` records fully satisfied (L0) states separately from the three-class MLP inputs. For states that need repair, run the independent post-route L1 planning stage:

```bash
python -m inference.router_data_collect.cli plan-l1-actions \
  --states work/router_flux2/input_states.jsonl \
  --planner-config work/router_flux2/reflector.json \
  --plan-root work/router_flux2/l1_plans \
  --project-root "$PWD"

# Run on the GPU
python -m inference.router_data_collect.execute_l1_plans_qwen \
  --plan-root work/router_flux2/l1_plans \
  --batch-size 8

python -m inference.router_data_collect.cli collect-l1-actions \
  --plan-root work/router_flux2/l1_plans \
  --output work/router_flux2/l1_action_plans.jsonl
```

### Sampling and class balance

Before GPU collection, stratify by the observed problem: local attributes or colors, counts or missing/extra objects, spatial relations or layout, global scene or style, and severe semantic mismatches. Use `error_family` for sampling and auditing only. If reusing a legacy success table, keep only prompt and error-type metadata; regenerate images, diagnoses, and outcomes for all three levels.

Derive labels after observing the outcomes. Preserve the natural validation and test distributions. The aggregator computes the three `class_factors` from training soft-label mass alone and records effective mass before and after weighting under `training_balance` in `derived/summary.json`. If L2 or L3 rarely wins, collect more examples of the relevant error types rather than duplicating samples or changing the test mix.

## 4. Freeze the counterfactual action plan

Run a CPU dry run, then freeze `3 × K` actions per state. You can repeat planning with identical inputs. If any contract bytes change, use a new dataset directory; the planner refuses to overwrite the existing plan.

```bash
CUDA_VISIBLE_DEVICES='' python -m inference.router_data_collect.cli plan \
  --config work/router_flux2/config.json \
  --states work/router_flux2/input_states.jsonl \
  --l1-plans work/router_flux2/l1_action_plans.jsonl \
  --output-dir work/router_flux2/dataset \
  --project-root "$PWD" \
  --dry-run

python -m inference.router_data_collect.cli plan \
  --config work/router_flux2/config.json \
  --states work/router_flux2/input_states.jsonl \
  --l1-plans work/router_flux2/l1_action_plans.jsonl \
  --output-dir work/router_flux2/dataset \
  --project-root "$PWD"
```

## 5. Execute counterfactual repairs on the GPU

```bash
python -m inference.router_data_collect.execute_flux2 \
  --dataset-root work/router_flux2/dataset \
  --device cuda:0 \
  --batch-size 2

# Run L3 for BAGEL or SD3.5; this is a no-op for a FLUX initial model.
python -m inference.router_data_collect.execute_l3_initial_base \
  --dataset-root work/router_flux2/dataset \
  --gpu-index 0
```

Verified successful jobs are skipped. Failed records are retried only with `--retry-errors`. FLUX.2 initial generation and text-only L3 generation support batching; the BAGEL and SD3.5 L3 executor runs one sample at a time. Qwen batches diagnosis and L1 planning. Blind evaluation batches pairs of images, splitting a failed batch down to individual pairs when necessary. L1 and L2 run one sample at a time because the Diffusers reference-image interface does not safely separate reference groups across prompts.

## 6. Evaluate blindly with a pinned Qwen revision

```bash
python -m inference.router_data_collect.evaluate_qwen \
  --dataset-root work/router_flux2/dataset \
  --model /absolute/path/to/qwen/snapshots/0123456789abcdef0123456789abcdef01234567 \
  --model-revision 0123456789abcdef0123456789abcdef01234567
```

The evaluation prompt randomizes image order and hides the action level, FLUX identity, seed, and rollout index. The first evaluation freezes `evaluator.lock.json`; later runs reject incompatible checkpoints, quantization, resize settings, or token limits. `success` is a fixed-threshold diagnostic of the outcome, not a model-selection label. Its default utility weight is zero to avoid counting satisfaction, fixes, and artifacts twice.

## 7. Aggregate and validate the dataset

```bash
python -m inference.router_data_collect.cli aggregate \
  --dataset-root work/router_flux2/dataset

python -m inference.router_data_collect.cli validate \
  --dataset-root work/router_flux2/dataset \
  --require-results --require-evaluations --require-targets
```

## Train the route head

The trainer reads `derived/{train,validation,test}.jsonl` and `derived/summary.json`. Validate the data, extract features with the pinned Qwen model, then train only the MLP.

```bash
python scripts/train_route_head.py validate-data \
  --dataset-root work/router_flux2/dataset --output-dir results/route_head

python scripts/train_route_head.py all \
  --dataset-root work/router_flux2/dataset --output-dir results/route_head \
  --feature-cache results/route_head/features.pt \
  --qwen-model /models/qwen/snapshots/COMMIT --qwen-revision COMMIT \
  --export-checkpoint models/route_head.pt
```

Replace `COMMIT` with the immutable model revision. With an existing feature cache, use the `train` subcommand with `--train-device cpu` without loading Qwen. See `--help` for all options. The example inputs illustrate the format; they are not validation data or evidence for the paper.

## Sharding and resuming collection

Use `cli partition-shards`, `validate-shards`, `merge-shards`, and `merge-bundles` to split or combine data while keeping prompt families together. See `orchestrate_shard --help` for stage scheduling and use `--print-only` to inspect commands. The automated pipeline for fresh initial images supports FLUX.2. For BAGEL or SD3.5, start with states that have exact initial-image records and run L3 with `execute_l3_initial_base`.

`cli export-online-states` exports visited states from completed online trajectories; use `--verify-only` to check them. `existing_initial_assets.py` supports external initial images with manifests and hashes. Legacy experiment-directory migration tools are not included. `advanced_prompts.py` and `balanced_prompt_sets.py` construct prompts deterministically from structured metadata; the fixed development set is not distributed.

Models, preprocessing, prompts, executors, and data hashes are validated together. Preserve internal format identifiers for compatibility. Changing those contracts requires fresh collection and training.
