# MLIE-CoT bundled route head

| Item | Value |
| --- | --- |
| File | `route_head.pt` |
| Size | 2,186,077 bytes |
| SHA-256 | `f37377348584c3f6fe5f82307a350cb5a83611e1302bcb660ad3f7e5c94e14b7` |
| Architecture | MLP, 4,107 input features, 128 hidden units, 3 outputs (L1/L2/L3) |
| Supported deployment | `inference/self_route_version/main.py --semantic-router head --initial-generator flux` |
| Qwen model | `Qwen/Qwen3.5-9B` |
| Qwen revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Image model | `black-forest-labs/FLUX.2-dev` |
| Image-model revision | `d72c4754e67efcaca88832fc18ea694547a28bd2` |

This file contains `state_dict` and configuration metadata. It does not contain
Qwen, image-generation backbones, baseline weights, or evaluation data. Obtain
those resources from the official repositories linked in the [root README](../../README.md).

The checkpoint is deployment-adapted to the self-route package's action
contract. The `main_method` loader rejects it; do not edit the metadata to
bypass compatibility validation. Historical training-machine paths in metadata
are provenance only; use your local immutable snapshots in CLI arguments.

Verified on CPU: loader and state-dict compatibility, `(1, 3)` forward output,
Qwen distribution signature, and encoder-source fingerprint. Runtime checks
against downloaded backbone/processor assets and full GPU inference still
require the actual model environment.
