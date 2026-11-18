"""Constants and prompt templates for the MLIE-CoT pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from .contracts import (
	L1_PLAN_SYSTEM_PROMPT_V10,
	L1_PLAN_USER_TEMPLATE_V10,
	QWEN_ANALYZER_L1_INPUT_MAX_SIDE_V10,
	TRIAGE_SYSTEM_PROMPT_V10,
	TRIAGE_USER_TEMPLATE_V10,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

MODULE_DIR = Path(__file__).resolve().parent


SCRIPT_DIR = MODULE_DIR.parent


PROJECT_ROOT = SCRIPT_DIR.parents[1]


DEFAULT_QWEN_ROUTE_HEAD = PROJECT_ROOT / "data" / "router_heads" / "route_head.pt"


LOCAL_BENCHMARK_SCRIPT = MODULE_DIR / "local_edit_worker.py"


DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "data" / "prompts.txt"


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "self_route"


DEFAULT_FLUX_MODEL = "black-forest-labs/FLUX.2-dev"


DEFAULT_FLUX_MODEL_REVISION = "d72c4754e67efcaca88832fc18ea694547a28bd2"


DEFAULT_SD35_TURBO_MODEL = "stabilityai/stable-diffusion-3.5-large-turbo"


DEFAULT_BAGEL_MODEL_PATH = str(PROJECT_ROOT / "models" / "BAGEL-7B-MoT")


DEFAULT_BAGEL_FINETUNED_PATH = DEFAULT_BAGEL_MODEL_PATH


DEFAULT_UNICOT_ROOT = str(PROJECT_ROOT / "third_party" / "UniCoT")


DEFAULT_EDITOR_GPU_INDICES = "0,1,2,3"


DEFAULT_EDITOR_PARALLEL_WORKERS = 4


DEFAULT_RETRY_MIN_SIDE = 300


DEFAULT_RETRY_SCALE = 0.8


QWEN_INPUT_MAX_SIDE = QWEN_ANALYZER_L1_INPUT_MAX_SIDE_V10


QWEN_REVIEW_INPUT_MAX_SIDE = 1024


QWEN_MULTI_IMAGE_INPUT_MAX_SIDE = 1024


QWEN_MULTI_IMAGE_SIDE_SEQUENCE = [1024, 768, 512, 384, 256]


CATEGORY_RANK = {
	"L0": 0,
	"L1": 1,
	"L2": 2,
	"L3": 3,
}


LOCAL_MODEL_ORDER = ["flux2", "bagel"]


LOCAL_REPAIR_MODEL_ORDER = ["flux2"]


OBJECT_REPAIR_MODEL_ORDER = ["flux2"]


DIFFICULTY_RANK = {
	"attribute": 1,
	"remove_count": 2,
	"add_count": 3,
	"add_object": 4,
	"background": 5,
	"other": 6,
}


MIN_QUAD_SPAN = 8


PROMPT_REASONING_SYSTEM_PROMPT = """You are the scene-planning controller for initial image generation.

Rewrite the user's target prompt into one concrete image-generation prompt. Keep the target semantics exactly: preserve every requested count, color, object/person type, spatial relation, attribute, action, state, and setting. Any added descriptive detail must be compatible with those requested facts.

Use any necessary implicit world knowledge internally. This response directly guides the image generator before the first image is made.

The complete response is exactly one valid JSON object with one key:
{
  "generation_prompt": "one polished English image-generation prompt, concrete and detailed enough for a generator"
}
"""


PROMPT_REASONING_USER_TEMPLATE = """Original target prompt:
{prompt}

Rewrite this into a concrete image-generation prompt. Preserve the exact requested counts, colors, objects/people, relations, actions, states, and scene setting. If the text contains awkward plurals or ambiguous words, interpret them naturally for an image while keeping the target facts.

The complete response is JSON with the single key generation_prompt.
"""


# Public aliases are sourced from the lightweight shared contract so online
# inference and offline collection cannot silently drift.
TRIAGE_SYSTEM_PROMPT = TRIAGE_SYSTEM_PROMPT_V10
TRIAGE_USER_TEMPLATE = TRIAGE_USER_TEMPLATE_V10
L1_PLAN_SYSTEM_PROMPT = L1_PLAN_SYSTEM_PROMPT_V10
L1_PLAN_USER_TEMPLATE = L1_PLAN_USER_TEMPLATE_V10


L2_PLAN_SYSTEM_PROMPT = """You are the reflective execution planner for the selected image executor.

Convert the current factual diagnosis into one direct, atomic instruction for the requested repair level. If prior attempts are supplied, explicitly learn from what changed and what remained wrong.

Rules:
- Start with a concrete edit verb such as Change, Add, Remove, Move, Replace, or Rebuild.
- Name the exact target entity, its observed state, and its required final state.
- For color or attribute errors, state the explicit value transition, for example green to brown.
- Never use an abstract goal such as "improve", "blend", or "better match" as a substitute for the observable edit.
- If a prior instruction produced insufficient progress, do not repeat it. Increase specificity by naming the full affected scope and explicitly say not to preserve the incorrect attribute.
- Preserve only content that is already correct. Never issue a generic preservation rule that could preserve the diagnosed error.
- For L2, return one clean full-image edit instruction.
- For L3, return a complete prompt for a new image generated without conditioning on the current image.
- Return an executable instruction, not analysis prose.

Return exactly one JSON object:
{
  "reason": "brief explanation of how this instruction improves on the diagnosis and prior attempts",
  "progress_assessment": "first_attempt | partial_progress | no_progress | regression",
  "instruction": "the executable image edit or generation instruction",
  "do_not_preserve": ["incorrect attributes that must be overwritten"],
  "preserve": ["specific already-correct content"]
}
"""


L2_PLAN_USER_TEMPLATE = """Requested repair level: {repair_level}

Target prompt:
{prompt}

Canonical diagnosis JSON:
{diagnosis_json}

Prior execution memory JSON:
{repair_memory_json}

Create the next direct image-executor instruction for the requested repair level.

/no_think
"""


GLOBAL_COMPARE_USER_TEMPLATE = """Target prompt:
{prompt}

Main issue from previous analysis: {issue}
Current intended correction: {fix}
Executed image-executor instruction: {executed_instruction}

Available candidate labels: {candidate_labels}

Each candidate image includes x/y tick marks. For left/right relations, compare object centers by x coordinate; "right of" means the subject center has a larger x coordinate than the anchor center. For above/below relations, compare y coordinate; "below" means the subject center has a larger y coordinate.

Select the best candidate from the available labels according to the schema.

/no_think
"""


LOCAL_REVIEW_USER_TEMPLATE = """Target prompt:
{prompt}

Current single-call joint local-edit plan ({edit_total} localized edits):
{edit_lines}

Select the winner according to the schema.

/no_think
"""


BOX_REVIEW_SYSTEM_PROMPT = """You are reviewing proposed joint L1 bounding boxes.

You receive the full original image with a coordinate ruler and every proposed box drawn directly on the full image. Each box uses a dynamically selected fluorescent red, yellow, blue, or green outline that contrasts with the pixels underneath it. The box label and the edit list identify which outline belongs to which edit. Use the original image coordinate system for every returned coordinate; the outer ruler canvas is a visual reference.

Decide whether each proposed box fully covers the local region needed for its planned edit while avoiding unrelated content as much as practical. You may keep a box unchanged or adjust its coordinates. Preserve the edit order, edit ids, target objects, and edit prompts exactly.

The complete response is exactly one valid JSON object:
{
  "satisfied": true | false,
  "reason": "brief explanation",
  "edits": [
    {
      "id": "same id as the input edit",
      "satisfied": true | false,
      "reason": "brief explanation for this box",
      "quad_points": [[x, y], [x, y], [x, y], [x, y]]
    }
  ]
}

Rules:
- Return exactly one edit entry for each proposed edit, in the same order.
- Each quad_points must define an axis-aligned bbox in clockwise order: top-left, top-right, bottom-right, bottom-left.
- The box should fully cover the local region that must change. Expand slightly if the target is small or if context is needed.
- If all boxes are good, set satisfied=true and return the original coordinates.
- The JSON object above is the complete response.
"""


BOX_REVIEW_USER_TEMPLATE = """Target prompt:
{prompt}

Original image size: width={width}, height={height}.
Review round: {review_round} of {max_rounds}.

Proposed micro edits:
{edit_lines}

Inspect the labeled fluorescent boxes on the full image. Confirm whether each box is an accurate edit region, or return adjusted coordinates according to the schema.

/no_think
"""


Point = Tuple[int, int]
