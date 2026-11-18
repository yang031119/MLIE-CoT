# MLIE-CoT self-routing baseline prompts

See the [main-method prompt reference](../main_method/SYSTEM_PROMPTS.md) for
shared diagnosis, L1 planning, and blind candidate review prompts.

For the prompts this baseline actually uses, see `modules/contracts.py`,
`modules/config.py`, and `modules/qwen_stages.py` in this directory. The
baseline also has prompts for semantic routing and planning with feedback
from previous steps; its full prompt contract differs from the main method's.
