"""Utility-aware loss for the three-logit router head.

Torch is imported inside the function so the data-building package and CLI stay
CPU/lightweight until training is explicitly started.
"""

from __future__ import annotations

import math
from typing import Any


def utility_aware_router_loss(
	logits: Any,
	soft_targets: Any,
	utility_targets: Any,
	*,
	regret_targets: Any | None = None,
	efficiency_regret_targets: Any | None = None,
	pairwise_mask: Any | None = None,
	safe_pairwise_mask: Any | None = None,
	efficiency_costs: Any | None = None,
	confidence: Any | None = None,
	sample_weight: Any | None = None,
	quality_row_weight: Any | None = None,
	class_weight: Any | None = None,
	utility_temperature: float = 0.5,
	pairwise_temperature: float = 0.5,
	policy_temperature: float = 1.0,
	quality_tie_epsilon: float = 0.0,
	soft_ce_weight: float = 1.0,
	utility_alignment_weight: float = 0.15,
	pairwise_weight: float = 0.20,
	expected_regret_weight: float = 0.0,
	efficiency_weight: float = 0.0,
	uncertain_safe_prior_weight: float = 0.0,
	safe_action_index: int = 1,
) -> tuple[Any, dict[str, Any]]:
	"""Combine value fitting, quality regret, reliable ranking and tie efficiency.

	Shapes are ``[batch, 3]``.  The logits remain deployment-compatible class
	logits; utility alignment centers both sides because adding a constant to all
	three logits does not change the route distribution. ``class_weight`` is the
	three-value inverse train soft-mass vector emitted in the aggregation summary.
	"""

	import torch
	from torch.nn import functional as F

	if logits.ndim != 2 or logits.shape[-1] != 3:
		raise ValueError("logits must have shape [batch,3]")
	if soft_targets.shape != logits.shape or utility_targets.shape != logits.shape:
		raise ValueError("soft_targets and utility_targets must match logits shape")
	if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (
		utility_temperature,
		pairwise_temperature,
		policy_temperature,
	)):
		raise ValueError("temperatures must be positive")
	if not math.isfinite(float(quality_tie_epsilon)) or quality_tie_epsilon < 0:
		raise ValueError("quality_tie_epsilon must be non-negative")
	for name, value in (
		("soft_ce_weight", soft_ce_weight),
		("utility_alignment_weight", utility_alignment_weight),
		("pairwise_weight", pairwise_weight),
		("expected_regret_weight", expected_regret_weight),
		("efficiency_weight", efficiency_weight),
		("uncertain_safe_prior_weight", uncertain_safe_prior_weight),
	):
		if not math.isfinite(float(value)) or float(value) < 0.0:
			raise ValueError(f"{name} must be finite and non-negative")
	if not torch.isfinite(logits).all():
		raise ValueError("logits must be finite")
	if int(safe_action_index) not in (0, 1, 2):
		raise ValueError("safe_action_index must be 0, 1, or 2")

	soft_targets = soft_targets.to(device=logits.device, dtype=logits.dtype)
	utility_targets = utility_targets.to(device=logits.device, dtype=logits.dtype)
	if not torch.isfinite(soft_targets).all() or (soft_targets < 0).any():
		raise ValueError("soft_targets must be finite and non-negative")
	if not torch.isfinite(utility_targets).all():
		raise ValueError("utility_targets must be finite")
	if regret_targets is None:
		regret_targets = utility_targets.max(dim=-1, keepdim=True).values - utility_targets
	else:
		regret_targets = regret_targets.to(device=logits.device, dtype=logits.dtype)
		if regret_targets.shape != logits.shape:
			raise ValueError("regret_targets must match logits shape")
	if not torch.isfinite(regret_targets).all() or (regret_targets < -1e-6).any():
		raise ValueError("regret_targets must be finite and non-negative")
	regret_targets = regret_targets.clamp_min(0.0)
	if not torch.allclose(
		soft_targets.sum(dim=-1),
		torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype),
		atol=1e-4,
	):
		raise ValueError("each soft target row must sum to one")
	class_weights = torch.ones(3, device=logits.device, dtype=logits.dtype)
	if class_weight is not None:
		class_weights = class_weight.to(device=logits.device, dtype=logits.dtype).reshape(-1)
		if class_weights.shape != (3,):
			raise ValueError("class_weight must contain exactly three values")
		if not torch.isfinite(class_weights).all() or (class_weights <= 0).any():
			raise ValueError("class_weight must be finite and positive")
	base_weights = torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
	if confidence is not None:
		base_weights = confidence.to(device=logits.device, dtype=logits.dtype).reshape(-1).clamp(0.0, 1.0)
		if base_weights.shape[0] != logits.shape[0]:
			raise ValueError("confidence must contain one value per row")
	if sample_weight is not None:
		sample_weight = sample_weight.to(device=logits.device, dtype=logits.dtype).reshape(-1)
		if sample_weight.shape[0] != logits.shape[0]:
			raise ValueError("sample_weight must contain one value per row")
		if not torch.isfinite(sample_weight).all() or (sample_weight < 0).any():
			raise ValueError("sample_weight must be finite and non-negative")
		base_weights = base_weights * sample_weight
	if not torch.isfinite(base_weights).all():
		raise ValueError("combined row weights must be finite")
	if quality_row_weight is None:
		quality_weights = base_weights
	else:
		quality_row_weight = quality_row_weight.to(device=logits.device, dtype=logits.dtype).reshape(-1)
		if quality_row_weight.shape[0] != logits.shape[0]:
			raise ValueError("quality_row_weight must contain one value per row")
		if not torch.isfinite(quality_row_weight).all() or (quality_row_weight < 0).any():
			raise ValueError("quality_row_weight must be finite and non-negative")
		quality_weights = base_weights * quality_row_weight
	quality_denominator = quality_weights.sum().clamp_min(torch.finfo(logits.dtype).eps)

	soft_ce_rows = -(
		soft_targets * class_weights.unsqueeze(0) * F.log_softmax(logits, dim=-1)
	).sum(dim=-1)
	soft_ce = (soft_ce_rows * quality_weights).sum() / quality_denominator

	pred_centered = logits - logits.mean(dim=-1, keepdim=True)
	target_centered = utility_targets - utility_targets.mean(dim=-1, keepdim=True)
	target_centered = target_centered / float(utility_temperature)
	alignment_rows = F.smooth_l1_loss(pred_centered, target_centered, reduction="none").mean(dim=-1)
	alignment = (alignment_rows * quality_weights).sum() / quality_denominator

	pair_losses = []
	for left, right in ((0, 1), (0, 2), (1, 2)):
		target_probability = torch.sigmoid(
			(utility_targets[:, left] - utility_targets[:, right]) / float(pairwise_temperature)
		)
		pair_losses.append(
			F.binary_cross_entropy_with_logits(
				logits[:, left] - logits[:, right], target_probability, reduction="none"
			)
		)
	pairwise_matrix = torch.stack(pair_losses, dim=-1)
	if pairwise_mask is None:
		pair_mask = torch.ones_like(pairwise_matrix)
	else:
		pair_mask = pairwise_mask.to(device=logits.device, dtype=logits.dtype)
		if pair_mask.shape != pairwise_matrix.shape:
			raise ValueError("pairwise_mask must have shape [batch,3]")
		if not torch.isfinite(pair_mask).all() or (pair_mask < 0).any():
			raise ValueError("pairwise_mask must be finite and non-negative")
	if safe_pairwise_mask is None:
		safe_pair_mask = pair_mask
	else:
		safe_pair_mask = safe_pairwise_mask.to(device=logits.device, dtype=logits.dtype)
		if safe_pair_mask.shape != pairwise_matrix.shape:
			raise ValueError("safe_pairwise_mask must have shape [batch,3]")
		if not torch.isfinite(safe_pair_mask).all() or (safe_pair_mask < 0).any():
			raise ValueError("safe_pairwise_mask must be finite and non-negative")
	pair_denominator = (pair_mask * quality_weights.unsqueeze(-1)).sum().clamp_min(torch.finfo(logits.dtype).eps)
	pairwise = (pairwise_matrix * pair_mask * quality_weights.unsqueeze(-1)).sum() / pair_denominator

	policy = torch.softmax(logits / float(policy_temperature), dim=-1)
	expected_regret_rows = (policy * regret_targets).sum(dim=-1)
	expected_regret = (expected_regret_rows * quality_weights).sum() / quality_denominator

	if efficiency_regret_targets is None:
		efficiency_regrets = regret_targets
	else:
		efficiency_regrets = efficiency_regret_targets.to(device=logits.device, dtype=logits.dtype)
		if efficiency_regrets.shape != logits.shape:
			raise ValueError("efficiency_regret_targets must match logits shape")
		if not torch.isfinite(efficiency_regrets).all() or (efficiency_regrets < -1e-6).any():
			raise ValueError("efficiency_regret_targets must be finite and non-negative")
		efficiency_regrets = efficiency_regrets.clamp_min(0.0)

	efficiency = logits.sum() * 0.0
	acceptable = efficiency_regrets <= float(quality_tie_epsilon) + 1e-7
	tie_rows = acceptable.sum(dim=-1) > 1
	if efficiency_costs is not None:
		costs = efficiency_costs.to(device=logits.device, dtype=logits.dtype).reshape(-1)
		if costs.shape != (3,) or not torch.isfinite(costs).all():
			raise ValueError("efficiency_costs must contain three finite values")
		masked_costs = costs.unsqueeze(0).expand_as(regret_targets).masked_fill(~acceptable, float("inf"))
		efficiency_targets = masked_costs.argmin(dim=-1)
		efficiency_rows = F.cross_entropy(logits, efficiency_targets, reduction="none")
		efficiency_weights = base_weights * tie_rows.to(dtype=base_weights.dtype)
		efficiency_denominator = efficiency_weights.sum().clamp_min(torch.finfo(logits.dtype).eps)
		efficiency = (efficiency_rows * efficiency_weights).sum() / efficiency_denominator

	# On rows where paired rollouts do not reliably show that an alternative is
	# better than the designated safe action, discourage a speculative escape.
	# This term vanishes automatically once a reliable quality advantage exists.
	safe_prior = logits.sum() * 0.0
	if float(uncertain_safe_prior_weight) > 0.0:
		safe_index = int(safe_action_index)
		reliable_escape = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.bool)
		pair_lookup = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
		for action_index in range(3):
			if action_index == safe_index:
				continue
			left, right = sorted((action_index, safe_index))
			pair_index = pair_lookup[(left, right)]
			reliable_pair = safe_pair_mask[:, pair_index] > 0.0
			better = utility_targets[:, action_index] > (
				utility_targets[:, safe_index] + float(quality_tie_epsilon)
			)
			reliable_escape |= reliable_pair & better
		safe_rows = ~reliable_escape & ~tie_rows
		safe_targets = torch.full(
			(logits.shape[0],),
			safe_index,
			device=logits.device,
			dtype=torch.long,
		)
		safe_rows_loss = F.cross_entropy(logits, safe_targets, reduction="none")
		safe_weights = quality_weights * safe_rows.to(dtype=quality_weights.dtype)
		safe_denominator = safe_weights.sum().clamp_min(torch.finfo(logits.dtype).eps)
		safe_prior = (safe_rows_loss * safe_weights).sum() / safe_denominator

	total = (
		float(soft_ce_weight) * soft_ce
		+ float(utility_alignment_weight) * alignment
		+ float(pairwise_weight) * pairwise
		+ float(expected_regret_weight) * expected_regret
		+ float(efficiency_weight) * efficiency
		+ float(uncertain_safe_prior_weight) * safe_prior
	)
	return total, {
		"loss": total.detach(),
		"soft_ce": soft_ce.detach(),
		"utility_alignment": alignment.detach(),
		"pairwise": pairwise.detach(),
		"expected_regret": expected_regret.detach(),
		"efficiency": efficiency.detach(),
		"safe_prior": safe_prior.detach(),
	}
