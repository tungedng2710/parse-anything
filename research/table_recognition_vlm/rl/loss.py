"""Reference implementation of the clipped DR-GRPO policy loss used by TRL."""


def grpo_clipped_loss(
    log_probs,
    old_log_probs,
    advantages,
    completion_mask,
    *,
    max_completion_length,
    epsilon=0.2,
    ref_log_probs=None,
    beta=0.0,
):
    """Return a length-unbiased clipped policy loss with an optional KL penalty.

    Training uses TRL's optimized equivalent via ``GRPOTrainer``. This function
    keeps the objective explicit and is useful for unit tests and experiments.
    """
    import torch

    if log_probs.shape != old_log_probs.shape or log_probs.shape != completion_mask.shape:
        raise ValueError("log_probs, old_log_probs, and completion_mask must have equal shapes")
    if advantages.ndim != 1 or advantages.shape[0] != log_probs.shape[0]:
        raise ValueError("advantages must contain one value per completion")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if beta < 0:
        raise ValueError("beta must be non-negative")

    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    advantages = advantages.to(log_probs.dtype).unsqueeze(1)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    per_token_loss = -torch.minimum(unclipped, clipped)

    if ref_log_probs is not None:
        if ref_log_probs.shape != log_probs.shape:
            raise ValueError("ref_log_probs must have the same shape as log_probs")
        log_ref_ratio = ref_log_probs - log_probs
        per_token_kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1.0
        per_token_loss = per_token_loss + beta * per_token_kl

    if max_completion_length <= 0:
        raise ValueError("max_completion_length must be positive")
    mask = completion_mask.to(per_token_loss.dtype)
    return (per_token_loss * mask).sum() / (
        log_probs.shape[0] * max_completion_length
    )
