"""PAPO terms added to the actual PPO/DAPO loss returned for backward."""

import torch

from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.metric import Metric
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding


def papo_loss(config, papo, model_output, data, dp_group=None):
    loss, metrics = ppo_loss(config=config, model_output=model_output, data=data, dp_group=dp_group)
    if not papo.active:
        return loss, metrics

    mask = data.select("response_mask").to_padded_tensor()["response_mask"].bool()
    clean = no_padding_2_padding(model_output["log_probs"], data).float()
    # Sanitize padding before exp, so masked-out NaNs/infinities cannot leak into gradients.
    clean = torch.where(mask, clean, 0.0)
    terms = {"clean_nll_proxy": -clean}
    if papo.needs_masked_forward:
        if papo.gradient_mode == "cached_old":
            if "papo_masked_old_log_probs" not in data:
                raise ValueError("cached_old requires pre-update papo_masked_old_log_probs")
            cached = data["papo_masked_old_log_probs"]
            if cached.requires_grad:
                raise ValueError("Cached masked log probabilities must not require gradients")
            masked = no_padding_2_padding(cached, data).float().detach()
        else:
            masked = no_padding_2_padding(model_output["papo_masked_log_probs"], data).float()
        if papo.gradient_mode == "stop_gradient":
            masked = masked.detach()
        masked = torch.where(mask, masked, 0.0)
        delta = (masked - clean).clamp(-20, 20)
        terms["perception_k3"] = (delta.exp() - delta - 1).clamp(-10, 10)
        terms["masked_nll_proxy"] = -masked

    reduced = {
        name: agg_loss(values, mask, config.loss_agg_mode, **config.global_batch_info) for name, values in terms.items()
    }
    auxiliary = papo.clean_entropy_coef * reduced["clean_nll_proxy"]
    if papo.needs_masked_forward:
        auxiliary = (
            auxiliary
            - papo.perception_coef * reduced["perception_k3"]
            + papo.masked_entropy_coef * reduced["masked_nll_proxy"]
        )
    total = loss + auxiliary
    aggregation = metrics["actor/pg_loss"].aggregation
    for name, value in {**reduced, "aux_loss": auxiliary, "total_loss": total}.items():
        metrics[f"papo/{name}"] = Metric(value=value, aggregation=aggregation)
    return total, metrics
