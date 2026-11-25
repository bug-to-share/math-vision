"""Apply MoCA routing after standard GRPO normalization, immediately before update."""

import torch
from recipe.moca.advantages import route_advantages
from recipe.moca.evidence import recognition_token_mask

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class MoCATrainer(RayPPOTrainer):
    def _update_actor(self, batch):
        if batch.meta_info.get("moca_routed", False):
            raise ValueError("MoCA advantages have already been routed")
        mask = torch.zeros_like(batch.batch["response_mask"], dtype=torch.bool)
        for i, row in enumerate(batch.batch["responses"]):
            length = int(batch.batch["response_mask"][i].sum())
            selected = recognition_token_mask(self.tokenizer, row[:length].tolist())
            mask[i, :length] = torch.tensor(selected, device=mask.device)
        outcome = torch.as_tensor(
            batch.non_tensor_batch["moca_outcome"].astype(float), device=mask.device, dtype=torch.float32
        )
        perception = torch.as_tensor(
            batch.non_tensor_batch["moca_perception"].astype(float), device=mask.device, dtype=torch.float32
        )
        batch.batch["advantages"], metrics = route_advantages(
            batch.batch["advantages"],
            batch.batch["response_mask"],
            mask,
            outcome,
            perception,
            protect=float(self.config.moca.protect),
            punish=float(self.config.moca.punish),
            route_only_negative_advantages=self.config.moca.get("route_only_negative_advantages", False),
        )
        batch.meta_info["moca_routed"] = True
        result = super()._update_actor(batch)
        result.meta_info["metrics"].update(metrics)
        return result
