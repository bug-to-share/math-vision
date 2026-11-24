"""Actor specialization without modifying reference workers or global registries."""

from functools import partial
from time import monotonic
from uuid import uuid4

import numpy as np
import torch

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
from verl.workers.utils.losses import ppo_loss

from .fsdp_engine import PAPOFSDPEngine
from .losses import papo_loss


class PAPOTrainingWorker(TrainingWorker):
    def _create_engine(self):
        return PAPOFSDPEngine(
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        if not isinstance(loss_fn, partial) or loss_fn.func is not ppo_loss:
            raise ValueError("PAPO requires the standard PPO loss; distillation/custom losses are unsupported")
        self.loss_fn = partial(papo_loss, config=loss_fn.keywords["config"], papo=self.engine_config.papo)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data):
        papo = self.engine_config.papo
        if papo.gradient_mode != "cached_old" or not papo.needs_masked_forward:
            return super().train_mini_batch(data)
        if "papo_masked_old_log_probs" in data or "papo_cache_version" in data:
            raise ValueError("Actor update must receive a fresh batch without an existing masked cache")

        # This runs on the whole DP-local batch before the base worker creates
        # its minibatch iterator. Inference restores dynamic microbatch ordering.
        # Clone the container so inference budgets/flags cannot leak into training.
        cache_data = data.clone(recurse=False)
        tu.assign_non_tensor(cache_data, papo_cache_forward=True, compute_loss=False, calculate_entropy=False)
        optimizer_step = self.engine._mask_step()
        started = monotonic()
        with torch.no_grad():
            cached_output = self.infer_batch(cache_data)
        cache_seconds = monotonic() - started
        if self.engine._mask_step() != optimizer_step:
            raise RuntimeError("Policy changed while computing the masked cache")
        cached = cached_output["log_probs"].detach()
        if not cached.is_nested or not torch.equal(
            cached.offsets().diff().cpu(), data["input_ids"].offsets().diff().cpu()
        ):
            raise ValueError("Masked cache does not align with the packed training sequences")
        if not torch.isfinite(cached.values()).all():
            raise ValueError("Masked cache contains non-finite log probabilities")
        data["papo_masked_old_log_probs"] = cached
        version = uuid4().hex
        tu.assign_non_tensor(data, papo_cache_version=version, papo_cache_optimizer_step=optimizer_step)
        self.engine._papo_active_cache_version = version
        cache_metrics = tu.get(cached_output, "metrics")
        try:
            output = super().train_mini_batch(data)
            if output is not None:
                metrics = tu.get(output, "metrics")
                metrics["papo/cache_seconds"] = cache_seconds
                metrics["papo/cache_passes"] = 1
                metrics["papo/cache_optimizer_step"] = optimizer_step
                metrics["papo/cache_mask_fraction"] = float(np.mean(cache_metrics["papo/cache_mask_fraction"]))
            return output
        finally:
            self.engine._papo_active_cache_version = None
            for key in ("papo_masked_old_log_probs", "papo_cache_version", "papo_cache_optimizer_step"):
                data.pop(key)


class PAPOActorRolloutRefWorker(ActorRolloutRefWorker):
    actor_worker_cls = PAPOTrainingWorker
