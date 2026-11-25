"""Train-only paired teacher forcing, scoped to this recipe's engine instance."""

from contextlib import nullcontext
from functools import partial

import torch

from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.utils.losses import ppo_loss

from .losses import papo_loss
from .masking import mask_image_inputs


class PAPOFSDPEngine(FSDPEngineWithLMHead):
    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        if engine_config.strategy != "fsdp2" or engine_config.ulysses_sequence_parallel_size != 1:
            raise ValueError("PAPO currently requires FSDP2 and ulysses_sequence_parallel_size=1")
        if engine_config.use_fused_kernels or model_config.use_fused_kernels:
            raise ValueError("PAPO currently requires use_fused_kernels=False")
        if model_config.hf_config.model_type != "qwen3_vl":
            raise ValueError("PAPO currently supports Qwen3-VL image models only")
        if model_config.processor is None:
            raise ValueError("PAPO requires the model's image processor")
        self.papo = engine_config.papo
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _mask_step(self):
        # Adam's step is saved/restored in the standard engine optimizer checkpoint.
        for state in self.optimizer.state.values():
            if "step" in state:
                return int(state["step"])
        return 0

    def _mask_inputs(self, model_inputs):
        return mask_image_inputs(
            model_inputs,
            self.model_config.processor.image_processor,
            self.papo.mask_probability,
            self.papo.seed,
            self._mask_step(),
            mask_strategy=self.papo.mask_strategy,
            attention_temperature=self.papo.attention_temperature,
            attention_mix_alpha=self.papo.attention_mix_alpha,
        )

    def prepare_model_inputs(self, micro_batch):
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        if tu.get(micro_batch, "papo_cache_forward", default=False):
            model_inputs, mask_rate = self._mask_inputs(model_inputs)
            output_args["papo_cache_mask_fraction"] = mask_rate
        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
        mask_rate = output_args.pop("papo_cache_mask_fraction", None)
        result = super().prepare_model_outputs(output, output_args, micro_batch, logits_processor_func)
        if mask_rate is not None:
            # Removed by forward_step before generic nested-output concatenation.
            result["papo_cache_mask_fraction"] = mask_rate
        return result

    def forward_step(self, micro_batch, loss_function, forward_only):
        if tu.get(micro_batch, "papo_cache_forward", default=False):
            if not forward_only or loss_function is not None or self.papo.gradient_mode != "cached_old":
                raise ValueError("Masked cache generation requires a loss-free cached_old inference pass")
            loss, output = super().forward_step(micro_batch, None, True)
            output["metrics"]["papo/cache_mask_fraction"] = output["model_output"].pop("papo_cache_mask_fraction")
            return loss, output
        if forward_only and isinstance(loss_function, partial) and loss_function.func is papo_loss:
            # infer_batch(compute_loss=True) still evaluates the clean policy loss.
            loss_function = partial(ppo_loss, config=loss_function.keywords["config"])
        if forward_only or not self.papo.needs_masked_forward:
            return super().forward_step(micro_batch, loss_function, forward_only)
        if self.papo.gradient_mode == "cached_old":
            version = tu.get(micro_batch, "papo_cache_version", default=None)
            if version is None or version != getattr(self, "_papo_active_cache_version", None):
                raise ValueError("Missing or stale masked cache: generate it before this actor update")
            loss, output = super().forward_step(micro_batch, loss_function, forward_only)
            output["metrics"]["papo/cache_age_optimizer_steps"] = self._mask_step() - tu.get(
                micro_batch, "papo_cache_optimizer_step"
            )
            return loss, output
        if loss_function is None:
            raise ValueError("PAPO training requires a loss function")

        def paired_loss(model_output, data, dp_group):
            # Executed inside the base forward's autocast context. Reuse exactly
            # the same tokens, positions, grids and temperature for the second view.
            model_inputs, output_args = self.prepare_model_inputs(data)
            model_inputs, mask_rate = self._mask_inputs(model_inputs)
            grad_context = torch.no_grad() if self.papo.gradient_mode == "stop_gradient" else nullcontext()
            with grad_context:
                raw_output = self.module(**model_inputs, use_cache=False)
                masked_output = self.prepare_model_outputs(
                    output=raw_output,
                    output_args=output_args,
                    micro_batch=data,
                    logits_processor_func=loss_function,
                )
            paired_output = {**model_output, "papo_masked_log_probs": masked_output["log_probs"]}
            loss, metrics = loss_function(model_output=paired_output, data=data, dp_group=dp_group)
            metrics["papo/mask_fraction"] = mask_rate
            return loss, metrics

        # Base forward detaches reported outputs, while the returned joint loss
        # remains live for the engine's one backward call per microbatch.
        return super().forward_step(micro_batch, paired_loss, forward_only)
