# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Local MoCA entry point; standard worker classes and the standard PPO loss."""

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.device import auto_set_device


def validate_moca_config(config):
    if config.algorithm.adv_estimator != "grpo" or not config.algorithm.norm_adv_by_std_in_grpo:
        raise ValueError("This experiment uses standard GRPO mean/std normalization")
    if config.trainer.use_v1 or config.actor_rollout_ref.rollout.multi_turn.enable:
        raise ValueError("Initial MoCA integration supports the v0 single-turn trainer")
    if config.distillation.enabled or config.algorithm.use_kl_in_reward:
        raise ValueError("Initial MoCA experiment does not combine distillation or reward KL")
    if int(config.actor_rollout_ref.rollout.n) < 2:
        raise ValueError("GRPO requires at least two responses per prompt")
    if not 0 < float(config.moca.pv_success_threshold) <= 1:
        raise ValueError("pv_success_threshold must be in (0, 1]")


class MoCATaskRunner(BaseTaskRunner):
    """Adapt the standard v0 task setup, substituting only the trainer class."""

    def _trainer_class(self, config):
        from recipe.moca.trainer import MoCATrainer

        return MoCATrainer

    def run(self, config):
        from verl.utils.dataset.rl_dataset import collate_fn

        OmegaConf.resolve(config)
        validate_moca_config(config)
        actor_cls, group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_cls)
        validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
        model_config = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer, processor = model_config.tokenizer, model_config.processor
        train = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        trainer = self._trainer_class(config)(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=self.init_resource_pool_mgr(config),
            ray_worker_group_cls=group_cls,
            train_dataset=train,
            val_dataset=val,
            collate_fn=collate_fn,
            train_sampler=create_rl_sampler(config.data, train),
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="moca_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    validate_moca_config(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(MoCATaskRunner))


if __name__ == "__main__":
    main()
