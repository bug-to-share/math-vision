"""Add the PAPO actor to the existing Geo3K DAPO trainer and reward pipeline."""

import hydra
import ray

from recipe.papo.main_papo import PAPOActorTaskRunnerMixin
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device

from .main_dapo import DAPOTaskRunner


class Geo3KPAPOTaskRunner(PAPOActorTaskRunnerMixin, DAPOTaskRunner):
    """Keep Geo3K sampling metrics, accuracy filtering and checkpoint semantics."""


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    if "papo" not in config.actor_rollout_ref.actor.fsdp_config:
        raise ValueError("Select +papo=default or use recipe/papo/run_qwen3_vl_4b_geo3k_papo_4gpu.sh")
    if config.distillation.enabled:
        raise ValueError("This PAPO-D experiment requires distillation.enabled=False")
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(Geo3KPAPOTaskRunner))


if __name__ == "__main__":
    main()
