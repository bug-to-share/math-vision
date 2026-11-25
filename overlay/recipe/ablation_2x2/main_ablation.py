"""Run a matched GRPO arm with the PAPO actor and optional MoCA token routing."""

import hydra
import ray
from recipe.moca.main_moca import MoCATaskRunner, validate_moca_config
from recipe.papo.main_papo import PAPOActorTaskRunnerMixin

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.device import auto_set_device


class AblationTaskRunner(PAPOActorTaskRunnerMixin, MoCATaskRunner):
    def _trainer_class(self, config):
        if config.ablation.route_tokens:
            return super()._trainer_class(config)
        return RayPPOTrainer


@hydra.main(config_path="config", config_name="ablation_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    validate_moca_config(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(AblationTaskRunner))


if __name__ == "__main__":
    main()
