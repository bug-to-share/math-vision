"""Run the existing DAPO trainer with an explicitly selected PAPO actor."""

import hydra
import ray
from recipe.dapo.main_dapo import DAPOTaskRunner

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import Role
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import auto_set_device


class PAPOActorTaskRunnerMixin:
    """Select the PAPO actor without replacing the experiment's trainer."""

    def add_actor_rollout_worker(self, config):
        actor_cls, group_cls = super().add_actor_rollout_worker(config)
        papo = omega_conf_to_dataclass(config.actor_rollout_ref.actor.fsdp_config.papo)
        if not papo.active:
            return actor_cls, group_cls
        from .workers import PAPOActorRolloutRefWorker

        for role in (Role.ActorRollout, Role.ActorRolloutRef):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = ray.remote(PAPOActorRolloutRefWorker)
        return PAPOActorRolloutRefWorker, group_cls


class PAPOTaskRunner(PAPOActorTaskRunnerMixin, DAPOTaskRunner):
    pass


@hydra.main(config_path="config", config_name="papo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(PAPOTaskRunner))


if __name__ == "__main__":
    main()
