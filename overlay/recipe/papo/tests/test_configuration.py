from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from recipe.papo.config import PAPOConfig, PAPOFSDPEngineConfig
from recipe.papo.main_papo import PAPOTaskRunner

from verl.trainer.ppo.utils import Role
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.engine_workers import ActorRolloutRefWorker


@pytest.mark.parametrize("enabled", [True, False])
def test_hydra_nested_dataclasses_and_recipe_worker_routing(enabled):
    with initialize_config_dir(config_dir=str(Path("recipe/papo/config").resolve()), version_base=None):
        config = compose(
            config_name="papo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                "actor_rollout_ref.rollout.n=2",
                f"actor_rollout_ref.actor.fsdp_config.papo.enabled={str(enabled).lower()}",
            ],
        )
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert isinstance(actor.fsdp_config, PAPOFSDPEngineConfig)
    assert isinstance(actor.fsdp_config.papo, PAPOConfig)
    assert actor.engine is actor.fsdp_config
    with patch("ray.remote", side_effect=lambda cls: cls):
        runner = PAPOTaskRunner()
        worker, _ = runner.add_actor_rollout_worker(config)
    if enabled:
        from recipe.papo.workers import PAPOActorRolloutRefWorker

        assert worker is PAPOActorRolloutRefWorker
    else:
        assert worker is ActorRolloutRefWorker
    assert runner.role_worker_mapping[Role.ActorRollout] is worker


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mask_probability": -0.1},
        {"mask_probability": float("nan")},
        {"perception_coef": float("inf")},
        {"gradient_mode": "cached"},
    ],
)
def test_invalid_algorithm_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        PAPOConfig(**kwargs)
