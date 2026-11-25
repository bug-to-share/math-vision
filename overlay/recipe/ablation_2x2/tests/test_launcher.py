import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from recipe.ablation_2x2.main_ablation import AblationTaskRunner, validate_moca_config
from recipe.moca.trainer import MoCATrainer
from recipe.papo.workers import PAPOActorRolloutRefWorker

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import omega_conf_to_dataclass, validate_config

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "recipe/ablation_2x2/run_qwen3_vl_4b_geo3k_2x2.sh"


@pytest.mark.parametrize(
    "arm,control,occlusion,evidence,route",
    [
        ("A", False, False, False, False),
        ("B", False, True, False, False),
        ("C", False, False, True, True),
        ("D", False, True, True, True),
        ("B", True, False, False, False),
        ("C-reward-only", False, False, True, False),
    ],
)
def test_launcher_composes_and_routes_actual_arm(arm, control, occlusion, evidence, route):
    env = {
        **os.environ,
        "ABLATION_PYTHON": sys.executable,
        "CUDA_VISIBLE_DEVICES": "",
        "MOCA_DATA": str(ROOT / "data/moca"),
        "RUN_TAG": "test",
    }
    env.pop("PAPO_PERCEPTION_COEF", None)
    if control:
        env["PAPO_PERCEPTION_COEF"] = "0"
    completed = subprocess.run(
        ["bash", str(LAUNCHER), arm, "--cfg", "job", "--resolve"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    config = OmegaConf.create(completed.stdout)
    validate_moca_config(config)
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    papo = omega_conf_to_dataclass(config.actor_rollout_ref.actor).engine.papo
    assert papo.active and papo.needs_masked_forward == occlusion
    assert papo.perception_coef == (0.01 if occlusion else 0.0)
    assert papo.gradient_mode == "cached_old"
    assert papo.clean_entropy_coef == 0.03 and papo.masked_entropy_coef == 0
    assert config.moca.use_rule_outcome and config.moca.use_evidence_reward == evidence
    assert "data/moca/train.parquet" in config.data.train_files
    assert config.actor_rollout_ref.rollout.n == 8
    assert config.data.max_response_length == 4096
    assert config.trainer.experiment_name.startswith("B-clean-nll-" if control else f"{arm}-")
    with patch("ray.remote", side_effect=lambda cls: cls):
        runner = AblationTaskRunner()
        worker, _ = runner.add_actor_rollout_worker(config)
    assert worker is PAPOActorRolloutRefWorker
    assert runner._trainer_class(config) is (MoCATrainer if route else RayPPOTrainer)
