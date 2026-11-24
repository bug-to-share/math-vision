"""Exercise the actual 4B launchers without starting Ray or allocating a GPU."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset
from examples.on_policy_distillation_trainer.dapo_geo3k.main_dapo import DAPOTaskRunner
from examples.on_policy_distillation_trainer.dapo_geo3k.main_papo import Geo3KPAPOTaskRunner
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from recipe.papo.config import PAPOFSDPEngineConfig
from recipe.papo.eval_dataset import PAPOEvaluationDataset
from recipe.papo.main_papo import PAPOTaskRunner

from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.dataset.rl_dataset import get_dataset_class
from verl.workers.engine_workers import ActorRolloutRefWorker

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples/on_policy_distillation_trainer"


def capture_launcher(name, directory, extra_args=()):
    """Replace only the interpreter so the real shell expands all its defaults."""
    launcher_dir = ROOT / "recipe/papo" if name.startswith("papo") else EXAMPLES
    launcher = launcher_dir / f"run_qwen3_vl_4b_geo3k_{name}_4gpu.sh"
    interpreter = directory / "python3"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'ray_address': os.environ.get('RAY_ADDRESS')}))\n"
    )
    interpreter.chmod(0o755)
    env = dict(os.environ)
    for key in re.findall(r"\$\{([A-Z][A-Z0-9_]*):-", launcher.read_text()):
        env.pop(key, None)
    env.update(
        PATH=f"{directory}:{os.environ.get('PATH', '')}",
        PAPO_PYTHON=str(interpreter),
        CUDA_VISIBLE_DEVICES="-1",
        RUN_TAG="config_parity",
    )
    result = subprocess.run(
        ["bash", str(launcher), *extra_args], cwd=ROOT, env=env, text=True, capture_output=True, check=True
    )
    return json.loads(result.stdout)


def compose_launcher(captured, extra_overrides=()):
    with initialize_config_dir(config_dir=str(EXAMPLES / "dapo_geo3k/config"), version_base=None):
        return compose(config_name="dapo_trainer", overrides=[*captured["argv"][2:], *extra_overrides])


def config_differences(baseline, papo, prefix=""):
    """Report every resolved field that changes, including unexpected additions."""
    differences = {}
    for key in sorted(set(baseline) | set(papo)):
        path = f"{prefix}.{key}" if prefix else key
        before, after = baseline.get(key), papo.get(key)
        if isinstance(before, dict) and isinstance(after, dict):
            differences.update(config_differences(before, after, path))
        elif key not in baseline or key not in papo or before != after:
            differences[path] = {"dapo": before, "papo": after}
    return differences


@pytest.fixture(scope="module")
def launchers(tmp_path_factory):
    directory = tmp_path_factory.mktemp("geo3k_launchers")
    return {name: capture_launcher(name, directory) for name in ("dapo", "papo")}


def test_actual_launchers_preserve_baseline_except_documented_changes(launchers):
    baseline = compose_launcher(launchers["dapo"])
    papo = compose_launcher(launchers["papo"])
    differences = config_differences(
        OmegaConf.to_container(baseline, resolve=True), OmegaConf.to_container(papo, resolve=True)
    )
    assert set(differences) == {
        "actor_rollout_ref.actor.fsdp_config._target_",
        "actor_rollout_ref.actor.fsdp_config.papo",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
        "actor_rollout_ref.rollout.gpu_memory_utilization",
        "actor_rollout_ref.rollout.trace.experiment_name",
        "data.val_files",
        "data.custom_cls.path",
        "data.custom_cls.name",
        "data.papo_eval_max_pixels",
        "reward.custom_reward_function.path",
        "trainer.experiment_name",
        "trainer.default_local_dir",
        "trainer.rollout_data_dir",
        "trainer.validation_data_dir",
        "ray_kwargs.ray_init.address",
    }, differences
    assert differences["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] == {"dapo": 12288, "papo": 6144}
    assert launchers["papo"]["argv"][:2] == [
        "-m",
        "examples.on_policy_distillation_trainer.dapo_geo3k.main_papo",
    ]
    assert launchers["papo"]["ray_address"] == "local"
    assert "papo" not in baseline.actor_rollout_ref.actor.fsdp_config
    assert papo.actor_rollout_ref.model.lora_rank == 0
    assert not papo.actor_rollout_ref.actor.freeze_vision_tower
    assert papo.trainer.resume_mode == "disable"
    assert papo.actor_rollout_ref.rollout.gpu_memory_utilization == 0.75
    assert len(papo.data.val_files) == 4
    validate_config(papo, use_reference_policy=need_reference_policy(papo), use_critic=need_critic(papo))


@pytest.mark.parametrize("mode", ["enabled", "disabled", "zero_coefficients"])
@pytest.mark.parametrize("use_reference", [False, True])
def test_geo3k_actor_routing_and_baseline_trainer_are_preserved(launchers, mode, use_reference):
    overrides = [f"actor_rollout_ref.actor.use_kl_loss={str(use_reference).lower()}"]
    if mode == "disabled":
        overrides.append("actor_rollout_ref.actor.fsdp_config.papo.enabled=false")
    elif mode == "zero_coefficients":
        overrides.extend(
            f"actor_rollout_ref.actor.fsdp_config.papo.{name}=0"
            for name in ("perception_coef", "clean_entropy_coef", "masked_entropy_coef")
        )
    config = compose_launcher(launchers["papo"], overrides)
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert isinstance(actor.engine, PAPOFSDPEngineConfig)
    assert actor.engine is actor.fsdp_config
    assert Geo3KPAPOTaskRunner.run is DAPOTaskRunner.run
    assert PAPOTaskRunner.run is not DAPOTaskRunner.run
    with patch("ray.remote", side_effect=lambda cls: cls):
        runner = Geo3KPAPOTaskRunner()
        worker, _ = runner.add_actor_rollout_worker(config)
    if mode == "enabled":
        from recipe.papo.workers import PAPOActorRolloutRefWorker

        assert worker is PAPOActorRolloutRefWorker
    else:
        assert worker is ActorRolloutRefWorker
    role = Role.ActorRolloutRef if use_reference else Role.ActorRollout
    assert runner.role_worker_mapping[role] is worker


def test_launcher_allows_explicit_overrides_after_defaults(tmp_path):
    captured = capture_launcher(
        "papo", tmp_path, ("trainer.total_training_steps=2", "actor_rollout_ref.actor.fsdp_config.papo.enabled=false")
    )
    config = compose_launcher(captured)
    assert config.trainer.total_training_steps == 2
    assert not config.actor_rollout_ref.actor.fsdp_config.papo.enabled


def test_cached_launcher_changes_only_mode_and_artifact_names(tmp_path):
    original = compose_launcher(capture_launcher("papo", tmp_path))
    cached = compose_launcher(capture_launcher("papo_cached", tmp_path))
    differences = config_differences(
        OmegaConf.to_container(original, resolve=True), OmegaConf.to_container(cached, resolve=True)
    )
    assert set(differences) == {
        "actor_rollout_ref.actor.fsdp_config.papo.gradient_mode",
        "actor_rollout_ref.rollout.trace.experiment_name",
        "trainer.experiment_name",
        "trainer.default_local_dir",
        "trainer.rollout_data_dir",
        "trainer.validation_data_dir",
    }, differences
    assert cached.actor_rollout_ref.actor.fsdp_config.papo.gradient_mode == "cached_old"


def test_launcher_dataset_survives_hf_multiprocess_filter(launchers):
    """File-only imports lose the super() class closure when dill rebuilds them."""
    config = compose_launcher(launchers["papo"])
    dataset_cls = get_dataset_class(config.data)
    assert dataset_cls is PAPOEvaluationDataset
    dataset = object.__new__(dataset_cls)
    dataset.config = config.data
    dataset.serialize_dataset = True
    dataset.image_key, dataset.video_key, dataset.audio_key = "images", "videos", "audios"
    examples = Dataset.from_list(
        [
            {
                "data_source": source,
                "prompt": [{"role": "user", "content": [{"type": "image", "image": "sentinel"}]}],
            }
            for source in ("hiyouga/geometry3k", "papo_eval/mathvista")
        ]
    )

    def accepts(row):
        image = dataset._build_messages(row, "prompt")[0]["content"][0]
        if row["data_source"].startswith("papo_eval/"):
            return image["max_pixels"] == 1048576
        return "max_pixels" not in image

    assert len(examples.filter(accepts, num_proc=1, load_from_cache_file=False)) == 2
