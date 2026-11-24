from functools import partial

import pytest
import torch
from recipe.papo.config import PAPOConfig
from recipe.papo.losses import papo_loss
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import FSDPActorConfig
from verl.workers.utils.losses import ppo_loss


def make_data(batch_size=2):
    # Three prompt tokens, two response tokens; second response is masked out
    # in sample 1, exercising normalization with unequal valid lengths.
    data = TensorDict(
        {
            "prompts": torch.zeros(batch_size, 3, dtype=torch.long),
            "responses": torch.zeros(batch_size, 2, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 5, dtype=torch.long),
            "response_mask": torch.tensor([[1, 1], [1, 0]])[:batch_size],
            "old_log_probs": torch.full((batch_size, 2), -1.0),
            "advantages": torch.ones(batch_size, 2),
        },
        batch_size=[batch_size],
    )
    return tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=3, global_batch_size=2)


def actor_config():
    return FSDPActorConfig(
        strategy="fsdp2", rollout_n=2, use_dynamic_bsz=True, ppo_mini_batch_size=2, loss_agg_mode="token-mean"
    )


def outputs(clean, masked=None):
    result = {"log_probs": clean}
    if masked is not None:
        result["papo_masked_log_probs"] = masked
    return result


@pytest.mark.parametrize("disabled", [False, True])
def test_zero_or_disabled_is_exact_base_loss_and_gradient(disabled):
    clean = torch.full((10,), -1.0, requires_grad=True)
    data, config = make_data(), actor_config()
    base, _ = ppo_loss(config, outputs(clean), data)
    papo = PAPOConfig(enabled=not disabled, perception_coef=0, clean_entropy_coef=0, masked_entropy_coef=0)
    actual, metrics = papo_loss(config, papo, outputs(clean), data)
    assert torch.equal(base, actual)
    assert torch.equal(torch.autograd.grad(base, clean)[0], torch.autograd.grad(actual, clean)[0])
    assert not any(key.startswith("papo/") for key in metrics)


@pytest.mark.parametrize("mode", ["recompute_grad", "stop_gradient"])
def test_perception_term_changes_actual_gradient_and_has_expected_sign(mode):
    clean = torch.full((10,), -1.0, requires_grad=True)
    masked = torch.full((10,), -2.0, requires_grad=True)
    papo = PAPOConfig(perception_coef=0.1, clean_entropy_coef=0, masked_entropy_coef=0, gradient_mode=mode)
    data, config = make_data(), actor_config()
    base, _ = ppo_loss(config, outputs(clean), data)
    joint, metrics = papo_loss(config, papo, outputs(clean, masked), data)
    base_grad = torch.autograd.grad(base, clean)[0]
    grad, masked_grad = torch.autograd.grad(joint, (clean, masked), allow_unused=True)
    expected = -0.1 * torch.exp(torch.tensor(-1.0))
    torch.testing.assert_close(joint - base, expected)
    assert not torch.allclose(grad, base_grad)
    assert torch.all(grad[[0, 1, 4, 5, 6, 8, 9]] == 0)  # prompts, masked response, final unused token
    if mode == "stop_gradient":
        assert masked_grad is None
    else:
        assert masked_grad.abs().sum() > 0
        torch.testing.assert_close((grad - base_grad)[[2, 3, 7]], -masked_grad[[2, 3, 7]])
    assert metrics["papo/aux_loss"].aggregate() < 0


def test_entropy_proxies_and_global_microbatch_normalization():
    clean = torch.full((10,), -1.0, requires_grad=True)
    masked = torch.full((10,), -2.0, requires_grad=True)
    config, data = actor_config(), make_data()
    papo = PAPOConfig(perception_coef=0, clean_entropy_coef=0.03, masked_entropy_coef=0.03)
    full, _ = papo_loss(config, papo, outputs(clean, masked), data)
    pieces = []
    for index in range(2):
        part, _ = papo_loss(
            config,
            papo,
            outputs(clean[index * 5 : (index + 1) * 5], masked[index * 5 : (index + 1) * 5]),
            data[index : index + 1],
        )
        pieces.append(part)
    torch.testing.assert_close(sum(pieces), full)
    grad = torch.autograd.grad(full, (clean, masked), retain_graph=True)
    split_grad = torch.autograd.grad(sum(pieces), (clean, masked))
    for actual, expected in zip(split_grad, grad, strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(grad[1][[2, 3, 7]], torch.full((3,), -0.01))


def test_worker_wraps_the_final_loss_and_reference_stays_standard():
    from types import SimpleNamespace

    from recipe.papo.workers import PAPOActorRolloutRefWorker, PAPOTrainingWorker

    from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

    worker = object.__new__(PAPOTrainingWorker)
    worker.engine_config = SimpleNamespace(papo=PAPOConfig())
    worker.set_loss_fn(partial(ppo_loss, config=actor_config()))
    assert worker.loss_fn.func is papo_loss
    assert PAPOActorRolloutRefWorker.actor_worker_cls is PAPOTrainingWorker
    assert PAPOActorRolloutRefWorker.ref_worker_cls is TrainingWorker
    assert ActorRolloutRefWorker.actor_worker_cls is TrainingWorker
    with pytest.raises(ValueError, match="standard PPO"):
        worker.set_loss_fn(lambda: None)
