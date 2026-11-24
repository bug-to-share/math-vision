from functools import partial
from types import SimpleNamespace

import pytest
import torch
from recipe.papo.config import PAPOConfig
from recipe.papo.fsdp_engine import PAPOFSDPEngine
from recipe.papo.losses import papo_loss
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import FSDPActorConfig


class ImageConditionedLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
        self.grad_modes = []

    def forward(self, pixel_values, image_grid_thw, use_cache=False):
        self.grad_modes.append(torch.is_grad_enabled())
        logits = self.weight * pixel_values.mean()
        return logits.log_softmax(-1)[0].expand(5)


def engine_fixture(monkeypatch, papo):
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.get_device_id", lambda: "cpu")
    engine = object.__new__(PAPOFSDPEngine)
    engine.papo = papo
    engine._autocast_dtype = torch.float32
    engine.module = ImageConditionedLM()
    engine.optimizer = SimpleNamespace(state={})
    engine.model_config = SimpleNamespace(
        processor=SimpleNamespace(
            image_processor=SimpleNamespace(
                patch_size=1,
                temporal_patch_size=1,
                image_mean=[0.5],
                image_std=[0.5],
                do_normalize=True,
            )
        )
    )
    engine.prepare_model_inputs = lambda micro_batch: (
        {"pixel_values": torch.ones(4, 1), "image_grid_thw": torch.tensor([[1, 2, 2]])},
        {},
    )
    engine.prepare_model_outputs = lambda output, **kwargs: {"log_probs": output}
    engine.get_data_parallel_group = lambda: None
    return engine


def data_and_loss(papo):
    data = TensorDict(
        {
            "prompts": torch.zeros(1, 3, dtype=torch.long),
            "responses": torch.zeros(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "response_mask": torch.ones(1, 2),
            "old_log_probs": torch.full((1, 2), -1.0),
            "advantages": torch.ones(1, 2),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=2, global_batch_size=1)
    actor = FSDPActorConfig(strategy="fsdp2", rollout_n=1, use_dynamic_bsz=True, ppo_mini_batch_size=1)
    return data, partial(papo_loss, config=actor, papo=papo)


@pytest.mark.parametrize(
    "mode,expected_grad_modes", [("recompute_grad", [True, True]), ("stop_gradient", [True, False])]
)
def test_paired_forward_loss_remains_live_and_outputs_are_detached(monkeypatch, mode, expected_grad_modes):
    papo = PAPOConfig(mask_probability=1, gradient_mode=mode)
    engine = engine_fixture(monkeypatch, papo)
    data, loss_fn = data_and_loss(papo)
    loss, output = engine.forward_step(data, loss_fn, forward_only=False)
    loss.backward()
    assert engine.module.weight.grad.abs().sum() > 0
    assert engine.module.grad_modes == expected_grad_modes
    assert output["model_output"]["log_probs"].grad_fn is None
    assert "papo_masked_log_probs" not in output["model_output"]
    assert output["metrics"]["papo/mask_fraction"] == 1


def test_inference_compute_loss_uses_clean_forward_only(monkeypatch):
    papo = PAPOConfig()
    engine = engine_fixture(monkeypatch, papo)
    data, loss_fn = data_and_loss(papo)
    with torch.no_grad():
        _, output = engine.forward_step(data, loss_fn, forward_only=True)
    assert len(engine.module.grad_modes) == 1
    assert not any(key.startswith("papo/") for key in output["metrics"])
