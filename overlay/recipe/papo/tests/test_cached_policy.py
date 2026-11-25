"""Cache lifecycle and gradients, including two updates from one rollout batch."""

from types import SimpleNamespace

import pytest
import torch
from recipe.papo.config import PAPOConfig
from recipe.papo.losses import papo_loss
from recipe.papo.tests.test_engine import data_and_loss, engine_fixture
from recipe.papo.tests.test_papo_loss import actor_config, make_data, outputs
from recipe.papo.workers import PAPOTrainingWorker
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine_workers import TrainingWorker


def nested(values):
    return torch.nested.as_nested_tensor(values, layout=torch.jagged)


def test_cached_loss_matches_fixed_target_and_rejects_missing_or_live_cache():
    data = make_data()
    clean = torch.full((10,), -1.0, requires_grad=True)
    masked = torch.full((10,), -2.0)
    data["papo_masked_old_log_probs"] = nested([masked[:5], masked[5:]])
    cached_loss, _ = papo_loss(actor_config(), PAPOConfig(gradient_mode="cached_old"), outputs(clean), data)
    stopped_loss, _ = papo_loss(actor_config(), PAPOConfig(gradient_mode="stop_gradient"), outputs(clean, masked), data)
    torch.testing.assert_close(cached_loss, stopped_loss)
    torch.testing.assert_close(torch.autograd.grad(cached_loss, clean)[0], torch.autograd.grad(stopped_loss, clean)[0])
    data.pop("papo_masked_old_log_probs")
    with pytest.raises(ValueError, match="requires pre-update"):
        papo_loss(actor_config(), PAPOConfig(gradient_mode="cached_old"), outputs(clean), data)
    data["papo_masked_old_log_probs"] = nested([masked[:5], masked[5:]]).requires_grad_()
    with pytest.raises(ValueError, match="must not require gradients"):
        papo_loss(actor_config(), PAPOConfig(gradient_mode="cached_old"), outputs(clean), data)


@pytest.mark.parametrize("strategy,mask_fraction", [("random", 1.0), ("hybrid_attention", 0.5)])
def test_engine_masks_once_then_only_clean_forward_across_updates(monkeypatch, strategy, mask_fraction):
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.get_device_name", lambda: "cpu")
    papo = PAPOConfig(gradient_mode="cached_old", mask_probability=mask_fraction, mask_strategy=strategy)
    engine = engine_fixture(monkeypatch, papo)
    prepare_inputs = engine.prepare_model_inputs
    prepare_outputs = engine.prepare_model_outputs
    del engine.prepare_model_inputs
    del engine.prepare_model_outputs
    monkeypatch.setattr(
        FSDPEngineWithLMHead, "prepare_model_inputs", lambda self, micro_batch: prepare_inputs(micro_batch)
    )
    monkeypatch.setattr(
        FSDPEngineWithLMHead,
        "prepare_model_outputs",
        lambda self, output, output_args, micro_batch, logits_processor_func: prepare_outputs(output),
    )
    data, loss_fn = data_and_loss(papo)
    tu.assign_non_tensor(data, papo_cache_forward=True)
    with torch.no_grad():
        _, output = engine.forward_step(data, None, forward_only=True)
    cached = output["model_output"]["log_probs"].clone()
    assert output["metrics"]["papo/cache_mask_fraction"] == mask_fraction
    assert engine.module.grad_modes == [False]
    data.pop("papo_cache_forward")
    data["papo_masked_old_log_probs"] = nested([cached])
    tu.assign_non_tensor(data, papo_cache_version="round-a", papo_cache_optimizer_step=0)
    engine._papo_active_cache_version = "round-a"
    for step in (0, 1):
        engine.optimizer.state = {0: {"step": step}}
        loss, output = engine.forward_step(data, loss_fn, forward_only=False)
        loss.backward()
        assert engine.module.weight.grad.abs().sum() > 0
        assert output["metrics"]["papo/cache_age_optimizer_steps"] == step
        torch.testing.assert_close(data["papo_masked_old_log_probs"].values(), cached)
        with torch.no_grad():
            engine.module.weight -= 0.1 * engine.module.weight.grad
            engine.module.weight.grad = None
    assert engine.module.grad_modes == [False, True, True]
    assert not torch.allclose((-engine.module.weight).log_softmax(-1)[0].expand(5), cached)
    engine._papo_active_cache_version = "round-b"
    with pytest.raises(ValueError, match="Missing or stale"):
        engine.forward_step(data, loss_fn, forward_only=False)


def worker_fixture():
    worker = object.__new__(PAPOTrainingWorker)
    worker.engine_config = SimpleNamespace(papo=PAPOConfig(gradient_mode="cached_old"))
    state = {"step": 7, "inferences": 0}
    worker.engine = SimpleNamespace(_mask_step=lambda: state["step"])
    data = TensorDict({"input_ids": nested([torch.arange(3), torch.arange(5)])}, batch_size=[2])
    return worker, state, data


def test_worker_caches_before_minibatching_and_invalidates_afterward(monkeypatch):
    worker, state, data = worker_fixture()
    cache = nested([torch.full((3,), -1.0), torch.full((5,), -2.0)])

    def infer(batch):
        state["inferences"] += 1
        assert state["step"] == 7
        assert tu.get(batch, "papo_cache_forward")
        assert not tu.get(batch, "compute_loss")
        assert not torch.is_grad_enabled()
        tu.assign_non_tensor(batch, max_token_len_per_gpu=12288)
        return tu.get_tensordict(
            tensor_dict={"log_probs": cache}, non_tensor_dict={"metrics": {"papo/cache_mask_fraction": [[0.6]]}}
        )

    def train(self, batch):
        assert "papo_cache_forward" not in batch and "max_token_len_per_gpu" not in batch
        assert tu.get(batch, "papo_cache_optimizer_step") == 7
        assert tu.get(batch, "papo_cache_version") == self.engine._papo_active_cache_version
        # Both minibatches retain their per-sample targets despite reversed order.
        for index in (1, 0):
            torch.testing.assert_close(batch["papo_masked_old_log_probs"][index], cache[index])
            state["step"] += 1
        return tu.get_tensordict({}, {"metrics": {}})

    worker.infer_batch = infer
    monkeypatch.setattr(TrainingWorker, "train_mini_batch", train)
    output = worker.train_mini_batch(data)
    assert state == {"step": 9, "inferences": 1}
    assert tu.get(output, "metrics")["papo/cache_passes"] == 1
    assert tu.get(output, "metrics")["papo/cache_mask_fraction"] == 0.6
    assert worker.engine._papo_active_cache_version is None
    assert not any(key.startswith("papo_") for key in data.keys())


def test_worker_does_not_reuse_incoming_cache():
    worker, _, data = worker_fixture()
    tu.assign_non_tensor(data, papo_cache_version="old-round")
    with pytest.raises(ValueError, match="fresh batch"):
        worker.train_mini_batch(data)


def test_cache_invalidated_if_training_fails(monkeypatch):
    worker, _, data = worker_fixture()
    worker.infer_batch = lambda batch: tu.get_tensordict(
        {"log_probs": nested([torch.zeros(3), torch.zeros(5)])},
        {"metrics": {"papo/cache_mask_fraction": [0.6]}},
    )

    def fail(self, data):
        raise RuntimeError("training failed")

    monkeypatch.setattr(TrainingWorker, "train_mini_batch", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        worker.train_mini_batch(data)
    assert worker.engine._papo_active_cache_version is None
    assert "papo_masked_old_log_probs" not in data
