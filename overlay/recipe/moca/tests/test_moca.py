import asyncio
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from recipe.moca.advantages import route_advantages
from recipe.moca.evidence import extract_evidence, recognition_token_mask
from recipe.moca.prepare_data import prepare_row
from recipe.moca.verifier import VerifierClient, parse_svv_score


@pytest.mark.parametrize(
    "text",
    [
        "plain answer",
        "<recognition>x",
        "</recognition>x<recognition>",
        "<recognition> </recognition>",
        "<recognition><recognition>x</recognition></recognition>",
        "<recognition><think>inferred</think></recognition>",
        "<think><recognition>not separated</recognition></think>",
    ],
)
def test_invalid_evidence(text):
    assert not extract_evidence(text).valid


def test_multiple_evidence_blocks_exclude_reasoning():
    result = extract_evidence("<recognition>AB=3</recognition><think>deduction</think><recognition>BC=4</recognition>")
    assert result.valid and result.text == "AB=3\nBC=4"


@pytest.mark.parametrize("text", [r"\boxed{2}", r"\boxed{-0.1}", r"\boxed{0.1}\boxed{1}", "1.0", r"\boxed{NaN}"])
def test_invalid_svv_score_is_not_a_negative_reward(text):
    with pytest.raises(ValueError):
        parse_svv_score(text)


def test_fractional_svv():
    assert parse_svv_score(r"The answer is right but unboxed. \boxed{0.9}") == 0.9


def test_quoted_student_answer_cannot_become_the_judge_score():
    assert parse_svv_score("Prediction quotes \\boxed{12}.\nConclusion: equivalent.\n\\boxed{1}\n") == 1
    with pytest.raises(ValueError):
        parse_svv_score("The student's answer is \\boxed{12}; I found no final score.")


def test_grpo_normalization_and_routing_gradient():
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    rewards = torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, 1.3], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    mask = torch.ones_like(rewards)
    base, _ = compute_grpo_outcome_advantage(rewards, mask, np.array([0, 0, 1, 1]), norm_adv_by_std_in_grpo=True)
    assert torch.allclose(base[:, 0], torch.tensor([-1.0, 1.0, -1.0, 1.0]) / 2**0.5, atol=1e-5)
    selected = torch.tensor([[1, 0, 0]] * 4)
    outcome, perception = torch.tensor([0.0, 1.0, 0.0, 1.0]), torch.tensor([1.0, 1.0, 0.0, 0.0])
    routed, _ = route_advantages(base, mask, selected, outcome, perception)
    logits = torch.zeros_like(base, requires_grad=True)
    old_logits = torch.zeros_like(base, requires_grad=True)
    (-torch.exp(logits) * routed).sum().backward()
    (-torch.exp(old_logits) * base).sum().backward()
    assert torch.allclose(logits.grad[0, 0], 0.8 * old_logits.grad[0, 0])
    assert torch.allclose(logits.grad[2, 0], 1.2 * old_logits.grad[2, 0])
    assert torch.equal(logits.grad[:, 1:], old_logits.grad[:, 1:])
    assert torch.equal(logits.grad[[1, 3]], old_logits.grad[[1, 3]])


def test_zero_advantages_and_padding_stay_zero():
    result, _ = route_advantages(
        torch.zeros(2, 4), torch.tensor([[1, 1, 0, 0]] * 2), torch.ones(2, 4), torch.zeros(2), torch.tensor([0.0, 1.0])
    )
    assert not result.any()


def test_zero_coefficients_leave_grpo_unchanged():
    base = torch.tensor([[-1.0, 2.0, 0.0]])
    result, _ = route_advantages(
        base, torch.tensor([[1, 1, 0]]), torch.ones_like(base), torch.zeros(1), torch.ones(1), protect=0, punish=0
    )
    assert torch.equal(result, base)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    path = os.environ.get("MOCA_TEST_TOKENIZER")
    if not path or not Path(path).is_dir():
        pytest.skip("Local Qwen3-VL tokenizer is unavailable")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def test_real_tokenizer_offsets_and_padding(tokenizer):
    text = "<think>plan</think><recognition>角 A=30°，AB=3.</recognition><think>calculate</think>\\boxed{6}"
    ids = tokenizer.encode(text, add_special_tokens=False)
    selected = recognition_token_mask(tokenizer, ids)
    content = tokenizer.decode([token for token, use in zip(ids, selected, strict=True) if use])
    assert "30" in content and "AB" in content
    assert "recognition" not in content and "calculate" not in content and "boxed" not in content


def test_trainer_routes_once_before_worker_update(tokenizer, monkeypatch):
    from recipe.moca.trainer import MoCATrainer

    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    ids = tokenizer.encode("<recognition>AB=3.</recognition><think>x</think>", add_special_tokens=False)
    length = len(ids)
    data = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([ids, ids]),
            "response_mask": torch.ones(2, length),
            "advantages": -torch.ones(2, length),
        },
        non_tensors={"moca_outcome": np.array([0.0, 0.0]), "moca_perception": np.array([1.0, 0.0])},
    )
    observed = []

    def worker_update(self, batch):
        observed.append(batch.batch["advantages"].clone())
        return DataProto(meta_info={"metrics": {}})

    monkeypatch.setattr(RayPPOTrainer, "_update_actor", worker_update)
    trainer = object.__new__(MoCATrainer)
    trainer.tokenizer = tokenizer
    trainer.config = OmegaConf.create({"moca": {"protect": 0.2, "punish": 0.2}})
    result = trainer._update_actor(data)
    selected = torch.tensor(recognition_token_mask(tokenizer, ids))
    assert torch.allclose(observed[0][0, selected], torch.full_like(observed[0][0, selected], -0.8))
    assert torch.allclose(observed[0][1, selected], torch.full_like(observed[0][1, selected], -1.2))
    assert "moca/perception_mean" in result.meta_info["metrics"]
    with pytest.raises(ValueError, match="already"):
        trainer._update_actor(data)


def test_composed_config_has_standard_grpo_and_no_papo():
    from hydra import compose, initialize_config_dir
    from recipe.moca.main_moca import validate_moca_config

    directory = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(config_dir=directory, version_base=None):
        config = compose(config_name="moca_trainer", overrides=["actor_rollout_ref.rollout.n=8"])
    validate_moca_config(config)
    assert config.algorithm.norm_adv_by_std_in_grpo
    assert "papo" not in config.actor_rollout_ref.actor.fsdp_config
    assert not config.actor_rollout_ref.actor.use_kl_loss


def test_actual_reward_and_dataset_factory_loads(tokenizer):
    from hydra import compose, initialize_config_dir

    from verl.trainer.ppo.reward import load_reward_manager
    from verl.utils.dataset.rl_dataset import get_dataset_class

    directory = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(config_dir=directory, version_base=None):
        config = compose(
            config_name="moca_trainer",
            overrides=[
                "data.custom_cls.path=recipe/moca/dataset.py",
                "data.custom_cls.name=MoCADataset",
            ],
        )

    async def load():
        manager = load_reward_manager(config, tokenizer)
        assert manager.__class__.__name__ == "MoCARewardManager"
        assert manager.verifier.config.model == "verifier"

    asyncio.run(load())
    assert get_dataset_class(config.data).__name__ == "MoCADataset"


def test_prepare_uses_clean_question_without_old_format_or_gold():
    original = {
        "data_source": "hiyouga/geometry3k",
        "prompt": [{"role": "user", "content": "old format"}],
        "extra_info": {"question": "<image>Find x.", "answer": "SECRET-GOLD"},
        "images": [b"image"],
    }
    result = prepare_row(original, training=True)
    assert original["prompt"][0]["content"] == "old format"
    assert result["prompt"][-1]["content"] == "<image>Find x."
    assert result["extra_info"]["moca_question"] == "Find x."
    assert "SECRET-GOLD" not in str(result["prompt"])
    assert result["images"] == original["images"]


def test_proxy_request_cannot_include_gold_or_original_cot():
    class CapturingClient(VerifierClient):
        async def _chat(self, system, payload, max_tokens):
            assert set(payload) == {"question", "observations"}
            assert payload == {"question": "q", "observations": "facts"}
            return {"text": "ok"}

    assert asyncio.run(CapturingClient({}).solve("q", "facts"))["text"] == "ok"


def test_reward_validation_skips_llm_and_training_separates_roles(tokenizer):
    from recipe.moca.reward_manager import MoCARewardManager

    from verl import DataProto

    class FakeVerifier:
        def __init__(self):
            self.solves = []
            self.scores = []

        async def solve(self, question, evidence):
            self.solves.append((question, evidence))
            assert "my reasoning" not in evidence
            return {"text": r"\boxed{12}", "usage": {}}

        async def score(self, question, ground_truth, response):
            self.scores.append((question, ground_truth, response))
            return {"text": r"\boxed{1}", "score": 1.0, "usage": {}}

    text = r"<recognition>AB=3,BC=4,CA=5</recognition><think>my reasoning</think>\boxed{12}"
    ids = tokenizer.encode(text, add_special_tokens=False)
    row = DataProto.from_dict(
        tensors={"responses": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids) + 2, dtype=torch.long)},
        non_tensors={
            "data_source": np.array(["hiyouga/geometry3k"], dtype=object),
            "reward_model": np.array([{"ground_truth": "12"}], dtype=object),
            "extra_info": np.array([{"moca_train": False, "moca_question": "perimeter?"}], dtype=object),
        },
    )
    manager = object.__new__(MoCARewardManager)
    manager.tokenizer = tokenizer
    manager.settings = OmegaConf.create({"perception_weight": 0.3, "pv_success_threshold": 1.0, "audit_dir": None})
    manager.verifier = FakeVerifier()
    result = asyncio.run(manager.run_single(row))
    assert result["reward_score"] == 1 and not manager.verifier.scores
    row.non_tensor_batch["extra_info"][0]["moca_train"] = True
    result = asyncio.run(manager.run_single(row))
    assert result["reward_score"] == pytest.approx(1.3)
    assert len(manager.verifier.solves) == 1 and len(manager.verifier.scores) == 2
    assert result["reward_extra_info"]["acc"] == 1
