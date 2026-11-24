from copy import deepcopy

import pytest
from recipe.papo.eval_dataset import PAPOEvaluationDataset

from verl.utils.dataset.rl_dataset import RLHFDataset


@pytest.mark.parametrize("source", ["hiyouga/geometry3k", "papo_eval/mathvista", "papo_eval/mathvision"])
def test_image_budget_applies_only_to_added_evaluation_sources(monkeypatch, source):
    messages = [{"role": "user", "content": [{"type": "image", "image": "sentinel"}, {"type": "text", "text": "x"}]}]
    monkeypatch.setattr(RLHFDataset, "_build_messages", lambda *args: deepcopy(messages))
    dataset = object.__new__(PAPOEvaluationDataset)
    dataset.config = {"papo_eval_max_pixels": 1048576}
    result = dataset._build_messages({"data_source": source}, "prompt")
    if source == "hiyouga/geometry3k":
        assert result == messages
    else:
        assert result[0]["content"][0]["max_pixels"] == 1048576
        assert result[0]["content"][1] == messages[0]["content"][1]
