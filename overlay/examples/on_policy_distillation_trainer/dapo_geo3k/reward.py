"""Geo3K reward with separate answer accuracy for DAPO group filtering."""

from verl.utils.reward_score.geo3k import acc_reward, format_reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    accuracy = acc_reward(solution_str, ground_truth)
    formatting = format_reward(solution_str)
    return {"score": 0.9 * accuracy + 0.1 * formatting, "acc": accuracy, "format": formatting}
