"""Separate generalization evaluation rules from the unchanged Geo3K reward."""

import ast
import json
import math
import re
from decimal import Decimal, InvalidOperation

from examples.on_policy_distillation_trainer.dapo_geo3k.reward import compute_score as geo3k_score
from mathruler.grader import extract_boxed_content, grade_answer

from verl.utils.reward_score.geo3k import format_reward

EVALUATION_SOURCES = {"papo_eval/geoqa", "papo_eval/mathvista", "papo_eval/mathvision"}


def normalize_math(answer):
    """Normalize notation differences without searching reasoning for the answer."""
    value = str(answer).strip().strip("$").replace("−", "-").replace("π", r"\pi")
    value = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\^\{?\\circ\}?|\\degree|°", "", value)
    value = re.sub(r"√\{([^{}]+)\}", r"\\sqrt{\1}", value)
    value = re.sub(r"√\(([^()]+)\)", r"\\sqrt{\1}", value)
    value = re.sub(r"√([0-9]+)", r"\\sqrt{\1}", value)
    return value.strip()


def choice_index(answer, choices):
    normalized = normalize_math(answer)
    letter = re.fullmatch(r"\(?([A-Z])\)?", normalized)
    if letter and ord(letter[1]) - ord("A") < len(choices):
        return ord(letter[1]) - ord("A")
    matches = [i for i, choice in enumerate(choices) if normalize_math(choice) == normalized]
    return matches[0] if matches else None


def evaluation_accuracy(solution_str, target):
    answer = normalize_math(extract_boxed_content(solution_str))
    if not answer:
        return 0.0
    truth = normalize_math(target["answer"])
    answer_type = target["answer_type"]
    if answer_type == "choice":
        predicted = choice_index(answer, target["choices"])
        expected = ord(truth) - ord("A")
        # Some source questions repeat an option; either matching letter is correct.
        return float(
            predicted is not None
            and normalize_math(target["choices"][predicted]) == normalize_math(target["choices"][expected])
        )
    if answer_type == "integer":
        try:
            predicted, expected = Decimal(answer), Decimal(truth)
            return float(predicted.is_finite() and expected.is_finite() and predicted == expected)
        except InvalidOperation:
            return 0.0
    if answer_type == "float":
        try:
            predicted, expected = float(answer), float(truth)
        except ValueError:
            return 0.0
        if not math.isfinite(predicted) or not math.isfinite(expected):
            return 0.0
        precision = target["precision"]
        return float(round(predicted, precision) == round(expected, precision))
    if answer_type == "list":
        try:
            predicted, expected = ast.literal_eval(answer), ast.literal_eval(truth)
        except (ValueError, SyntaxError):
            return 0.0
        return float(isinstance(predicted, list) and isinstance(expected, list) and predicted == expected)
    if answer_type == "math":
        return float(grade_answer(answer, truth))
    raise ValueError(f"Unsupported evaluation answer type: {answer_type}")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == "hiyouga/geometry3k":
        return geo3k_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
    if data_source not in EVALUATION_SOURCES:
        raise ValueError(f"Unexpected PAPO evaluation data source: {data_source}")
    accuracy = evaluation_accuracy(solution_str, json.loads(ground_truth))
    return {"score": accuracy, "acc": accuracy, "format": format_reward(solution_str)}
