import json

import pytest
from examples.on_policy_distillation_trainer.dapo_geo3k.reward import compute_score as baseline_score
from recipe.papo.eval_reward import compute_score, evaluation_accuracy


@pytest.mark.parametrize(
    "solution,truth",
    [(r"<thinking>3</thinking>\boxed{3}", "3"), (r"\boxed{4}", "3"), ("3", "3"), (r"\boxed{1/2}", "0.5")],
)
def test_training_reward_is_unchanged(solution, truth):
    assert compute_score("hiyouga/geometry3k", solution, truth) == baseline_score("hiyouga/geometry3k", solution, truth)


@pytest.mark.parametrize(
    "solution,expected",
    [
        (r"Consider A then B. \boxed{B}", 1),
        (r"A is right but \boxed{C}", 0),
        ("B", 0),
        (r"\boxed{A} then \boxed{B}", 1),
        (r"\boxed{\text{B}}", 1),
        (r"\boxed{(B)}", 1),
        (r"\boxed{five}", 1),
        (r"\boxed{Option B is correct}", 0),
    ],
)
def test_choice_uses_only_final_box_and_unambiguous_option(solution, expected):
    target = {"answer": "B", "choices": ["four", "five", "six"], "answer_type": "choice", "precision": None}
    assert evaluation_accuracy(solution, target) == expected


@pytest.mark.parametrize(
    "kind,truth,predicted,precision,expected",
    [
        ("integer", "4", "4.0", None, 1),
        ("integer", "4", "4.9", None, 0),
        ("integer", "4", "NaN", None, 0),
        ("float", "0.33", "0.33333", 2, 1),
        ("float", "0.33", "0.34", 2, 0),
        ("float", "0.33", "inf", 2, 0),
        ("list", "[1, 2]", "[1,2]", None, 1),
        ("list", "[1, 2]", "[2,1]", None, 0),
        ("math", "5√{3}", r"5\sqrt{3}", None, 1),
        ("math", "78°", "78", None, 1),
        ("math", "(3,-4)", "(3,-4)", None, 1),
        ("math", "5", "6", None, 0),
    ],
)
def test_benchmark_specific_answer_rules(kind, truth, predicted, precision, expected):
    target = {"answer": truth, "choices": [], "answer_type": kind, "precision": precision}
    assert evaluation_accuracy(r"\boxed{" + predicted + "}", target) == expected


def test_evaluation_score_has_no_partial_format_credit():
    target = {"answer": "2", "choices": [], "answer_type": "integer", "precision": None}
    score = compute_score("papo_eval/mathvista", r"<thinking>...</thinking>\boxed{3}", json.dumps(target))
    assert score == {"score": 0.0, "acc": 0.0, "format": 1.0}


@pytest.mark.parametrize("answer", ["C", "D", "18"])
def test_duplicate_options_are_scored_by_original_answer_content(answer):
    target = {"answer": "C", "choices": ["9", "12", "18", "18"], "answer_type": "choice", "precision": None}
    assert evaluation_accuracy(r"\boxed{" + answer + "}", target) == 1.0
