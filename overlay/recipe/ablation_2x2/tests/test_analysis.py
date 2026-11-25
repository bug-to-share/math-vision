import json
import subprocess
import sys
from pathlib import Path

import pytest
from recipe.ablation_2x2.analyze_2x2_interaction import ARMS, analyze_2x2_interaction


def write_results(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def arm_paths(tmp_path):
    arms = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in (42, 43, 44):
            path = tmp_path / f"{arm}-{seed}.jsonl"
            rows = [
                {
                    "data_source": "correlated",
                    "sample_id": f"q{i}",
                    "sampling_seed": seed,
                    "acc": int(i == (1 if arm == "B" else 0)),
                    "omission": int(i == 1),
                }
                for i in range(2)
            ]
            # Known nonzero contrasts with a seed effect shared across arms.
            rows.append(
                {
                    "data_source": "effects",
                    "sample_id": "q0",
                    "sampling_seed": seed,
                    "acc": {"A": 0.0, "B": 0.25, "C": 0.5, "D": 0.875}[arm] + (seed - 42) / 32,
                    "omission": {"A": 0.75, "B": 0.75, "C": 0.5, "D": 0.25}[arm],
                }
            )
            write_results(path, list(reversed(rows)) if arm in ("B", "D") else rows)
            arms[arm].append(path)
    return arms


def test_bootstrap_resamples_questions_with_all_their_seeds(arm_paths):
    report = analyze_2x2_interaction(arm_paths, resamples=1000)
    assert report == analyze_2x2_interaction(arm_paths, resamples=1000)
    assert report["decode_seeds"] == [42, 43, 44]
    benchmark = report["benchmarks"]["correlated"]
    assert benchmark["questions"] == 2
    effects = benchmark["metrics"]["acc"]
    # Two independent clusters contribute -1 and +1 for all three seeds.
    # Drawing two clusters gives -1, 0, +1 with probabilities 1/4, 1/2, 1/4.
    # Resampling the six rows independently would yield a narrower interval.
    assert effects["occlusion"] == {"estimate": 0.0, "ci95": [-1.0, 1.0]}
    assert effects["decoupling"] == {"estimate": 0.0, "ci95": [0.0, 0.0]}
    assert effects["interaction"] == {"estimate": 0.0, "ci95": [-1.0, 1.0]}


def test_cli_reports_signed_contrasts_by_benchmark_and_metric(arm_paths, tmp_path):
    script = Path(__file__).resolve().parents[1] / "analyze_2x2_interaction.py"
    output = tmp_path / "report.json"
    command = [sys.executable, str(script)]
    for arm, paths in arm_paths.items():
        command.extend([f"--{arm.lower()}", *map(str, paths)])
    subprocess.run(
        [
            *command,
            "--metric",
            "acc",
            "--metric",
            "omission",
            "--seeds",
            "42",
            "43",
            "44",
            "--resamples",
            "20",
            "--output",
            str(output),
        ],
        check=True,
    )
    effects = json.loads(output.read_text())["benchmarks"]["effects"]["metrics"]
    assert effects["acc"] == {
        "occlusion": {"estimate": 0.25, "ci95": [0.25, 0.25]},
        "decoupling": {"estimate": 0.5, "ci95": [0.5, 0.5]},
        "interaction": {"estimate": 0.125, "ci95": [0.125, 0.125]},
    }
    assert effects["omission"]["interaction"]["estimate"] == -0.25


def test_unmatched_duplicate_and_incomplete_inputs_fail(arm_paths):
    path = arm_paths["B"][0]
    data = path.read_text()
    first, rest = data.split("\n", 1)
    path.write_text(rest)
    with pytest.raises(ValueError, match="Arm B differs from A"):
        analyze_2x2_interaction(arm_paths)
    path.write_text(data + first + "\n")
    with pytest.raises(ValueError, match="duplicate question/seed key"):
        analyze_2x2_interaction(arm_paths)
    for paths in arm_paths.values():
        path = paths[0]
        path.write_text(
            "".join(
                line + "\n" for line in path.read_text().splitlines() if json.loads(line)["data_source"] != "effects"
            )
        )
    with pytest.raises(ValueError, match="Incomplete or unexpected decode seeds"):
        analyze_2x2_interaction(arm_paths)
