"""Compute paired PAPO × MoCA contrasts from matched-evaluation JSONL files."""

import argparse
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

ARMS = ("A", "B", "C", "D")
EFFECTS = ("occlusion", "decoupling", "interaction")
RowKey = tuple[str, str, int]
Contrasts = tuple[float, float, float]


def load_arm(paths: Iterable[Path], metrics: tuple[str, ...]) -> dict[RowKey, tuple[float, ...]]:
    rows = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    source, sample, seed = row["data_source"], row["sample_id"], row["sampling_seed"]
                    if not (isinstance(source, str) and source and isinstance(sample, str) and sample):
                        raise ValueError("data_source and sample_id must be nonempty strings")
                    if type(seed) is not int:
                        raise ValueError("sampling_seed must be an integer")
                    scores = tuple(row[metric] for metric in metrics)
                    if not all(isinstance(score, (int, float)) and math.isfinite(score) for score in scores):
                        raise ValueError("metric values must be finite numbers")
                    key = (source, sample, seed)
                    if key in rows:
                        raise ValueError(f"duplicate question/seed key {key}")
                    rows[key] = tuple(float(score) for score in scores)
                except (KeyError, ValueError, TypeError, OverflowError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError("Every arm must contain at least one scored record")
    return rows


def percentile(ordered: list[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def paired_bootstrap(
    question_contrasts: list[Contrasts], resamples: int, bootstrap_seed: int
) -> dict[str, dict[str, float | list[float]]]:
    """Resample questions jointly for all arms and effects.

    Each contrast already averages the question's complete paired seed block,
    so selecting that question repeats all of its seeds as a unit.
    """
    count = len(question_contrasts)
    rng = random.Random(bootstrap_seed)
    draws = [[], [], []]
    for _ in range(resamples):
        sample = rng.choices(question_contrasts, k=count)
        for values, distribution in zip(zip(*sample, strict=True), draws, strict=True):
            distribution.append(math.fsum(values) / count)
    effects = {}
    for effect, values, distribution in zip(EFFECTS, zip(*question_contrasts, strict=True), draws, strict=True):
        distribution.sort()
        effects[effect] = {
            "estimate": math.fsum(values) / count,
            "ci95": [percentile(distribution, 0.025), percentile(distribution, 0.975)],
        }
    return effects


def analyze_2x2_interaction(
    arm_paths: dict[str, list[Path]],
    *,
    metrics: tuple[str, ...] = ("acc",),
    resamples: int = 10_000,
    bootstrap_seed: int = 42,
    expected_seeds: Iterable[int] | None = None,
) -> dict:
    if set(arm_paths) != set(ARMS):
        raise ValueError("Exactly four arms are required: A, B, C and D")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("Select at least one metric, without duplicates")
    arms = {arm: load_arm(arm_paths[arm], metrics) for arm in ARMS}
    keys = arms["A"].keys()
    for arm in ARMS[1:]:
        missing, extra = keys - arms[arm].keys(), arms[arm].keys() - keys
        if missing or extra:
            raise ValueError(
                f"Arm {arm} differs from A: {len(missing)} missing and {len(extra)} extra question/seed keys"
            )

    question_seeds = defaultdict(set)
    for source, sample, seed in keys:
        question_seeds[source, sample].add(seed)
    seeds = sorted(set(expected_seeds) if expected_seeds is not None else {key[2] for key in keys})
    if len(seeds) < 2:
        raise ValueError("At least two decode seeds are required")
    seed_set = set(seeds)
    summaries = {}
    for (source, sample), observed in sorted(question_seeds.items()):
        if observed != seed_set:
            raise ValueError(f"Incomplete or unexpected decode seeds for {(source, sample)}: {sorted(observed)}")
        summary = summaries.setdefault(source, {metric: [] for metric in metrics})
        for index, metric in enumerate(metrics):
            a, b, c, d = (
                math.fsum(arms[arm][source, sample, seed][index] for seed in seeds) / len(seeds) for arm in ARMS
            )
            summary[metric].append((b - a, c - a, (d - c) - (b - a)))

    return {
        "decode_seeds": seeds,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": resamples,
        "benchmarks": {
            source: {
                "questions": len(by_metric[metrics[0]]),
                "metrics": {
                    metric: paired_bootstrap(contrasts, resamples, bootstrap_seed)
                    for metric, contrasts in by_metric.items()
                },
            }
            for source, by_metric in summaries.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm.lower()}", nargs="+", type=Path, required=True, metavar="JSONL", help=f"Arm {arm}")
    parser.add_argument("--metric", action="append", help="Numeric row field to analyze. Repeatable (default: acc).")
    parser.add_argument("--seeds", nargs="+", type=int, help="Require these exact decode seeds for every question.")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Write JSON to this file (default: stdout).")
    args = parser.parse_args()
    try:
        results = analyze_2x2_interaction(
            {arm: getattr(args, arm.lower()) for arm in ARMS},
            metrics=tuple(args.metric or ["acc"]),
            resamples=args.resamples,
            bootstrap_seed=args.bootstrap_seed,
            expected_seeds=args.seeds,
        )
        output = json.dumps(results, indent=2) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
