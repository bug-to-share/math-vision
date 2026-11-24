"""Small text-only service check; this is not a calibration or training run."""

import argparse
import asyncio
import json
from pathlib import Path

from recipe.moca.verifier import VerifierClient


async def run(args):
    client = VerifierClient(
        {
            "base_url": args.base_url,
            "model": args.model,
            "pv_max_tokens": 2048,
            "svv_max_tokens": 2048,
            "max_concurrency": 1,
            "max_retries": 0,
        }
    )
    question = "What is the perimeter of triangle ABC?"
    evidence = "The three side lengths shown are AB=3, BC=4, and CA=5."
    proxy = await client.solve(question, evidence)
    positive = await client.score(question, "12", proxy["text"])
    negative = await client.score(question, "12", r"The perimeter is \boxed{11}.")
    absent = await client.solve(question, "No side lengths or other measurements are available.")
    report = {
        "purpose": "Four-request service/plumbing smoke; not judge-accuracy calibration",
        "model": args.model,
        "proxy": proxy,
        "positive": positive,
        "negative": negative,
        "no_evidence": absent,
        "checks": {
            "correct_answer_score": positive["score"] == 1,
            "wrong_answer_score": negative["score"] == 0,
            "missing_information_acknowledged": "INSUFFICIENT" in absent["text"],
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "checks": report["checks"]}, ensure_ascii=False))
    if not all(report["checks"].values()):
        raise SystemExit("Verifier smoke has failed checks; inspect the saved answers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="verifier")
    parser.add_argument("--output", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
