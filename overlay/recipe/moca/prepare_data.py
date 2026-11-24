"""Prepare separate MoCA input files without changing existing benchmark files."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from recipe.moca.prompts import POLICY_PROMPT


def prepare_row(original: dict, training: bool) -> dict:
    row = copy.deepcopy(original)
    extra = dict(row.get("extra_info") or {})
    if training and row["data_source"] != "hiyouga/geometry3k":
        raise ValueError("Initial MoCA training data must be Geo3K")
    # Geo3K provides a clean question separate from its old <thinking> suffix.
    question = extra.get("question")
    if question is None:
        if training:
            raise ValueError("Missing clean question; refusing to infer it from labels or answers")
        # Generalization evaluation uses its original question and formatting.
        question = "\n".join(m["content"] for m in row["prompt"] if m["role"] == "user")
    extra["moca_question"] = question.replace("<image>", "").strip()
    extra["moca_train"] = training
    row["extra_info"] = extra
    if row["data_source"] == "hiyouga/geometry3k":
        row["prompt"] = [
            {"role": "system", "content": POLICY_PROMPT},
            {"role": "user", "content": question},
        ]
    else:
        row["prompt"].insert(0, {"role": "system", "content": POLICY_PROMPT})
    return row


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources = [args.train, *args.validation]
    names = ["train.parquet", *[f"validation-{i}.parquet" for i in range(1, len(sources))]]
    for source in sources:
        if not Path(source).is_file():
            raise FileNotFoundError(source)
    for name in [*names, "manifest.json"]:
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output / name}")
    manifest = []
    for index, (source, name) in enumerate(zip(sources, names, strict=True)):
        destination = output / name
        rows = pq.read_table(source).to_pylist()
        converted = [prepare_row(row, training=index == 0) for row in rows]
        pq.write_table(pa.Table.from_pylist(converted), destination)
        with Path(source).open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        with destination.open("rb") as stream:
            output_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest.append(
            {
                "source": source,
                "output": str(destination),
                "rows": len(rows),
                "source_sha256": source_hash,
                "output_sha256": output_hash,
            }
        )
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
