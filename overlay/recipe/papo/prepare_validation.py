"""Convert the already-downloaded HF validation split to the existing train schema."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="data/raw-geometry3k/data/validation-00000-of-00001.parquet"
    )
    parser.add_argument("--train-file", default="data/geo3k/train.parquet")
    parser.add_argument("--output", default="data/geo3k/validation.parquet")
    args = parser.parse_args()
    training_rows = pq.read_table(args.train_file).to_pylist()
    template = training_rows[0]
    instruction = template["prompt"][0]["content"].removeprefix(template["extra_info"]["question"])
    rows = []
    for index, example in enumerate(pq.read_table(args.source).to_pylist()):
        rows.append(
            {
                "data_source": template["data_source"],
                "ability": "math",
                "images": example["images"],
                "prompt": [{"role": "user", "content": example["problem"] + instruction}],
                "reward_model": {"style": "rule", "ground_truth": example["answer"]},
                "extra_info": {
                    "split": "validation",
                    "index": index,
                    "answer": example["answer"],
                    "question": example["problem"],
                },
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(args.train_file)), output)
    manifest = {
        "source": args.source,
        "rows": len(rows),
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "prompt_instruction": instruction,
        "note": "Full validation 300; use E0 clean-161 membership for the primary independent metric.",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    def pixel_hash(image):
        rgb = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()

    training_images = {pixel_hash(image) for row in training_rows for image in row["images"]}
    clean_rows = [row for row in rows if not any(pixel_hash(image) in training_images for image in row["images"])]
    clean_output = output.with_name(f"{output.stem}-clean{len(clean_rows)}.parquet")
    pq.write_table(pa.Table.from_pylist(clean_rows, schema=pq.read_schema(args.train_file)), clean_output)
    clean_manifest = {
        "source_validation": str(output),
        "training_file": args.train_file,
        "criterion": "No RGB-pixel-identical training image (including width and height)",
        "rows": len(clean_rows),
        "validation_indices": [row["extra_info"]["index"] for row in clean_rows],
        "sha256": hashlib.sha256(clean_output.read_bytes()).hexdigest(),
        "training_sha256": hashlib.sha256(Path(args.train_file).read_bytes()).hexdigest(),
    }
    clean_output.with_suffix(".manifest.json").write_text(json.dumps(clean_manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Clean validation: {len(clean_rows)} rows -> {clean_output}")


if __name__ == "__main__":
    main()
