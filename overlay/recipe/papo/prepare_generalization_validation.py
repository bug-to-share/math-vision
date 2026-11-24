"""Prepare local, image-preserving generalization evaluation splits for PAPO.

Only evaluation rows are produced. The training Parquet supplies both the exact
Arrow schema and the existing reasoning/boxed-answer instruction. Structured
labels are JSON strings consumed by PAPO's evaluation-specific reward router.
"""

import argparse
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IMAGE_MARKER = re.compile(r"<image\d*>", re.IGNORECASE)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_hf_split(path, split):
    """Read embedded images as bytes, without decoding/re-encoding or HF caches."""
    files = sorted((path / split).glob("data-*.arrow"))
    if not files:
        raise FileNotFoundError(f"No Arrow shards in {path / split}")
    rows = []
    for file in files:
        with pa.memory_map(str(file), "r") as source:
            rows.extend(pa.ipc.open_stream(source).read_all().to_pylist())
    return rows, files


def image_record(value):
    if not isinstance(value, dict) or not value.get("bytes"):
        raise ValueError("Every evaluation image must have its original embedded bytes")
    return {"bytes": value["bytes"], "path": value.get("path")}


def pixel_hash(value):
    with Image.open(io.BytesIO(value["bytes"])) as source:
        rgb = source.convert("RGB")
        # Match prepare_validation.py's existing image-overlap definition.
        digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        return digest, rgb.size


def choice_letter(answer, choices):
    exact = [index for index, choice in enumerate(choices) if choice.strip() == answer.strip()]
    if not exact:
        raise ValueError(f"Gold answer must map to a choice: {answer!r}, {choices!r}")
    # Some upstream rows repeat an identical option. Preserve them; the scorer
    # compares the selected option content and the manifest records ambiguity.
    return chr(ord("A") + exact[0])


def question_and_target(name, example):
    """Construct prompts from question fields, never from the ground-truth answer."""
    if name == "geoqa":
        question = example.get("extra_info", {}).get("question")
        if not question:
            question = "\n".join(message["content"] for message in example["prompt"])
        # The local GeoQA copy already appends a different <think>/<answer> format.
        question = question.split("You FIRST think about the reasoning process", 1)[0].strip()
        answer = str(example["reward_model"]["ground_truth"])
        images = [image_record(image) for image in example["images"]]
        target = {"answer": answer, "choices": [], "answer_type": "math", "precision": None}
        origin_id = str(example["extra_info"]["index"])
        source_name = "GeoQA"
    elif name == "mathvista":
        question = example["question"].strip()
        answer = str(example["answer"])
        choices = list(example.get("choices") or [])
        kind = example["answer_type"]
        precision = example.get("precision")
        if precision is not None:
            if int(precision) != precision or precision < 0:
                raise ValueError(f"Invalid MathVista precision: {precision}")
            precision = int(precision)
        if example["question_type"] == "multi_choice":
            normalized_answer = choice_letter(answer, choices)
            kind = "choice"
        else:
            if choices or kind not in {"integer", "float", "list"}:
                raise ValueError(f"Unsupported MathVista answer schema: {example['pid']}")
            normalized_answer = answer
        target = {"answer": normalized_answer, "choices": choices, "answer_type": kind, "precision": precision}
        if example.get("unit"):
            question += f"\nAnswer unit: {example['unit']}."
        images = [image_record(example["decoded_image"])]
        origin_id = str(example["pid"])
        source_name = example["metadata"]["source"]
    elif name == "mathvision":
        question = example["question"].strip()
        answer = str(example["answer"])
        choices = list(example.get("options") or [])
        kind = "choice" if choices else "math"
        if choices and answer not in [chr(ord("A") + index) for index in range(len(choices))]:
            raise ValueError(f"Invalid MathVision choice label: {example['id']}: {answer!r}")
        target = {"answer": answer, "choices": choices, "answer_type": kind, "precision": None}
        images = [image_record(example["decoded_image"])]
        origin_id = str(example["id"])
        source_name = "MathVision"
    else:
        raise ValueError(name)
    if len(images) != 1:
        raise ValueError(f"Expected exactly one image in {name}:{origin_id}")
    if not question:
        raise ValueError(f"Empty question in {name}:{origin_id}")
    if not answer.strip():
        raise ValueError(f"Empty answer in {name}:{origin_id}")
    marker_count = len(IMAGE_MARKER.findall(question))
    # Numbered markers refer to labelled panels inside MathVision's single
    # composite image. Keep their meaning as plain text, with one actual image
    # placeholder for the embedded bytes below.
    question = IMAGE_MARKER.sub(
        lambda match: "[" + match.group()[1:-1] + "]" if re.search(r"\d", match.group()) else "", question
    ).strip()
    if target["choices"]:
        question += "\nChoices:\n" + "\n".join(
            f"({chr(ord('A') + index)}) {choice}" for index, choice in enumerate(target["choices"])
        )
        question += "\n" + r"For the final answer, put only the correct option letter in \boxed{}."
    elif target["answer_type"] == "float":
        if target["precision"] is None:
            raise ValueError(f"Float question without precision: {name}:{origin_id}")
        question += (
            f"\nGive the final numeric value rounded to {target['precision']} decimal places; "
            r"put only that value in \boxed{}."
        )
    elif target["answer_type"] == "integer":
        question += "\n" + r"Give the final integer value; put only that value in \boxed{}."
    elif target["answer_type"] == "list":
        question += "\n" + r"Give the final Python list; put only that list in \boxed{}."
    else:
        question += "\n" + r"Put only the final answer value or expression in \boxed{}."
    question = "<image>" + question
    return question, target, answer, images, origin_id, source_name, marker_count


def prepare(data_root, train_file, output_dir):
    train_table = pq.read_table(train_file)
    schema = train_table.schema
    train_rows = train_table.to_pylist()
    template = train_rows[0]
    prompt = template["prompt"][0]["content"]
    question = template["extra_info"]["question"]
    if not prompt.startswith(question):
        raise ValueError("Cannot identify the existing Geo3K prompt instruction")
    instruction = prompt.removeprefix(question)
    if "<thinking>" not in instruction or "\\boxed" not in instruction:
        raise ValueError("Unexpected Geo3K reasoning/answer format")
    training_hashes = defaultdict(list)
    for row in train_rows:
        for image in row["images"]:
            digest, _ = pixel_hash(image)
            training_hashes[digest].append(row["extra_info"]["index"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file": str(train_file.resolve()),
        "train_sha256": file_hash(train_file),
        "train_rows": len(train_rows),
        "prompt_instruction": instruction,
        "target_schema": {
            "answer": "string",
            "choices": "list[string]",
            "answer_type": "string",
            "precision": "int|null",
        },
        "overlap_criterion": "RGB pixels and dimensions exactly equal to a Geo3K training image; retain all rows",
        "notes": [
            "Exact image overlap does not detect resized, cropped, recompressed-lossy, or redrawn images.",
            "These are benchmark test/testmini splits used for monitoring; they are not new training data.",
            "Data sources are reported independently; these scores do not establish clean OOD generalization.",
            "All original image bytes are retained without resizing or recompression.",
            "Numbered image references become [imageN] text identifying panels in the original composite image.",
        ],
        "datasets": {},
    }
    for name, split, expected_rows in (
        ("geoqa", "test", 740),
        ("mathvista", "testmini", 1000),
        ("mathvision", "testmini", 304),
    ):
        if name == "mathvision":
            files = sorted((data_root / name / "data").glob(f"{split}-*.parquet"))
            examples = [row for file in files for row in pq.read_table(file).to_pylist()]
        else:
            examples, files = read_hf_split(data_root / "benchmarks" / name, split)
        if len(examples) != expected_rows:
            raise ValueError(f"{name}: expected {expected_rows} rows, found {len(examples)}")
        rows, overlap_rows, anomalies, origin_ids = [], [], [], []
        types, sources, sizes, markers = Counter(), Counter(), Counter(), Counter()
        within_split_hashes = Counter()
        for index, example in enumerate(examples):
            question, target, answer, images, origin_id, source_name, marker_count = question_and_target(name, example)
            digest, size = pixel_hash(images[0])
            within_split_hashes[digest] += 1
            sizes[size] += 1
            markers[marker_count] += 1
            types[target["answer_type"]] += 1
            sources[source_name] += 1
            origin_ids.append(origin_id)
            if target["choices"]:
                equivalent = [
                    chr(ord("A") + choice_index)
                    for choice_index, choice in enumerate(target["choices"])
                    if choice.strip() == target["choices"][ord(target["answer"]) - ord("A")].strip()
                ]
                if len(equivalent) > 1:
                    anomalies.append(
                        {
                            "index": index,
                            "source_id": origin_id,
                            "kind": "duplicate_correct_option_content",
                            "equivalent_letters": equivalent,
                        }
                    )
            if digest in training_hashes:
                overlap_rows.append(
                    {"index": index, "source_id": origin_id, "training_indices": training_hashes[digest]}
                )
            if marker_count > 1:
                anomalies.append(
                    {
                        "index": index,
                        "source_id": origin_id,
                        "kind": "composite_image_numbered_panel_references",
                        "count": marker_count,
                    }
                )
            if not target["choices"] and re.fullmatch(r"[A-Z]", answer):
                anomalies.append(
                    {
                        "index": index,
                        "source_id": origin_id,
                        "kind": "letter_answer_without_text_choices",
                        "answer": answer,
                    }
                )
            rows.append(
                {
                    "images": images,
                    "data_source": f"papo_eval/{name}",
                    "prompt": [{"role": "user", "content": question + instruction}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": json.dumps(target, ensure_ascii=False)},
                    "extra_info": {"split": split, "index": index, "answer": answer, "question": question},
                }
            )
        if len(set(origin_ids)) != len(origin_ids):
            raise ValueError(f"Non-unique original IDs in {name}")
        table = pa.Table.from_pylist(rows, schema=schema)
        output = output_dir / f"{name}-{split}.parquet"
        temporary = output.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary)
        if not pq.read_schema(temporary).equals(schema, check_metadata=True):
            raise ValueError(f"Output schema differs from training data: {output}")
        temporary.replace(output)
        result = {
            "split": split,
            "rows": len(rows),
            "data_source": f"papo_eval/{name}",
            "source_files": [{"path": str(file.resolve()), "sha256": file_hash(file)} for file in files],
            "output": str(output.resolve()),
            "sha256": file_hash(output),
            "answer_types": dict(types),
            "metadata_sources": dict(sources),
            "original_ids_in_row_order": origin_ids,
            "raw_image_marker_counts": dict(markers),
            "image_size_min": [min(size[axis] for size in sizes) for axis in (0, 1)],
            "image_size_max": [max(size[axis] for size in sizes) for axis in (0, 1)],
            "image_decode_errors": 0,
            "pixel_duplicate_training_rows": len(overlap_rows),
            "pixel_duplicate_training_details": overlap_rows,
            "duplicate_image_rows_within_split": sum(count - 1 for count in within_split_hashes.values()),
            "anomalies": anomalies,
        }
        manifest["datasets"][name] = result
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "data_source",
                        "rows",
                        "output",
                        "answer_types",
                        "pixel_duplicate_training_rows",
                        "anomalies",
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    output = output_dir / "manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw-benchmarks")
    parser.add_argument("--train-file", type=Path, default=ROOT / "data/geo3k/train.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/papo-eval")
    args = parser.parse_args()
    prepare(args.data_root, args.train_file, args.output_dir)


if __name__ == "__main__":
    main()
