"""Download the named benchmark splits in the layout expected by the converter."""

from argparse import ArgumentParser
from pathlib import Path

from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    for name, repository, split, expected in (
        ('geoqa', 'minglingfeng/geoqa_test', 'test', 740),
        ('mathvista', 'AI4Math/MathVista', 'testmini', 1000),
    ):
        dataset = load_dataset(repository, split=split)
        if len(dataset) != expected:
            raise ValueError(f'{repository} {split}: expected {expected}, found {len(dataset)}')
        DatasetDict({split: dataset}).save_to_disk(str(args.output / 'benchmarks' / name))

    snapshot_download(
        repo_id='MathLLMs/MathVision', repo_type='dataset',
        revision='2837ddb3f13abaf6b3997c12d80753e5470bd46a',
        allow_patterns='data/testmini-*.parquet', local_dir=args.output / 'mathvision',
    )


if __name__ == '__main__':
    main()
