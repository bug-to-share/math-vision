"""Prepare immutable matched inputs and evaluate checkpoints on isolated GPUs."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in path.open()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run, config_path, models):
    run.mkdir(parents=True, exist_ok=False)
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from recipe.papo.eval_dataset import PAPOEvaluationDataset
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids
    from verl.utils.tokenizer.chat_template import apply_chat_template
    from verl.workers.rollout.utils import qwen2_5_vl_dedup_image_tokens

    data = OmegaConf.load(config_path).data
    data.shuffle = False
    processor = AutoProcessor.from_pretrained(models["base4b"], local_files_only=True)
    tokenizer = processor.tokenizer
    for label, path in models.items():
        other = AutoProcessor.from_pretrained(path, local_files_only=True)
        assert other.tokenizer.get_vocab() == tokenizer.get_vocab(), label
        assert other.chat_template == processor.chat_template, label
        for key in ("patch_size", "merge_size", "size", "min_pixels", "max_pixels", "image_mean", "image_std"):
            assert getattr(other.image_processor, key, None) == getattr(processor.image_processor, key, None), (label, key)
    image_dir = run / "images"
    image_dir.mkdir(exist_ok=True)

    def encode(messages, sample_id, provided_images=None):
        images, videos, audios = RLHFDataset._process_multi_modal_info(messages, image_patch_size=16, config=data)
        assert not videos and not audios
        if provided_images is not None:
            images = provided_images
        prompt_text = apply_chat_template(processor, messages, add_generation_prompt=True, tokenize=False,
                                         **data.apply_chat_template_kwargs)
        inputs = build_multimodal_processor_inputs(processor, text=[prompt_text], images=images or None,
                                                   mm_processor_kwargs=dict(data.mm_processor_kwargs))
        ids = normalize_token_ids(inputs["input_ids"])
        assert len(ids) <= 2048, (sample_id, len(ids))
        paths = []
        for i, im in enumerate(images or []):
            path = image_dir / f"{sample_id.replace(':', '_').replace('/', '_')}_{i}.png"
            im.save(path)
            paths.append(str(path))
        return {"prompt_text": prompt_text, "prompt_token_ids": qwen2_5_vl_dedup_image_tokens(ids, processor),
                "expanded_prompt_length": len(ids), "image_paths": paths,
                "image_sha256": [sha(Path(p)) for p in paths]}

    benchmarks = []
    dataset = PAPOEvaluationDataset(list(data.val_files), tokenizer, data, processor=processor)
    assert len(dataset) == 2645, len(dataset)
    for index in range(len(dataset)):
        row = dataset[index]
        sid = f"benchmark:{index}"
        benchmarks.append({"sample_id": sid, "index": index, "suite": "benchmark", "condition": "original",
                           "data_source": row["data_source"], "ground_truth": str(row["reward_model"]["ground_truth"]),
                           **encode(row["raw_prompt"], sid)})
        if (index + 1) % 250 == 0:
            print(f"Prepared benchmark {index+1}/{len(dataset)}", flush=True)

    with (run / "benchmark-inputs.jsonl").open("x") as stream:
        for row in benchmarks:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    protocol = {"models": models, "benchmark_seeds": [42, 43, 44], "benchmark_sampling": {
        "temperature": .7, "top_p": .8, "top_k": 20, "n": 1, "max_tokens": 4096,
        "presence_penalty": 1.5, "repetition_penalty": 1., "frequency_penalty": 0., "ignore_eos": False},
        "engine": {"dtype": "bfloat16", "max_model_len": 6144, "tensor_parallel_size": 1,
        "gpu_memory_utilization": .75, "max_num_seqs": 64, "max_num_batched_tokens": 4096,
        "enforce_eager": True, "enable_chunked_prefill": True, "enable_prefix_caching": True, "seed": 42},
        "original_config": str(config_path), "original_config_sha256": sha(config_path),
        "dataset_sha256": {p: sha(Path(p)) for p in data.val_files},
        "input_sha256": {"benchmark-inputs.jsonl": sha(run / "benchmark-inputs.jsonl")},
        "counts": {"benchmark": len(benchmarks)},
        "seed_rule": "seed + 1000003 * immutable global input index", "code_sha256": sha(Path(__file__))}
    dump(run / "protocol.json", protocol)
    print(json.dumps(protocol["counts"]), flush=True)


def score(text, row):
    from recipe.papo.eval_reward import compute_score
    return compute_score(row["data_source"], text, row["ground_truth"])


async def evaluate(run, label, seeds=None):
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from verl.workers.rollout.utils import get_vision_placeholder_token_ids

    protocol = json.loads((run / "protocol.json").read_text())
    for name, expected in protocol["input_sha256"].items():
        assert sha(run / name) == expected, name
    model = protocol["models"][label]
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    tokenizer = processor.tokenizer
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=model, tokenizer=model, **protocol["engine"],
        generation_config="vllm", limit_mm_per_prompt={"image": 3}, distributed_executor_backend="mp",
        worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"))
    try:
        await engine.reset_mm_cache()
        await engine.collective_rpc(method="monkey_patch_model", kwargs={"vocab_size": len(tokenizer),
            "banned_token_ids": get_vision_placeholder_token_ids(processor)})
        chosen = protocol["benchmark_seeds"] if seeds is None else seeds
        if not set(chosen).issubset(protocol["benchmark_seeds"]):
            raise ValueError("Requested seed is outside the frozen protocol")
        phases = [("benchmark", seed) for seed in chosen]
        for suite, seed in phases:
            inputs = rows(run / f"{suite}-inputs.jsonl")
            output = run / f"{label}-{suite}-seed{seed}.jsonl"
            if output.exists():
                raise FileExistsError(output)
            status_path = run / f"{label}-status.json"
            started = time.monotonic()
            completed = 0
            semaphore = asyncio.Semaphore(protocol["engine"]["max_num_seqs"])
            print(f"Starting {label}/{suite}/{seed}: {len(inputs)}", flush=True)
            with output.open("x", buffering=1) as stream:
                async def one(row):
                    nonlocal completed
                    async with semaphore:
                        prompt = {"prompt_token_ids": row["prompt_token_ids"]}
                        images = []
                        for path, expected in zip(row["image_paths"], row["image_sha256"], strict=True):
                            assert sha(Path(path)) == expected
                            with Image.open(path) as im:
                                images.append(im.copy())
                        if images:
                            prompt["multi_modal_data"] = {"image": images}
                        sampling = dict(protocol[f"{suite}_sampling"])
                        request_seed = seed + 1000003 * row["index"]
                        params = SamplingParams(**sampling, seed=request_seed, output_kind=RequestOutputKind.FINAL_ONLY)
                        t0 = time.monotonic()
                        final = None
                        async for result in engine.generate(prompt, params, request_id=f"{label}:{seed}:{row['sample_id']}"):
                            if result.finished:
                                final = result
                        assert final is not None and len(final.outputs) == 1
                        completion = final.outputs[0]
                        text = tokenizer.decode(completion.token_ids, skip_special_tokens=True)
                        scores = await asyncio.to_thread(score, text, row)
                        record = {k: v for k, v in row.items() if k not in ("prompt_text", "prompt_token_ids", "image_paths")}
                        record.update(model=label, sampling_seed=seed, request_seed=request_seed, output=text,
                                      output_token_ids=list(completion.token_ids), output_tokens=len(completion.token_ids),
                                      prompt_tokens=len(final.prompt_token_ids), finish_reason=completion.finish_reason,
                                      length_capped=completion.finish_reason == "length", generation_seconds=time.monotonic()-t0,
                                      **scores)
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        completed += 1
                        dump(status_path, {"state": "running", "suite": suite, "seed": seed, "completed": completed,
                                           "total": len(inputs), "elapsed_seconds": time.monotonic()-started})
                        if completed % 100 == 0 or completed == len(inputs):
                            print(f"{label}/{suite}/{seed} {completed}/{len(inputs)} {time.monotonic()-started:.1f}s", flush=True)
                await asyncio.gather(*(one(row) for row in inputs))
            dump(run / f"{label}-{suite}-seed{seed}-complete.json", {"completed": completed, "seconds": time.monotonic()-started})
        dump(run / f"{label}-status.json", {"state": "completed", "phases": phases})
    except BaseException as error:
        dump(run / f"{label}-error.json", {"error": repr(error)})
        raise
    finally:
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "evaluate"])
    parser.add_argument("run", type=Path)
    parser.add_argument("--model", choices=["papo120", "dapo120", "base4b"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--dapo-model", type=Path)
    parser.add_argument("--mask-model", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.config or not args.base_model:
            parser.error("prepare requires --config and --base-model")
        models = {label: str(path.resolve()) for label, path in (
            ("base4b", args.base_model), ("dapo120", args.dapo_model), ("papo120", args.mask_model),
        ) if path is not None}
        prepare(args.run.resolve(), args.config, models)
    else:
        if not args.model:
            parser.error("evaluate requires --model")
        asyncio.run(evaluate(args.run, args.model, seeds=args.seeds))
