"""Blacken processed Qwen3-VL patches without changing visual token geometry."""

import hashlib
import math
from typing import Any, Mapping, Protocol

import torch


class ImageProcessor(Protocol):
    patch_size: int
    temporal_patch_size: int
    image_mean: list[float]
    image_std: list[float]
    do_normalize: bool


def validate_mask_strategy(mask_strategy: str, attention_temperature: float, attention_mix_alpha: float) -> None:
    if mask_strategy not in {"random", "high_attention", "hybrid_attention"}:
        raise ValueError("PAPO mask_strategy must be random, high_attention or hybrid_attention")
    if not math.isfinite(attention_temperature) or attention_temperature <= 0:
        raise ValueError("PAPO attention_temperature must be finite and positive")
    if not 0 <= attention_mix_alpha <= 1:
        raise ValueError("PAPO attention_mix_alpha must be in [0, 1]")


def mask_image_inputs(
    model_inputs: Mapping[str, Any],
    image_processor: ImageProcessor,
    probability: float,
    seed: int,
    step: int,
    *,
    mask_strategy: str = "random",
    attention_weights: torch.Tensor | None = None,
    attention_temperature: float = 1.0,
    attention_mix_alpha: float = 0.1,
) -> tuple[dict[str, Any], float]:
    """Return a shallow input copy with independent pixels and a realized mask rate.

    Qwen packs each row as [channel, temporal_patch, patch_y, patch_x]. The
    processor's normalization determines black; zero is generally not black.
    A content/step seed keeps a given image's mask invariant to microbatch order,
    rollout repetition and resumption with restored optimizer state. Hashing the
    pixels entails a CPU copy, deliberately trading some speed for reproducibility.

    Attention scores follow the packed pixel row order. The attention strategies
    select the nearest whole number of patches (rounding halves up) per image.
    The hybrid strategy blends a softmax of the scores divided by temperature
    with a uniform distribution of mass attention_mix_alpha before sampling
    without replacement.
    """
    if not 0 <= probability <= 1:
        raise ValueError("mask probability must be in [0, 1]")
    validate_mask_strategy(mask_strategy, attention_temperature, attention_mix_alpha)
    if model_inputs.get("pixel_values_videos") is not None or model_inputs.get("video_grid_thw") is not None:
        raise ValueError("PAPO currently supports images, not videos")
    pixels = model_inputs.get("pixel_values")
    grid = model_inputs.get("image_grid_thw")
    if pixels is None or grid is None or grid.numel() == 0:
        raise ValueError("PAPO requires image pixel_values and image_grid_thw")
    if grid.ndim != 2 or grid.shape[1] != 3 or (grid <= 0).any() or (grid[:, 0] != 1).any():
        raise ValueError("PAPO requires positive [1, height, width] image grids")
    patch = image_processor.patch_size
    temporal = image_processor.temporal_patch_size
    mean = torch.as_tensor(image_processor.image_mean, device=pixels.device, dtype=pixels.dtype)
    std = torch.as_tensor(image_processor.image_std, device=pixels.device, dtype=pixels.dtype)
    width = mean.numel() * temporal * patch * patch
    if pixels.ndim != 2 or pixels.shape[1] != width or not pixels.is_floating_point():
        raise ValueError(f"Unexpected Qwen3-VL pixel layout: {pixels.shape}, expected (*, {width})")
    if not image_processor.do_normalize or (std <= 0).any():
        raise ValueError("PAPO requires a normalized Qwen3-VL image processor with positive std")
    lengths = grid.prod(-1).tolist()
    if sum(lengths) != pixels.shape[0]:
        raise ValueError("image_grid_thw does not match the number of pixel patches")
    scores_by_image = [None] * len(lengths)
    if mask_strategy != "random" and attention_weights is not None:
        if attention_weights.shape != (pixels.shape[0],) or attention_weights.is_complex():
            raise ValueError("attention_weights must contain one real score per pixel patch")
        scores = attention_weights.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(scores).all():
            raise ValueError("attention_weights must be finite")
        scores_by_image = scores.split(lengths)
    black = (-mean / std).repeat_interleave(temporal * patch * patch)
    masks = []
    for image, scores in zip(pixels.detach().split(lengths), scores_by_image, strict=True):
        count = image.shape[0]
        if mask_strategy != "high_attention":
            digest = hashlib.blake2b(digest_size=8)
            digest.update(f"{seed}:{step}:".encode())
            digest.update(image.float().cpu().contiguous().numpy().tobytes())
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int.from_bytes(digest.digest(), "little"))
        if mask_strategy == "random":
            masks.append(torch.rand(image.shape[0], generator=generator) < probability)
            continue

        selected_count = math.floor(count * probability + 0.5)
        if selected_count in (0, count):
            masks.append(torch.full((count,), selected_count == count, dtype=torch.bool, device="cpu"))
            continue
        if scores is None:
            # Within-channel contrast in pre-normalization pixel units. A
            # uniformly colored patch should score zero regardless of its hue.
            channels = image.to(device="cpu", dtype=torch.float64).reshape(count, mean.numel(), -1)
            pixel_std = torch.as_tensor(image_processor.image_std, dtype=torch.float64, device="cpu").square()
            scores = (channels.var(dim=-1, correction=0) * pixel_std).mean(dim=-1)
        if mask_strategy == "high_attention":
            selected = torch.argsort(scores, descending=True, stable=True)[:selected_count]
        else:
            weights = torch.softmax((scores - scores.max()) / attention_temperature, dim=0)
            weights = (1 - attention_mix_alpha) * weights + attention_mix_alpha / count
            # Even without a uniform floor, softmax underflow must not exclude
            # patches when the requested quota exceeds the nonzero weights.
            weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
            selected = torch.multinomial(weights, selected_count, replacement=False, generator=generator)
        mask = torch.zeros(count, dtype=torch.bool, device="cpu")
        mask[selected] = True
        masks.append(mask)
    mask = torch.cat(masks).to(pixels.device)
    result = dict(model_inputs)
    result["pixel_values"] = torch.where(mask[:, None], black[None, :], pixels)
    return result, mask.float().mean().item()
