"""Blacken processed Qwen3-VL patches without changing visual token geometry."""

import hashlib

import torch


def mask_image_inputs(model_inputs, image_processor, probability, seed, step):
    """Return a shallow input copy with independent pixels and a realized mask rate.

    Qwen packs each row as [channel, temporal_patch, patch_y, patch_x]. The
    processor's normalization determines black; zero is generally not black.
    A content/step seed keeps a given image's mask invariant to microbatch order,
    rollout repetition and resumption with restored optimizer state. Hashing the
    pixels entails a CPU copy, deliberately trading some speed for reproducibility.
    """
    if not 0 <= probability <= 1:
        raise ValueError("mask probability must be in [0, 1]")
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
    black = (-mean / std).repeat_interleave(temporal * patch * patch)
    masks = []
    for image in pixels.detach().split(lengths):
        digest = hashlib.blake2b(digest_size=8)
        digest.update(f"{seed}:{step}:".encode())
        digest.update(image.float().cpu().contiguous().numpy().tobytes())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest.digest(), "little"))
        masks.append(torch.rand(image.shape[0], generator=generator) < probability)
    mask = torch.cat(masks).to(pixels.device)
    result = dict(model_inputs)
    result["pixel_values"] = torch.where(mask[:, None], black[None, :], pixels)
    return result, mask.float().mean().item()
