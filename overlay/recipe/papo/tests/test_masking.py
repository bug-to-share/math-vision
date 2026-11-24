from types import SimpleNamespace

import pytest
import torch
from recipe.papo.masking import mask_image_inputs


def processor():
    return SimpleNamespace(
        patch_size=2, temporal_patch_size=2, do_normalize=True, image_mean=[0.2, 0.5, 0.8], image_std=[0.5, 0.25, 0.4]
    )


def inputs():
    return {
        "pixel_values": torch.arange(48 * 24).reshape(48, 24).float(),
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 8]]),
        "position_ids": torch.arange(30),
        "input_ids": torch.arange(30),
    }


def test_black_values_geometry_and_original_unchanged():
    source = inputs()
    snapshot = {key: value.clone() for key, value in source.items()}
    masked, fraction = mask_image_inputs(source, processor(), 1.0, 42, 0)
    expected = torch.tensor([-0.4, -2.0, -2.0]).repeat_interleave(8)
    torch.testing.assert_close(masked["pixel_values"], expected.expand_as(source["pixel_values"]))
    assert fraction == 1
    for key, value in snapshot.items():
        torch.testing.assert_close(source[key], value)
        if key != "pixel_values":
            assert masked[key] is source[key]
    assert masked["pixel_values"].data_ptr() != source["pixel_values"].data_ptr()


def test_seed_is_local_reproducible_and_independent_of_microbatch_partition():
    source = inputs()
    rng_state = torch.get_rng_state().clone()
    masked, _ = mask_image_inputs(source, processor(), 0.6, 42, 2)
    repeated, _ = mask_image_inputs(source, processor(), 0.6, 42, 2)
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert torch.equal(masked["pixel_values"], repeated["pixel_values"])
    parts = []
    for index, pixels in enumerate(source["pixel_values"].split([16, 32])):
        split = {**source, "pixel_values": pixels, "image_grid_thw": source["image_grid_thw"][index : index + 1]}
        parts.append(mask_image_inputs(split, processor(), 0.6, 42, 2)[0]["pixel_values"])
    assert torch.equal(masked["pixel_values"], torch.cat(parts))
    different, _ = mask_image_inputs(source, processor(), 0.6, 42, 3)
    assert not torch.equal(masked["pixel_values"], different["pixel_values"])
    unchanged, fraction = mask_image_inputs(source, processor(), 0.0, 42, 0)
    assert fraction == 0 and torch.equal(unchanged["pixel_values"], source["pixel_values"])


def test_invalid_video_and_geometry_fail_explicitly():
    with pytest.raises(ValueError, match="videos"):
        mask_image_inputs({**inputs(), "video_grid_thw": torch.ones(1, 3)}, processor(), 0.6, 42, 0)
    with pytest.raises(ValueError, match="number of pixel patches"):
        mask_image_inputs({**inputs(), "image_grid_thw": torch.tensor([[1, 2, 2]])}, processor(), 0.6, 42, 0)
