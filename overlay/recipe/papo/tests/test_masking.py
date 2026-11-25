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


def selected_patches(source, masked):
    return (source["pixel_values"] != masked["pixel_values"]).any(dim=1).nonzero().flatten().tolist()


def test_high_attention_uses_per_image_top_k_and_keeps_qwen_layout():
    source = inputs()
    attention = torch.cat([torch.arange(16).flip(0), torch.arange(32)])
    masked, fraction = mask_image_inputs(
        source, processor(), 0.25, 42, 0, mask_strategy="high_attention", attention_weights=attention
    )
    chosen = [*range(4), *range(40, 48)]
    assert selected_patches(source, masked) == chosen
    assert fraction == 0.25
    assert masked["image_grid_thw"] is source["image_grid_thw"]
    black = torch.tensor([-0.4, -2.0, -2.0]).repeat_interleave(8)
    torch.testing.assert_close(masked["pixel_values"][chosen], black.expand(12, -1))


def test_high_attention_proxy_ranks_within_channel_contrast_and_breaks_ties_stably():
    source = {
        "pixel_values": torch.cat(
            [
                torch.tensor([10.0, 20.0, 30.0]).repeat_interleave(8).view(1, -1),
                torch.arange(24).view(1, -1) * 3.0,
                torch.arange(24).view(1, -1) * 3.0,
                torch.ones(1, 24),
            ]
        ),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    for step in (0, 3):
        masked, _ = mask_image_inputs(source, processor(), 0.25, 42, step, mask_strategy="high_attention")
        assert selected_patches(source, masked) == [1]


@pytest.mark.parametrize("use_scores", [False, True])
def test_hybrid_attention_is_local_reproducible_and_partition_invariant(use_scores):
    source = inputs()
    weights = torch.arange(48).float().div(12) if use_scores else None
    options = dict(mask_strategy="hybrid_attention", attention_mix_alpha=0.3, attention_temperature=0.7)
    rng_state = torch.get_rng_state()
    masked, fraction = mask_image_inputs(source, processor(), 0.25, 42, 2, attention_weights=weights, **options)
    assert fraction == 0.25
    assert torch.equal(
        masked["pixel_values"],
        mask_image_inputs(source, processor(), 0.25, 42, 2, attention_weights=weights, **options)[0]["pixel_values"],
    )
    changed, _ = mask_image_inputs(source, processor(), 0.25, 42, 3, attention_weights=weights, **options)
    assert not torch.equal(masked["pixel_values"], changed["pixel_values"])
    for index, (start, end) in enumerate(((0, 16), (16, 48))):
        part = {
            "pixel_values": source["pixel_values"][start:end],
            "image_grid_thw": source["image_grid_thw"][index : index + 1],
        }
        result, _ = mask_image_inputs(
            part, processor(), 0.25, 42, 2, attention_weights=weights[start:end] if use_scores else None, **options
        )
        torch.testing.assert_close(result["pixel_values"], masked["pixel_values"][start:end])
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_hybrid_uniform_floor_and_temperature_control_concentration():
    source = inputs()
    scores = torch.zeros(48)
    scores[[0, 16]] = 100.0

    def select(step, mix, temperature):
        result, _ = mask_image_inputs(
            source,
            processor(),
            0.05,
            42,
            step,
            mask_strategy="hybrid_attention",
            attention_weights=scores,
            attention_mix_alpha=mix,
            attention_temperature=temperature,
        )
        return selected_patches(source, result)

    for step in range(10):
        assert {0, 16} <= set(select(step, 0.0, 0.01))
        assert len(select(step, 0.0, 0.01)) == 3
    assert any(0 not in select(step, 1.0, 0.01) for step in range(10))
    assert any(0 not in select(step, 0.0, 1e6) for step in range(10))


@pytest.mark.parametrize("strategy", ["high_attention", "hybrid_attention"])
def test_attention_scores_must_match_packed_patch_count_and_be_finite(strategy):
    with pytest.raises(ValueError, match="one real score"):
        mask_image_inputs(inputs(), processor(), 0.6, 42, 0, mask_strategy=strategy, attention_weights=torch.ones(47))
    with pytest.raises(ValueError, match="finite"):
        mask_image_inputs(
            inputs(), processor(), 0.6, 42, 0, mask_strategy=strategy, attention_weights=torch.full((48,), torch.nan)
        )
