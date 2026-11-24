"""Parse recognition blocks and align their contents with actual response tokens."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    text: str
    spans: tuple[tuple[int, int], ...]
    valid: bool


def extract_evidence(response: str) -> Evidence:
    # Evidence nested inside a reasoning block is not a separated observation.
    depth = 0
    for tag in re.finditer(r"</?(?:recognition|think|thinking)>", response):
        if tag.group() in ("<think>", "<thinking>"):
            depth += 1
        elif tag.group() in ("</think>", "</thinking>"):
            depth = max(depth - 1, 0)
        elif tag.group() == "<recognition>" and depth:
            return Evidence("", (), False)
    tags = list(re.finditer(r"</?recognition>", response))
    if not tags or len(tags) % 2:
        return Evidence("", (), False)
    spans = []
    for opening, closing in zip(tags[::2], tags[1::2], strict=True):
        if opening.group() != "<recognition>" or closing.group() != "</recognition>":
            return Evidence("", (), False)
        start, end = opening.end(), closing.start()
        if not response[start:end].strip():
            return Evidence("", (), False)
        # A recognition block may not contain a reasoning block.
        if re.search(r"</?think(?:ing)?>", response[start:end]):
            return Evidence("", (), False)
        spans.append((start, end))
    return Evidence("\n".join(response[a:b].strip() for a, b in spans), tuple(spans), True)


def recognition_token_mask(tokenizer, response_ids: list[int]) -> list[bool]:
    """Exclude delimiters/padding; fail explicitly if decoding cannot be aligned."""
    text = tokenizer.decode(response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    evidence = extract_evidence(text)
    if not evidence.valid:
        return [False] * len(response_ids)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != response_ids:
        raise ValueError("MoCA response tokens do not round-trip; refusing an ambiguous perception mask")
    return [
        end > start and any(a <= start and end <= b for a, b in evidence.spans)
        for start, end in encoded["offset_mapping"]
    ]
