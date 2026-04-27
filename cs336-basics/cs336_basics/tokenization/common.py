from __future__ import annotations

from functools import lru_cache

import regex as re

GPT2_PRETOKEN_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@lru_cache(maxsize=1)
def _gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(n_) for n_ in cs], strict=True))


@lru_cache(maxsize=128)
def _build_special_pattern(special_tokens: tuple[str, ...]) -> re.Pattern[str] | None:
    if not special_tokens:
        return None
    sorted_tokens = sorted(special_tokens, key=lambda token: (-len(token), token))
    return re.compile("|".join(re.escape(token) for token in sorted_tokens))


def _split_non_special_segments(text: str, special_pattern: re.Pattern[str] | None) -> list[str]:
    if special_pattern is None or text == "":
        return [text]
    segments: list[str] = []
    start = 0
    for match in special_pattern.finditer(text):
        if start < match.start():
            segments.append(text[start : match.start()])
        start = match.end()
    if start < len(text):
        segments.append(text[start:])
    return segments


def _longest_incomplete_special_suffix(
    text: str,
    special_tokens: tuple[str, ...],
    max_special_token_length: int,
) -> int:
    if not text or not special_tokens or max_special_token_length <= 1:
        return 0
    for suffix_len in range(min(max_special_token_length - 1, len(text)), 0, -1):
        suffix = text[-suffix_len:]
        if any(len(token) > suffix_len and token.startswith(suffix) for token in special_tokens):
            return suffix_len
    return 0


def _compute_stream_safe_prefix_length(
    text: str,
    special_tokens: tuple[str, ...],
    special_pattern: re.Pattern[str] | None,
    max_special_token_length: int,
) -> int:
    if text == "":
        return 0

    holdback = _longest_incomplete_special_suffix(text, special_tokens, max_special_token_length)
    stable_end = len(text) - holdback
    if stable_end <= 0:
        return 0

    stable_text = text[:stable_end]
    tail_non_special = stable_text
    if special_pattern is not None:
        last_match_end = 0
        for match in special_pattern.finditer(stable_text):
            last_match_end = match.end()
        if last_match_end == len(stable_text):
            return stable_end
        tail_non_special = stable_text[last_match_end:]

    last_pretoken: str | None = None
    for match in GPT2_PRETOKEN_PATTERN.finditer(tail_non_special):
        last_pretoken = match.group(0)
    if last_pretoken is None:
        return stable_end
    return max(stable_end - len(last_pretoken), 0)
