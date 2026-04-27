from __future__ import annotations

from collections import Counter
from pathlib import Path

from .common import GPT2_PRETOKEN_PATTERN, _build_special_pattern, _split_non_special_segments


def _merge_pair_in_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    left, right = pair
    merged: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
            merged.append(left + right)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def _pair_count_items(word: tuple[bytes, ...]) -> tuple[tuple[tuple[bytes, bytes], int], ...]:
    if len(word) < 2:
        return ()
    counts: Counter[tuple[bytes, bytes]] = Counter()
    for left, right in zip(word, word[1:], strict=False):
        counts[(left, right)] += 1
    return tuple(counts.items())


def _count_pretokens(text: str, special_tokens: tuple[str, ...]) -> Counter[tuple[bytes, ...]]:
    special_pattern = _build_special_pattern(special_tokens)
    word_freq: Counter[tuple[bytes, ...]] = Counter()
    for segment in _split_non_special_segments(text, special_pattern):
        for match in GPT2_PRETOKEN_PATTERN.finditer(segment):
            token_bytes = match.group(0).encode("utf-8")
            if token_bytes:
                word_freq[tuple(bytes([byte]) for byte in token_bytes)] += 1
    return word_freq


def _read_text(input_path: str | Path, max_lines: int | None) -> str:
    with open(input_path, encoding="utf-8") as f:
        if max_lines is None:
            return f.read()
        return "".join(line for _, line in zip(range(max_lines), f, strict=False))


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    *,
    max_lines: int | None = None,
    streaming: bool = False,
    stream_chunk_size: int = 1 << 20,
    progress: bool = False,
    log_interval_sec: float = 5.0,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    del streaming, stream_chunk_size, progress, log_interval_sec

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    token_values = set(vocab.values())
    for special_token in dict.fromkeys(special_tokens):
        special_bytes = special_token.encode("utf-8")
        if special_bytes not in token_values:
            vocab[len(vocab)] = special_bytes
            token_values.add(special_bytes)

    text = _read_text(input_path, max_lines=max_lines)
    word_freq = _count_pretokens(text, tuple(token for token in dict.fromkeys(special_tokens) if token))

    pair_freq: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], dict[tuple[bytes, ...], int]] = {}
    word_pair_items: dict[tuple[bytes, ...], tuple[tuple[tuple[bytes, bytes], int], ...]] = {}
    for word, freq in word_freq.items():
        items = _pair_count_items(word)
        word_pair_items[word] = items
        for pair, occ in items:
            pair_freq[pair] += occ * freq
            pair_to_words.setdefault(pair, {})[word] = occ

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and pair_freq:
        # Assignment tie-break: highest count, then lexicographically greatest pair.
        best_pair, _ = max(pair_freq.items(), key=lambda item: (item[1], item[0]))
        merges.append(best_pair)

        merged_token = best_pair[0] + best_pair[1]
        if merged_token not in token_values:
            vocab[len(vocab)] = merged_token
            token_values.add(merged_token)

        impacted_words = list(pair_to_words.pop(best_pair, {}).keys())
        pair_freq.pop(best_pair, None)
        impacted_word_freqs: list[tuple[tuple[bytes, ...], int]] = []
        for word in impacted_words:
            freq = word_freq.pop(word, 0)
            if freq > 0:
                impacted_word_freqs.append((word, freq))

        for word, freq in impacted_word_freqs:
            for pair, occ in word_pair_items.pop(word, ()):
                new_count = pair_freq.get(pair, 0) - occ * freq
                if new_count > 0:
                    pair_freq[pair] = new_count
                else:
                    pair_freq.pop(pair, None)
                words_for_pair = pair_to_words.get(pair)
                if words_for_pair is not None:
                    words_for_pair.pop(word, None)
                    if not words_for_pair:
                        pair_to_words.pop(pair, None)

        additions: Counter[tuple[bytes, ...]] = Counter()
        for word, freq in impacted_word_freqs:
            additions[_merge_pair_in_word(word, best_pair)] += freq

        for merged_word, freq in additions.items():
            word_freq[merged_word] += freq
            items = word_pair_items.get(merged_word)
            if items is None:
                items = _pair_count_items(merged_word)
                word_pair_items[merged_word] = items
            for pair, occ in items:
                pair_freq[pair] += occ * freq
                pair_to_words.setdefault(pair, {})[merged_word] = occ

    return vocab, merges
