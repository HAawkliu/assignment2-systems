from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .common import (
    GPT2_PRETOKEN_PATTERN,
    _build_special_pattern,
    _compute_stream_safe_prefix_length,
    _gpt2_bytes_to_unicode,
)


class Tokenizer:
    """GPT-2 style byte-pair tokenizer used by the assignment tests."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)

        self._id_to_bytes = dict(vocab)
        self._bytes_to_id: dict[bytes, int] = {}
        for token_id, token_bytes in self._id_to_bytes.items():
            if token_bytes in self._bytes_to_id:
                raise ValueError(f"Duplicate token bytes in vocab: {token_bytes!r}")
            self._bytes_to_id[token_bytes] = token_id
        self.id_to_bytes = self._id_to_bytes
        self.bytes_to_id = self._bytes_to_id

        self._merge_ranks: dict[tuple[bytes, bytes], int] = {}
        for rank, pair in enumerate(merges):
            self._merge_ranks.setdefault(pair, rank)
        self.merge_ranks = self._merge_ranks

        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        self._special_tokens_sorted = sorted(self.special_tokens, key=lambda token: (-len(token), token))
        self._special_token_to_id: dict[str, int] = {}
        for token in self.special_tokens:
            token_id = self._bytes_to_id.get(token.encode("utf-8"))
            if token_id is None:
                raise KeyError(f"Special token {token!r} not found in vocabulary.")
            self._special_token_to_id[token] = token_id
        self._max_special_token_length = max((len(token) for token in self._special_tokens_sorted), default=0)

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        byte_decoder = {char: byte for byte, char in _gpt2_bytes_to_unicode().items()}

        with open(vocab_path, encoding="utf-8") as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        vocab = {
            token_id: bytes(byte_decoder[char] for char in token_str)
            for token_str, token_id in gpt2_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_path, encoding="utf-8") as merges_f:
            for line in merges_f:
                parts = line.strip().split()
                if len(parts) == 2:
                    left, right = parts
                    merges.append(
                        (
                            bytes(byte_decoder[char] for char in left),
                            bytes(byte_decoder[char] for char in right),
                        )
                    )

        existing = set(vocab.values())
        for special_token in dict.fromkeys(special_tokens or []):
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in existing:
                vocab[len(vocab)] = special_bytes
                existing.add(special_bytes)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []

        ids: list[int] = []
        for segment, is_special in self._split_on_special_tokens(text):
            if segment == "":
                continue
            if is_special:
                ids.append(self._special_token_to_id[segment])
                continue
            for pretoken in self._pretokenize(segment):
                ids.extend(self._encode_pretoken(pretoken))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        output = bytearray()
        for token_id in ids:
            token_bytes = self._id_to_bytes.get(token_id)
            if token_bytes is None:
                raise KeyError(f"Token id {token_id} not found in vocabulary.")
            output.extend(token_bytes)
        return bytes(output).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        special_tokens = tuple(self._special_tokens_sorted)
        special_pattern = _build_special_pattern(special_tokens)

        for chunk in iterable:
            if chunk == "":
                continue
            buffer += chunk
            safe_len = _compute_stream_safe_prefix_length(
                text=buffer,
                special_tokens=special_tokens,
                special_pattern=special_pattern,
                max_special_token_length=self._max_special_token_length,
            )
            if safe_len > 0:
                yield from self.encode(buffer[:safe_len])
                buffer = buffer[safe_len:]

        if buffer:
            yield from self.encode(buffer)

    def _split_on_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        if text == "":
            return []
        if not self._special_tokens_sorted:
            return [(text, False)]

        segments: list[tuple[str, bool]] = []
        normal_start = 0
        i = 0
        while i < len(text):
            matched_token = next((token for token in self._special_tokens_sorted if text.startswith(token, i)), None)
            if matched_token is None:
                i += 1
                continue
            if normal_start < i:
                segments.append((text[normal_start:i], False))
            segments.append((matched_token, True))
            i += len(matched_token)
            normal_start = i

        if normal_start < len(text):
            segments.append((text[normal_start:], False))
        return segments

    def _pretokenize(self, text: str) -> list[str]:
        if text == "":
            return []
        return [match.group(0) for match in GPT2_PRETOKEN_PATTERN.finditer(text)]

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        pieces = [bytes([byte]) for byte in pretoken.encode("utf-8")]
        merged_pieces = self._apply_bpe_merges(pieces)
        return [self._bytes_to_id[piece] for piece in merged_pieces]

    def _apply_bpe_merges(self, pieces: list[bytes]) -> list[bytes]:
        if len(pieces) < 2:
            return pieces[:]

        word = pieces[:]
        while len(word) > 1:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break

            left, right = best_pair
            merged: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                    merged.append(left + right)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        return word
