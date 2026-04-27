from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenization import Tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
SYSTEMS_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from the TinyStories CS336 checkpoint.")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SYSTEMS_ROOT / "data/checkpoint/cs336_tinystories_20260408_235037.pt",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=SYSTEMS_ROOT / "data/tokenizer_outputs/tinystories_full_20260408",
    )
    return parser.parse_args()


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    special_tokens = json.loads((tokenizer_dir / "special_tokens.json").read_text())
    return Tokenizer.from_files(
        tokenizer_dir / "vocab.json",
        tokenizer_dir / "merges.txt",
        special_tokens=special_tokens,
    )


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.tokenizer_dir)
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,
        device=args.device,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = ids[:, -model.context_length :]
            logits = model(context)[:, -1, :]
            next_id = sample_next_token(logits, temperature=args.temperature, top_k=args.top_k)
            ids = torch.cat([ids, next_id], dim=-1)

    print(tokenizer.decode(ids[0].tolist()))


if __name__ == "__main__":
    main()
