# cs336_basics

A minimal CS336 assignment 1 implementation package intended as a clean carry-over for assignment 2.

It keeps only the code needed by the assignment 1 tests:

- model primitives: `Linear`, `Embedding`, `RMSNorm`, `SwiGLU`, RoPE, causal self-attention, Transformer block, Transformer LM
- training utilities: cross entropy, AdamW, LR schedule, gradient clipping, batch sampling, checkpoint save/load
- tokenizer utilities: GPT-2 style BPE `Tokenizer` and `train_bpe`
- adapter hooks: all tested `run_*` functions

It intentionally omits training scripts, generation code, KV cache, benchmark code, docs, and pipeline-specific helpers.

The main implementation is organized into `model`, `training`, and `tokenization` subpackages. Thin top-level compatibility modules such as `cs336_basics.layers` and `cs336_basics.tokenizer` re-export those implementations for older assignment 1 adapters.
