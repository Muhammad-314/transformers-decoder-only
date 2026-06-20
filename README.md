# Tiny Transformer Decoder-Only (Educational Implementation)

A minimal decoder-only Transformer implementation built from scratch in PyTorch for educational purposes.

## Purpose

This repository was created as a learning project to deepen my understanding of:

* Transformer architectures
* Self-attention and causal masking
* Multi-head attention
* Positional embeddings
* Decoder-only language models
* The mathematical concepts presented in the original Transformer literature

The goal is educational understanding rather than production use.

## Important Disclaimer

**This implementation is not intended to be production-ready, optimized, novel, or state-of-the-art.**

The code was written primarily to help me study and internalize how decoder-only Transformers work by implementing the core components myself.

This repository:

* Is a learning exercise
* Prioritizes readability over efficiency
* Does not claim any research contribution or novelty
* Does not attempt to reproduce modern LLM training pipelines
* Omits many engineering optimizations used in real-world systems

If you are looking for production-grade implementations, consider projects such as Hugging Face Transformers, nanoGPT, or other mature open-source frameworks.

---

## Architecture

The model includes the fundamental building blocks of a decoder-only Transformer:

### Token Embeddings

Converts token IDs into dense vector representations.

### Positional Embeddings

Learned positional embeddings are added to token embeddings to provide sequence position information.

### Masked Self-Attention

Implements causal (autoregressive) attention so tokens can only attend to previous tokens and themselves.

### Multi-Head Attention

Multiple attention heads operate in parallel and are concatenated before projection.

### Feed-Forward Network (FFN)

A two-layer feed-forward network with ReLU activation.

### Residual Connections + Layer Normalization

Each Transformer block uses residual connections and layer normalization.

### Language Modeling Head

Projects decoder outputs back into vocabulary space to produce token logits.

---

## Model Structure

```text
Input Tokens
      │
      ▼
Token Embedding
      +
Position Embedding
      │
      ▼
Transformer Block × N
 ├─ Multi-Head Attention
 ├─ Residual + LayerNorm
 ├─ Feed Forward Network
 └─ Residual + LayerNorm
      │
      ▼
Linear LM Head
      │
      ▼
Vocabulary Logits
```

---

## Example Usage

```python
import torch

vocab_size = 1000

x = torch.randint(0, vocab_size, (2, 16))

model = TransformerLM(vocab_size)

logits = model(x)

print(logits.shape)
```

Output:

```python
torch.Size([2, 16, 1000])
```

---

## Concepts Implemented

* Decoder-only architecture
* Self-attention
* Query, Key, Value projections
* Scaled dot-product attention
* Causal masking
* Multi-head attention
* Feed-forward networks
* Layer normalization
* Residual connections
* Learned positional embeddings
* Language modeling output projection

---

## Limitations

This implementation intentionally keeps things simple and therefore lacks many features found in modern language models, including:

* Training loop
* Optimizers and schedulers
* Weight tying
* Dropout
* Flash Attention
* Rotary positional embeddings (RoPE)
* KV caching
* Mixed precision training
* Distributed training
* Tokenization pipeline
* Model checkpointing
* Inference optimizations

---

## Learning References

* "Attention Is All You Need" (Vaswani et al., 2017)
* The Illustrated Transformer by Jay Alammar
* PyTorch Documentation
* Andrej Karpathy's educational transformer resources

---

## Why This Repository Exists

I learn best by building things from scratch.

Rather than only reading papers or watching explanations, I wanted to implement the core components myself to gain a more intuitive understanding of how decoder-only Transformers process sequences and generate predictions.

This repository documents that learning journey.
