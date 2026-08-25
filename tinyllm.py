"""
simple_llm.py - Train and run inference with a GPT-style decoder LLM.

USAGE
-----
Train:
    python simple_llm.py train --data path/to/text.txt --out model.pt --steps 3000

Generate:
    python simple_llm.py generate --model model.pt --prompt "Once upon a time"

Requirements:
    pip install torch tiktoken
"""

import argparse
import math
import os
import sys
from pathlib import Path
import json
from dataclasses import asdict, dataclass

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from safetensors.torch import save_model
from torch import nn

# ----------------------------------------------------------------------
# Global setup
# ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# GPT-2 BPE tokenizer (readymade)
TOKENIZER = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = TOKENIZER.n_vocab  # 50257


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class TinyLLMConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    max_len: int = 512
    dropout: float = 0.1
    tie_weights: bool = True
    bias: bool = True


# ----------------------------------------------------------------------
# Attention with KV cache (fused SDPA / Flash Attention)
# ----------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        assert config.d_model % config.nhead == 0
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.d_model = config.d_model

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, kv_cache=None, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)

        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v) if use_cache else None

        # When generating one token at a time with a cache, no mask needed.
        is_causal = kv_cache is None and T > 1

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y, new_cache


# ----------------------------------------------------------------------
# MLP + Transformer Block (Pre-LN)
# ----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.dim_feedforward, bias=config.bias)
        self.proj = nn.Linear(config.dim_feedforward, config.d_model, bias=config.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


# ----------------------------------------------------------------------
# Model: decoder-only transformer
# ----------------------------------------------------------------------
class TinyLLM(nn.Module):
    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.config = config
        self.max_len = config.max_len  # kept for backward-compat access

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_len, config.d_model)  # learned
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (GPT-2 style: NO sqrt(d_model) scaling)
        if config.tie_weights:
            self.output_head.weight = self.token_embedding.weight

        # GPT-2 style init
        self.apply(self._init_weights)
        # Scaled residual init on residual projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches=None, use_cache=False):
        _, T = idx.shape

        past_len = 0
        if kv_caches is not None and kv_caches[0] is not None:
            past_len = kv_caches[0][0].size(2)

        assert past_len + T <= self.config.max_len, (
            f"Sequence length {past_len + T} exceeds max_len {self.config.max_len}"
        )

        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.token_embedding(idx) + self.pos_embedding(pos)
        x = self.drop(x)

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cache, use_cache)
            new_caches.append(new_cache)

        x = self.ln_f(x)

        if targets is not None:
            logits = self.output_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            return logits, loss

        # Inference: only need logits for the last position
        logits = self.output_head(x[:, [-1], :])
        return logits, (new_caches if use_cache else None)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_embedding.weight.numel()
        return n

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token_id=None,
    ):
        self.eval()
        kv_caches = None
        cur = idx

        for _ in range(max_new_tokens):
            # Stop if we would exceed the context window
            if (
                kv_caches[0][0].size(2) if kv_caches and kv_caches[0] else cur.size(1)
            ) >= self.config.max_len:
                break

            input_ids = cur if kv_caches is None else cur[:, -1:]
            logits, kv_caches = self(input_ids, use_cache=True, kv_caches=kv_caches)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                remove_scattered = remove.scatter(1, sorted_idx, remove)
                logits[remove_scattered] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            cur = torch.cat([cur, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return cur


# ----------------------------------------------------------------------
# Optimizer with weight-decay grouping
# ----------------------------------------------------------------------
def configure_optimizers(model, weight_decay=0.1, lr=3e-4, betas=(0.9, 0.95)):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2:  # biases, LayerNorm scales/shifts
            no_decay.append(p)
        else:  # matmul weights + embeddings
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


class RandomTokenDataset:
    def __init__(self, shard_dir, device=None):
        self.shard_dir = Path(shard_dir)
        self.device = device if device is not None else DEVICE

        meta_path = self.shard_dir / "meta.json"

        if not meta_path.is_file():
            raise FileNotFoundError(
                f"No meta.json found in {self.shard_dir}. "
                f"Run build_token_shards first."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.dtype = np.dtype(meta["dtype"])

        self.paths = [
            self.shard_dir / shard["file"]
            for shard in meta["shards"]
        ]

        self.lengths = np.asarray(
            [shard["tokens"] for shard in meta["shards"]],
            dtype=np.int64,
        )

        self.total_tokens = int(meta["total_tokens"])
        self._mmap_cache = {}

        print(
            f"Loaded token dataset: {len(self.paths)} shards, "
            f"{self.total_tokens:,} tokens"
        )

    def _get_mmap(self, shard_idx):
        mmap = self._mmap_cache.get(shard_idx)

        if mmap is None:
            mmap = np.memmap(
                self.paths[shard_idx],
                mode="r",
                dtype=self.dtype,
                shape=(int(self.lengths[shard_idx]),),
            )
            self._mmap_cache[shard_idx] = mmap

        return mmap

    def get_batch(self, block_size, batch_size):
        """
        Returns:
            x: [batch_size, block_size]
            y: [batch_size, block_size]
        """
        valid = self.lengths - block_size

        valid_shards = np.where(valid > 0)[0]

        if len(valid_shards) == 0:
            raise ValueError(
                f"No shard has enough tokens for block_size={block_size}"
            )

        # Sample shards proportional to number of valid starting positions.
        weights = valid[valid_shards].astype(np.float64)
        weights /= weights.sum()

        chosen_shards = np.random.choice(
            valid_shards,
            size=batch_size,
            replace=True,
            p=weights,
        )

        x = torch.empty((batch_size, block_size), dtype=torch.long)
        y = torch.empty((batch_size, block_size), dtype=torch.long)

        for b, shard_idx in enumerate(chosen_shards):
            shard_len = self.lengths[shard_idx]

            # Need block_size + 1 tokens to make x and y.
            start = np.random.randint(0, shard_len - block_size)

            mmap = self._get_mmap(shard_idx)

            seq = np.asarray(
                mmap[start : start + block_size + 1],
                dtype=np.int64,
            )

            x[b] = torch.from_numpy(seq[:-1])
            y[b] = torch.from_numpy(seq[1:])

        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def load_data(
    path,
    token_cache_dir=None,
    text_key="text",
    rebuild=False,
    shard_tokens=50_000_000,
):
    """
    Replacement for your old load_data.

    path:
        Directory containing many .jsonl.zst files.

    token_cache_dir:
        Directory where tokenized shards are stored. If None, uses:
            path/_token_shards

    text_key:
        JSON field containing the training text.

    rebuild:
        If True, rebuild token shards even if they already exist.
    """
    path = Path(path)

    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    if token_cache_dir is None:
        token_cache_dir = path / "_token_shards"

    token_cache_dir = Path(token_cache_dir)
    meta_path = token_cache_dir / "meta.json"

    if rebuild or not meta_path.is_file():
        build_token_shards(
            jsonl_zst_dir=path,
            out_dir=token_cache_dir,
            text_key=text_key,
            shard_tokens=shard_tokens,
        )

    return RandomTokenDataset(token_cache_dir, device=DEVICE)


def get_batch(data, block_size, batch_size):
    """
    Replacement for your old get_batch.

    data is now a RandomTokenDataset returned by load_data.
    """
    return data.get_batch(block_size, batch_size)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train(args):
    data = load_data("/home/vishpat/data/openwebtext2/openwebtext2/zst", token_cache_dir="/home/vishpat/data/openwebtext2/token_cache", rebuild=False)

    # Config saved with the checkpoint so inference rebuilds an identical model
    config = TinyLLMConfig(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        max_len=args.block_size,
        dropout=args.dropout,
    )

    model = TinyLLM(config).to(DEVICE)
    print(f"Model parameters: {model.num_params() / 1e6:.2f}M (non-embedding)")

    optimizer = configure_optimizers(model, weight_decay=0.1, lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        xb, yb = get_batch(data, args.block_size, args.batch_size)

        # New forward signature: pass targets to get (logits, loss)
        _, loss = model(xb, targets=yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_interval == 0 or step == 1:
            print(f"Step {step:>5}/{args.steps} | loss {loss.item():.4f}")

    # Save weights + config together (store config as a dict for portability)
    torch.save(
        {"model_state": model.state_dict(), "config": asdict(config)},
        args.out,
    )
    save_model(model, f"{args.out}.safetensors")
    print(f"\n✓ Model saved to: {args.out}")


# ----------------------------------------------------------------------
# Generation / Inference
# ----------------------------------------------------------------------
@torch.no_grad()
def generate(args):
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    ckpt = torch.load(args.model, map_location=DEVICE)
    config = TinyLLMConfig(**ckpt["config"])
    model = TinyLLM(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor([TOKENIZER.encode(args.prompt)], dtype=torch.long, device=DEVICE)

    # KV-cached sampling handled inside model.generate()
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=getattr(args, "top_p", None),
        eos_token_id=TOKENIZER.eot_token,  # 50256 for gpt2
    )

    print("\n" + "=" * 60)
    print(TOKENIZER.decode(out[0].tolist()))
    print("=" * 60)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="Simple GPT-style LLM")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    t = sub.add_parser("train", help="Train the model on a text file")
    t.add_argument("--data", required=True, help="Path to training .txt file")
    t.add_argument("--out", default="model.pt", help="Output checkpoint path")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--block_size", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--d_model", type=int, default=256)
    t.add_argument("--nhead", type=int, default=8)
    t.add_argument("--num_layers", type=int, default=6)
    t.add_argument("--dim_feedforward", type=int, default=1024)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--log_interval", type=int, default=100)

    # --- generate ---
    g = sub.add_parser("generate", help="Generate text from a prompt")
    g.add_argument("--model", required=True, help="Path to saved checkpoint")
    g.add_argument("--prompt", required=True, help="Starting text prompt")
    g.add_argument("--max_new_tokens", type=int, default=100)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top_k", type=int, default=40)
    g.add_argument("--top_p", type=float, default=None)

    return parser


def main():
    if not torch.cuda.is_available():
        print("Unable to use GPU")
        sys.exit(1)
    else:
        print("Will use GPU for training")

    args = build_parser().parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)


if __name__ == "__main__":
    main()
