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

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ----------------------------------------------------------------------
# Global setup
# ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# GPT-2 BPE tokenizer (readymade)
TOKENIZER = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = TOKENIZER.n_vocab  # 50257


# ----------------------------------------------------------------------
# Positional Encoding
# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ----------------------------------------------------------------------
# Model: decoder-only transformer (encoder stack + causal mask)
# ----------------------------------------------------------------------
class TinyLLM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, nhead=8,
                 num_layers=6, dim_feedforward=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

        # Weight tying: share embedding & output weights (saves ~12.9M params)
        self.output_head.weight = self.token_embedding.weight

    def forward(self, idx):
        seq_len = idx.size(1)
        x = self.token_embedding(idx) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_encoder(x))
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=idx.device),
            diagonal=1,
        )
        x = self.transformer(x, mask=mask)
        return self.output_head(self.ln_f(x))


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_data(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Text file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = TOKENIZER.encode(text, allowed_special={"<|endoftext|>"})
    print(f"Loaded {len(text):,} characters -> {len(tokens):,} tokens")
    return torch.tensor(tokens, dtype=torch.long)


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train(args):
    data = load_data(args.data)
    if len(data) < args.block_size + 2:
        raise ValueError("Dataset too small for the given block size.")

    # Config saved with the checkpoint so inference rebuilds an identical model
    config = dict(
        vocab_size=VOCAB_SIZE, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        max_len=args.block_size, dropout=args.dropout,
    )

    model = TinyLLM(**config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        xb, yb = get_batch(data, args.block_size, args.batch_size)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_interval == 0 or step == 1:
            print(f"Step {step:>5}/{args.steps} | loss {loss.item():.4f}")

    # Save weights + config together
    torch.save({"model_state": model.state_dict(), "config": config}, args.out)
    print(f"\n✓ Model saved to: {args.out}")


# ----------------------------------------------------------------------
# Generation / Inference
# ----------------------------------------------------------------------
@torch.no_grad()
def generate(args):
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    ckpt = torch.load(args.model, map_location=DEVICE)
    model = TinyLLM(**ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor(
        [TOKENIZER.encode(args.prompt)], dtype=torch.long, device=DEVICE
    )

    for _ in range(args.max_new_tokens):
        idx_cond = idx[:, -model.max_len :]
        logits = model(idx_cond)[:, -1, :] / args.temperature

        if args.top_k > 0:
            v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    print("\n" + "=" * 60)
    print(TOKENIZER.decode(idx[0].tolist()))
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
