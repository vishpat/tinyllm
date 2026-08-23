use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{
    embedding, layer_norm, linear, ops, Embedding, LayerNorm, Linear, Module, VarBuilder,
};
use candle_nn::{activation::Activation};
use clap::Parser;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::Ordering;
use std::path::PathBuf;
use tiktoken_rs::get_bpe_from_model;

#[derive(Debug, Clone)]
struct TinyLLMConfig {
    vocab_size: usize,
    d_model: usize,
    nhead: usize,
    num_layers: usize,
    dim_feedforward: usize,
    max_len: usize,
}

impl Default for TinyLLMConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50_257,
            d_model: 256,
            nhead: 8,
            num_layers: 6,
            dim_feedforward: 1024,

            // Your safetensors contains pos_embedding.weight [128, 256].
            // Therefore the real context length of this checkpoint is 128.
            max_len: 128,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to the safetensors model file.
    #[arg(long)]
    model: PathBuf,

    /// Prompt text.
    #[arg(long)]
    prompt: String,

    /// Maximum number of new tokens to generate.
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,

    /// Sampling temperature. Use 0 for greedy decoding.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Top-k sampling. Use 0 to disable top-k filtering.
    #[arg(long, default_value_t = 50)]
    top_k: usize,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Force CPU.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Stop when GPT-2 EOS token 50256 is generated.
    #[arg(long, default_value_t = true)]
    stop_eos: bool,
}

struct CausalSelfAttention {
    qkv: Linear,
    proj: Linear,
    nhead: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    fn load(cfg: &TinyLLMConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let qkv = linear(
            cfg.d_model,
            3 * cfg.d_model,
            vb.pp("qkv"),
        )?;

        let proj = linear(
            cfg.d_model,
            cfg.d_model,
            vb.pp("proj"),
        )?;

        Ok(Self {
            qkv,
            proj,
            nhead: cfg.nhead,
            head_dim: cfg.d_model / cfg.nhead,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let device = x.device();
        let dtype = x.dtype();

        let bsz;
        let seq_len;
        let d_model;
        {
            let dims = x.dims3()?;
            bsz = dims.0;
            seq_len = dims.1;
            d_model = dims.2;
        }

        let qkv = self.qkv.forward(x)?;

        let q = qkv.narrow(2, 0, d_model)?;
        let k = qkv.narrow(2, d_model, d_model)?;
        let v = qkv.narrow(2, 2 * d_model, d_model)?;

        let q = q
            .reshape((bsz, seq_len, self.nhead, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let k = k
            .reshape((bsz, seq_len, self.nhead, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let v = v
            .reshape((bsz, seq_len, self.nhead, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // q: [B, H, T, Hd]
        // k: [B, H, T, Hd]
        // att: [B, H, T, T]
        let k_t = k.transpose(2, 3)?;
        let att = q.matmul(&k_t)?;
        let att = (att / (self.head_dim as f64).sqrt())?;

        // Causal mask: positions cannot attend to future tokens.
        let mut mask = vec![0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask[i * seq_len + j] = -1.0e9;
                }
            }
        }

        let mask = Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?;
        let mask = if dtype != DType::F32 {
            mask.to_dtype(dtype)?
        } else {
            mask
        };

        let att = att.broadcast_add(&mask)?;
        let att = ops::softmax_last_dim(&att)?;

        let y = att.matmul(&v)?;
        let y = y
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bsz, seq_len, d_model))?;

        self.proj.forward(&y)
    }
}

struct Mlp {
    fc: Linear,
    proj: Linear,
}

impl Mlp {
    fn load(cfg: &TinyLLMConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let fc = linear(
            cfg.d_model,
            cfg.dim_feedforward,
            vb.pp("fc"),
        )?;

        let proj = linear(
            cfg.dim_feedforward,
            cfg.d_model,
            vb.pp("proj"),
        )?;

        Ok(Self { fc, proj })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.fc.forward(x)?;
        let x = Activation::Gelu.forward(&x)?;
        self.proj.forward(&x)
    }
}

struct Block {
    ln1: LayerNorm,
    ln2: LayerNorm,
    attn: CausalSelfAttention,
    mlp: Mlp,
}

impl Block {
    fn load(cfg: &TinyLLMConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let ln1 = layer_norm(cfg.d_model, 1e-5, vb.pp("ln1"))?;
        let ln2 = layer_norm(cfg.d_model, 1e-5, vb.pp("ln2"))?;
        let attn = CausalSelfAttention::load(cfg, vb.pp("attn"))?;
        let mlp = Mlp::load(cfg, vb.pp("mlp"))?;

        Ok(Self {
            ln1,
            ln2,
            attn,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Pre-LN transformer block:
        // x = x + attn(ln1(x))
        // x = x + mlp(ln2(x))
        let h = self.ln1.forward(x)?;
        let h = self.attn.forward(&h)?;
        let x = (x + &h)?;

        let h = self.ln2.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        let x = (&x + &h)?;

        Ok(x)
    }
}

struct TinyLLM {
    cfg: TinyLLMConfig,

    // Because your checkpoint only has output_head.weight and no separate
    // token_embedding.weight, we use output_head.weight as tied input embeddings.
    token_embedding: Embedding,

    pos_embedding: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,

    // Shape: [vocab_size, d_model]
    output_weight: Tensor,
}

impl TinyLLM {
    fn load(cfg: TinyLLMConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let output_weight = vb.get(
            (cfg.vocab_size, cfg.d_model),
            "output_head.weight",
        )?;

        let token_embedding = Embedding::new(output_weight.clone(), cfg.d_model);

        let pos_embedding = embedding(
            cfg.max_len,
            cfg.d_model,
            vb.pp("pos_embedding"),
        )?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for layer_idx in 0..cfg.num_layers {
            let block = Block::load(&cfg, vb.pp("blocks").pp(layer_idx))?;
            blocks.push(block);
        }

        let ln_f = layer_norm(cfg.d_model, 1e-5, vb.pp("ln_f"))?;

        Ok(Self {
            cfg,
            token_embedding,
            pos_embedding,
            blocks,
            ln_f,
            output_weight,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        let device = input_ids.device();

        let dims = input_ids.dims2()?;
        let bsz = dims.0;
        let seq_len = dims.1;

        if seq_len > self.cfg.max_len {
            candle_core::bail!(
                "sequence length {} exceeds model context length {}",
                seq_len,
                self.cfg.max_len
            );
        }

        let tok_emb = self.token_embedding.forward(input_ids)?;

        let pos_ids = Tensor::arange(0u32, seq_len as u32, device)?;
        let pos_emb = self.pos_embedding.forward(&pos_ids)?;
        let pos_emb = pos_emb.unsqueeze(0)?;

        let mut x = tok_emb.broadcast_add(&pos_emb)?;

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        x = self.ln_f.forward(&x)?;

        // output_head without bias:
        // x: [B, T, C]
        // output_weight: [V, C]
        // logits: [B, T, V]
        let x = x.reshape((bsz * seq_len, self.cfg.d_model))?;
        let logits = x.matmul(&self.output_weight.t()?)?;
        let logits = logits.reshape((bsz, seq_len, self.cfg.vocab_size))?;

        Ok(logits)
    }
}

fn sample_next_token(
    logits: &[f32],
    temperature: f64,
    top_k: usize,
    rng: &mut StdRng,
) -> usize {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap();
    }

    let mut pairs: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .collect();

    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    if top_k > 0 && top_k < pairs.len() {
        pairs.truncate(top_k);
    }

    let inv_temp = 1.0 / temperature as f32;

    let max_logit = pairs
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut probs = Vec::with_capacity(pairs.len());
    let mut sum = 0f32;

    for &(_, logit) in &pairs {
        let p = ((logit - max_logit) * inv_temp).exp();
        probs.push(p);
        sum += p;
    }

    let mut sample = rng.r#gen::<f32>() * sum;

    for ((token_id, _), p) in pairs.iter().zip(probs.iter()) {
        sample -= *p;
        if sample <= 0.0 {
            return *token_id;
        }
    }

    pairs.last().unwrap().0
}

//
// cargo run --release --  --model ../tinyllm/model.pt.safetensors --prompt "On a dark night" --max-new-tokens 120
//
fn main() -> Result<()> {
    let args = Args::parse();

    let cfg = TinyLLMConfig::default();

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    eprintln!("Using device: {:?}", device);

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[args.model.clone()],
            DType::F32,
            &device,
        )?
    };

    let model = TinyLLM::load(cfg.clone(), vb)?;

    let bpe = get_bpe_from_model("gpt2")?;

    let mut tokens = bpe.encode_with_special_tokens(&args.prompt);

    if tokens.is_empty() {
        bail!("prompt tokenized to an empty sequence");
    }

    if tokens.len() >= cfg.max_len {
        bail!(
            "prompt is {} tokens, but model context length is {}",
            tokens.len(),
            cfg.max_len
        );
    }

    let mut rng = StdRng::seed_from_u64(args.seed);

    for _step in 0..args.max_new_tokens {
        if tokens.len() >= cfg.max_len {
            eprintln!(
                "\nReached model context length {}; stopping generation.",
                cfg.max_len
            );
            break;
        }

        let input: Vec<u32> = tokens.iter().map(|&x| x as u32).collect();
        let input = Tensor::new(input.as_slice(), &device)?.unsqueeze(0)?;

        let logits = model.forward(&input)?;

        let seq_len = tokens.len();
        let last_logits = logits.i((0, seq_len - 1))?;
        let last_logits = last_logits.to_vec1::<f32>()?;

        let next = sample_next_token(
            &last_logits,
            args.temperature,
            args.top_k,
            &mut rng,
        );

        tokens.push(next.try_into().unwrap());

        if args.stop_eos && next == 50_256 {
            break;
        }
    }

    let text = bpe.decode(tokens)?;
    println!("{text}");

    Ok(())
}
