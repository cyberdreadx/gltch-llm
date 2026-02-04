"""
GLTCH-Pro: Larger Model Configuration
======================================
A beefier version of GLTCH for better quality text generation.

Comparison:
    GLTCH-2.7M:  2.7M params, ~1hr CPU / ~5min GPU
    GLTCH-10M:   10M params, ~3hr CPU / ~15min GPU  
    GLTCH-25M:   25M params, ~8hr CPU / ~30min GPU

Usage:
    python train_pro.py --size 10m
    python train_pro.py --size 25m
    python train_pro.py --size 10m --resume

Created by: cyberdreadx
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

# ============================================
# MODEL CONFIGURATIONS
# ============================================

CONFIGS = {
    '2.7m': {
        'name': 'GLTCH-2.7M',
        'batch_size': 64,
        'block_size': 128,
        'n_embd': 192,
        'n_head': 6,
        'n_layer': 6,
        'dropout': 0.1,
        'learning_rate': 3e-4,
        'max_iters': 3000,
    },
    '10m': {
        'name': 'GLTCH-10M',
        'batch_size': 48,
        'block_size': 256,
        'n_embd': 384,
        'n_head': 8,
        'n_layer': 8,
        'dropout': 0.1,
        'learning_rate': 2e-4,
        'max_iters': 5000,
    },
    '25m': {
        'name': 'GLTCH-25M',
        'batch_size': 32,
        'block_size': 512,
        'n_embd': 512,
        'n_head': 8,
        'n_layer': 12,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'max_iters': 8000,
    },
    '50m': {
        'name': 'GLTCH-50M',
        'batch_size': 24,
        'block_size': 512,
        'n_embd': 768,
        'n_head': 12,
        'n_layer': 12,
        'dropout': 0.1,
        'learning_rate': 6e-5,
        'max_iters': 10000,
    },
}

# Will be set based on selected size
config = None


# ============================================
# MODEL ARCHITECTURE
# ============================================

class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config['n_embd'], head_size, bias=False)
        self.query = nn.Linear(config['n_embd'], head_size, bias=False)
        self.value = nn.Linear(config['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config['block_size'], config['block_size'])))
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return self.dropout(F.softmax(wei, dim=-1)) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']),
            nn.GELU(),
            nn.Linear(4 * config['n_embd'], config['n_embd']),
            nn.Dropout(config['dropout']),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = config['n_embd'] // config['n_head']
        self.attention = MultiHeadAttention(config['n_head'], head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
    
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GLTCH(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, config['n_embd'])
        self.position_embedding = nn.Embedding(config['block_size'], config['n_embd'])
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1)) if targets is not None else None
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


# ============================================
# TRAINING
# ============================================

def train(size='10m', resume=False):
    global config
    config = CONFIGS[size].copy()
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_name = config['name'].lower().replace('-', '_')
    checkpoint_path = f"{model_name}_checkpoint.pt"
    final_path = f"{model_name}.pt"
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   {config['name']:^73} ║
║   Generative Language Transformer with Contextual Hierarchy                  ║
║   Created by: cyberdreadx                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🚀 Device: {config['device']}")
    print(f"📐 Config: {config['n_layer']} layers, {config['n_head']} heads, {config['n_embd']} dim")
    print(f"📦 Batch: {config['batch_size']}, Context: {config['block_size']}")
    
    # Load data
    print("\n📥 Loading training data...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '?') for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    
    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - config['block_size'], (config['batch_size'],))
        x = torch.stack([d[i:i+config['block_size']] for i in ix])
        y = torch.stack([d[i+1:i+config['block_size']+1] for i in ix])
        return x.to(config['device']), y.to(config['device'])
    
    # Create model
    model = GLTCH(vocab_size).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    start_step = 0
    
    # Resume from checkpoint
    if resume and os.path.exists(checkpoint_path):
        print(f"📂 Resuming from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt['step']
        print(f"   Resumed at step {start_step}")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: {n_params:,} parameters")
    print(f"📊 Training: {start_step} → {config['max_iters']} steps")
    print("-" * 60)
    
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ['train', 'val']:
            losses = torch.zeros(50)
            for k in range(50):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out
    
    start_time = time.time()
    
    for step in range(start_step, config['max_iters']):
        if step % 500 == 0:
            losses = estimate_loss()
            elapsed = time.time() - start_time
            eta = (config['max_iters'] - step) * (elapsed / max(step - start_step, 1))
            print(f"Step {step:5d} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | ETA: {eta/60:.1f}m")
            
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'vocab_size': vocab_size,
                'chars': chars,
                'config': config
            }, checkpoint_path)
        
        _, loss = model(*get_batch('train'))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Final save
    print("-" * 60)
    losses = estimate_loss()
    print(f"✅ Final | Train: {losses['train']:.4f} | Val: {losses['val']:.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'chars': chars,
        'config': config
    }, final_path)
    print(f"💾 Saved: {final_path}")
    
    # Generate sample
    print("\n✨ Sample:")
    print("=" * 60)
    prompt = "ROMEO:"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=config['device'])
    model.eval()
    generated = model.generate(context, max_new_tokens=300)
    print(decode(generated[0].tolist()))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GLTCH Pro models")
    parser.add_argument("--size", choices=['2.7m', '10m', '25m', '50m'], default='10m',
                        help="Model size to train")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    train(size=args.size, resume=args.resume)
