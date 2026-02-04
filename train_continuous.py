"""
GLTCH-2.7M Continuous Training
================================
Train, save checkpoints, and resume training from where you left off.

Usage:
    # Start fresh training
    python train_continuous.py

    # Resume from checkpoint
    python train_continuous.py --resume

    # Train for more steps
    python train_continuous.py --resume --steps 5000

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
# CONFIGURATION
# ============================================

config = {
    'batch_size': 64,
    'block_size': 128,
    'n_embd': 192,
    'n_head': 6,
    'n_layer': 6,
    'dropout': 0.1,
    'learning_rate': 3e-4,
    'max_iters': 3000,
    'eval_interval': 300,
    'checkpoint_interval': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

CHECKPOINT_PATH = "gltch_checkpoint.pt"
FINAL_MODEL_PATH = "gltch_2_7m.pt"


# ============================================
# MODEL (same architecture)
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
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -config['block_size']:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


# ============================================
# CHECKPOINT FUNCTIONS
# ============================================

def save_checkpoint(model, optimizer, step, total_steps, vocab_info, loss):
    """Save training checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'total_steps': total_steps,
        'vocab_size': vocab_info['vocab_size'],
        'chars': vocab_info['chars'],
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"💾 Checkpoint saved at step {step}")


def load_checkpoint():
    """Load training checkpoint"""
    if os.path.exists(CHECKPOINT_PATH):
        return torch.load(CHECKPOINT_PATH, weights_only=False)
    return None


# ============================================
# TRAINING
# ============================================

def train(resume=False, additional_steps=None):
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   GLTCH-2.7M Continuous Training                                              ║
║   Created by: cyberdreadx                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🚀 Device: {config['device']}")
    
    # Load data
    print("📥 Loading training data...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    
    # Check for checkpoint
    checkpoint = load_checkpoint() if resume else None
    
    if checkpoint:
        print(f"📂 Resuming from checkpoint (step {checkpoint['step']})")
        chars = checkpoint['chars']
        vocab_size = checkpoint['vocab_size']
        start_step = checkpoint['step']
        total_steps = checkpoint['total_steps']
        if additional_steps:
            total_steps = start_step + additional_steps
    else:
        print("🆕 Starting fresh training")
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        start_step = 0
        total_steps = additional_steps or config['max_iters']
    
    # Tokenizer
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '?') for i in l])
    
    vocab_info = {'vocab_size': vocab_size, 'chars': chars}
    
    # Prepare data
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
    
    # Load checkpoint weights
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"   Loaded model weights (loss was {checkpoint['loss']:.4f})")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: {n_params:,} parameters")
    print(f"📊 Training steps: {start_step} → {total_steps}")
    print("-" * 60)
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(50)
            for k in range(50):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out
    
    # Training loop
    start_time = time.time()
    
    for step in range(start_step, total_steps):
        # Evaluate
        if step % config['eval_interval'] == 0:
            losses = estimate_loss()
            elapsed = time.time() - start_time
            print(f"Step {step:5d}/{total_steps} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | Time: {elapsed:.0f}s")
        
        # Save checkpoint
        if step > 0 and step % config['checkpoint_interval'] == 0:
            losses = estimate_loss()
            save_checkpoint(model, optimizer, step, total_steps, vocab_info, losses['val'])
        
        # Training step
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Final save
    print("-" * 60)
    losses = estimate_loss()
    print(f"✅ Final | Train: {losses['train']:.4f} | Val: {losses['val']:.4f}")
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, total_steps, total_steps, vocab_info, losses['val'])
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'chars': chars,
        'config': config
    }, FINAL_MODEL_PATH)
    print(f"💾 Final model saved to: {FINAL_MODEL_PATH}")
    
    # Generate sample
    print("\n✨ Generated sample:")
    print("=" * 60)
    prompt = "ROMEO:"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=config['device'])
    model.eval()
    generated = model.generate(context, max_new_tokens=300)
    print(decode(generated[0].tolist()))
    print("=" * 60)
    
    return model, encode, decode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLTCH Continuous Training")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--steps", type=int, help="Additional steps to train")
    
    args = parser.parse_args()
    
    train(resume=args.resume, additional_steps=args.steps)
