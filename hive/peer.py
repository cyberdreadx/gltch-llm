"""
GLTCH HIVE — Training Peer
===========================
Run this script to join the training hive and contribute GPU power.

Usage:
    python peer.py --server ws://YOUR_VPS:8765 --key SECRET --name my-gpu
    python peer.py --server ws://YOUR_VPS:8765 --key SECRET --name my-gpu --size 10m

Model Sizes:
    2.7m  - 2.7M params (default)
    10m   - ~10M params
    25m   - ~25M params  
    50m   - ~50M params

Created by: cyberdreadx
"""

import asyncio
import json
import argparse
import platform
import random
import time
import requests

# Try imports
try:
    import websockets
except ImportError:
    print("❌ websockets not installed. Run: pip install websockets")
    exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not installed. Running in simulation mode.")


# ============================================
# MODEL CONFIGURATIONS
# ============================================

CONFIGS = {
    '1m': {
        'name': 'GLTCH-1M',
        'batch_size': 32,
        'block_size': 64,
        'n_embd': 128,
        'n_head': 4,
        'n_layer': 4,
        'dropout': 0.1,
        'learning_rate': 5e-4,
        'max_iters': 2000,
    },
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

config = None  # Set based on --size


# ============================================
# GPU DETECTION
# ============================================

def get_gpu_info():
    """Detect GPU and return info string"""
    if HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    elif HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "Apple Silicon (MPS)"
    else:
        return f"CPU ({platform.processor() or 'Unknown'})"


def get_device():
    """Get the best available device"""
    if HAS_TORCH:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
    return torch.device('cpu') if HAS_TORCH else None


# ============================================
# GLTCH MODEL ARCHITECTURE
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
            idx_cond = idx[:, -config['block_size']:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


# ============================================
# TRAINING PEER
# ============================================

class TrainingPeer:
    def __init__(self, server_url: str, name: str, size: str, secret_key: str):
        global config
        config = CONFIGS[size].copy()
        config['device'] = get_device()
        
        self.server_url = server_url
        self.name = name
        self.size = size
        self.secret_key = secret_key
        self.gpu = get_gpu_info()
        self.device = get_device()
        self.peer_id = None
        self.ws = None
        self.running = True
        self.step = 0
        
        # Load training data
        self.data = None
        self.train_data = None
        self.vocab_size = None
        self.encode = None
        self.decode = None
        self.stoi = None
        self.itos = None
        
        # Model will be initialized after data is loaded
        self.model = None
        self.optimizer = None
    
    def load_data(self, stoi=None, itos=None):
        """Load training data. Uses server-provided vocab if available."""
        # Download training data (always needed for training)
        print("📥 Loading training data...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
        
        if stoi and itos:
            # Use vocabulary from server (ensures compatibility across peers)
            print("   Using vocabulary from server...")
            self.vocab_size = len(stoi)
            # itos keys are strings in JSON, convert to int
            itos_int = {int(k): v for k, v in itos.items()}
            self.encode = lambda s: [stoi.get(c, 0) for c in s]
            self.decode = lambda l: ''.join([itos_int.get(i, '?') for i in l])
            self.stoi = stoi
            self.itos = itos_int
        else:
            # Build vocabulary locally
            chars = sorted(list(set(text)))
            self.vocab_size = len(chars)
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
            self.encode = lambda s: [self.stoi.get(c, 0) for c in s]
            self.decode = lambda l: ''.join([self.itos.get(i, '?') for i in l])
        
        # Encode and split training data
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        
        print(f"   Loaded {len(text):,} characters, vocab size: {self.vocab_size}")
    
    def init_model(self):
        """Initialize model and optimizer"""
        print(f"🧠 Creating {config['name']} model...")
        self.model = GLTCH(self.vocab_size).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['learning_rate'])
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Parameters: {n_params:,}")
    
    def get_batch(self):
        """Get a random training batch"""
        ix = torch.randint(len(self.train_data) - config['block_size'], (config['batch_size'],))
        x = torch.stack([self.train_data[i:i+config['block_size']] for i in ix])
        y = torch.stack([self.train_data[i+1:i+config['block_size']+1] for i in ix])
        return x.to(self.device), y.to(self.device)
    
    async def connect(self):
        """Connect to the hive server"""
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   {config['name']:^73} ║
║   GLTCH HIVE — Distributed Training Peer                                     ║
║   Created by: cyberdreadx                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """)
        
        print(f"🖥️  Node: {self.name}")
        print(f"🎮 GPU: {self.gpu}")
        print(f"📐 Config: {config['n_layer']} layers, {config['n_head']} heads, {config['n_embd']} dim")
        print(f"🔗 Server: {self.server_url}")
        print("-" * 50)
        
        try:
            self.ws = await websockets.connect(self.server_url)
            
            # Register with server
            await self.ws.send(json.dumps({
                "type": "register",
                "name": self.name,
                "gpu": self.gpu,
                "model_size": self.size,
                "key": self.secret_key
            }))
            
            # Wait for confirmation
            response = await self.ws.recv()
            data = json.loads(response)
            
            if data.get("type") == "error":
                print(f"❌ {data.get('message', 'Connection rejected')}")
                return False
            
            if data["type"] == "registered":
                self.peer_id = data["peer_id"]
                self.step = data.get("training_step", 0)
                
                # Get vocabulary from server if provided
                vocab = data.get("vocab")
                if vocab:
                    self.load_data(stoi=vocab.get("stoi"), itos=vocab.get("itos"))
                else:
                    self.load_data()  # Fallback to local loading
                
                # Initialize model with correct vocab size
                self.init_model()
                
                print(f"✅ Registered as: {self.peer_id}")
                print(f"📊 Current training step: {self.step}")
                print("-" * 50)
                return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
        
        return False
    
    async def training_loop(self):
        """Main training loop"""
        print("🏋️ Starting training...")
        start_time = time.time()
        
        while self.running and self.step < config['max_iters']:
            try:
                # Perform local training step
                loss = await self.train_step()
                
                # Send update to server
                await self.ws.send(json.dumps({
                    "type": "gradient",
                    "step": self.step,
                    "loss": loss,
                    "gradient": self.get_gradient_summary()
                }))
                
                # Log progress
                if self.step % 100 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = (self.step * config['batch_size'] * config['block_size']) / elapsed
                    eta = (config['max_iters'] - self.step) * (elapsed / max(self.step, 1))
                    print(f"   Step {self.step:5d}/{config['max_iters']} | Loss: {loss:.4f} | Speed: {tokens_per_sec:.0f} tok/s | ETA: {eta/60:.1f}m")
                    
                    # Generate sample every 500 steps
                    if self.step % 500 == 0 and self.step > 0:
                        self.generate_sample()
                
                self.step += 1
                
                # Check for server messages (non-blocking)
                try:
                    response = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                    data = json.loads(response)
                    # Handle server commands here
                except asyncio.TimeoutError:
                    pass
                
            except websockets.exceptions.ConnectionClosed:
                print("🔌 Disconnected from server")
                self.running = False
                break
            except Exception as e:
                print(f"⚠️  Error: {e}")
                await asyncio.sleep(1)
        
        # Training complete
        self.save_model()
    
    async def train_step(self) -> float:
        """Perform one training step"""
        if self.model and self.device:
            self.model.train()
            
            xb, yb = self.get_batch()
            logits, loss = self.model(xb, yb)
            
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        else:
            # Simulation mode
            await asyncio.sleep(0.05)
            return 4.0 - (self.step * 0.0005) + random.random() * 0.1
    
    def get_gradient_summary(self) -> dict:
        """Get summary of gradients (for transmission)"""
        if self.model:
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            return {"norm": total_norm ** 0.5}
        return {"norm": 0}
    
    def generate_sample(self):
        """Generate a text sample"""
        self.model.eval()
        prompt = "ROMEO:"
        context = torch.tensor([self.encode(prompt)], dtype=torch.long, device=self.device)
        generated = self.model.generate(context, max_new_tokens=100)
        text = self.decode(generated[0].tolist())
        print(f"\n   ✨ Sample: {text[:150]}...\n")
        self.model.train()
    
    def save_model(self):
        """Save the trained model"""
        filename = config['name'].lower().replace('-', '_') + '.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.vocab_size,
            'chars': [chr(i) for i in range(self.vocab_size)],  # Simplified
            'config': config
        }, filename)
        print(f"\n💾 Model saved to: {filename}")
    
    async def run(self):
        """Main entry point"""
        if await self.connect():
            try:
                await self.training_loop()
            except KeyboardInterrupt:
                print("\n👋 Shutting down peer...")
            finally:
                self.save_model()
                if self.ws:
                    await self.ws.close()


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="GLTCH Hive Training Peer")
    parser.add_argument("--server", default="ws://localhost:8765", help="Server WebSocket URL")
    parser.add_argument("--key", required=True, help="Secret key from coordinator")
    parser.add_argument("--name", default=f"node-{random.randint(1000, 9999)}", help="Peer name")
    parser.add_argument("--size", choices=['2.7m', '10m', '25m', '50m'], default='2.7m',
                        help="Model size to train (default: 2.7m)")
    
    args = parser.parse_args()
    
    peer = TrainingPeer(args.server, args.name, args.size, args.key)
    asyncio.run(peer.run())


if __name__ == "__main__":
    main()
