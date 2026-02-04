"""
GLTCH Training Dashboard
==============================
Web-based UI for monitoring training progress in real-time.

Usage:
    python train_with_ui.py                 # Train GLTCH-2.7M (default)
    python train_with_ui.py --size 10m      # Train GLTCH-10M
    python train_with_ui.py --size 25m      # Train GLTCH-25M
    python train_with_ui.py --size 50m      # Train GLTCH-50M

Opens a browser dashboard showing:
- Live loss curve
- Training speed (tokens/sec)
- Current step and ETA
- Generated text samples

Created by: cyberdreadx
"""

import argparse
import threading
import webbrowser
import json
import time
import http.server
import socketserver
from pathlib import Path

# ============================================
# TRAINING BACKEND
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

# Global state for UI updates
training_state = {
    "running": False,
    "step": 0,
    "max_steps": 3000,
    "loss": 0,
    "loss_history": [],
    "tokens_per_sec": 0,
    "eta_seconds": 0,
    "current_sample": "",
    "status": "Initializing...",
    "model_name": "GLTCH-2.7M"
}

# MODEL CONFIGURATIONS
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
        'eval_interval': 100,
        'sample_interval': 500,
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
        'eval_interval': 200,
        'sample_interval': 500,
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
        'eval_interval': 300,
        'sample_interval': 1000,
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
        'eval_interval': 500,
        'sample_interval': 1000,
    },
}

# Will be set based on selected size
config = None


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
        wei = F.softmax(wei, dim=-1)
        return self.dropout(wei) @ v


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
        x = x + self.ffwd(self.ln2(x))
        return x


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
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


def train_model():
    """Main training function that updates global state"""
    global training_state
    
    training_state["status"] = "Loading data..."
    training_state["running"] = True
    
    # Load data
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    
    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - config['block_size'], (config['batch_size'],))
        x = torch.stack([d[i:i+config['block_size']] for i in ix])
        y = torch.stack([d[i+1:i+config['block_size']+1] for i in ix])
        return x.to(config['device']), y.to(config['device'])
    
    training_state["status"] = "Creating model..."
    
    model = GLTCH(vocab_size).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    training_state["status"] = "Training..."
    training_state["max_steps"] = config['max_iters']
    
    start_time = time.time()
    tokens_processed = 0
    
    for step in range(config['max_iters']):
        if not training_state["running"]:
            break
        
        # Training step
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Update stats
        tokens_processed += config['batch_size'] * config['block_size']
        elapsed = time.time() - start_time
        
        training_state["step"] = step + 1
        training_state["loss"] = loss.item()
        training_state["tokens_per_sec"] = int(tokens_processed / elapsed)
        training_state["eta_seconds"] = int((config['max_iters'] - step) * (elapsed / (step + 1)))
        
        # Record loss history
        if step % 10 == 0:
            training_state["loss_history"].append({
                "step": step,
                "loss": loss.item()
            })
            # Keep last 300 points
            if len(training_state["loss_history"]) > 300:
                training_state["loss_history"] = training_state["loss_history"][-300:]
        
        # Generate sample
        if step % config['sample_interval'] == 0 and step > 0:
            model.eval()
            prompt = "ROMEO:"
            context = torch.tensor([encode(prompt)], dtype=torch.long, device=config['device'])
            generated = model.generate(context, max_new_tokens=150)
            training_state["current_sample"] = decode(generated[0].tolist())
            model.train()
    
    training_state["status"] = "Complete!"
    training_state["running"] = False
    
    # Final generation
    model.eval()
    prompt = "ROMEO:"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=config['device'])
    generated = model.generate(context, max_new_tokens=300)
    training_state["current_sample"] = decode(generated[0].tolist())
    
    # Save model
    model_filename = config['name'].lower().replace('-', '_') + '.pt'
    torch.save(model.state_dict(), model_filename)
    training_state["status"] = f"Complete! Model saved to {model_filename}"


# ============================================
# WEB SERVER
# ============================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GLTCH-2.7M Training Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0a0a0f;
            --card: #12121a;
            --cyan: #00f5ff;
            --purple: #bf00ff;
            --green: #00ff88;
            --text: #ffffff;
            --text-dim: #8888aa;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Rajdhani', sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 2rem;
        }
        
        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background: 
                linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px),
                linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
        }
        
        .container { max-width: 1200px; margin: 0 auto; position: relative; z-index: 1; }
        
        header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--cyan), var(--purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle { color: var(--text-dim); margin-top: 0.5rem; }
        
        .status-bar {
            background: var(--card);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(0,245,255,0.2);
        }
        
        .status-text {
            font-size: 1.2rem;
            color: var(--cyan);
            text-align: center;
        }
        
        .progress-container {
            margin-top: 1rem;
            background: rgba(0,245,255,0.1);
            border-radius: 0.5rem;
            height: 20px;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--cyan), var(--purple));
            width: 0%;
            transition: width 0.3s;
            box-shadow: 0 0 20px var(--cyan);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .stat-card {
            background: var(--card);
            border-radius: 0.75rem;
            padding: 1.5rem;
            border: 1px solid rgba(0,245,255,0.1);
            text-align: center;
        }
        
        .stat-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 2rem;
            color: var(--cyan);
        }
        
        .stat-label {
            color: var(--text-dim);
            text-transform: uppercase;
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }
        
        .chart-container {
            background: var(--card);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(0,245,255,0.1);
        }
        
        .chart-container h2 {
            font-size: 1rem;
            color: var(--text-dim);
            margin-bottom: 1rem;
        }
        
        #loss-chart {
            width: 100%;
            height: 200px;
        }
        
        .sample-container {
            background: var(--card);
            border-radius: 0.75rem;
            padding: 1.5rem;
            border: 1px solid rgba(0,245,255,0.1);
        }
        
        .sample-container h2 {
            font-size: 1rem;
            color: var(--text-dim);
            margin-bottom: 1rem;
        }
        
        #sample-text {
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.6;
            white-space: pre-wrap;
            color: var(--green);
            max-height: 300px;
            overflow-y: auto;
        }
        
        .device-badge {
            display: inline-block;
            background: var(--purple);
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>GLTCH-2.7M</h1>
            <p class="subtitle">Generative Language Transformer with Contextual Hierarchy</p>
            <span class="device-badge" id="device-badge">CPU</span>
        </header>
        
        <div class="status-bar">
            <div class="status-text" id="status">Initializing...</div>
            <div class="progress-container">
                <div class="progress-bar" id="progress-bar"></div>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="step">0</div>
                <div class="stat-label">Step</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="loss">—</div>
                <div class="stat-label">Loss</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="speed">0</div>
                <div class="stat-label">Tokens/sec</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="eta">—</div>
                <div class="stat-label">ETA</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>📉 Loss Curve</h2>
            <canvas id="loss-chart"></canvas>
        </div>
        
        <div class="sample-container">
            <h2>✨ Generated Sample</h2>
            <div id="sample-text">Training will generate samples periodically...</div>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('loss-chart');
        const ctx = canvas.getContext('2d');
        
        function formatTime(seconds) {
            if (seconds < 60) return seconds + 's';
            if (seconds < 3600) return Math.floor(seconds / 60) + 'm ' + (seconds % 60) + 's';
            return Math.floor(seconds / 3600) + 'h ' + Math.floor((seconds % 3600) / 60) + 'm';
        }
        
        function drawChart(history) {
            const w = canvas.width = canvas.offsetWidth;
            const h = canvas.height = 200;
            
            ctx.clearRect(0, 0, w, h);
            
            if (history.length < 2) return;
            
            const maxLoss = Math.max(...history.map(p => p.loss));
            const minLoss = Math.min(...history.map(p => p.loss));
            const range = maxLoss - minLoss || 1;
            
            // Grid
            ctx.strokeStyle = 'rgba(0,245,255,0.1)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = (i / 4) * h;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }
            
            // Line
            ctx.beginPath();
            ctx.strokeStyle = '#00f5ff';
            ctx.lineWidth = 2;
            
            history.forEach((point, i) => {
                const x = (i / (history.length - 1)) * w;
                const y = h - ((point.loss - minLoss) / range) * (h - 20) - 10;
                
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            
            ctx.stroke();
            
            // Glow
            ctx.shadowColor = '#00f5ff';
            ctx.shadowBlur = 10;
            ctx.stroke();
            ctx.shadowBlur = 0;
        }
        
        async function fetchState() {
            try {
                const res = await fetch('/api/state');
                const state = await res.json();
                
                document.getElementById('status').textContent = state.status;
                document.getElementById('step').textContent = state.step.toLocaleString() + ' / ' + state.max_steps.toLocaleString();
                document.getElementById('loss').textContent = state.loss.toFixed(4);
                document.getElementById('speed').textContent = state.tokens_per_sec.toLocaleString();
                document.getElementById('eta').textContent = formatTime(state.eta_seconds);
                
                const progress = (state.step / state.max_steps) * 100;
                document.getElementById('progress-bar').style.width = progress + '%';
                
                if (state.current_sample) {
                    document.getElementById('sample-text').textContent = state.current_sample;
                }
                
                drawChart(state.loss_history);
                
            } catch (e) {
                console.error(e);
            }
        }
        
        // Update device badge
        fetch('/api/state').then(r => r.json()).then(s => {
            const device = s.device || 'CPU';
            document.getElementById('device-badge').textContent = device.toUpperCase();
        });
        
        // Poll for updates
        setInterval(fetchState, 500);
        fetchState();
    </script>
</body>
</html>
"""


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logs
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        
        elif self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            state = dict(training_state)
            state['device'] = config['device']
            self.wfile.write(json.dumps(state).encode())
        
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port=8888):
    """Start the dashboard server"""
    with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
        httpd.serve_forever()


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="GLTCH Training Dashboard")
    parser.add_argument("--size", choices=['2.7m', '10m', '25m', '50m'], default='2.7m',
                        help="Model size to train (default: 2.7m)")
    args = parser.parse_args()
    
    # Set config based on size
    config = CONFIGS[args.size].copy()
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Update training state with model name
    training_state['model_name'] = config['name']
    training_state['max_steps'] = config['max_iters']
    
    # Calculate model file name
    model_filename = config['name'].lower().replace('-', '_') + '.pt'
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   {config['name']:^73} ║
║   Generative Language Transformer with Contextual Hierarchy                  ║
║   Training Dashboard — Created by: cyberdreadx                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"📐 Config: {config['n_layer']} layers, {config['n_head']} heads, {config['n_embd']} dim")
    print(f"📦 Batch: {config['batch_size']}, Context: {config['block_size']}")
    print(f"🚀 Device: {config['device']}")
    print(f"🌐 Dashboard: http://localhost:8888")
    print("-" * 50)
    
    # Start server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Open browser
    webbrowser.open("http://localhost:8888")
    
    # Start training
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n👋 Training stopped by user")
        training_state["running"] = False
        training_state["status"] = "Stopped by user"
    
    # Keep server running after training
    print(f"\n✅ Training complete! Model saved to {model_filename}")
    print("   Dashboard still available at http://localhost:8888")
    print("   Press Ctrl+C to exit")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

