"""
GLTCH-2.7M Chat Interface
==========================
Web-based chat UI for your trained GLTCH model.

Usage:
    python chat.py

Opens a browser with a chat interface where you can talk to GLTCH!

Created by: cyberdreadx
"""

import os
import json
import threading
import webbrowser
import http.server
import socketserver
from urllib.parse import parse_qs

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
# MODEL ARCHITECTURE (must match training)
# ============================================

config = {
    'block_size': 128,
    'n_embd': 192,
    'n_head': 6,
    'n_layer': 6,
    'dropout': 0.0,  # No dropout during inference
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


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
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, rep_penalty=1.1):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if rep_penalty != 1.0:
                for i in range(idx.shape[0]):
                    for token_id in set(idx[i].tolist()):
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= rep_penalty
                        else:
                            logits[i, token_id] *= rep_penalty
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


# ============================================
# LOAD MODEL
# ============================================

MODEL_PATH = "gltch_2_7m.pt"
model = None
encode = None
decode = None


def load_model():
    global model, encode, decode
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Train a model first with: python train_continuous.py")
        return False
    
    print(f"📂 Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=config['device'], weights_only=False)
    
    # Check if checkpoint has vocab info or is just state dict
    if 'vocab_size' in checkpoint:
        # New format with vocab info
        vocab_size = checkpoint['vocab_size']
        chars = checkpoint['chars']
        state_dict = checkpoint['model_state_dict']
    else:
        # Old format - just weights, need to rebuild vocab from training data
        print("   (Rebuilding vocabulary from training data...)")
        import requests
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        state_dict = checkpoint
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '?') for i in l])
    
    model = GLTCH(vocab_size).to(config['device'])
    model.load_state_dict(state_dict)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {n_params:,} parameters")
    return True


# ============================================
# WEB INTERFACE
# ============================================

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GLTCH-2.7M Chat</title>
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
            --user-bg: #1a1a2e;
            --bot-bg: #0f2027;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Rajdhani', sans-serif;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            display: flex;
            flex-direction: column;
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
            z-index: 0;
        }
        
        header {
            background: var(--card);
            padding: 1rem 2rem;
            border-bottom: 1px solid rgba(0,245,255,0.2);
            display: flex;
            align-items: center;
            gap: 1rem;
            z-index: 10;
        }
        
        .logo {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5rem;
            background: linear-gradient(135deg, var(--cyan), var(--purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            color: var(--text-dim);
            font-size: 0.9rem;
        }
        
        .status {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 1rem;
            color: var(--green);
            font-size: 0.85rem;
        }
        
        .voice-toggle {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(0,245,255,0.3);
            border-radius: 2rem;
            padding: 0.4rem 1rem;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .voice-toggle:hover {
            border-color: var(--cyan);
        }
        
        .voice-toggle.active {
            background: rgba(0,245,255,0.2);
            border-color: var(--cyan);
        }
        
        .voice-toggle .icon {
            font-size: 1.1rem;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--green);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            z-index: 1;
        }
        
        .message {
            max-width: 80%;
            padding: 1rem 1.5rem;
            border-radius: 1rem;
            line-height: 1.6;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            background: var(--user-bg);
            border: 1px solid rgba(191,0,255,0.3);
            align-self: flex-end;
            border-bottom-right-radius: 0.25rem;
        }
        
        .message.bot {
            background: var(--bot-bg);
            border: 1px solid rgba(0,245,255,0.3);
            align-self: flex-start;
            border-bottom-left-radius: 0.25rem;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
        }
        
        .message.bot .label {
            color: var(--cyan);
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            margin-bottom: 0.5rem;
        }
        
        .message.user .label {
            color: var(--purple);
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            margin-bottom: 0.5rem;
        }
        
        #input-container {
            background: var(--card);
            padding: 1.5rem 2rem;
            border-top: 1px solid rgba(0,245,255,0.2);
            display: flex;
            gap: 1rem;
            z-index: 10;
        }
        
        #message-input {
            flex: 1;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(0,245,255,0.3);
            border-radius: 0.5rem;
            padding: 1rem 1.5rem;
            color: var(--text);
            font-family: 'Rajdhani', sans-serif;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }
        
        #message-input:focus {
            border-color: var(--cyan);
        }
        
        #message-input::placeholder {
            color: var(--text-dim);
        }
        
        #send-btn {
            background: linear-gradient(135deg, var(--cyan), var(--purple));
            border: none;
            border-radius: 0.5rem;
            padding: 1rem 2rem;
            color: white;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9rem;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        #send-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(0,245,255,0.4);
        }
        
        #send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .controls {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        
        .control-group label {
            font-size: 0.7rem;
            color: var(--text-dim);
            text-transform: uppercase;
        }
        
        .control-group input {
            width: 60px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(0,245,255,0.2);
            border-radius: 0.25rem;
            padding: 0.3rem 0.5rem;
            color: var(--cyan);
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            text-align: center;
        }
        
        .thinking {
            color: var(--text-dim);
            font-style: italic;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">GLTCH-2.7M</div>
        <div class="subtitle">Generative Language Transformer with Contextual Hierarchy</div>
        <div class="status">
            <div class="voice-toggle" id="voice-toggle" onclick="toggleVoice()">
                <span class="icon">🔇</span>
                <span id="voice-label">Voice Off</span>
            </div>
            <div class="status-dot"></div>
            <span>Online</span>
        </div>
    </header>
    
    <div id="chat-container">
        <div class="message bot">
            <div class="label">GLTCH</div>
            Greetings! I am GLTCH-2.7M, a tiny language model trained on Shakespeare.
            
Try prompts like:
• ROMEO:
• HAMLET: To be or not to be
• The king said

I'll continue your text in Shakespearean style!
        </div>
    </div>
    
    <div id="input-container">
        <input type="text" id="message-input" placeholder="Enter a prompt for GLTCH to continue..." autofocus>
        <div class="controls">
            <div class="control-group">
                <label>Tokens</label>
                <input type="number" id="max-tokens" value="200" min="10" max="500">
            </div>
            <div class="control-group">
                <label>Temp</label>
                <input type="number" id="temperature" value="0.8" min="0.1" max="2.0" step="0.1">
            </div>
            <div class="control-group">
                <label>Top-K</label>
                <input type="number" id="top-k" value="40" min="1" max="100">
            </div>
            <div class="control-group">
                <label>Rep Pen</label>
                <input type="number" id="rep-penalty" value="1.1" min="1.0" max="2.0" step="0.1">
            </div>
        </div>
        <button id="send-btn">Generate</button>
    </div>
    
    <script>
        const chatContainer = document.getElementById('chat-container');
        const messageInput = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-btn');
        const maxTokensInput = document.getElementById('max-tokens');
        const temperatureInput = document.getElementById('temperature');
        const topKInput = document.getElementById('top-k');
        const repPenaltyInput = document.getElementById('rep-penalty');
        
        // Text-to-Speech
        let voiceEnabled = false;
        const synth = window.speechSynthesis;
        
        function toggleVoice() {
            voiceEnabled = !voiceEnabled;
            const toggle = document.getElementById('voice-toggle');
            const label = document.getElementById('voice-label');
            
            if (voiceEnabled) {
                toggle.classList.add('active');
                toggle.querySelector('.icon').textContent = '🔊';
                label.textContent = 'Voice On';
                speak('Voice enabled');
            } else {
                toggle.classList.remove('active');
                toggle.querySelector('.icon').textContent = '🔇';
                label.textContent = 'Voice Off';
                synth.cancel();
            }
        }
        
        function speak(text) {
            if (!voiceEnabled || !synth) return;
            synth.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.9;
            utterance.pitch = 0.8;
            // Try to get a nice voice
            const voices = synth.getVoices();
            const preferredVoice = voices.find(v => v.name.includes('Daniel') || v.name.includes('Google UK English Male'));
            if (preferredVoice) utterance.voice = preferredVoice;
            synth.speak(utterance);
        }
        
        // Load voices
        speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
        
        function addMessage(text, isUser) {
            const div = document.createElement('div');
            div.className = `message ${isUser ? 'user' : 'bot'}`;
            div.innerHTML = `<div class="label">${isUser ? 'YOU' : 'GLTCH'}</div>${text}`;
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return div;
        }
        
        async function sendMessage() {
            const prompt = messageInput.value.trim();
            if (!prompt) return;
            
            addMessage(prompt, true);
            messageInput.value = '';
            sendBtn.disabled = true;
            
            const thinkingDiv = addMessage('<span class="thinking">Generating...</span>', false);
            
            try {
                const params = new URLSearchParams({
                    prompt: prompt,
                    max_tokens: maxTokensInput.value,
                    temperature: temperatureInput.value,
                    top_k: topKInput.value,
                    rep_penalty: repPenaltyInput.value
                });
                
                const response = await fetch('/generate?' + params);
                const data = await response.json();
                
                thinkingDiv.innerHTML = `<div class="label">GLTCH</div>${data.text}`;
                speak(data.text);
            } catch (error) {
                thinkingDiv.innerHTML = `<div class="label">GLTCH</div><span style="color: #ff6b6b;">Error: ${error.message}</span>`;
            }
            
            sendBtn.disabled = false;
            messageInput.focus();
        }
        
        sendBtn.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
"""


class ChatHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(CHAT_HTML.encode())
        
        elif self.path.startswith('/generate'):
            # Parse query params
            query = self.path.split('?')[1] if '?' in self.path else ''
            params = parse_qs(query)
            
            prompt = params.get('prompt', [''])[0]
            max_tokens = int(params.get('max_tokens', ['200'])[0])
            temperature = float(params.get('temperature', ['0.8'])[0])
            top_k = int(params.get('top_k', ['40'])[0])
            rep_penalty = float(params.get('rep_penalty', ['1.1'])[0])
            
            # Generate response
            if model is None:
                response = {"text": "Model not loaded!"}
            else:
                try:
                    context = torch.tensor([encode(prompt)], dtype=torch.long, device=config['device'])
                    generated = model.generate(
                        context, 
                        max_new_tokens=max_tokens, 
                        temperature=temperature,
                        top_k=top_k,
                        rep_penalty=rep_penalty
                    )
                    text = decode(generated[0].tolist())
                    response = {"text": text}
                except Exception as e:
                    response = {"text": f"Error: {str(e)}"}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port=8889):
    with socketserver.TCPServer(("", port), ChatHandler) as httpd:
        httpd.serve_forever()


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   GLTCH-2.7M Chat Interface                                                   ║
║   Created by: cyberdreadx                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if not load_model():
        exit(1)
    
    PORT = 8889
    print(f"\n🌐 Chat interface: http://localhost:{PORT}")
    print("   Press Ctrl+C to exit")
    print("-" * 50)
    
    # Start server in background
    server_thread = threading.Thread(target=start_server, args=(PORT,), daemon=True)
    server_thread.start()
    
    # Open browser
    webbrowser.open(f"http://localhost:{PORT}")
    
    # Keep running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
