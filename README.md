# GLTCH

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║    ██████╗ ██╗  ████████╗ ██████╗██╗  ██╗                                     ║
║   ██╔════╝ ██║  ╚══██╔══╝██╔════╝██║  ██║                                     ║
║   ██║  ███╗██║     ██║   ██║     ███████║                                     ║
║   ██║   ██║██║     ██║   ██║     ██╔══██║                                     ║
║   ╚██████╔╝███████╗██║   ╚██████╗██║  ██║                                     ║
║    ╚═════╝ ╚══════╝╚═╝    ╚═════╝╚═╝  ╚═╝                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

**Generative Language Transformer with Contextual Hierarchy**

Build and train your own language model from scratch. Supports multiple model sizes and distributed training across multiple GPUs.

## Features

- 🧠 **Multiple model sizes** — 2.7M, 10M, 25M, or 50M parameters
- 📊 **Live training dashboard** — Real-time loss curve and generated samples
- 💬 **Chat interface** — Talk to your trained model with voice output
- 🌐 **Distributed training (Hive)** — Train across multiple machines/GPUs
- 🎮 **Works on consumer GPUs** — Or even CPU (just slower)

---

## Quick Start

### Option 1: Train with Dashboard (Recommended)

```bash
git clone https://github.com/cyberdreadx/gltch-2.7m.git
cd gltch-2.7m
pip install torch requests

# Train 2.7M model with live dashboard
python train_with_ui.py

# Or train a larger model
python train_with_ui.py --size 10m
```

Opens a browser dashboard showing loss curve, speed, ETA, and generated samples.

### Option 2: Chat with Trained Model

```bash
pip install torch requests
python chat.py
```

Opens a chat interface at `http://localhost:8889` with:
- 💬 Text generation from prompts
- 🔊 Text-to-speech (toggle on/off)
- 🎚️ Adjustable temperature, top-k, repetition penalty

---

## Model Sizes

| Size | Params | VRAM | Training Time (GPU) |
|------|--------|------|---------------------|
| `2.7m` | 2.7M | ~1GB | ~5 min |
| `10m` | ~10M | ~2GB | ~15 min |
| `25m` | ~25M | ~4GB | ~30 min |
| `50m` | ~50M | ~8GB | ~1 hour |

```bash
# Examples
python train_with_ui.py --size 2.7m   # Default
python train_with_ui.py --size 10m    # Larger
python train_with_ui.py --size 50m    # Largest
```

---

## Training Scripts

| Script | Purpose |
|--------|---------|
| `train_with_ui.py` | Dashboard UI + training |
| `train_continuous.py` | Resume training from checkpoint |
| `train_custom.py` | Train on your own text data |
| `train_pro.py` | CLI training with size selection |
| `gltch_2_7m.py` | Simple terminal training |

### Train on Custom Data

```bash
python train_custom.py --data your_novel.txt
python train_custom.py --data ./my_dataset/ --steps 10000
python train_custom.py --data https://example.com/text.txt
```

### Resume Training

```bash
python train_continuous.py --resume
python train_continuous.py --resume --steps 5000  # Add more steps
```

---

## Chat Interface

```bash
python chat.py
```

Open `http://localhost:8889` in your browser.

### Controls

| Slider | What It Does | Default |
|--------|--------------|---------|
| Tokens | Output length | 200 |
| Temp | Creativity (lower = focused) | 0.8 |
| Top-K | Only consider top K tokens | 40 |
| Rep Pen | Penalize repetition | 1.1 |

Click the **🔊 Voice** button to enable text-to-speech.

---

## Distributed Training (GLTCH Hive)

Train across multiple machines using the Hive network.

### Architecture

```
                     ┌─────────────────┐
                     │  Coordinator    │
                     │  (VPS/Server)   │
                     │  server.py      │
                     └────────┬────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Peer 1      │    │   Peer 2      │    │   Peer 3      │
│   RTX 4090    │    │   RTX 3080    │    │   M1 Mac      │
│   peer.py     │    │   peer.py     │    │   peer.py     │
└───────────────┘    └───────────────┘    └───────────────┘
```

### Step 1: Start Coordinator (on VPS)

```bash
# SSH to your VPS
curl -sSL https://raw.githubusercontent.com/cyberdreadx/gltch-2.7m/main/hive/setup_coordinator.sh | bash
```

Or manually:

```bash
git clone https://github.com/cyberdreadx/gltch-2.7m.git
cd gltch-2.7m/hive
pip install websockets
python server.py
```

Dashboard: `http://YOUR_VPS_IP:8080`

### Step 2: Connect Peers (on GPU machines)

```bash
git clone https://github.com/cyberdreadx/gltch-2.7m.git
cd gltch-2.7m
pip install torch websockets requests

# Connect and train
python hive/peer.py --server ws://YOUR_VPS_IP:8765 --name my-gpu --size 10m
```

### Peer Options

```bash
python hive/peer.py \
    --server ws://coordinator.example.com:8765 \
    --name office-4090 \
    --size 25m
```

| Option | Description |
|--------|-------------|
| `--server` | Coordinator WebSocket URL |
| `--name` | Your peer's display name |
| `--size` | Model size (2.7m/10m/25m/50m) |

---

## Project Structure

```
gltch-2.7m/
├── gltch_2_7m.py          # Core model + terminal training
├── gltch_2_7m_colab.py    # Google Colab version
├── train_with_ui.py       # Dashboard training (--size support)
├── train_continuous.py    # Resumable training
├── train_custom.py        # Train on custom data
├── train_pro.py           # CLI training with sizes
├── chat.py                # Chat interface + voice
├── README.md
├── LICENSE
└── hive/                  # Distributed training
    ├── server.py          # Coordinator
    ├── peer.py            # Training peer (--size support)
    ├── quick_peer.py      # Easy peer connect
    ├── setup_coordinator.sh
    ├── index.html         # Dashboard
    ├── style.css
    └── hive.js
```

---

## Model Architecture

```
GLTCH
├── Token Embedding
├── Position Embedding
├── N × Transformer Blocks
│   ├── Multi-Head Self-Attention
│   ├── Layer Norm
│   ├── Feed Forward (GELU)
│   └── Layer Norm
├── Final Layer Norm
└── Output Head
```

| Size | Layers | Heads | Dim | Context |
|------|--------|-------|-----|---------|
| 2.7M | 6 | 6 | 192 | 128 |
| 10M | 8 | 8 | 384 | 256 |
| 25M | 12 | 8 | 512 | 512 |
| 50M | 12 | 12 | 768 | 512 |

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- `requests` (for data loading)
- `websockets` (for Hive only)

```bash
pip install torch requests websockets
```

---

## Google Colab (Free GPU)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `gltch_2_7m_colab.py`
3. **Runtime → Change runtime type → T4 GPU**
4. Run cells in order

Training takes ~5 minutes on a free T4 GPU.

---

## License

MIT License — see [LICENSE](LICENSE)

## Author

Created by **cyberdreadx**

---

*GLTCH — Generative Language Transformer with Contextual Hierarchy*
