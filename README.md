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

---

## 🌐 Join the GLTCH Hive

**Contribute your GPU to train AI together!** Join our distributed training network and help build open-source language models.

### One-Click Join (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cyberdreadx/gltch-llm/blob/main/gltch_hive_peer.ipynb)

### Manual Join

```bash
pip install torch websockets requests
python hive/peer.py --server ws://76.13.121.10:8765 --key PUBLIC_KEY --name my-gpu
```

### 📊 Live Dashboard

Watch training progress in real-time: **[https://hive.gltch.app](https://hive.gltch.app)**

---

## Features

- 🧠 **Multiple model sizes** — 1M, 2.7M, 10M, 25M, or 50M parameters
- 📊 **Live training dashboard** — Real-time loss curve and generated samples
- 💬 **Chat interface** — Talk to your trained model with voice output
- 🌐 **Distributed training (Hive)** — Train across multiple machines/GPUs
- 🎮 **Works everywhere** — GPU, CPU, low VRAM, or Colab free tier
- 📱 **Mobile support** — Train on Android via Termux

---

## Quick Start

### Option 1: Train with Dashboard (Recommended)

```bash
git clone https://github.com/cyberdreadx/gltch-llm.git
cd gltch-llm
pip install torch requests

# Train 2.7M model with live dashboard
python train_with_ui.py

# Or train a larger model
python train_with_ui.py --size 10m
```

Opens a browser dashboard showing loss curve, speed, ETA, and generated samples.

### Option 2: Train on Custom Data

```bash
# Train on your own text file
python train_with_ui.py --data your_novel.txt

# Resume from a checkpoint
python train_with_ui.py --resume checkpoint_step_500.pt

# Custom checkpoint interval
python train_with_ui.py --data mydata.txt --checkpoint-interval 250
```

### Option 3: Chat with Trained Model

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

| Size | Params | VRAM | Training Time | Best For |
|------|--------|------|---------------|----------|
| `1m` | ~1M | <1GB | ~2 min | Mobile, CPU, testing |
| `2.7m` | 2.7M | ~1GB | ~5 min | Quick experiments |
| `10m` | ~10M | ~2GB | ~15 min | Good quality |
| `25m` | ~25M | ~4GB | ~30 min | Better quality |
| `50m` | ~50M | ~8GB | ~1 hour | Best quality |

```bash
# Examples
python train_with_ui.py --size 1m     # Micro (mobile/CPU)
python train_with_ui.py --size 2.7m   # Default
python train_with_ui.py --size 10m    # Larger
python train_with_ui.py --size 50m    # Largest
```

---

## Compatibility Options

### CPU-Only Training

GLTCH automatically detects if no GPU is available and runs on CPU:

```bash
# Force CPU mode
python train_with_ui.py --device cpu --size 1m
```

### Low VRAM Mode

For GPUs with limited memory (2-4GB):

```bash
python train_with_ui.py --low-vram --size 2.7m
```

This reduces batch size and context length to fit in memory.

### Android (Termux)

Train on your Android phone:

```bash
# Install Termux from F-Droid
pkg install python
pip install torch requests
git clone https://github.com/cyberdreadx/gltch-llm.git
cd gltch-llm
python train_with_ui.py --size 1m --device cpu
```

### Google Colab (Free GPU)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `gltch_2_7m_colab.py` or use our Hive notebook
3. **Runtime → Change runtime type → T4 GPU**
4. Run cells in order

Training takes ~5 minutes on a free T4 GPU.

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
│   RTX 4090    │    │   Colab T4    │    │   Android     │
│   peer.py     │    │   peer.py     │    │   peer.py     │
└───────────────┘    └───────────────┘    └───────────────┘
```

### How It Works

1. **Each peer trains independently** on their own device
2. **Peers compute gradients** (how to adjust the model)
3. **Gradients are sent to the coordinator**
4. **Coordinator averages all gradients** from every peer
5. **Updated weights are broadcast** back to everyone
6. **Repeat** — all peers stay perfectly in sync

### Why Is It Faster?

More peers = more gradient samples per training step = **faster convergence**.

| Peers | Combined Compute | Speedup |
|-------|------------------|---------|
| 1× RTX 4090 | 83 TFLOPS | 1× |
| + 1× Colab T4 | 91 TFLOPS | ~1.1× |
| + 2× more GPUs | 150+ TFLOPS | ~1.8× |

The model trains faster because you're effectively processing more data in parallel. This is called **data-parallel distributed training**.

### Join as a Peer

**Easiest**: Use Google Colab (click badge above)

**Manual**:
```bash
git clone https://github.com/cyberdreadx/gltch-llm.git
cd gltch-llm
pip install torch websockets requests

# Connect to public hive
python hive/peer.py --server ws://76.13.121.10:8765 --key PUBLIC_KEY --name my-gpu --size 10m
```

### Peer Options

| Option | Description | Default |
|--------|-------------|---------|
| `--server` | Coordinator WebSocket URL | Required |
| `--key` | Authentication key | Required |
| `--name` | Your peer's display name | `peer-XXXX` |
| `--size` | Model size (1m/2.7m/10m/25m/50m) | `2.7m` |
| `--low-vram` | Reduce memory usage | Off |

### Run Your Own Coordinator

```bash
# On your VPS
git clone https://github.com/cyberdreadx/gltch-llm.git
cd gltch-llm/hive
pip install websockets requests torch

# Start server with custom data
python server.py --data your_training_data.txt

# With checkpoint resume
python server.py --resume checkpoint.pt --checkpoint-interval 500
```

Dashboard: `http://YOUR_VPS_IP:8080`

---

## CLI Flags Reference

### train_with_ui.py

| Flag | Description |
|------|-------------|
| `--size` | Model size: 1m, 2.7m, 10m, 25m, 50m |
| `--data PATH` | Custom training data file |
| `--resume PATH` | Resume from checkpoint |
| `--checkpoint-interval N` | Save checkpoint every N steps |
| `--device` | Force cpu or cuda |
| `--low-vram` | Reduce memory usage |

### hive/peer.py

| Flag | Description |
|------|-------------|
| `--server URL` | Coordinator WebSocket URL |
| `--key KEY` | Authentication key |
| `--name NAME` | Peer display name |
| `--size` | Model size |
| `--low-vram` | Low memory mode |

### hive/server.py

| Flag | Description |
|------|-------------|
| `--data PATH` | Training data file |
| `--resume PATH` | Resume from checkpoint |
| `--checkpoint-interval N` | Checkpoint frequency |
| `--ws-port` | WebSocket port (default: 8765) |
| `--http-port` | Dashboard port (default: 8080) |

---

## Project Structure

```
gltch-llm/
├── train_with_ui.py       # Dashboard training
├── chat.py                # Chat interface + voice
├── gltch_2_7m.py          # Core model
├── gltch_2_7m_colab.py    # Colab version
├── gltch_hive_peer.ipynb  # Colab peer notebook
├── hive/
│   ├── server.py          # Coordinator
│   ├── peer.py            # Training peer
│   ├── index.html         # Dashboard
│   ├── style.css
│   └── hive.js
└── website/               # gltch.app site
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
| 1M | 4 | 4 | 128 | 64 |
| 2.7M | 6 | 6 | 192 | 128 |
| 10M | 8 | 8 | 384 | 256 |
| 25M | 12 | 8 | 512 | 512 |
| 50M | 12 | 12 | 768 | 512 |

---

## Requirements

- Python 3.8+
- PyTorch 2.0+ (or 1.x for CPU)
- `requests` (for data loading)
- `websockets` (for Hive only)

```bash
pip install torch requests websockets
```

---

## License

MIT License — see [LICENSE](LICENSE)

## Author

Created by **cyberdreadx**

---

*GLTCH — Generative Language Transformer with Contextual Hierarchy*
