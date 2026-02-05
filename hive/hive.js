/**
 * GLTCH HIVE — Distributed Training Visualization
 * ================================================
 * Animated node graph showing connected peers
 * Created by: cyberdreadx
 */

class HiveVisualization {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.nodes = [];
        this.connections = [];
        this.centerX = 0;
        this.centerY = 0;
        this.time = 0;

        // Drag state
        this.isDragging = false;
        this.dragNode = null;
        this.hoveredNode = null;
        this.mouseX = 0;
        this.mouseY = 0;

        this.resize();
        window.addEventListener('resize', () => this.resize());

        // Mouse events for dragging
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', () => this.onMouseUp());
        this.canvas.addEventListener('mouseleave', () => this.onMouseUp());

        // Start animation loop
        this.animate();
    }

    onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Find clicked node
        for (const node of this.nodes) {
            const dist = Math.sqrt((node.x - x) ** 2 + (node.y - y) ** 2);
            if (dist < node.size + 10) {
                this.isDragging = true;
                this.dragNode = node;
                this.canvas.style.cursor = 'grabbing';
                return;
            }
        }
    }

    onMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        this.mouseX = e.clientX - rect.left;
        this.mouseY = e.clientY - rect.top;

        if (this.isDragging && this.dragNode) {
            this.dragNode.x = this.mouseX;
            this.dragNode.y = this.mouseY;
            this.dragNode.targetX = this.mouseX;
            this.dragNode.targetY = this.mouseY;
        } else {
            // Check for hover
            this.hoveredNode = null;
            for (const node of this.nodes) {
                const dist = Math.sqrt((node.x - this.mouseX) ** 2 + (node.y - this.mouseY) ** 2);
                if (dist < node.size + 10) {
                    this.hoveredNode = node;
                    this.canvas.style.cursor = 'grab';
                    return;
                }
            }
            this.canvas.style.cursor = 'default';
        }
    }

    onMouseUp() {
        this.isDragging = false;
        this.dragNode = null;
        this.canvas.style.cursor = this.hoveredNode ? 'grab' : 'default';
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        // Set canvas size accounting for DPI
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;

        // Scale canvas CSS size to match
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';

        // Scale context to match DPI
        this.ctx.scale(dpr, dpr);

        this.centerX = rect.width / 2;
        this.centerY = rect.height / 2;
    }

    addNode(peer) {
        const angle = (this.nodes.length / 8) * Math.PI * 2 + Math.random() * 0.5;
        const radius = 120 + Math.random() * 100;

        const node = {
            id: peer.id,
            name: peer.name,
            gpu: peer.gpu,
            x: this.centerX + Math.cos(angle) * radius,
            y: this.centerY + Math.sin(angle) * radius,
            targetX: this.centerX + Math.cos(angle) * radius,
            targetY: this.centerY + Math.sin(angle) * radius,
            size: 20 + Math.random() * 10,
            pulse: Math.random() * Math.PI * 2,
            color: this.getGpuColor(peer.gpu),
            active: true
        };

        this.nodes.push(node);
        this.connections.push({
            node: node,
            activity: 0,
            lastPulse: 0
        });

        return node;
    }

    removeNode(peerId) {
        this.nodes = this.nodes.filter(n => n.id !== peerId);
        this.connections = this.connections.filter(c => c.node.id !== peerId);
    }

    getGpuColor(gpu) {
        if (gpu.includes('4090') || gpu.includes('H100')) return '#00ff88';
        if (gpu.includes('4080') || gpu.includes('A100')) return '#00f5ff';
        if (gpu.includes('3090') || gpu.includes('3080')) return '#bf00ff';
        return '#ff0080';
    }

    pulseConnection(peerId) {
        const conn = this.connections.find(c => c.node.id === peerId);
        if (conn) {
            conn.activity = 1;
            conn.lastPulse = this.time;
        }
    }

    animate() {
        this.time += 0.016;
        this.draw();
        requestAnimationFrame(() => this.animate());
    }

    draw() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Clear with fade effect
        ctx.fillStyle = 'rgba(10, 10, 15, 0.1)';
        ctx.fillRect(0, 0, w, h);

        // Draw honeycomb background pattern
        this.drawHoneycomb();

        // Draw connections
        this.connections.forEach(conn => {
            this.drawConnection(conn);
            conn.activity *= 0.95;
        });

        // Draw nodes
        this.nodes.forEach(node => {
            this.drawNode(node);
        });

        // Draw center node
        this.drawCenterNode();
    }

    drawHoneycomb() {
        const ctx = this.ctx;
        const size = 40;
        const rows = Math.ceil(this.canvas.height / (size * 1.5)) + 1;
        const cols = Math.ceil(this.canvas.width / (size * 1.73)) + 1;

        ctx.strokeStyle = 'rgba(0, 245, 255, 0.03)';
        ctx.lineWidth = 1;

        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const x = col * size * 1.73 + (row % 2) * size * 0.866;
                const y = row * size * 1.5;
                this.drawHexagon(x, y, size * 0.5);
            }
        }
    }

    drawHexagon(x, y, size) {
        const ctx = this.ctx;
        ctx.beginPath();
        for (let i = 0; i < 6; i++) {
            const angle = (i / 6) * Math.PI * 2 - Math.PI / 2;
            const px = x + Math.cos(angle) * size;
            const py = y + Math.sin(angle) * size;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.stroke();
    }

    drawConnection(conn) {
        const ctx = this.ctx;
        const node = conn.node;

        // Line from node to center
        const gradient = ctx.createLinearGradient(
            node.x, node.y, this.centerX, this.centerY
        );

        const alpha = 0.2 + conn.activity * 0.8;
        gradient.addColorStop(0, node.color);
        gradient.addColorStop(1, `rgba(0, 245, 255, ${alpha * 0.3})`);

        ctx.beginPath();
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 2 + conn.activity * 3;
        ctx.moveTo(node.x, node.y);
        ctx.lineTo(this.centerX, this.centerY);
        ctx.stroke();

        // Animated pulse along connection
        if (conn.activity > 0.1) {
            const pulseT = (this.time - conn.lastPulse) * 2 % 1;
            const pulseX = node.x + (this.centerX - node.x) * pulseT;
            const pulseY = node.y + (this.centerY - node.y) * pulseT;

            ctx.beginPath();
            ctx.arc(pulseX, pulseY, 5 * conn.activity, 0, Math.PI * 2);
            ctx.fillStyle = node.color;
            ctx.fill();
        }
    }

    drawNode(node) {
        const ctx = this.ctx;
        const isHovered = this.hoveredNode === node;
        const isDragged = this.dragNode === node;

        // Smooth movement (skip if dragging)
        if (!isDragged) {
            node.x += (node.targetX - node.x) * 0.05;
            node.y += (node.targetY - node.y) * 0.05;
        }

        // Floating animation (reduce when hovered/dragged)
        const floatY = (isHovered || isDragged) ? 0 : Math.sin(this.time * 2 + node.pulse) * 5;
        const y = node.y + floatY;

        // Size boost on hover
        const sizeMultiplier = isHovered ? 1.2 : 1;
        const size = node.size * sizeMultiplier;

        // Outer glow ring
        const glowSize = size + 20 + Math.sin(this.time * 3 + node.pulse) * 8;
        const glowGradient = ctx.createRadialGradient(node.x, y, size * 0.8, node.x, y, glowSize);
        glowGradient.addColorStop(0, node.color + '60');
        glowGradient.addColorStop(0.5, node.color + '20');
        glowGradient.addColorStop(1, 'transparent');

        ctx.beginPath();
        ctx.arc(node.x, y, glowSize, 0, Math.PI * 2);
        ctx.fillStyle = glowGradient;
        ctx.fill();

        // Main bubble - smooth circle with gradient (bubblemaps style)
        const bubbleGradient = ctx.createRadialGradient(
            node.x - size * 0.3, y - size * 0.3, 0,
            node.x, y, size
        );
        bubbleGradient.addColorStop(0, '#ffffff30');
        bubbleGradient.addColorStop(0.3, node.color);
        bubbleGradient.addColorStop(1, node.color + '80');

        ctx.beginPath();
        ctx.arc(node.x, y, size, 0, Math.PI * 2);
        ctx.fillStyle = bubbleGradient;
        ctx.fill();

        // Inner highlight (glass effect)
        const highlightGradient = ctx.createRadialGradient(
            node.x - size * 0.25, y - size * 0.25, 0,
            node.x - size * 0.25, y - size * 0.25, size * 0.5
        );
        highlightGradient.addColorStop(0, 'rgba(255, 255, 255, 0.4)');
        highlightGradient.addColorStop(1, 'transparent');

        ctx.beginPath();
        ctx.arc(node.x - size * 0.2, y - size * 0.2, size * 0.4, 0, Math.PI * 2);
        ctx.fillStyle = highlightGradient;
        ctx.fill();

        // Border ring
        ctx.beginPath();
        ctx.arc(node.x, y, size, 0, Math.PI * 2);
        ctx.strokeStyle = isHovered ? '#ffffff' : node.color;
        ctx.lineWidth = isHovered ? 3 : 2;
        ctx.stroke();

        // Node label
        ctx.fillStyle = '#fff';
        ctx.font = `${isHovered ? 12 : 11}px Orbitron`;
        ctx.textAlign = 'center';
        ctx.fillText(node.name, node.x, y + size + 18);

        // GPU label
        ctx.fillStyle = node.color;
        ctx.font = '9px Rajdhani';
        ctx.fillText(node.gpu, node.x, y + size + 30);
    }

    drawCenterNode() {
        const ctx = this.ctx;
        const pulse = Math.sin(this.time * 2) * 5;

        // Outer glow
        const gradient = ctx.createRadialGradient(
            this.centerX, this.centerY, 0,
            this.centerX, this.centerY, 80 + pulse
        );
        gradient.addColorStop(0, 'rgba(0, 245, 255, 0.3)');
        gradient.addColorStop(0.5, 'rgba(0, 245, 255, 0.1)');
        gradient.addColorStop(1, 'transparent');

        ctx.beginPath();
        ctx.arc(this.centerX, this.centerY, 80 + pulse, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
    }
}


// ============================================
// STATS & WEBSOCKET CONNECTION
// ============================================

class HiveNetwork {
    constructor() {
        this.hive = null;
        this.ws = null;
        this.stats = {
            peerCount: 0,
            currentLoss: null,
            totalTflops: 0,
            trainingStep: 0
        };
        this.lossHistory = [];
        this.peers = new Map();

        this.init();
    }

    init() {
        // Initialize hive visualization
        const canvas = document.getElementById('hive-canvas');
        this.hive = new HiveVisualization(canvas);

        // Initialize loss chart
        this.initLossChart();

        // Connect to WebSocket
        this.connect();
    }

    connect() {
        const wsUrl = `ws://${window.location.hostname}:8765`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                this.setConnectionStatus(true);
                console.log('Connected to GLTCH Hive server');

                // Register as dashboard client
                this.ws.send(JSON.stringify({ type: 'dashboard' }));
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            };

            this.ws.onclose = () => {
                this.setConnectionStatus(false);
                setTimeout(() => this.connect(), 5000);
            };

            this.ws.onerror = () => {
                this.setConnectionStatus(false);
            };
        } catch (e) {
            this.setConnectionStatus(false);
        }
    }

    setConnectionStatus(connected) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.getElementById('connection-status');

        if (connected) {
            statusDot.classList.remove('disconnected');
            statusText.textContent = 'Connected';
        } else {
            statusDot.classList.add('disconnected');
            statusText.textContent = 'Disconnected';
        }
    }

    handleMessage(data) {
        switch (data.type) {
            case 'initial_state':
                // Load existing peers and training state when dashboard connects
                if (data.peers) {
                    data.peers.forEach(peer => this.addPeer(peer));
                }
                if (data.training) {
                    this.stats.trainingStep = data.training.step || 0;
                    this.stats.currentLoss = data.training.loss;
                    this.updateStats();
                }
                break;
            case 'peer_joined':
                this.addPeer(data.peer);
                break;
            case 'peer_left':
                this.removePeer(data.peerId);
                break;
            case 'training_update':
                this.updateTraining(data);
                break;
            case 'gradient_sync':
                this.hive.pulseConnection(data.peerId);
                break;
        }
    }

    addPeer(peer) {
        this.peers.set(peer.id, peer);
        this.hive.addNode(peer);
        this.updatePeerList();
        this.updateStats();
    }

    removePeer(peerId) {
        this.peers.delete(peerId);
        this.hive.removeNode(peerId);
        this.updatePeerList();
        this.updateStats();
    }

    updateTraining(data) {
        this.stats.currentLoss = data.loss;
        this.stats.trainingStep = data.step;

        this.lossHistory.push({ x: data.step, y: data.loss });
        if (this.lossHistory.length > 100) {
            this.lossHistory.shift();
        }

        this.updateStats();
        this.updateLossChart();

        // Pulse a random connection to show activity
        if (this.peers.size > 0) {
            const peerIds = Array.from(this.peers.keys());
            const randomId = peerIds[Math.floor(Math.random() * peerIds.length)];
            this.hive.pulseConnection(randomId);
        }
    }

    updateStats() {
        document.getElementById('peer-count').textContent = this.peers.size;
        document.getElementById('current-loss').textContent =
            this.stats.currentLoss?.toFixed(4) || '—';
        document.getElementById('total-tflops').textContent =
            this.calculateTflops().toFixed(1);
        document.getElementById('training-step').textContent =
            this.stats.trainingStep.toLocaleString();
    }

    calculateTflops() {
        let total = 0;
        this.peers.forEach(peer => {
            // Rough TFLOPS estimates
            if (peer.gpu.includes('H100')) total += 1979;
            else if (peer.gpu.includes('A100')) total += 312;
            else if (peer.gpu.includes('4090')) total += 83;
            else if (peer.gpu.includes('4080')) total += 49;
            else if (peer.gpu.includes('3090')) total += 36;
            else if (peer.gpu.includes('3080')) total += 30;
            else if (peer.gpu.includes('T4')) total += 8;
            else total += 10;
        });
        return total;
    }

    updatePeerList() {
        const container = document.getElementById('peer-list-container');

        if (this.peers.size === 0) {
            container.innerHTML = '<div class="no-peers">Waiting for peers to connect...</div>';
            return;
        }

        container.innerHTML = '';
        this.peers.forEach(peer => {
            const div = document.createElement('div');
            div.className = 'peer-item';
            div.innerHTML = `
                <div class="peer-dot"></div>
                <span class="peer-name">${peer.name}</span>
                <span class="peer-gpu">${peer.gpu}</span>
            `;
            container.appendChild(div);
        });
    }

    initLossChart() {
        this.lossCanvas = document.getElementById('loss-chart');
        this.lossCtx = this.lossCanvas.getContext('2d');
        this.drawLossChart();
    }

    updateLossChart() {
        this.drawLossChart();
    }

    drawLossChart() {
        const ctx = this.lossCtx;
        const w = this.lossCanvas.width = this.lossCanvas.offsetWidth;
        const h = this.lossCanvas.height = 120;

        ctx.clearRect(0, 0, w, h);

        if (this.lossHistory.length < 2) return;

        const maxLoss = Math.max(...this.lossHistory.map(p => p.y));
        const minLoss = Math.min(...this.lossHistory.map(p => p.y));
        const range = maxLoss - minLoss || 1;

        ctx.beginPath();
        ctx.strokeStyle = '#00f5ff';
        ctx.lineWidth = 2;

        this.lossHistory.forEach((point, i) => {
            const x = (i / (this.lossHistory.length - 1)) * w;
            const y = h - ((point.y - minLoss) / range) * (h - 20) - 10;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });

        ctx.stroke();

        // Glow effect
        ctx.shadowColor = '#00f5ff';
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    // Demo mode for testing without server
    startDemoMode() {
        console.log('Starting demo mode...');
        document.getElementById('connection-status').textContent = 'Demo Mode';

        const demoGpus = ['RTX 4090', 'RTX 3090', 'RTX 4080', 'A100', 'T4 (Colab)'];
        const demoNames = ['node-alpha', 'node-beta', 'node-gamma', 'node-delta', 'node-epsilon'];

        // Add demo peers
        demoNames.forEach((name, i) => {
            setTimeout(() => {
                this.addPeer({
                    id: `demo-${i}`,
                    name: name,
                    gpu: demoGpus[i % demoGpus.length]
                });
            }, i * 1000);
        });

        // Simulate training
        let step = 0;
        let loss = 4.5;

        setInterval(() => {
            step += 1;
            loss = Math.max(0.5, loss - 0.01 + (Math.random() - 0.5) * 0.02);

            this.updateTraining({
                step: step,
                loss: loss
            });
        }, 200);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    window.hiveNetwork = new HiveNetwork();
});
