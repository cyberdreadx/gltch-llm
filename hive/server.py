"""
GLTCH HIVE — Coordinator Server
================================
WebSocket server for distributed training coordination.

Peers connect and send gradients, server aggregates and broadcasts.

Usage:
    python server.py                     # Generate new secret key
    python server.py --key YOUR_KEY      # Use custom key

Created by: cyberdreadx
"""

import asyncio
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Set
import http.server
import threading
import os
import secrets
import argparse

# Try to import websockets, provide install instructions if not found
try:
    import websockets
except ImportError:
    print("❌ websockets not installed. Run: pip install websockets")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GLTCH-Hive")

# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class Peer:
    id: str
    name: str
    gpu: str
    connected_at: str
    websocket: any = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "gpu": self.gpu,
            "connected_at": self.connected_at
        }


class TrainingState:
    def __init__(self):
        self.step = 0
        self.loss = 4.5
        self.gradients = {}
        self.batch_size = 0


# ============================================
# HIVE SERVER
# ============================================

class HiveServer:
    def __init__(self, host="0.0.0.0", ws_port=8765, http_port=8080, secret_key=None):
        self.host = host
        self.ws_port = ws_port
        self.http_port = http_port
        self.peers: Dict[str, Peer] = {}
        self.dashboard_clients: Set = set()
        self.training = TrainingState()
        self.peer_counter = 0
        
        # Authentication
        if secret_key:
            self.secret_key = secret_key
        else:
            self.secret_key = secrets.token_urlsafe(16)
        
    def verify_key(self, provided_key: str) -> bool:
        """Verify provided key matches server key"""
        return secrets.compare_digest(provided_key or "", self.secret_key)
    
    async def handle_peer(self, websocket):
        """Handle incoming peer connection"""
        peer_id = None
        
        try:
            # Wait for registration message
            message = await websocket.recv()
            data = json.loads(message)
            
            if data.get("type") == "register":
                # Verify secret key
                if not self.verify_key(data.get("key")):
                    logger.warning(f"❌ Rejected connection: Invalid key from {data.get('name', 'unknown')}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid secret key. Connection rejected."
                    }))
                    await websocket.close()
                    return
                
                self.peer_counter += 1
                peer_id = f"peer-{self.peer_counter:04d}"
                
                peer = Peer(
                    id=peer_id,
                    name=data.get("name", f"node-{self.peer_counter}"),
                    gpu=data.get("gpu", "Unknown GPU"),
                    connected_at=datetime.now().isoformat(),
                    websocket=websocket
                )
                
                self.peers[peer_id] = peer
                logger.info(f"🔗 Peer joined: {peer.name} ({peer.gpu})")
                
                # Send confirmation
                await websocket.send(json.dumps({
                    "type": "registered",
                    "peer_id": peer_id,
                    "training_step": self.training.step
                }))
                
                # Notify dashboard
                await self.broadcast_to_dashboard({
                    "type": "peer_joined",
                    "peer": peer.to_dict()
                })
                
                # Handle messages from peer
                async for message in websocket:
                    await self.handle_peer_message(peer_id, message)
            
            elif data.get("type") == "dashboard":
                # Dashboard is public (read-only view)
                self.dashboard_clients.add(websocket)
                logger.info("👁️  Viewer connected (public dashboard)")
                
                # Send current state
                await websocket.send(json.dumps({
                    "type": "initial_state",
                    "peers": [p.to_dict() for p in self.peers.values()],
                    "training": {
                        "step": self.training.step,
                        "loss": self.training.loss
                    }
                }))
                
                # Keep alive
                async for message in websocket:
                    pass  # Dashboard doesn't send messages
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if peer_id and peer_id in self.peers:
                logger.info(f"🔌 Peer disconnected: {self.peers[peer_id].name}")
                del self.peers[peer_id]
                await self.broadcast_to_dashboard({
                    "type": "peer_left",
                    "peerId": peer_id
                })
            
            if websocket in self.dashboard_clients:
                self.dashboard_clients.discard(websocket)
                logger.info("📊 Dashboard disconnected")
    
    async def handle_peer_message(self, peer_id: str, message: str):
        """Process message from a training peer"""
        data = json.loads(message)
        
        if data["type"] == "gradient":
            # Store gradient from this peer
            self.training.gradients[peer_id] = data["gradient"]
            
            # Notify dashboard of activity
            await self.broadcast_to_dashboard({
                "type": "gradient_sync",
                "peerId": peer_id
            })
            
            # Check if we have gradients from all peers
            if len(self.training.gradients) >= len(self.peers):
                await self.aggregate_and_broadcast()
        
        elif data["type"] == "heartbeat":
            # Peer is still alive
            pass
    
    async def aggregate_and_broadcast(self):
        """Aggregate gradients and broadcast to all peers"""
        if not self.training.gradients:
            return
        
        # Simple averaging of gradients (in real implementation, this would be proper gradient aggregation)
        self.training.step += 1
        self.training.loss = max(0.1, self.training.loss * 0.999)  # Simulate decreasing loss
        
        # Clear gradients for next round
        self.training.gradients = {}
        
        # Broadcast aggregated gradient to all peers
        update = {
            "type": "aggregated_gradient",
            "step": self.training.step,
            "loss": self.training.loss
        }
        
        await self.broadcast_to_peers(update)
        
        # Update dashboard
        await self.broadcast_to_dashboard({
            "type": "training_update",
            "step": self.training.step,
            "loss": self.training.loss
        })
        
        if self.training.step % 100 == 0:
            logger.info(f"📈 Step {self.training.step} | Loss: {self.training.loss:.4f}")
    
    async def broadcast_to_peers(self, message: dict):
        """Send message to all training peers"""
        if not self.peers:
            return
        
        msg = json.dumps(message)
        await asyncio.gather(*[
            peer.websocket.send(msg) 
            for peer in self.peers.values() 
            if peer.websocket
        ], return_exceptions=True)
    
    async def broadcast_to_dashboard(self, message: dict):
        """Send message to all dashboard clients"""
        if not self.dashboard_clients:
            return
        
        msg = json.dumps(message)
        await asyncio.gather(*[
            client.send(msg) 
            for client in self.dashboard_clients
        ], return_exceptions=True)
    
    def start_http_server(self):
        """Start simple HTTP server for dashboard files"""
        hive_dir = os.path.dirname(os.path.abspath(__file__))
        
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=hive_dir, **kwargs)
            
            def log_message(self, format, *args):
                pass  # Suppress HTTP logs
        
        server = http.server.HTTPServer((self.host, self.http_port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"🌐 Dashboard: http://{self.host}:{self.http_port}")
    
    async def run(self):
        """Start the hive server"""
        print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║    ██████╗ ██╗  ████████╗ ██████╗██╗  ██╗    ██╗  ██╗██╗██╗   ██╗███████╗     ║
║   ██╔════╝ ██║  ╚══██╔══╝██╔════╝██║  ██║    ██║  ██║██║██║   ██║██╔════╝     ║
║   ██║  ███╗██║     ██║   ██║     ███████║    ███████║██║██║   ██║█████╗       ║
║   ██║   ██║██║     ██║   ██║     ██╔══██║    ██╔══██║██║╚██╗ ██╔╝██╔══╝       ║
║   ╚██████╔╝███████╗██║   ╚██████╗██║  ██║    ██║  ██║██║ ╚████╔╝ ███████╗     ║
║    ╚═════╝ ╚══════╝╚═╝    ╚═════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝     ║
║                                                                               ║
║   Generative Language Transformer with Contextual Hierarchy — COORDINATOR    ║
║   Created by: cyberdreadx                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """)
        
        # Start HTTP server for dashboard
        self.start_http_server()
        
        # Start WebSocket server
        logger.info(f"🔌 WebSocket server: ws://{self.host}:{self.ws_port}")
        logger.info("⏳ Waiting for peers to connect...")
        
        async with websockets.serve(self.handle_peer, self.host, self.ws_port):
            await asyncio.Future()  # Run forever


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLTCH Hive Coordinator")
    parser.add_argument("--key", help="Secret key for authentication (auto-generated if not provided)")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--http-port", type=int, default=8080, help="HTTP dashboard port")
    args = parser.parse_args()
    
    server = HiveServer(secret_key=args.key, ws_port=args.ws_port, http_port=args.http_port)
    
    # Display secret key prominently
    print("\n" + "=" * 60)
    print("🔐 SECRET KEY (share only with trusted peers):")
    print(f"   {server.secret_key}")
    print("=" * 60 + "\n")
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down GLTCH Hive...")

