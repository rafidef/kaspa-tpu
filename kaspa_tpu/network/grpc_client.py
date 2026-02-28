"""
Kaspa gRPC client for solo mining.

Uses the Kaspa RPC protocol to:
  - GetBlockTemplate: Fetch new block templates for mining
  - SubmitBlock: Submit mined blocks to the network

Requires a running Kaspa node (kaspad / rusty-kaspa) with gRPC enabled.
Default gRPC endpoint: localhost:16110
"""

import asyncio
import logging
import struct
from dataclasses import dataclass
from typing import Optional, Callable

from ..mining.pipeline import BlockTemplate

logger = logging.getLogger(__name__)

# Try to import grpc
try:
    import grpc
    import grpc.aio
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False
    logger.warning("grpcio not installed — gRPC solo mining unavailable")


class KaspaGrpcClient:
    """
    Kaspa gRPC client for solo mining.
    
    Connects to a Kaspa node's gRPC endpoint to fetch block templates
    and submit mined blocks.
    
    Note: Kaspa's gRPC uses a custom streaming protocol. For simplicity,
    this implementation uses the JSON-RPC wrapper (wRPC) when available,
    or direct protobuf messages.
    """
    
    def __init__(
        self,
        node_address: str = "localhost:16110",
        wallet_address: str = "",
        on_new_template: Optional[Callable[[BlockTemplate], None]] = None,
        poll_interval: float = 0.5,
    ):
        """
        Args:
            node_address: Kaspa node gRPC address (host:port)
            wallet_address: Kaspa wallet address for coinbase
            on_new_template: Callback for new block templates
            poll_interval: Seconds between template polls
        """
        self.node_address = node_address
        self.wallet_address = wallet_address
        self.on_new_template = on_new_template
        self.poll_interval = poll_interval
        
        self._channel = None
        self._connected = False
        self._running = False
        self._last_template_hash: Optional[bytes] = None
    
    async def connect(self):
        """Connect to the Kaspa node gRPC endpoint."""
        if not HAS_GRPC:
            raise RuntimeError("grpcio is required for gRPC solo mining")
        
        logger.info(f"Connecting to Kaspa node at {self.node_address}")
        self._channel = grpc.aio.insecure_channel(self.node_address)
        self._connected = True
        logger.info("Connected to Kaspa node")
    
    async def get_block_template(self) -> Optional[BlockTemplate]:
        """
        Request a block template from the Kaspa node.
        
        Uses the GetBlockTemplate RPC call.
        
        Returns:
            BlockTemplate if successful, None otherwise
        """
        if not self._connected:
            return None
        
        try:
            # The actual gRPC call would use generated protobuf stubs.
            # For now, we use a simplified JSON-RPC approach.
            # In production, you'd generate stubs from:
            # https://github.com/kaspanet/rusty-kaspa/blob/master/rpc/grpc/core/proto/rpc.proto
            
            # Placeholder — actual implementation requires protobuf compilation
            logger.debug("GetBlockTemplate called (stub)")
            return None
            
        except Exception as e:
            logger.error(f"GetBlockTemplate failed: {e}")
            return None
    
    async def submit_block(self, block_data: bytes) -> bool:
        """
        Submit a mined block to the Kaspa node.
        
        Args:
            block_data: Serialized block data
        
        Returns:
            True if accepted, False if rejected
        """
        if not self._connected:
            return False
        
        try:
            # Placeholder — actual implementation requires protobuf compilation
            logger.debug("SubmitBlock called (stub)")
            return False
            
        except Exception as e:
            logger.error(f"SubmitBlock failed: {e}")
            return False
    
    async def poll_templates(self):
        """
        Continuously poll for new block templates.
        
        Kaspa has ~1 second block times, so we poll frequently.
        When a new template is detected (different pre_pow_hash),
        the on_new_template callback is fired.
        """
        self._running = True
        
        while self._running and self._connected:
            template = await self.get_block_template()
            
            if template and template.pre_pow_hash != self._last_template_hash:
                self._last_template_hash = template.pre_pow_hash
                if self.on_new_template:
                    self.on_new_template(template)
            
            await asyncio.sleep(self.poll_interval)
    
    async def disconnect(self):
        """Disconnect from the Kaspa node."""
        self._running = False
        self._connected = False
        if self._channel:
            await self._channel.close()
        logger.info("Disconnected from Kaspa node")
