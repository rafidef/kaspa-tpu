"""
Kaspa Stratum protocol client for pool mining.

Implements the Kaspa Common Stratum Protocol (JSON-RPC 2.0 over TCP).
Compatible with kaspa-stratum-bridge and major pools (WoolyPooly, F2Pool, etc).

Protocol flow:
  1. mining.subscribe → server info
  2. mining.authorize(address, "tpu-miner") → authentication
  3. Server sends mining.notify with new jobs
  4. Client sends mining.submit with found nonces
"""

import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass
from typing import Optional, Callable

from ..mining.pipeline import BlockTemplate

logger = logging.getLogger(__name__)


@dataclass
class StratumJob:
    """A mining job received from the stratum server."""
    job_id: str
    pre_pow_hash: bytes      # 32 bytes
    timestamp: int
    target_bits: int
    nonce_start: int = 0
    nonce_end: int = (1 << 64)
    clean_jobs: bool = True


class StratumClient:
    """
    Kaspa Stratum protocol client.
    
    Connects to a Kaspa mining pool via the Common Stratum Protocol.
    Receives block templates as mining.notify messages and submits
    solutions via mining.submit.
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        wallet_address: str,
        worker_name: str = "kaspa-tpu",
        on_new_job: Optional[Callable[[BlockTemplate], None]] = None,
    ):
        """
        Args:
            host: Stratum server hostname
            port: Stratum server port
            wallet_address: Kaspa wallet address for mining rewards
            worker_name: Worker identifier
            on_new_job: Callback when a new mining job is received
        """
        self.host = host
        self.port = port
        self.wallet_address = wallet_address
        self.worker_name = worker_name
        self.on_new_job = on_new_job
        
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._request_id = 0
        self._connected = False
        self._authorized = False
        self._current_job: Optional[StratumJob] = None
        self._extranonce: str = ""
    
    async def connect(self):
        """Connect to the stratum server and start the mining session."""
        logger.info(f"Connecting to stratum://{self.host}:{self.port}")
        
        self._reader, self._writer = await asyncio.open_connection(
            self.host, self.port
        )
        self._connected = True
        logger.info("Connected to stratum server")
        
        # Subscribe
        await self._subscribe()
        
        # Authorize
        await self._authorize()
    
    async def _subscribe(self):
        """Send mining.subscribe."""
        response = await self._call("mining.subscribe", [
            "kaspa-tpu-miner/0.1.0",
        ])
        if response and response.get("result"):
            result = response["result"]
            if isinstance(result, list) and len(result) > 1:
                self._extranonce = result[1] if isinstance(result[1], str) else ""
            logger.info(f"Subscribed, extranonce: {self._extranonce}")
        else:
            logger.warning("Subscribe response unexpected, continuing...")
    
    async def _authorize(self):
        """Send mining.authorize."""
        response = await self._call("mining.authorize", [
            self.wallet_address,
            self.worker_name,
        ])
        if response and response.get("result"):
            self._authorized = True
            logger.info(f"Authorized as {self.wallet_address}")
        else:
            raise ConnectionError(
                f"Authorization failed: {response.get('error') if response else 'no response'}"
            )
    
    async def submit_solution(self, job_id: str, nonce: int) -> bool:
        """
        Submit a mining solution to the pool.
        
        Args:
            job_id: The job ID from the mining.notify
            nonce: The valid nonce found
        
        Returns:
            True if accepted, False if rejected
        """
        nonce_hex = format(nonce, '016x')
        response = await self._call("mining.submit", [
            self.wallet_address,
            job_id,
            nonce_hex,
        ])
        
        if response and response.get("result"):
            logger.info(f"Share accepted! nonce={nonce_hex}")
            return True
        else:
            error = response.get("error") if response else "no response"
            logger.warning(f"Share rejected: {error}")
            return False
    
    async def listen(self):
        """
        Listen for server notifications (mining.notify, etc.)
        
        Runs continuously, dispatching new jobs to on_new_job callback.
        """
        while self._connected:
            try:
                line = await self._reader.readline()
                if not line:
                    logger.warning("Connection closed by server")
                    self._connected = False
                    break
                
                message = json.loads(line.decode('utf-8').strip())
                await self._handle_message(message)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON from server: {e}")
            except Exception as e:
                logger.error(f"Error in stratum listener: {e}")
                self._connected = False
                break
    
    async def _handle_message(self, message: dict):
        """Handle an incoming stratum message."""
        method = message.get("method")
        
        if method == "mining.notify":
            await self._handle_notify(message.get("params", []))
        elif method == "mining.set_difficulty":
            await self._handle_set_difficulty(message.get("params", []))
        elif "id" in message:
            # This is a response, handled by _call
            pass
        else:
            logger.debug(f"Unknown message: {message}")
    
    async def _handle_notify(self, params: list):
        """
        Handle mining.notify — new job from the pool.
        
        Kaspa stratum notify params typically:
          [job_id, pre_pow_hash_hex, timestamp_hex, target_bits_hex, clean_jobs]
        """
        if len(params) < 4:
            logger.warning(f"mining.notify with insufficient params: {params}")
            return
        
        job_id = str(params[0])
        pre_pow_hash_hex = str(params[1])
        
        # Parse timestamp and target_bits (may be hex strings or integers)
        timestamp = (
            int(params[2], 16) if isinstance(params[2], str) else int(params[2])
        )
        target_bits = (
            int(params[3], 16) if isinstance(params[3], str) else int(params[3])
        )
        clean_jobs = bool(params[4]) if len(params) > 4 else True
        
        # Convert pre_pow_hash from hex
        pre_pow_hash = bytes.fromhex(pre_pow_hash_hex)
        assert len(pre_pow_hash) == 32
        
        self._current_job = StratumJob(
            job_id=job_id,
            pre_pow_hash=pre_pow_hash,
            timestamp=timestamp,
            target_bits=target_bits,
            clean_jobs=clean_jobs,
        )
        
        logger.info(f"New job: {job_id}, clean={clean_jobs}")
        
        # Convert to BlockTemplate and notify pipeline
        if self.on_new_job:
            template = BlockTemplate(
                pre_pow_hash=pre_pow_hash,
                timestamp=timestamp,
                target_bits=target_bits,
                template_id=job_id,
            )
            self.on_new_job(template)
    
    async def _handle_set_difficulty(self, params: list):
        """Handle mining.set_difficulty."""
        if params:
            logger.info(f"Difficulty set to: {params[0]}")
    
    async def _call(self, method: str, params: list) -> Optional[dict]:
        """
        Send a JSON-RPC 2.0 request and wait for the response.
        
        Args:
            method: RPC method name
            params: Method parameters
        
        Returns:
            Response dict, or None on error
        """
        self._request_id += 1
        request = {
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        
        line = json.dumps(request) + "\n"
        self._writer.write(line.encode('utf-8'))
        await self._writer.drain()
        
        # Wait for response
        try:
            response_line = await asyncio.wait_for(
                self._reader.readline(), timeout=10.0
            )
            if response_line:
                return json.loads(response_line.decode('utf-8').strip())
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for response to {method}")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON response: {e}")
        
        return None
    
    async def disconnect(self):
        """Disconnect from the stratum server."""
        self._connected = False
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        logger.info("Disconnected from stratum server")
    
    @property
    def current_job_id(self) -> Optional[str]:
        """Current job ID, if any."""
        return self._current_job.job_id if self._current_job else None
