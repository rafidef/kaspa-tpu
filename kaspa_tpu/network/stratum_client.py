"""
Kaspa Stratum protocol client for pool mining.

Implements the Kaspa Common Stratum Protocol (JSON-RPC 2.0 over TCP).
Compatible with kaspa-stratum-bridge, Kryptex, WoolyPooly, F2Pool, etc.

The client uses proper async message routing — server notifications
(mining.notify, mining.set_difficulty) can arrive at any time, even
between a request and its response. Messages are routed by their `id`
field (responses) or `method` field (notifications).

Protocol flow:
  1. mining.subscribe → server info
  2. mining.authorize(address, worker) → authentication
  3. Server sends mining.notify with new jobs
  4. Client sends mining.submit with found nonces
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Callable, Dict

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
    
    Uses async message routing: a background reader task dispatches
    incoming messages to either pending request futures (by id) or
    notification handlers (by method).
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        wallet_address: str,
        worker_name: str = "kaspa-tpu",
        on_new_job: Optional[Callable[[BlockTemplate], None]] = None,
    ):
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
        
        # Pending request futures, keyed by request id
        self._pending: Dict[int, asyncio.Future] = {}
        # Background reader task
        self._reader_task: Optional[asyncio.Task] = None
        # Current difficulty
        self._difficulty: float = 1.0
    
    async def connect(self):
        """Connect to the stratum server and start the mining session."""
        logger.info(f"Connecting to stratum://{self.host}:{self.port}")
        
        self._reader, self._writer = await asyncio.open_connection(
            self.host, self.port
        )
        self._connected = True
        logger.info("Connected to stratum server")
        
        # Start the background message reader
        self._reader_task = asyncio.create_task(self._read_loop())
        
        # Subscribe
        await self._subscribe()
        
        # Authorize
        await self._authorize()
    
    async def _subscribe(self):
        """Send mining.subscribe."""
        response = await self._call("mining.subscribe", [
            "kaspa-tpu-miner/0.1.0",
        ])
        logger.info(f"Subscribe response: {response}")
        
        if response:
            result = response.get("result")
            if isinstance(result, list) and len(result) > 1:
                self._extranonce = str(result[1]) if result[1] else ""
            elif isinstance(result, str):
                self._extranonce = result
            logger.info(f"Subscribed successfully, extranonce: {self._extranonce}")
        else:
            # Some pools don't send a real subscribe response — continue anyway
            logger.warning("No subscribe response, continuing...")
    
    async def _authorize(self):
        """Send mining.authorize."""
        response = await self._call("mining.authorize", [
            self.wallet_address,
            self.worker_name,
        ])
        logger.info(f"Authorize response: {response}")
        
        if response is None:
            # Some pools authorize implicitly or don't respond with id match
            logger.warning("No explicit auth response — assuming authorized")
            self._authorized = True
            return
        
        result = response.get("result")
        error = response.get("error")
        
        if error:
            raise ConnectionError(f"Authorization failed: {error}")
        
        # result can be True, or just present
        self._authorized = True
        logger.info(f"Authorized as {self.wallet_address}")
    
    async def submit_solution(self, job_id: str, nonce: int) -> bool:
        """Submit a mining solution to the pool."""
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
        Wait for the reader task to finish (i.e., connection closes).
        
        The actual message handling happens in _read_loop which runs
        as a background task started in connect().
        """
        if self._reader_task:
            await self._reader_task
    
    async def _read_loop(self):
        """
        Background reader: continuously reads lines and dispatches messages.
        
        - Messages with an `id` matching a pending request → resolve that future
        - Messages with a `method` field → handle as server notification
        """
        while self._connected:
            try:
                line = await self._reader.readline()
                if not line:
                    logger.warning("Connection closed by server")
                    self._connected = False
                    break
                
                line_str = line.decode('utf-8').strip()
                if not line_str:
                    continue
                
                logger.debug(f"<< {line_str}")
                
                try:
                    message = json.loads(line_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON: {e} — {line_str[:200]}")
                    continue
                
                # Route the message
                msg_id = message.get("id")
                method = message.get("method")
                
                if msg_id is not None and msg_id in self._pending:
                    # This is a response to a pending request
                    future = self._pending.pop(msg_id)
                    if not future.done():
                        future.set_result(message)
                elif method:
                    # This is a server notification
                    await self._handle_notification(method, message.get("params", []))
                else:
                    # Unknown message or response to unknown id
                    logger.debug(f"Unrouted message: {message}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                self._connected = False
                break
    
    async def _handle_notification(self, method: str, params):
        """Handle a server notification."""
        if method == "mining.notify":
            await self._handle_notify(params)
        elif method == "mining.set_difficulty":
            await self._handle_set_difficulty(params)
        elif method == "mining.set_extranonce":
            if params and len(params) >= 1:
                self._extranonce = str(params[0])
                logger.info(f"Extranonce updated: {self._extranonce}")
        else:
            logger.debug(f"Unhandled notification: {method} {params}")
    
    async def _handle_notify(self, params):
        """
        Handle mining.notify — new job from the pool.
        
        Kaspa stratum notify params can vary by pool implementation.
        Common formats:
          [job_id, pre_pow_hash_hex, timestamp_hex/int, nonce_hex/int, clean_jobs]
          [job_id, header_data, ...]
        
        We try to parse flexibly.
        """
        if not params:
            logger.warning(f"mining.notify with empty params")
            return
        
        if isinstance(params, dict):
            # Some pools send params as a dict
            job_id = str(params.get("id", params.get("jobId", "0")))
            pre_pow_hash_hex = str(params.get("prePowHash", params.get("header", "")))
            timestamp = params.get("timestamp", 0)
            target_bits = params.get("bits", params.get("nBits", 0))
            clean_jobs = params.get("cleanJobs", True)
        elif isinstance(params, list):
            if len(params) < 2:
                logger.warning(f"mining.notify with insufficient params: {params}")
                return
            
            job_id = str(params[0])
            
            # params[1] could be pre_pow_hash or full header data
            pre_pow_hash_hex = str(params[1])
            
            # Parse remaining fields with flexibility
            timestamp = 0
            target_bits = 0
            clean_jobs = True
            
            if len(params) > 2:
                try:
                    val = params[2]
                    timestamp = int(val, 16) if isinstance(val, str) else int(val)
                except (ValueError, TypeError):
                    pass
            
            if len(params) > 3:
                try:
                    val = params[3]
                    target_bits = int(val, 16) if isinstance(val, str) else int(val)
                except (ValueError, TypeError):
                    pass
            
            if len(params) > 4:
                clean_jobs = bool(params[4])
        else:
            logger.warning(f"mining.notify unexpected params type: {type(params)}")
            return
        
        # Validate and convert pre_pow_hash
        # Strip 0x prefix if present
        if pre_pow_hash_hex.startswith("0x"):
            pre_pow_hash_hex = pre_pow_hash_hex[2:]
        
        try:
            pre_pow_hash = bytes.fromhex(pre_pow_hash_hex)
        except ValueError:
            logger.warning(f"Invalid pre_pow_hash hex: {pre_pow_hash_hex[:64]}...")
            return
        
        # Pad or truncate to 32 bytes
        if len(pre_pow_hash) < 32:
            pre_pow_hash = pre_pow_hash.ljust(32, b'\x00')
        elif len(pre_pow_hash) > 32:
            pre_pow_hash = pre_pow_hash[:32]
        
        self._current_job = StratumJob(
            job_id=job_id,
            pre_pow_hash=pre_pow_hash,
            timestamp=timestamp,
            target_bits=target_bits,
            clean_jobs=clean_jobs,
        )
        
        logger.info(f"New job: {job_id}, clean={clean_jobs}, hash={pre_pow_hash_hex[:16]}...")
        
        # Convert to BlockTemplate and notify pipeline
        if self.on_new_job:
            template = BlockTemplate(
                pre_pow_hash=pre_pow_hash,
                timestamp=timestamp,
                target_bits=target_bits,
                template_id=job_id,
            )
            self.on_new_job(template)
    
    async def _handle_set_difficulty(self, params):
        """Handle mining.set_difficulty."""
        if params:
            try:
                self._difficulty = float(params[0])
            except (ValueError, TypeError, IndexError):
                pass
            logger.info(f"Difficulty set to: {self._difficulty}")
    
    async def _call(self, method: str, params: list, timeout: float = 10.0) -> Optional[dict]:
        """
        Send a JSON-RPC 2.0 request and wait for the matching response.
        
        Uses the async message router: creates a Future for this request id,
        sends the request, then awaits the Future (which the read loop
        will resolve when a response with matching id arrives).
        """
        self._request_id += 1
        req_id = self._request_id
        
        request = {
            "id": req_id,
            "method": method,
            "params": params,
        }
        
        # Create a future for this request
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending[req_id] = future
        
        # Send the request
        line = json.dumps(request) + "\n"
        logger.debug(f">> {line.strip()}")
        self._writer.write(line.encode('utf-8'))
        await self._writer.drain()
        
        # Wait for the response (with timeout)
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            logger.warning(f"Timeout waiting for response to {method} (id={req_id})")
            return None
    
    async def disconnect(self):
        """Disconnect from the stratum server."""
        self._connected = False
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
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
