"""
Hybrid CPU↔TPU mining pipeline for Kaspa kHeavyHash.

Architecture: Triple-buffered async pipeline
  Stage 1 (CPU): PrePowHash — batch of nonces → Keccak → hashes
  Stage 2 (TPU): MatMul — hashes → nibbles → batched MatMul → products
  Stage 3 (CPU): PostHash — XOR + cSHAKE256("HeavyHash") + difficulty check

Kaspa has ~1 second block times, so rapid template switching is critical.
The pipeline uses asyncio for coordination with ThreadPoolExecutor for
CPU-bound Keccak and JAX for TPU MatMul.
"""

import asyncio
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple

from ..crypto.keccak import PowHash, HeavyHash
from ..crypto.pow_hash import MiningState, uint256_from_le_bytes, compact_target_to_uint256
from ..core.matrix import Matrix
from ..core.tpu_matmul import TPUMatMul, hashes_to_nibbles, products_xor_hashes, get_tpu_device

logger = logging.getLogger(__name__)


@dataclass
class BlockTemplate:
    """Block template received from the Kaspa node or pool."""
    pre_pow_hash: bytes          # 32 bytes
    timestamp: int               # u64
    target_bits: int             # compact target
    nonce_start: int = 0         # nonce range start
    nonce_end: int = (1 << 64)   # nonce range end
    extra_data: bytes = b""      # extra data for block
    block_data: bytes = b""      # full serialized block (for submission)
    template_id: str = ""        # identifier for this template


@dataclass
class MiningResult:
    """Result of finding a valid nonce."""
    nonce: int
    pow_hash: bytes
    pow_value: int
    template: BlockTemplate


@dataclass  
class PipelineStats:
    """Real-time pipeline performance statistics."""
    total_hashes: int = 0
    start_time: float = field(default_factory=time.time)
    last_report_time: float = field(default_factory=time.time)
    last_report_hashes: int = 0
    solutions_found: int = 0
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    @property
    def hashrate(self) -> float:
        """Overall average hashrate in H/s."""
        elapsed = self.elapsed
        return self.total_hashes / elapsed if elapsed > 0 else 0
    
    def instant_hashrate(self) -> float:
        """Hashrate since last report."""
        now = time.time()
        dt = now - self.last_report_time
        dh = self.total_hashes - self.last_report_hashes
        rate = dh / dt if dt > 0 else 0
        self.last_report_time = now
        self.last_report_hashes = self.total_hashes
        return rate


class MiningPipeline:
    """
    Hybrid CPU↔TPU mining pipeline for Kaspa kHeavyHash.
    
    The pipeline processes nonces in configurable batch sizes:
      1. CPU threads compute PrePowHash (Keccak) for a batch of nonces
      2. TPU computes batched MatMul on the hash nibbles
      3. CPU threads XOR results, compute final HeavyHash, check difficulty
    
    A callback is fired when a valid solution is found.
    """
    
    def __init__(
        self,
        batch_size: int = 8192,
        cpu_threads: int = 4,
        on_solution: Optional[Callable[[MiningResult], None]] = None,
        on_stats: Optional[Callable[[PipelineStats], None]] = None,
        stats_interval: float = 5.0,
    ):
        """
        Args:
            batch_size: Number of nonces per pipeline batch
            cpu_threads: Number of CPU threads for Keccak work
            on_solution: Callback when a valid nonce is found
            on_stats: Callback for periodic stats reporting
            stats_interval: Seconds between stats reports
        """
        self.batch_size = batch_size
        self.cpu_threads = cpu_threads
        self.on_solution = on_solution
        self.on_stats = on_stats
        self.stats_interval = stats_interval
        
        self._tpu_device = get_tpu_device()
        self._tpu_engine: Optional[TPUMatMul] = None
        self._executor = ThreadPoolExecutor(max_workers=cpu_threads)
        self._current_template: Optional[BlockTemplate] = None
        self._current_matrix: Optional[Matrix] = None
        self._target: int = 0
        self._running = False
        self._stats = PipelineStats()
        self._nonce_counter: int = 0
    
    def update_template(self, template: BlockTemplate):
        """
        Update the block template (called when a new block arrives).
        
        Regenerates the matrix and updates the TPU engine.
        This is called approximately once per second for Kaspa.
        
        Args:
            template: New block template
        """
        logger.info(f"New block template: {template.template_id}")
        
        self._current_template = template
        self._target = compact_target_to_uint256(template.target_bits)
        self._nonce_counter = template.nonce_start
        
        # Generate matrix on CPU (fast, once per block)
        self._current_matrix = Matrix.generate(template.pre_pow_hash)
        
        # Transfer matrix to TPU
        if self._tpu_engine is None:
            self._tpu_engine = TPUMatMul(
                self._current_matrix.data, 
                self._tpu_device
            )
        else:
            self._tpu_engine.update_matrix(self._current_matrix.data)
    
    async def run(self):
        """
        Main mining loop — runs until stopped.
        
        Continuously processes batches of nonces through the pipeline.
        """
        self._running = True
        self._stats = PipelineStats()
        
        logger.info(
            f"Mining pipeline started — batch_size={self.batch_size}, "
            f"cpu_threads={self.cpu_threads}, "
            f"device={'TPU' if self._tpu_device else 'CPU'}"
        )
        
        last_stats_time = time.time()
        
        while self._running:
            if self._current_template is None:
                await asyncio.sleep(0.1)
                continue
            
            # Get the next batch of nonces
            nonces = self._get_nonce_batch()
            if nonces is None:
                # Exhausted nonce range for this template
                logger.warning("Nonce range exhausted, waiting for new template")
                await asyncio.sleep(0.1)
                continue
            
            # Process the batch through the pipeline
            solution = await self._process_batch(nonces)
            
            if solution is not None:
                self._stats.solutions_found += 1
                if self.on_solution:
                    self.on_solution(solution)
            
            # Stats reporting
            now = time.time()
            if now - last_stats_time >= self.stats_interval:
                if self.on_stats:
                    self.on_stats(self._stats)
                last_stats_time = now
    
    def stop(self):
        """Stop the mining pipeline."""
        self._running = False
        logger.info("Mining pipeline stopping...")
    
    def _get_nonce_batch(self) -> Optional[np.ndarray]:
        """Get the next batch of nonces to process."""
        template = self._current_template
        start = self._nonce_counter
        end = min(start + self.batch_size, template.nonce_end)
        
        if start >= end:
            return None
        
        self._nonce_counter = end
        return np.arange(start, end, dtype=np.uint64)
    
    async def _process_batch(self, nonces: np.ndarray) -> Optional[MiningResult]:
        """
        Process a batch of nonces through the full pipeline.
        
        Stage 1 (CPU): PrePowHash for all nonces
        Stage 2 (TPU): Batched MatMul
        Stage 3 (CPU): XOR + HeavyHash + difficulty check
        
        Returns:
            MiningResult if a valid nonce was found, None otherwise
        """
        loop = asyncio.get_event_loop()
        template = self._current_template
        
        # --- Stage 1: PrePowHash on CPU ---
        hasher = PowHash(template.pre_pow_hash, template.timestamp)
        pre_hashes = await loop.run_in_executor(
            self._executor,
            hasher.finalize_batch,
            nonces,
        )
        
        # --- Stage 2: MatMul on TPU (or CPU fallback) ---
        nibbles = hashes_to_nibbles(pre_hashes)
        products = await loop.run_in_executor(
            self._executor,
            self._tpu_engine.batched_matmul,
            nibbles,
        )
        
        # --- Stage 3: XOR + Final Hash + Difficulty Check on CPU ---
        xored = products_xor_hashes(products, pre_hashes)
        
        final_hashes = await loop.run_in_executor(
            self._executor,
            HeavyHash.hash_batch,
            xored,
        )
        
        # Update stats
        self._stats.total_hashes += len(nonces)
        
        # Check difficulty for each result
        for i in range(len(nonces)):
            pow_value = uint256_from_le_bytes(bytes(final_hashes[i]))
            if pow_value <= self._target:
                logger.info(f"*** SOLUTION FOUND! nonce={nonces[i]} ***")
                return MiningResult(
                    nonce=int(nonces[i]),
                    pow_hash=bytes(final_hashes[i]),
                    pow_value=pow_value,
                    template=template,
                )
        
        return None
    
    @property
    def stats(self) -> PipelineStats:
        """Current pipeline statistics."""
        return self._stats
