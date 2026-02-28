"""
Hybrid CPU↔TPU mining pipeline for Kaspa kHeavyHash.

Architecture: Triple-buffered async pipeline
  Stage 1 (CPU): PrePowHash — batch of nonces → C Keccak → hashes
  Stage 2 (TPU): MatMul — hashes → nibbles → batched MatMul → products
  Stage 3 (CPU): PostHash — XOR + C HeavyHash + difficulty check

Uses C-accelerated Keccak (100-1000x faster than pure Python),
multiprocessing to saturate all CPU cores, and JAX for TPU MatMul.
"""

import asyncio
import logging
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List

from ..crypto.pow_hash import uint256_from_le_bytes, compact_target_to_uint256
from ..core.matrix import Matrix
from ..core.tpu_matmul import TPUMatMul, hashes_to_nibbles, products_xor_hashes, get_tpu_device

logger = logging.getLogger(__name__)

# Try to use fast C Keccak, fall back to pure Python
try:
    from ..crypto.fast_keccak import FastPowHash, FastHeavyHash, is_available as fast_keccak_available
    if fast_keccak_available():
        USE_FAST_KECCAK = True
        logger.info("Using C-accelerated Keccak (fast)")
    else:
        USE_FAST_KECCAK = False
except ImportError:
    USE_FAST_KECCAK = False

if not USE_FAST_KECCAK:
    from ..crypto.keccak import PowHash, HeavyHash
    logger.warning("Using pure Python Keccak (SLOW) — install gcc for C acceleration")


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


def _cpu_prehash_chunk(args):
    """Worker function for multiprocessing: compute PrePowHash for a chunk."""
    pre_pow_hash, timestamp, nonces = args
    if USE_FAST_KECCAK:
        hasher = FastPowHash(pre_pow_hash, timestamp)
    else:
        hasher = PowHash(pre_pow_hash, timestamp)
    return hasher.finalize_batch(nonces)


def _cpu_heavyhash_chunk(inputs):
    """Worker function for multiprocessing: compute HeavyHash for a chunk."""
    if USE_FAST_KECCAK:
        return FastHeavyHash.hash_batch(inputs)
    else:
        return HeavyHash.hash_batch(inputs)


class MiningPipeline:
    """
    Hybrid CPU↔TPU mining pipeline for Kaspa kHeavyHash.
    
    The pipeline processes nonces in configurable batch sizes:
      1. CPU cores compute PrePowHash (C Keccak) for a batch of nonces
      2. TPU computes batched MatMul on the hash nibbles
      3. CPU cores XOR results, compute final HeavyHash, check difficulty
    
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
        
        Uses multi-chunk parallel processing: splits large batches
        across CPU threads for Keccak, then consolidates for TPU MatMul.
        """
        self._running = True
        self._stats = PipelineStats()
        
        # Determine optimal chunk size based on threads
        # Each thread gets an equal share of the batch
        n_chunks = max(1, self.cpu_threads)
        chunk_size = max(256, self.batch_size // n_chunks)
        total_batch = chunk_size * n_chunks
        
        logger.info(
            f"Mining pipeline started — total_batch={total_batch}, "
            f"chunks={n_chunks}x{chunk_size}, "
            f"cpu_threads={self.cpu_threads}, "
            f"device={'TPU' if self._tpu_device else 'CPU'}, "
            f"keccak={'C (fast)' if USE_FAST_KECCAK else 'Python (slow)'}"
        )
        
        last_stats_time = time.time()
        
        while self._running:
            if self._current_template is None:
                await asyncio.sleep(0.1)
                continue
            
            template = self._current_template
            
            # Get a big batch of nonces
            start = self._nonce_counter
            end = min(start + total_batch, template.nonce_end)
            if start >= end:
                await asyncio.sleep(0.1)
                continue
            self._nonce_counter = end
            
            nonces = np.arange(start, end, dtype=np.uint64)
            
            # Process through the pipeline
            solution = await self._process_batch_parallel(nonces, template, n_chunks)
            
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
            
            # Yield to event loop for template updates
            await asyncio.sleep(0)
    
    async def _process_batch_parallel(
        self, nonces: np.ndarray, template: BlockTemplate, n_chunks: int
    ) -> Optional[MiningResult]:
        """
        Process a batch using parallel CPU workers for Keccak.
        
        1. Split nonces into chunks and distribute to CPU threads
        2. Concatenate results for TPU MatMul
        3. Split again for parallel HeavyHash + difficulty check
        """
        loop = asyncio.get_event_loop()
        
        # --- Stage 1: Parallel PrePowHash on CPU ---
        chunks = np.array_split(nonces, n_chunks)
        args = [(template.pre_pow_hash, template.timestamp, c) for c in chunks if len(c) > 0]
        
        pre_hash_parts = await asyncio.gather(*[
            loop.run_in_executor(self._executor, _cpu_prehash_chunk, a)
            for a in args
        ])
        pre_hashes = np.concatenate(pre_hash_parts, axis=0)
        
        # --- Stage 2: MatMul on TPU ---
        nibbles = hashes_to_nibbles(pre_hashes)
        products = self._tpu_engine.batched_matmul(nibbles)
        
        # --- Stage 3: XOR + Parallel HeavyHash on CPU ---
        xored = products_xor_hashes(products, pre_hashes)
        
        xor_chunks = np.array_split(xored, n_chunks)
        heavy_parts = await asyncio.gather(*[
            loop.run_in_executor(self._executor, _cpu_heavyhash_chunk, c)
            for c in xor_chunks if len(c) > 0
        ])
        final_hashes = np.concatenate(heavy_parts, axis=0)
        
        # Update stats
        self._stats.total_hashes += len(nonces)
        
        # Check difficulty
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
    
    def stop(self):
        """Stop the mining pipeline."""
        self._running = False
        logger.info("Mining pipeline stopping...")
    
    @property
    def stats(self) -> PipelineStats:
        """Current pipeline statistics."""
        return self._stats
