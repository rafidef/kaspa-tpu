"""
Fully integrated C mining engine for Kaspa kHeavyHash.

Executes the ENTIRE mining pipeline in C with zero Python overhead:
  PrePowHash(Keccak) → nibble extraction → 64x64 MatMul → truncation →
  XOR → HeavyHash(Keccak) → difficulty check

One C function call processes a batch of nonces and returns any solutions.
Uses ctypes + GIL release for true thread parallelism.

The 64×64 integer matmul is done in C on CPU (faster than TPU for such
tiny matrices due to data transfer overhead being larger than compute).
"""

import ctypes
import logging
import numpy as np
import os
import struct
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

_lib: Optional[ctypes.CDLL] = None
_init_done = False

# ─────────────────────────────────────────────────────────────────────
# Fully integrated C mining engine
# ─────────────────────────────────────────────────────────────────────
_MINER_C_SOURCE = r"""
#include <stdint.h>
#include <string.h>

/* ──────── Keccak-f[1600] ──────── */
static const uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};
static const int ROTATIONS[25] = {
     0,  1, 62, 28, 27, 36, 44,  6, 55, 20,
     3, 10, 43, 25, 39, 41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};
static const int PI[25] = {
     0, 6, 12, 18, 24, 3, 9, 10, 16, 22,
     1, 7, 13, 19, 20, 4, 5, 11, 17, 23,
     2, 8, 14, 15, 21,
};

static inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

static void keccak_f1600(uint64_t state[25]) {
    uint64_t t, bc[5], temp[25];
    int round, i, j;
    for (round = 0; round < 24; round++) {
        for (i = 0; i < 5; i++)
            bc[i] = state[i] ^ state[i+5] ^ state[i+10] ^ state[i+15] ^ state[i+20];
        for (i = 0; i < 5; i++) {
            t = bc[(i+4)%5] ^ rotl64(bc[(i+1)%5], 1);
            for (j = 0; j < 25; j += 5) state[j+i] ^= t;
        }
        for (i = 0; i < 25; i++)
            temp[PI[i]] = rotl64(state[i], ROTATIONS[i]);
        memcpy(state, temp, sizeof(temp));
        for (j = 0; j < 25; j += 5) {
            uint64_t s0=state[j],s1=state[j+1],s2=state[j+2],s3=state[j+3],s4=state[j+4];
            state[j+0]=s0^((~s1)&s2); state[j+1]=s1^((~s2)&s3);
            state[j+2]=s2^((~s3)&s4); state[j+3]=s3^((~s4)&s0);
            state[j+4]=s4^((~s0)&s1);
        }
        state[0] ^= RC[round];
    }
}

/* Extract 32 LE bytes from state */
static inline void state_to_bytes(const uint64_t state[4], uint8_t out[32]) {
    int j, k;
    for (j = 0; j < 4; j++) {
        uint64_t w = state[j];
        for (k = 0; k < 8; k++) {
            out[j*8+k] = (uint8_t)(w >> (k*8));
        }
    }
}

/* ──────── Integrated Mining Function ──────── */
/*
 * mine_range: Process a range of nonces through the ENTIRE kHeavyHash pipeline.
 *
 * For each nonce:
 *   1. PrePowHash (Keccak-f1600)
 *   2. Hash → 64 nibbles
 *   3. Matrix × vector (64×64 × 64), truncate
 *   4. XOR product with original hash
 *   5. HeavyHash (Keccak-f1600) → final hash
 *   6. Compare final hash ≤ target
 *
 * Arguments:
 *   pow_state:   25 uint64 — initial Keccak state with hash+timestamp absorbed
 *   heavy_state: 25 uint64 — initial Keccak state for HeavyHash domain
 *   matrix:      64×64 uint16 — row-major nibble matrix
 *   target:      32 bytes — LE target (hash must be ≤ this)
 *   nonce_start: first nonce to try
 *   nonce_count: number of nonces to try
 *   out_nonce:   output — first valid nonce found (or unchanged)
 *   out_hash:    output — 32-byte hash for the solution
 *
 * Returns: number of solutions found (0 or 1)
 */
int mine_range(
    const uint64_t pow_state[25],
    const uint64_t heavy_state[25],
    const uint16_t matrix[64*64],
    const uint8_t target[32],
    uint64_t nonce_start,
    int nonce_count,
    uint64_t *out_nonce,
    uint8_t *out_hash
) {
    int n;
    uint64_t state[25];
    uint8_t pre_hash[32];
    uint8_t nibbles[64];
    uint8_t product[32];
    uint8_t xored[32];
    uint8_t final_hash[32];

    for (n = 0; n < nonce_count; n++) {
        uint64_t nonce = nonce_start + (uint64_t)n;

        /* ── Stage 1: PrePowHash ── */
        memcpy(state, pow_state, 200);
        state[9] ^= nonce;
        keccak_f1600(state);
        state_to_bytes(state, pre_hash);

        /* ── Stage 2: Hash → 64 nibbles ── */
        {
            int i;
            for (i = 0; i < 32; i++) {
                nibbles[2*i]     = pre_hash[i] >> 4;
                nibbles[2*i + 1] = pre_hash[i] & 0x0F;
            }
        }

        /* ── Stage 3: Matrix × vector, truncate ── */
        {
            int i, j;
            for (i = 0; i < 32; i++) {
                uint32_t sum1 = 0, sum2 = 0;
                const uint16_t *row1 = &matrix[(2*i)*64];
                const uint16_t *row2 = &matrix[(2*i+1)*64];
                for (j = 0; j < 64; j++) {
                    sum1 += (uint32_t)row1[j] * (uint32_t)nibbles[j];
                    sum2 += (uint32_t)row2[j] * (uint32_t)nibbles[j];
                }
                product[i] = (uint8_t)((((sum1 >> 10) & 0x0F) << 4) | ((sum2 >> 10) & 0x0F));
            }
        }

        /* ── Stage 4: XOR with original hash ── */
        {
            int i;
            for (i = 0; i < 32; i++)
                xored[i] = product[i] ^ pre_hash[i];
        }

        /* ── Stage 5: HeavyHash ── */
        memcpy(state, heavy_state, 200);
        {
            int j, k;
            for (j = 0; j < 4; j++) {
                uint64_t w = 0;
                for (k = 0; k < 8; k++)
                    w |= (uint64_t)xored[j*8+k] << (k*8);
                state[j] ^= w;
            }
        }
        keccak_f1600(state);
        state_to_bytes(state, final_hash);

        /* ── Stage 6: Compare final_hash <= target (LE, MSB first) ── */
        {
            int i;
            int less_or_equal = 1;
            for (i = 31; i >= 0; i--) {
                if (final_hash[i] < target[i]) {
                    break;  /* definitely <= */
                } else if (final_hash[i] > target[i]) {
                    less_or_equal = 0;
                    break;
                }
                /* equal, continue to next byte */
            }
            if (less_or_equal) {
                *out_nonce = nonce;
                memcpy(out_hash, final_hash, 32);
                return 1;
            }
        }
    }
    return 0;
}

/* For benchmarking: count total hashes processed */
void mine_range_bench(
    const uint64_t pow_state[25],
    const uint64_t heavy_state[25],
    const uint16_t matrix[64*64],
    uint64_t nonce_start,
    int nonce_count,
    uint64_t *hash_count
) {
    int n;
    uint64_t state[25];
    uint8_t pre_hash[32];
    uint8_t nibbles[64];
    uint8_t product[32];
    uint8_t xored[32];
    uint8_t final_hash[32];

    for (n = 0; n < nonce_count; n++) {
        uint64_t nonce = nonce_start + (uint64_t)n;
        memcpy(state, pow_state, 200);
        state[9] ^= nonce;
        keccak_f1600(state);
        state_to_bytes(state, pre_hash);
        {
            int i;
            for (i = 0; i < 32; i++) {
                nibbles[2*i] = pre_hash[i] >> 4;
                nibbles[2*i+1] = pre_hash[i] & 0x0F;
            }
        }
        {
            int i, j;
            for (i = 0; i < 32; i++) {
                uint32_t sum1 = 0, sum2 = 0;
                const uint16_t *row1 = &matrix[(2*i)*64];
                const uint16_t *row2 = &matrix[(2*i+1)*64];
                for (j = 0; j < 64; j++) {
                    sum1 += (uint32_t)row1[j] * (uint32_t)nibbles[j];
                    sum2 += (uint32_t)row2[j] * (uint32_t)nibbles[j];
                }
                product[i] = (uint8_t)((((sum1 >> 10) & 0x0F) << 4) | ((sum2 >> 10) & 0x0F));
            }
        }
        {
            int i;
            for (i = 0; i < 32; i++)
                xored[i] = product[i] ^ pre_hash[i];
        }
        memcpy(state, heavy_state, 200);
        {
            int j, k;
            for (j = 0; j < 4; j++) {
                uint64_t w = 0;
                for (k = 0; k < 8; k++)
                    w |= (uint64_t)xored[j*8+k] << (k*8);
                state[j] ^= w;
            }
        }
        keccak_f1600(state);
        state_to_bytes(state, final_hash);
    }
    *hash_count = (uint64_t)nonce_count;
}
"""


def _find_or_compile() -> Optional[ctypes.CDLL]:
    """Find or compile the integrated C miner library."""
    global _lib, _init_done
    if _init_done:
        return _lib

    lib_dir = Path(tempfile.gettempdir()) / "kaspa_tpu_c"
    lib_dir.mkdir(parents=True, exist_ok=True)

    lib_name = "libkaspa_miner.so" if sys.platform != "win32" else "libkaspa_miner.dll"
    lib_path = lib_dir / lib_name
    c_path = lib_dir / "kaspa_miner.c"

    # Try loading existing
    if lib_path.exists():
        try:
            lib = ctypes.CDLL(str(lib_path))
            _setup_lib(lib)
            _lib = lib
            _init_done = True
            logger.info(f"Loaded pre-compiled C miner from {lib_path}")
            return lib
        except OSError:
            # Recompile
            lib_path.unlink(missing_ok=True)

    # Write and compile
    logger.info("Compiling integrated C mining engine...")
    c_path.write_text(_MINER_C_SOURCE)

    try:
        result = subprocess.run(
            ["gcc", "-O3", "-march=native", "-shared", "-fPIC",
             "-o", str(lib_path), str(c_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            lib = ctypes.CDLL(str(lib_path))
            _setup_lib(lib)
            _lib = lib
            _init_done = True
            logger.info(f"Compiled C miner: {lib_path}")
            return lib
        else:
            logger.error(f"gcc failed: {result.stderr}")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.error(f"Compilation failed: {e}")

    _init_done = True
    return None


def _setup_lib(lib: ctypes.CDLL):
    """Set up ctypes function signatures."""
    # int mine_range(pow_state, heavy_state, matrix, target,
    #                nonce_start, nonce_count, out_nonce, out_hash)
    lib.mine_range.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),   # pow_state[25]
        ctypes.POINTER(ctypes.c_uint64),   # heavy_state[25]
        ctypes.POINTER(ctypes.c_uint16),   # matrix[64*64]
        ctypes.POINTER(ctypes.c_uint8),    # target[32]
        ctypes.c_uint64,                   # nonce_start
        ctypes.c_int,                      # nonce_count
        ctypes.POINTER(ctypes.c_uint64),   # out_nonce
        ctypes.POINTER(ctypes.c_uint8),    # out_hash[32]
    ]
    lib.mine_range.restype = ctypes.c_int

    lib.mine_range_bench.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.mine_range_bench.restype = None


class IntegratedMiner:
    """
    Fully integrated C mining engine.
    
    Processes the entire kHeavyHash pipeline in a single C function call
    with zero Python overhead. Uses ThreadPoolExecutor for multi-core
    parallelism (ctypes releases the GIL).
    """
    
    def __init__(
        self,
        pre_pow_hash: bytes,
        timestamp: int,
        matrix_data: np.ndarray,
        target_bytes: bytes,
        num_threads: int = 32,
    ):
        from ..crypto.keccak import POWHASH_INITIAL_STATE, HEAVYHASH_INITIAL_STATE
        
        self._lib = _find_or_compile()
        if self._lib is None:
            raise RuntimeError("Failed to compile C mining engine (gcc required)")
        
        # Build PrePowHash initial state
        pow_state = list(POWHASH_INITIAL_STATE)
        for i in range(4):
            word = int.from_bytes(pre_pow_hash[i*8:(i+1)*8], 'little')
            pow_state[i] ^= word
        pow_state[4] ^= (timestamp & 0xFFFFFFFFFFFFFFFF)
        self._pow_state = (ctypes.c_uint64 * 25)(*pow_state)
        
        # HeavyHash initial state
        self._heavy_state = (ctypes.c_uint64 * 25)(*HEAVYHASH_INITIAL_STATE)
        
        # Matrix (64×64, uint16, row-major)
        mat = np.ascontiguousarray(matrix_data.flatten(), dtype=np.uint16)
        self._matrix = mat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        self._matrix_ref = mat  # prevent GC
        
        # Target (32 bytes LE)
        assert len(target_bytes) == 32
        self._target = (ctypes.c_uint8 * 32)(*target_bytes)
        
        self.num_threads = num_threads
        self._executor = ThreadPoolExecutor(max_workers=num_threads)
    
    def update_state(self, pre_pow_hash: bytes, timestamp: int,
                     matrix_data: np.ndarray, target_bytes: bytes):
        """Update the mining state for a new block."""
        from ..crypto.keccak import POWHASH_INITIAL_STATE
        
        pow_state = list(POWHASH_INITIAL_STATE)
        for i in range(4):
            word = int.from_bytes(pre_pow_hash[i*8:(i+1)*8], 'little')
            pow_state[i] ^= word
        pow_state[4] ^= (timestamp & 0xFFFFFFFFFFFFFFFF)
        self._pow_state = (ctypes.c_uint64 * 25)(*pow_state)
        
        mat = np.ascontiguousarray(matrix_data.flatten(), dtype=np.uint16)
        self._matrix = mat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        self._matrix_ref = mat
        
        self._target = (ctypes.c_uint8 * 32)(*target_bytes)
    
    def mine_batch(self, nonce_start: int, nonce_count: int) -> Optional[Tuple[int, bytes]]:
        """
        Mine a batch of nonces using multiple threads.
        
        Returns (nonce, hash) if solution found, None otherwise.
        """
        chunk_size = nonce_count // self.num_threads
        if chunk_size < 64:
            chunk_size = nonce_count
            actual_threads = 1
        else:
            actual_threads = self.num_threads
        
        futures = []
        for i in range(actual_threads):
            start = nonce_start + i * chunk_size
            count = chunk_size if i < actual_threads - 1 else nonce_count - i * chunk_size
            futures.append(
                self._executor.submit(self._mine_chunk, start, count)
            )
        
        for f in futures:
            result = f.result()
            if result is not None:
                # Cancel remaining futures
                for other in futures:
                    other.cancel()
                return result
        return None
    
    def _mine_chunk(self, nonce_start: int, nonce_count: int) -> Optional[Tuple[int, bytes]]:
        """Mine a chunk of nonces — runs in a thread, GIL released during C call."""
        out_nonce = ctypes.c_uint64(0)
        out_hash = (ctypes.c_uint8 * 32)()
        
        found = self._lib.mine_range(
            self._pow_state,
            self._heavy_state,
            self._matrix,
            self._target,
            ctypes.c_uint64(nonce_start),
            ctypes.c_int(nonce_count),
            ctypes.byref(out_nonce),
            out_hash,
        )
        
        if found > 0:
            return (out_nonce.value, bytes(out_hash))
        return None
    
    def benchmark(self, duration: float = 10.0) -> float:
        """Run a benchmark and return hashes/second."""
        import time
        
        batch_per_thread = 100000
        total_hashes = 0
        nonce = 0
        
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            futures = []
            for i in range(self.num_threads):
                s = nonce + i * batch_per_thread
                futures.append(
                    self._executor.submit(self._bench_chunk, s, batch_per_thread)
                )
            for f in futures:
                total_hashes += f.result()
            nonce += self.num_threads * batch_per_thread
        
        elapsed = time.perf_counter() - start
        return total_hashes / elapsed
    
    def _bench_chunk(self, nonce_start: int, nonce_count: int) -> int:
        """Benchmark a chunk — runs entire pipeline without difficulty check."""
        out_count = ctypes.c_uint64(0)
        self._lib.mine_range_bench(
            self._pow_state,
            self._heavy_state,
            self._matrix,
            ctypes.c_uint64(nonce_start),
            ctypes.c_int(nonce_count),
            ctypes.byref(out_count),
        )
        return out_count.value
    
    def shutdown(self):
        """Shut down the thread pool."""
        self._executor.shutdown(wait=False)


def is_available() -> bool:
    """Check if the integrated C miner is available."""
    return _find_or_compile() is not None
