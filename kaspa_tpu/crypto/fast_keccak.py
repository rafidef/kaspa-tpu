"""
Fast Keccak-f1600 via C extension (ctypes).

Auto-compiles a C implementation of Keccak-f1600 on first import,
then provides fast batch operations for PowHash and HeavyHash that
are 100-1000x faster than pure Python.

The C source is embedded directly in this file, so it works even
when installed via pip (which doesn't package .c files in wheels).

Falls back to the pure Python implementation if compilation fails.
"""

import ctypes
import logging
import numpy as np
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# The compiled library handle
_lib: Optional[ctypes.CDLL] = None
_init_done = False

# ─────────────────────────────────────────────────────────────────────
# Embedded C source — the Keccak-f[1600] permutation + batch wrappers
# ─────────────────────────────────────────────────────────────────────
_KECCAK_C_SOURCE = r"""
#include <stdint.h>
#include <string.h>

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
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

static const int PI[25] = {
     0, 6, 12, 18, 24,
     3, 9, 10, 16, 22,
     1, 7, 13, 19, 20,
     4, 5, 11, 17, 23,
     2, 8, 14, 15, 21,
};

static inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccak_f1600(uint64_t state[25]) {
    uint64_t t, bc[5];
    int round, i, j;
    for (round = 0; round < 24; round++) {
        for (i = 0; i < 5; i++)
            bc[i] = state[i] ^ state[i+5] ^ state[i+10] ^ state[i+15] ^ state[i+20];
        for (i = 0; i < 5; i++) {
            t = bc[(i+4)%5] ^ rotl64(bc[(i+1)%5], 1);
            for (j = 0; j < 25; j += 5)
                state[j+i] ^= t;
        }
        {
            uint64_t temp[25];
            for (i = 0; i < 25; i++)
                temp[PI[i]] = rotl64(state[i], ROTATIONS[i]);
            memcpy(state, temp, sizeof(temp));
        }
        for (j = 0; j < 25; j += 5) {
            uint64_t s0=state[j],s1=state[j+1],s2=state[j+2],
                     s3=state[j+3],s4=state[j+4];
            state[j+0] = s0 ^ ((~s1) & s2);
            state[j+1] = s1 ^ ((~s2) & s3);
            state[j+2] = s2 ^ ((~s3) & s4);
            state[j+3] = s3 ^ ((~s4) & s0);
            state[j+4] = s4 ^ ((~s0) & s1);
        }
        state[0] ^= RC[round];
    }
}

void keccak_f1600_batch(uint64_t *states, int n) {
    int i;
    for (i = 0; i < n; i++)
        keccak_f1600(states + i * 25);
}

void powhash_batch(const uint64_t initial_state[25],
                   const uint64_t *nonces, int n,
                   uint8_t *out_hashes) {
    int i, j;
    uint64_t state[25];
    for (i = 0; i < n; i++) {
        memcpy(state, initial_state, 200);
        state[9] ^= nonces[i];
        keccak_f1600(state);
        for (j = 0; j < 4; j++) {
            uint64_t w = state[j];
            out_hashes[i*32+j*8+0]=(uint8_t)(w);
            out_hashes[i*32+j*8+1]=(uint8_t)(w>>8);
            out_hashes[i*32+j*8+2]=(uint8_t)(w>>16);
            out_hashes[i*32+j*8+3]=(uint8_t)(w>>24);
            out_hashes[i*32+j*8+4]=(uint8_t)(w>>32);
            out_hashes[i*32+j*8+5]=(uint8_t)(w>>40);
            out_hashes[i*32+j*8+6]=(uint8_t)(w>>48);
            out_hashes[i*32+j*8+7]=(uint8_t)(w>>56);
        }
    }
}

void heavyhash_batch(const uint64_t heavy_initial_state[25],
                     const uint8_t *inputs, int n,
                     uint8_t *out_hashes) {
    int i, j, k;
    uint64_t state[25];
    for (i = 0; i < n; i++) {
        memcpy(state, heavy_initial_state, 200);
        for (j = 0; j < 4; j++) {
            uint64_t w = 0;
            for (k = 0; k < 8; k++)
                w |= (uint64_t)inputs[i*32+j*8+k] << (k*8);
            state[j] ^= w;
        }
        keccak_f1600(state);
        for (j = 0; j < 4; j++) {
            uint64_t w = state[j];
            out_hashes[i*32+j*8+0]=(uint8_t)(w);
            out_hashes[i*32+j*8+1]=(uint8_t)(w>>8);
            out_hashes[i*32+j*8+2]=(uint8_t)(w>>16);
            out_hashes[i*32+j*8+3]=(uint8_t)(w>>24);
            out_hashes[i*32+j*8+4]=(uint8_t)(w>>32);
            out_hashes[i*32+j*8+5]=(uint8_t)(w>>40);
            out_hashes[i*32+j*8+6]=(uint8_t)(w>>48);
            out_hashes[i*32+j*8+7]=(uint8_t)(w>>56);
        }
    }
}
"""


def _find_or_compile() -> Optional[ctypes.CDLL]:
    """Find or compile the C keccak library."""
    global _lib, _init_done
    if _init_done:
        return _lib

    lib_dir = Path(tempfile.gettempdir()) / "kaspa_tpu_c"
    lib_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        lib_name = "libkeccak.dll"
    elif sys.platform == "darwin":
        lib_name = "libkeccak.dylib"
    else:
        lib_name = "libkeccak.so"

    lib_path = lib_dir / lib_name
    c_path = lib_dir / "keccak_f1600.c"

    # Try loading existing compiled library
    if lib_path.exists():
        try:
            lib = ctypes.CDLL(str(lib_path))
            _setup_lib(lib)
            _lib = lib
            _init_done = True
            logger.info(f"Loaded pre-compiled C Keccak from {lib_path}")
            return lib
        except OSError:
            pass

    # Write embedded C source to temp
    logger.info("Compiling C Keccak extension...")
    c_path.write_text(_KECCAK_C_SOURCE)

    # Compile
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
            logger.info(f"Compiled and loaded C Keccak: {lib_path}")
            return lib
        else:
            logger.warning(f"gcc compilation failed: {result.stderr}")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"Compilation failed: {e}")

    _init_done = True
    logger.warning("Could not compile C Keccak — using pure Python (SLOW)")
    return None


def _setup_lib(lib: ctypes.CDLL):
    """Set up ctypes function signatures."""
    lib.keccak_f1600.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    lib.keccak_f1600.restype = None

    lib.keccak_f1600_batch.argtypes = [
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_int
    ]
    lib.keccak_f1600_batch.restype = None

    lib.powhash_batch.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.powhash_batch.restype = None

    lib.heavyhash_batch.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.heavyhash_batch.restype = None


class FastPowHash:
    """
    C-accelerated PowHash (cSHAKE256 "ProofOfWorkHash").

    ~100-1000x faster than pure Python Keccak.
    ctypes calls release the GIL, so ThreadPoolExecutor actually works.
    """

    def __init__(self, pre_pow_hash: bytes, timestamp: int):
        from .keccak import POWHASH_INITIAL_STATE

        self._state = list(POWHASH_INITIAL_STATE)

        for i in range(4):
            word = int.from_bytes(pre_pow_hash[i*8:(i+1)*8], 'little')
            self._state[i] ^= word

        self._state[4] ^= (timestamp & 0xFFFFFFFFFFFFFFFF)
        self._c_state = (ctypes.c_uint64 * 25)(*self._state)

    def finalize_batch(self, nonces: np.ndarray) -> np.ndarray:
        """
        Compute PowHash for a batch of nonces using C.
        Returns (N, 32) uint8 array of hashes.
        """
        lib = _find_or_compile()
        if lib is None:
            from .keccak import PowHash
            py = PowHash.__new__(PowHash)
            py._state = list(self._state)
            return py.finalize_batch(nonces)

        n = len(nonces)
        nonces_c = np.ascontiguousarray(nonces, dtype=np.uint64)
        out = np.empty(n * 32, dtype=np.uint8)

        lib.powhash_batch(
            self._c_state,
            nonces_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_int(n),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )

        return out.reshape(n, 32)


class FastHeavyHash:
    """C-accelerated HeavyHash (cSHAKE256 "HeavyHash")."""

    _c_state = None

    @classmethod
    def _ensure_state(cls):
        if cls._c_state is None:
            from .keccak import HEAVYHASH_INITIAL_STATE
            cls._c_state = (ctypes.c_uint64 * 25)(*HEAVYHASH_INITIAL_STATE)

    @classmethod
    def hash_batch(cls, inputs: np.ndarray) -> np.ndarray:
        """
        Compute HeavyHash for a batch of inputs using C.
        Returns (N, 32) uint8 array of hashes.
        """
        lib = _find_or_compile()
        if lib is None:
            from .keccak import HeavyHash
            return HeavyHash.hash_batch(inputs)

        cls._ensure_state()
        n = inputs.shape[0]
        inputs_c = np.ascontiguousarray(inputs, dtype=np.uint8)
        out = np.empty(n * 32, dtype=np.uint8)

        lib.heavyhash_batch(
            cls._c_state,
            inputs_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(n),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )

        return out.reshape(n, 32)


def is_available() -> bool:
    """Check if the C extension is available."""
    return _find_or_compile() is not None
