"""
Fast Keccak-f1600 via C extension (ctypes).

Auto-compiles keccak_f1600.c on first import, then provides
fast batch operations for PowHash and HeavyHash that are
100-1000x faster than pure Python.

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


def _find_or_compile() -> Optional[ctypes.CDLL]:
    """Find or compile the C keccak library."""
    global _lib
    if _lib is not None:
        return _lib

    c_source = Path(__file__).parent / "keccak_f1600.c"
    if not c_source.exists():
        logger.warning(f"C source not found: {c_source}")
        return None

    # Look for existing compiled library next to the source
    if sys.platform == "win32":
        lib_name = "libkeccak.dll"
    elif sys.platform == "darwin":
        lib_name = "libkeccak.dylib"
    else:
        lib_name = "libkeccak.so"

    lib_path = c_source.parent / lib_name

    # Also check /tmp for pip-installed packages (read-only site-packages)
    tmp_lib_path = Path(tempfile.gettempdir()) / "kaspa_tpu_libkeccak" / lib_name

    # Try loading existing library
    for candidate in [lib_path, tmp_lib_path]:
        if candidate.exists():
            try:
                lib = ctypes.CDLL(str(candidate))
                _setup_lib(lib)
                logger.info(f"Loaded pre-compiled C Keccak from {candidate}")
                _lib = lib
                return lib
            except OSError:
                pass

    # Compile it
    logger.info(f"Compiling C Keccak extension: {c_source}")
    
    # Try compiling to the source directory first, then /tmp
    for target in [lib_path, tmp_lib_path]:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["gcc", "-O3", "-march=native", "-shared", "-fPIC",
                 "-o", str(target), str(c_source)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                lib = ctypes.CDLL(str(target))
                _setup_lib(lib)
                logger.info(f"Compiled and loaded C Keccak: {target}")
                _lib = lib
                return lib
            else:
                logger.debug(f"gcc failed: {result.stderr}")
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            logger.debug(f"Compilation attempt failed: {e}")

    logger.warning("Could not compile C Keccak extension — using pure Python (SLOW)")
    return None


def _setup_lib(lib: ctypes.CDLL):
    """Set up ctypes function signatures."""
    # void keccak_f1600(uint64_t state[25])
    lib.keccak_f1600.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    lib.keccak_f1600.restype = None

    # void keccak_f1600_batch(uint64_t *states, int n)
    lib.keccak_f1600_batch.argtypes = [
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_int
    ]
    lib.keccak_f1600_batch.restype = None

    # void powhash_batch(const uint64_t initial_state[25],
    #                    const uint64_t *nonces, int n,
    #                    uint8_t *out_hashes)
    lib.powhash_batch.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.powhash_batch.restype = None

    # void heavyhash_batch(const uint64_t heavy_initial_state[25],
    #                      const uint8_t *inputs, int n,
    #                      uint8_t *out_hashes)
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
    """
    
    def __init__(self, pre_pow_hash: bytes, timestamp: int):
        """
        Build the initial state (same as Python PowHash).
        
        Args:
            pre_pow_hash: 32-byte block hash
            timestamp: Block timestamp
        """
        from .keccak import POWHASH_INITIAL_STATE
        
        self._state = list(POWHASH_INITIAL_STATE)
        
        # XOR pre_pow_hash into words 0-3
        for i in range(4):
            word = int.from_bytes(pre_pow_hash[i*8:(i+1)*8], 'little')
            self._state[i] ^= word
        
        # XOR timestamp into word 4
        self._state[4] ^= (timestamp & 0xFFFFFFFFFFFFFFFF)
        
        # Convert to C array
        self._c_state = (ctypes.c_uint64 * 25)(*self._state)
    
    def finalize_batch(self, nonces: np.ndarray) -> np.ndarray:
        """
        Compute PowHash for a batch of nonces using C.
        
        Args:
            nonces: 1D array of uint64 nonce values
            
        Returns:
            (N, 32) uint8 array of hashes
        """
        lib = _find_or_compile()
        if lib is None:
            # Fallback to pure Python
            from .keccak import PowHash
            py_hasher = PowHash.__new__(PowHash)
            py_hasher._state = list(self._state)
            return py_hasher.finalize_batch(nonces)
        
        n = len(nonces)
        nonces_c = nonces.astype(np.uint64)
        out = np.zeros(n * 32, dtype=np.uint8)
        
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
        
        Args:
            inputs: (N, 32) uint8 array
            
        Returns:
            (N, 32) uint8 array of hashes
        """
        lib = _find_or_compile()
        if lib is None:
            from .keccak import HeavyHash
            return HeavyHash.hash_batch(inputs)
        
        cls._ensure_state()
        n = inputs.shape[0]
        inputs_c = np.ascontiguousarray(inputs, dtype=np.uint8)
        out = np.zeros(n * 32, dtype=np.uint8)
        
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
