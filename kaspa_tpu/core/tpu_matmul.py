"""
TPU-accelerated batched MatMul kernel for kHeavyHash.

This is the core TPU offload module. It takes a batch of 32-byte hashes,
performs the 64x64 matrix-vector multiplication on the TPU's MXU (systolic
array), and returns the truncated products.

Architecture:
  - The 64x64 matrix stays resident on TPU HBM for the entire block
  - Batches of nibble vectors are streamed to TPU, MatMul'd, and returned
  - The right-shift and truncation happen on-device to minimize data transfer

The CPU handles:
  - Keccak pre/post hashing (not suitable for TPU)
  - Nibble extraction and result combination
  - Difficulty checking

JAX is used for TPU compilation. The module gracefully falls back to
CPU/GPU if no TPU is available.
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Import JAX with graceful fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import devices as jax_devices
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    logger.warning("JAX not available — TPU acceleration disabled, using NumPy fallback")


def get_tpu_device():
    """
    Detect and return a TPU device if available.
    
    Returns:
        JAX device object, or None if no TPU
    """
    if not HAS_JAX:
        return None
    
    try:
        tpus = jax_devices('tpu')
        if tpus:
            logger.info(f"Found {len(tpus)} TPU device(s): {tpus}")
            return tpus[0]
    except RuntimeError:
        pass
    
    logger.info("No TPU found, will use default JAX backend")
    return None


def _build_matmul_kernel():
    """
    Build the JIT-compiled TPU MatMul kernel.
    
    Returns a function that takes:
      - matrix: (64, 64) int32 on device
      - nibbles: (N, 64) int32 on device
    And returns:
      - products: (N, 32) uint8 — the truncated MatMul result
    
    The kernel does:
      1. full_product = matrix @ nibbles.T  → (64, N)
      2. For each output byte pair (2i, 2i+1):
         high = (full_product[2i] >> 10) & 0x0F
         low  = (full_product[2i+1] >> 10) & 0x0F
         byte = (high << 4) | low
    """
    if not HAS_JAX:
        return None
    
    @jax.jit
    def matmul_kernel(matrix, nibbles):
        """
        TPU MatMul kernel for kHeavyHash.
        
        Args:
            matrix: (64, 64) int32 — the block's nibble matrix
            nibbles: (N, 64) int32 — nibble vectors for N nonces
        
        Returns:
            products: (N, 32) int32 — truncated MatMul products (before XOR)
        """
        # Matrix-vector multiply: (64, 64) × (64, N) → (64, N)
        # nibbles is (N, 64), transpose to (64, N)
        full_product = jnp.matmul(matrix, nibbles.T)  # (64, N)
        
        # Truncation: right-shift by 10, mask to 4 bits
        # Combine pairs into bytes
        # full_product[2*i] → high nibble, full_product[2*i+1] → low nibble
        
        # Reshape to (32, 2, N) to pair even/odd rows
        paired = full_product.reshape(32, 2, -1)  # (32, 2, N)
        
        high_nibbles = (paired[:, 0, :] >> 10) & 0x0F  # (32, N)
        low_nibbles = (paired[:, 1, :] >> 10) & 0x0F   # (32, N)
        
        # Combine: (high << 4) | low → (32, N)
        combined = (high_nibbles << 4) | low_nibbles  # (32, N)
        
        # Transpose to (N, 32) for output
        return combined.T.astype(jnp.int32)
    
    return matmul_kernel


class TPUMatMul:
    """
    TPU-accelerated MatMul engine for kHeavyHash.
    
    Usage:
        engine = TPUMatMul(matrix_data)  # matrix_data is (64,64) uint16
        products = engine.batched_matmul(nibble_vectors)  # (N, 64) → (N, 32)
    
    The matrix is transferred to TPU once per block and stays resident.
    """
    
    def __init__(self, matrix_data: np.ndarray, device=None):
        """
        Initialize the TPU MatMul engine with a block's matrix.
        
        Args:
            matrix_data: (64, 64) uint16 array — the block's nibble matrix
            device: Optional JAX device to place the matrix on
        """
        assert matrix_data.shape == (64, 64)
        
        self._matrix_np = matrix_data.astype(np.int32)
        self._device = device
        self._use_tpu = HAS_JAX
        self._kernel = None
        self._matrix_device = None
        
        if self._use_tpu:
            self._kernel = _build_matmul_kernel()
            # Transfer matrix to device (stays resident for entire block)
            if device is not None:
                self._matrix_device = jax.device_put(
                    jnp.array(self._matrix_np), device
                )
            else:
                self._matrix_device = jnp.array(self._matrix_np)
            
            logger.info(f"Matrix loaded to {'TPU' if device else 'default JAX device'}")
        else:
            logger.info("Using NumPy fallback for MatMul")
    
    def batched_matmul(self, nibble_vectors: np.ndarray) -> np.ndarray:
        """
        Perform batched MatMul: matrix (64×64) × nibbles (N×64) → products (N×32)
        
        This is the operation that runs on TPU. It computes the integer
        multiply-accumulate, right-shift-by-10, nibble truncation, and
        byte combination — all on-device.
        
        Args:
            nibble_vectors: (N, 64) uint8/uint16 array of nibble values
        
        Returns:
            (N, 32) uint8 array of products (before XOR with original hash)
        """
        if self._use_tpu:
            return self._batched_matmul_jax(nibble_vectors)
        else:
            return self._batched_matmul_numpy(nibble_vectors)
    
    def _batched_matmul_jax(self, nibble_vectors: np.ndarray) -> np.ndarray:
        """JAX/TPU accelerated MatMul."""
        nibs_jax = jnp.array(nibble_vectors.astype(np.int32))
        
        # Run the JIT-compiled kernel on TPU
        products = self._kernel(self._matrix_device, nibs_jax)
        
        # Transfer result back to CPU
        return np.array(products, dtype=np.uint8)
    
    def _batched_matmul_numpy(self, nibble_vectors: np.ndarray) -> np.ndarray:
        """NumPy CPU fallback MatMul."""
        mat = self._matrix_np.astype(np.int64)
        nibs = nibble_vectors.astype(np.int64)
        
        # (64, 64) × (64, N) → (64, N)
        full_product = mat @ nibs.T
        
        # Truncation and combination
        products = np.zeros((nibble_vectors.shape[0], 32), dtype=np.uint8)
        for i in range(32):
            high = (full_product[2 * i, :] >> 10) & 0x0F
            low = (full_product[2 * i + 1, :] >> 10) & 0x0F
            products[:, i] = ((high << 4) | low).astype(np.uint8)
        
        return products
    
    def update_matrix(self, matrix_data: np.ndarray):
        """
        Update the matrix for a new block template.
        
        Transfers the new matrix to TPU, replacing the old one.
        
        Args:
            matrix_data: (64, 64) uint16 array
        """
        assert matrix_data.shape == (64, 64)
        self._matrix_np = matrix_data.astype(np.int32)
        
        if self._use_tpu:
            if self._device is not None:
                self._matrix_device = jax.device_put(
                    jnp.array(self._matrix_np), self._device
                )
            else:
                self._matrix_device = jnp.array(self._matrix_np)


def hashes_to_nibbles(hashes: np.ndarray) -> np.ndarray:
    """
    Convert an array of 32-byte hashes to 64-element nibble vectors.
    
    Each byte is split into high and low nibbles:
      byte[i] → nibble[2i] = byte >> 4, nibble[2i+1] = byte & 0x0F
    
    Args:
        hashes: (N, 32) uint8 array
    
    Returns:
        (N, 64) uint16 array of nibble values
    """
    n = hashes.shape[0]
    nibbles = np.zeros((n, 64), dtype=np.uint16)
    for i in range(32):
        nibbles[:, 2 * i] = hashes[:, i] >> 4
        nibbles[:, 2 * i + 1] = hashes[:, i] & 0x0F
    return nibbles


def products_xor_hashes(products: np.ndarray, hashes: np.ndarray) -> np.ndarray:
    """
    XOR the MatMul products with the original hashes.
    
    This is step 3 of heavy_hash, done on CPU after receiving
    products back from TPU.
    
    Args:
        products: (N, 32) uint8 — MatMul output
        hashes: (N, 32) uint8 — original PrePowHash output
    
    Returns:
        (N, 32) uint8 — XOR'd result, ready for final cSHAKE256
    """
    return products ^ hashes
