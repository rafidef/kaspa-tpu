"""
64x64 Matrix for Kaspa kHeavyHash — CPU-side generation, CPU+TPU heavy_hash.

The matrix is generated deterministically from the block's pre_pow_hash using
xoshiro256++. It is a 64x64 matrix of 4-bit nibbles (0-15). The matrix must
have full rank (64) or it is regenerated.

The heavy_hash operation:
  1. Split 32-byte input hash → 64 nibbles
  2. MatMul: matrix (64x64 u16) × vector (64 u8) → products
  3. Right-shift by 10, truncate to 4 bits, combine to bytes
  4. XOR with original hash
  5. Final cSHAKE256("HeavyHash")

Reference: rusty-kaspa/consensus/pow/src/matrix.rs
"""

import numpy as np
from typing import Optional

from ..crypto.xoshiro import XoShiRo256PlusPlus
from ..crypto.keccak import HeavyHash


class Matrix:
    """
    64x64 nibble matrix for kHeavyHash.
    
    Generated once per block template on CPU, then reused for all nonce trials.
    The matrix can be transferred to TPU memory for batched MatMul operations.
    """
    
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: 64x64 array of uint16 values (each 0-15)
        """
        assert data.shape == (64, 64)
        self._data = data.astype(np.uint16)
    
    @classmethod
    def generate(cls, pre_pow_hash: bytes) -> 'Matrix':
        """
        Generate the matrix from a pre_pow_hash, matching Rusty Kaspa exactly.
        
        Uses xoshiro256++ seeded from pre_pow_hash, generates 64x64 nibble matrix.
        Rejects matrices with rank < 64 and regenerates.
        
        Args:
            pre_pow_hash: 32-byte hash seed
        
        Returns:
            Matrix instance with full rank
        """
        assert len(pre_pow_hash) == 32
        generator = XoShiRo256PlusPlus(pre_pow_hash)
        
        while True:
            mat = cls._rand_matrix(generator)
            if mat._compute_rank() == 64:
                return mat
    
    @classmethod
    def _rand_matrix(cls, generator: XoShiRo256PlusPlus) -> 'Matrix':
        """Generate a random matrix without rank checking."""
        data = np.zeros((64, 64), dtype=np.uint16)
        
        for i in range(64):
            for j in range(0, 64, 16):
                val = generator.u64()
                for shift in range(16):
                    data[i, j + shift] = (val >> (4 * shift)) & 0x0F
        
        return cls(data)
    
    def _compute_rank(self) -> int:
        """
        Compute the rank of the matrix using Gaussian elimination over floats.
        
        Exact match with Rusty Kaspa's compute_rank() method.
        
        Returns:
            Matrix rank (0-64)
        """
        EPS = 1e-9
        mat = self._data.astype(np.float64).copy()
        rank = 0
        row_selected = [False] * 64
        
        for i in range(64):
            # Find a non-zero entry in column i
            j = 0
            while j < 64:
                if not row_selected[j] and abs(mat[j, i]) > EPS:
                    break
                j += 1
            
            if j != 64:
                rank += 1
                row_selected[j] = True
                
                # Normalize row j by mat[j, i]
                pivot = mat[j, i]
                for p in range(i + 1, 64):
                    mat[j, p] /= pivot
                
                # Eliminate column i from all other rows
                for k in range(64):
                    if k != j and abs(mat[k, i]) > EPS:
                        factor = mat[k, i]
                        for p in range(i + 1, 64):
                            mat[k, p] -= mat[j, p] * factor
        
        return rank
    
    @property
    def data(self) -> np.ndarray:
        """The raw 64x64 uint16 matrix data."""
        return self._data
    
    def heavy_hash(self, hash_bytes: bytes) -> bytes:
        """
        CPU-side heavy_hash for a single hash — reference implementation.
        
        Used for verification. For mining, use the batched TPU version.
        
        Args:
            hash_bytes: 32-byte input hash (output of PrePowHash)
        
        Returns:
            32-byte heavy hash result
        """
        assert len(hash_bytes) == 32
        
        # Step 1: Split 32 bytes → 64 nibbles (high nibble first per byte)
        vec = np.zeros(64, dtype=np.uint8)
        for i, byte_val in enumerate(hash_bytes):
            vec[2 * i] = byte_val >> 4          # High nibble
            vec[2 * i + 1] = byte_val & 0x0F    # Low nibble
        
        # Step 2: Matrix-vector multiplication with truncation
        # For each output byte i:
        #   sum1 = Σ matrix[2i][j] * vec[j]   for j in 0..63
        #   sum2 = Σ matrix[2i+1][j] * vec[j] for j in 0..63
        #   product[i] = ((sum1 >> 10) << 4) | (sum2 >> 10)
        product = np.zeros(32, dtype=np.uint8)
        for i in range(32):
            sum1 = 0
            sum2 = 0
            for j in range(64):
                sum1 += int(self._data[2 * i, j]) * int(vec[j])
                sum2 += int(self._data[2 * i + 1, j]) * int(vec[j])
            product[i] = (((sum1 >> 10) & 0x0F) << 4) | ((sum2 >> 10) & 0x0F)
        
        # Step 3: XOR with original hash
        result = bytes(p ^ h for p, h in zip(product, hash_bytes))
        
        # Step 4: Final cSHAKE256("HeavyHash")
        return HeavyHash.hash(result)
    
    def heavy_hash_batch_cpu(self, hashes: np.ndarray) -> np.ndarray:
        """
        CPU-side batched heavy_hash using NumPy vectorization.
        
        This is the CPU reference for verification and fallback.
        For production mining, use tpu_matmul.batched_heavy_hash().
        
        Args:
            hashes: (N, 32) uint8 array of input hashes
        
        Returns:
            (N, 32) uint8 array of heavy hash results
        """
        n = hashes.shape[0]
        
        # Step 1: Split bytes → nibbles: (N, 32) → (N, 64)
        nibbles = np.zeros((n, 64), dtype=np.uint16)
        for i in range(32):
            nibbles[:, 2 * i] = hashes[:, i] >> 4
            nibbles[:, 2 * i + 1] = hashes[:, i] & 0x0F
        
        # Step 2: MatMul — (64, 64) × (N, 64)^T = (64, N)
        # matrix is (64, 64) uint16, nibbles is (N, 64) uint16
        # We need: for each row pair (2i, 2i+1), dot product with each nibble vector
        mat = self._data.astype(np.uint32)
        nibs = nibbles.astype(np.uint32)
        
        # Full matmul: (64, 64) × (64, N) → (64, N) 
        # = mat @ nibbles.T → shape (64, N)
        full_product = mat @ nibs.T  # (64, N)
        
        # Step 3: Truncation and combination
        products = np.zeros((n, 32), dtype=np.uint8)
        for i in range(32):
            sum1 = full_product[2 * i, :]  # (N,)
            sum2 = full_product[2 * i + 1, :]  # (N,)
            products[:, i] = (((sum1 >> 10) & 0x0F) << 4).astype(np.uint8) | \
                             ((sum2 >> 10) & 0x0F).astype(np.uint8)
        
        # Step 4: XOR with original hashes
        xored = products ^ hashes
        
        # Step 5: Final cSHAKE256("HeavyHash") — batch
        return HeavyHash.hash_batch(xored)
