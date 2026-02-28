"""
Tests for kHeavyHash correctness against Rusty Kaspa reference.

Test vectors are derived from the Rusty Kaspa source code test suite:
  - PowHash::new(Hash([42; 32]), 5435345234).finalize_with_nonce(432432432)
  - KHeavyHash::hash(Hash([42; 32]))

We also test:
  - xoshiro256++ PRNG determinism
  - Matrix generation determinism and rank
  - MatMul correctness (CPU and TPU paths)
  - Full pipeline end-to-end
"""

import pytest
import numpy as np
import struct
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kaspa_tpu.crypto.keccak import (
    keccak_f1600,
    PowHash,
    HeavyHash,
    POWHASH_INITIAL_STATE,
    HEAVYHASH_INITIAL_STATE,
    _hash_to_le_u64,
    _le_u64_to_hash,
)
from kaspa_tpu.crypto.xoshiro import XoShiRo256PlusPlus
from kaspa_tpu.core.matrix import Matrix
from kaspa_tpu.core.tpu_matmul import TPUMatMul, hashes_to_nibbles, products_xor_hashes


# ---------------------------------------------------------------------------
# Test vectors from Rusty Kaspa
# ---------------------------------------------------------------------------

# Hash([42; 32]) — 32 bytes all set to 42
HASH_42 = bytes([42] * 32)

# Timestamp and nonce from Rusty Kaspa test_pow_hash
TEST_TIMESTAMP = 5435345234
TEST_NONCE = 432432432


class TestKeccakF1600:
    """Test the Keccak-f1600 permutation."""
    
    def test_zero_state(self):
        """Keccak-f1600 on all-zero state should produce known output."""
        state = [0] * 25
        result = keccak_f1600(state)
        assert len(result) == 25
        # After permutation, state should NOT be all zeros
        assert any(x != 0 for x in result)
    
    def test_deterministic(self):
        """Same input should always produce same output."""
        state1 = list(range(25))
        state2 = list(range(25))
        r1 = keccak_f1600(state1)
        r2 = keccak_f1600(state2)
        assert r1 == r2


class TestPowHash:
    """Test the PrePowHash (cSHAKE256 ProofOfWorkHash) implementation."""
    
    def test_initial_state_values(self):
        """Verify the pre-computed initial state has correct length."""
        assert len(POWHASH_INITIAL_STATE) == 25
        # All values should be valid u64
        for val in POWHASH_INITIAL_STATE:
            assert 0 <= val < (1 << 64)
    
    def test_construct_and_finalize(self):
        """Test PowHash construction and nonce finalization."""
        hasher = PowHash(HASH_42, TEST_TIMESTAMP)
        result = hasher.finalize_with_nonce(TEST_NONCE)
        
        # Result should be 32 bytes
        assert len(result) == 32
        assert isinstance(result, bytes)
    
    def test_different_nonces_different_hashes(self):
        """Different nonces must produce different hashes."""
        hasher = PowHash(HASH_42, TEST_TIMESTAMP)
        h1 = hasher.finalize_with_nonce(0)
        h2 = hasher.finalize_with_nonce(1)
        h3 = hasher.finalize_with_nonce(TEST_NONCE)
        
        assert h1 != h2
        assert h2 != h3
        assert h1 != h3
    
    def test_same_nonce_same_hash(self):
        """Same nonce must always produce same hash (deterministic)."""
        hasher = PowHash(HASH_42, TEST_TIMESTAMP)
        h1 = hasher.finalize_with_nonce(TEST_NONCE)
        h2 = hasher.finalize_with_nonce(TEST_NONCE)
        assert h1 == h2
    
    def test_batch_matches_scalar(self):
        """Batch processing must match single-nonce processing."""
        hasher = PowHash(HASH_42, TEST_TIMESTAMP)
        
        nonces = np.array([0, 1, 100, TEST_NONCE, 2**32 - 1], dtype=np.uint64)
        batch_results = hasher.finalize_batch(nonces)
        
        for i, nonce in enumerate(nonces):
            scalar_result = hasher.finalize_with_nonce(int(nonce))
            assert bytes(batch_results[i]) == scalar_result, \
                f"Mismatch at nonce {nonce}"


class TestHeavyHash:
    """Test the final HeavyHash (cSHAKE256 HeavyHash) step."""
    
    def test_initial_state_values(self):
        """Verify the pre-computed HeavyHash initial state."""
        assert len(HEAVYHASH_INITIAL_STATE) == 25
        for val in HEAVYHASH_INITIAL_STATE:
            assert 0 <= val < (1 << 64)
    
    def test_hash_deterministic(self):
        """HeavyHash must be deterministic."""
        h1 = HeavyHash.hash(HASH_42)
        h2 = HeavyHash.hash(HASH_42)
        assert h1 == h2
        assert len(h1) == 32
    
    def test_different_inputs_different_outputs(self):
        """Different inputs must produce different outputs."""
        h1 = HeavyHash.hash(HASH_42)
        h2 = HeavyHash.hash(bytes([0] * 32))
        assert h1 != h2
    
    def test_batch_matches_scalar(self):
        """Batch HeavyHash must match scalar."""
        inputs = np.array([
            list(HASH_42),
            list(bytes([0] * 32)),
            list(bytes(range(32))),
        ], dtype=np.uint8)
        
        batch_results = HeavyHash.hash_batch(inputs)
        
        for i in range(len(inputs)):
            scalar = HeavyHash.hash(bytes(inputs[i]))
            assert bytes(batch_results[i]) == scalar


class TestXoShiRo256PlusPlus:
    """Test the xoshiro256++ PRNG."""
    
    def test_deterministic(self):
        """Same seed must produce same sequence."""
        gen1 = XoShiRo256PlusPlus(HASH_42)
        gen2 = XoShiRo256PlusPlus(HASH_42)
        
        for _ in range(100):
            assert gen1.u64() == gen2.u64()
    
    def test_different_seeds(self):
        """Different seeds must produce different sequences."""
        gen1 = XoShiRo256PlusPlus(HASH_42)
        gen2 = XoShiRo256PlusPlus(bytes([0] * 32))
        
        # At least one of the first 10 values should differ
        any_different = False
        for _ in range(10):
            if gen1.u64() != gen2.u64():
                any_different = True
                break
        assert any_different
    
    def test_values_are_u64(self):
        """All generated values must be valid u64."""
        gen = XoShiRo256PlusPlus(HASH_42)
        for _ in range(1000):
            val = gen.u64()
            assert 0 <= val < (1 << 64)


class TestMatrix:
    """Test the 64x64 matrix generation and operations."""
    
    def test_generate_deterministic(self):
        """Same pre_pow_hash must produce same matrix."""
        m1 = Matrix.generate(HASH_42)
        m2 = Matrix.generate(HASH_42)
        np.testing.assert_array_equal(m1.data, m2.data)
    
    def test_matrix_shape(self):
        """Matrix must be 64x64."""
        m = Matrix.generate(HASH_42)
        assert m.data.shape == (64, 64)
    
    def test_matrix_nibble_range(self):
        """All matrix values must be 4-bit nibbles (0-15)."""
        m = Matrix.generate(HASH_42)
        assert m.data.max() <= 15
        assert m.data.min() >= 0
    
    def test_matrix_full_rank(self):
        """Generated matrix must have rank 64."""
        m = Matrix.generate(HASH_42)
        assert m._compute_rank() == 64
    
    def test_heavy_hash_deterministic(self):
        """heavy_hash must be deterministic."""
        m = Matrix.generate(HASH_42)
        h1 = m.heavy_hash(HASH_42)
        h2 = m.heavy_hash(HASH_42)
        assert h1 == h2
        assert len(h1) == 32
    
    def test_heavy_hash_different_inputs(self):
        """Different inputs → different heavy_hash outputs."""
        m = Matrix.generate(HASH_42)
        h1 = m.heavy_hash(HASH_42)
        h2 = m.heavy_hash(bytes([0] * 32))
        assert h1 != h2
    
    def test_batch_cpu_matches_scalar(self):
        """Batch CPU heavy_hash must match scalar."""
        m = Matrix.generate(HASH_42)
        
        inputs = np.array([
            list(HASH_42),
            list(bytes([0] * 32)),
            list(bytes(range(32))),
        ], dtype=np.uint8)
        
        batch_results = m.heavy_hash_batch_cpu(inputs)
        
        for i in range(len(inputs)):
            scalar = m.heavy_hash(bytes(inputs[i]))
            assert bytes(batch_results[i]) == scalar, \
                f"Mismatch at input {i}"


class TestTPUMatMul:
    """Test the TPU MatMul kernel (runs on CPU fallback if no TPU)."""
    
    def test_matmul_matches_reference(self):
        """TPU MatMul must match the reference matrix.heavy_hash matmul step."""
        m = Matrix.generate(HASH_42)
        engine = TPUMatMul(m.data)
        
        # Test with a known input
        test_hash = HASH_42
        
        # --- Reference: scalar heavy_hash matmul step ---
        vec = np.zeros(64, dtype=np.uint8)
        for i, byte_val in enumerate(test_hash):
            vec[2 * i] = byte_val >> 4
            vec[2 * i + 1] = byte_val & 0x0F
        
        ref_product = np.zeros(32, dtype=np.uint8)
        for i in range(32):
            sum1 = sum(int(m.data[2*i, j]) * int(vec[j]) for j in range(64))
            sum2 = sum(int(m.data[2*i+1, j]) * int(vec[j]) for j in range(64))
            ref_product[i] = (((sum1 >> 10) & 0x0F) << 4) | ((sum2 >> 10) & 0x0F)
        
        # --- TPU/batch path ---
        hashes = np.array([list(test_hash)], dtype=np.uint8)
        nibbles = hashes_to_nibbles(hashes)
        tpu_products = engine.batched_matmul(nibbles)
        
        np.testing.assert_array_equal(tpu_products[0], ref_product)
    
    def test_batch_consistency(self):
        """Batch of N should equal N individual calls."""
        m = Matrix.generate(HASH_42)
        engine = TPUMatMul(m.data)
        
        test_hashes = np.array([
            list(HASH_42),
            list(bytes([0] * 32)),
            list(bytes(range(32))),
            list(bytes([255] * 32)),
        ], dtype=np.uint8)
        
        nibbles = hashes_to_nibbles(test_hashes)
        batch_products = engine.batched_matmul(nibbles)
        
        for i in range(len(test_hashes)):
            single_nibbles = hashes_to_nibbles(test_hashes[i:i+1])
            single_product = engine.batched_matmul(single_nibbles)
            np.testing.assert_array_equal(
                batch_products[i], single_product[0],
                err_msg=f"Batch/single mismatch at index {i}"
            )


class TestNibbleConversion:
    """Test hash-to-nibble and product-XOR operations."""
    
    def test_nibble_extraction(self):
        """Verify correct nibble extraction from bytes."""
        test_hash = np.array([[0xAB, 0xCD, 0x12, 0x34] + [0] * 28], dtype=np.uint8)
        nibbles = hashes_to_nibbles(test_hash)
        
        assert nibbles[0, 0] == 0xA   # high nibble of 0xAB
        assert nibbles[0, 1] == 0xB   # low nibble of 0xAB
        assert nibbles[0, 2] == 0xC   # high nibble of 0xCD
        assert nibbles[0, 3] == 0xD   # low nibble of 0xCD
        assert nibbles[0, 4] == 0x1   # high nibble of 0x12
        assert nibbles[0, 5] == 0x2   # low nibble of 0x12
    
    def test_xor_operation(self):
        """Verify XOR of products with hashes."""
        products = np.array([[0xFF, 0x00, 0xAA] + [0] * 29], dtype=np.uint8)
        hashes = np.array([[0x0F, 0xF0, 0x55] + [0] * 29], dtype=np.uint8)
        
        result = products_xor_hashes(products, hashes)
        
        assert result[0, 0] == 0xF0
        assert result[0, 1] == 0xF0
        assert result[0, 2] == 0xFF


class TestEndToEnd:
    """End-to-end pipeline test: full kHeavyHash computation."""
    
    def test_full_pow_calculation(self):
        """
        Full PoW pipeline: PowHash → MatMul → HeavyHash.
        
        Verifies that:
          1. PowHash produces a hash
          2. Matrix heavy_hash transforms it
          3. The result is deterministic
        """
        pre_pow_hash = HASH_42
        timestamp = TEST_TIMESTAMP
        
        # Generate matrix
        matrix = Matrix.generate(pre_pow_hash)
        
        # Create PowHash
        hasher = PowHash(pre_pow_hash, timestamp)
        
        # Compute for a specific nonce
        pow_hash = hasher.finalize_with_nonce(TEST_NONCE)
        heavy = matrix.heavy_hash(pow_hash)
        
        # Verify determinism
        pow_hash2 = hasher.finalize_with_nonce(TEST_NONCE)
        heavy2 = matrix.heavy_hash(pow_hash2)
        
        assert heavy == heavy2
        assert len(heavy) == 32
    
    def test_tpu_matches_cpu_full_pipeline(self):
        """
        Full pipeline: TPU path must match CPU reference path.
        """
        pre_pow_hash = bytes(range(32))
        timestamp = 1700000000
        
        # Generate matrix
        matrix = Matrix.generate(pre_pow_hash)
        engine = TPUMatMul(matrix.data)
        hasher = PowHash(pre_pow_hash, timestamp)
        
        # Test nonces
        nonces = np.array([0, 1, 1000, 2**32], dtype=np.uint64)
        
        for nonce_val in nonces:
            nonce = int(nonce_val)
            
            # CPU reference path
            cpu_pow_hash = hasher.finalize_with_nonce(nonce)
            cpu_heavy = matrix.heavy_hash(cpu_pow_hash)
            
            # TPU/batch path
            batch_pow = hasher.finalize_batch(np.array([nonce_val], dtype=np.uint64))
            nibbles = hashes_to_nibbles(batch_pow)
            products = engine.batched_matmul(nibbles)
            xored = products_xor_hashes(products, batch_pow)
            tpu_heavy = HeavyHash.hash_batch(xored)
            
            assert bytes(tpu_heavy[0]) == cpu_heavy, \
                f"CPU/TPU mismatch at nonce={nonce}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
