"""
PowHash — high-level interface combining cSHAKE256 PrePowHash with
the mining state, matching Rusty Kaspa's lib.rs State struct.

This module ties together:
  - Header hashing (pre_pow_hash generation)
  - PowHash (cSHAKE256("ProofOfWorkHash"))
  - Matrix generation
  - Heavy hash computation
  - Difficulty checking
"""

import struct
import numpy as np
from typing import Tuple

from .keccak import PowHash as KeccakPowHash, HeavyHash


def hash_override_nonce_time(header_hash: bytes, nonce: int = 0, timestamp: int = 0) -> bytes:
    """
    Placeholder for the header serialization hash.
    
    In Rusty Kaspa, this hashes the full block header with nonce and time
    zeroed out to produce the pre_pow_hash. In a real miner, this is
    received from the node/pool via GetBlockTemplate.
    
    Args:
        header_hash: Pre-computed pre_pow_hash from the block template
        nonce: Nonce override (usually 0 for template)
        timestamp: Timestamp override (usually 0 for template)
    
    Returns:
        32-byte pre_pow_hash
    """
    # In practice, the node provides this via GetBlockTemplate
    return header_hash


def uint256_from_le_bytes(b: bytes) -> int:
    """Convert 32 little-endian bytes to a 256-bit integer."""
    return int.from_bytes(b, byteorder='little')


def compact_target_to_uint256(bits: int) -> int:
    """
    Convert compact target representation to full 256-bit target.
    
    Same as Bitcoin's nBits format used by Kaspa.
    
    Args:
        bits: Compact target representation (32-bit)
    
    Returns:
        256-bit target value
    """
    exponent = (bits >> 24) & 0xFF
    mantissa = bits & 0x7FFFFF
    
    if exponent <= 3:
        mantissa >>= 8 * (3 - exponent)
        target = mantissa
    else:
        target = mantissa << (8 * (exponent - 3))
    
    # Sign bit
    if bits & 0x800000:
        target = -target
    
    return target & ((1 << 256) - 1)


class MiningState:
    """
    Pre-computed mining state for a block template.
    
    Mirrors Rusty Kaspa's `State` struct:
      - matrix: The 64x64 nibble matrix (generated once per block)
      - target: The difficulty target as Uint256
      - hasher: The PowHash with pre_pow_hash and timestamp absorbed
    
    Usage:
        state = MiningState(pre_pow_hash, timestamp, target_bits, matrix)
        found, pow_value = state.check_pow(nonce)
    """
    
    def __init__(self, pre_pow_hash: bytes, timestamp: int, target_bits: int, matrix):
        """
        Args:
            pre_pow_hash: 32-byte pre_pow_hash from block template
            timestamp: Block timestamp
            target_bits: Compact target bits
            matrix: Matrix instance from core.matrix
        """
        self.target = compact_target_to_uint256(target_bits)
        self.hasher = KeccakPowHash(pre_pow_hash, timestamp)
        self.matrix = matrix
        self.pre_pow_hash = pre_pow_hash
    
    def calculate_pow(self, nonce: int) -> int:
        """
        Calculate the PoW value for a single nonce.
        
        Pipeline: PowHash(nonce) → matrix.heavy_hash() → Uint256
        
        Args:
            nonce: 64-bit nonce
        
        Returns:
            256-bit PoW value
        """
        # Stage 1: PrePowHash (Keccak on CPU)
        hash_result = self.hasher.finalize_with_nonce(nonce)
        
        # Stage 2: Heavy hash (MatMul + final Keccak)
        heavy = self.matrix.heavy_hash(hash_result)
        
        return uint256_from_le_bytes(heavy)
    
    def check_pow(self, nonce: int) -> Tuple[bool, int]:
        """
        Check if a nonce produces a valid PoW.
        
        Args:
            nonce: 64-bit nonce
        
        Returns:
            Tuple of (is_valid, pow_value)
        """
        pow_value = self.calculate_pow(nonce)
        return pow_value <= self.target, pow_value
