"""
xoshiro256++ PRNG — exact port from Rusty Kaspa.

Used to generate the 64x64 matrix from a block's pre_pow_hash.
This runs on CPU since it's called once per block template.

Reference: rusty-kaspa/consensus/pow/src/xoshiro.rs
"""

import struct
import numpy as np

_MASK64 = 0xFFFFFFFFFFFFFFFF


def _rotl64(x: int, k: int) -> int:
    """Rotate left a 64-bit unsigned integer."""
    k = k % 64
    return ((x << k) | (x >> (64 - k))) & _MASK64


class XoShiRo256PlusPlus:
    """
    xoshiro256++ PRNG, seeded from a 32-byte (256-bit) hash.
    
    Exactly matches the Rust implementation in rusty-kaspa.
    Used for deterministic matrix generation from pre_pow_hash.
    """
    
    __slots__ = ('s0', 's1', 's2', 's3')
    
    def __init__(self, hash_bytes: bytes):
        """
        Seed the PRNG from a 32-byte hash (interpreted as 4 LE u64 words).
        
        Args:
            hash_bytes: 32-byte seed hash
        """
        assert len(hash_bytes) == 32
        words = struct.unpack('<4Q', hash_bytes)
        self.s0 = words[0]
        self.s1 = words[1]
        self.s2 = words[2]
        self.s3 = words[3]
    
    def u64(self) -> int:
        """
        Generate the next 64-bit pseudorandom value.
        
        Exactly matches Rusty Kaspa:
            res = s0 + rotl(s0 + s3, 23)
            t = s1 << 17
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3
            s2 ^= t
            s3 = rotl(s3, 45)
        
        Returns:
            64-bit unsigned integer
        """
        # Result computation (wrapping arithmetic)
        inner = (self.s0 + self.s3) & _MASK64
        res = (self.s0 + _rotl64(inner, 23)) & _MASK64
        
        t = (self.s1 << 17) & _MASK64
        
        self.s2 ^= self.s0
        self.s3 ^= self.s1
        self.s1 ^= self.s2
        self.s0 ^= self.s3
        
        self.s2 = (self.s2 ^ t) & _MASK64
        self.s3 = _rotl64(self.s3, 45)
        
        return res
