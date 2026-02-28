"""
Keccak-f1600 and cSHAKE256 implementation for Kaspa kHeavyHash.

This is a performance-critical CPU-side module. The implementation uses
optimized constants and follows the exact same logic as Rusty Kaspa's
pow_hashers.rs to ensure hash compatibility.

Two cSHAKE256 domain-separated instances:
  - "ProofOfWorkHash" : PrePowHash (absorbs pre_pow_hash || timestamp || padding || nonce)
  - "HeavyHash"       : Final hash after MatMul
"""

import struct
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Keccak-f1600 constants
# ---------------------------------------------------------------------------

# Round constants (RC) for 24 rounds of Keccak-f1600
_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

# Rotation offsets for the rho step
_ROT = [
    0, 1, 62, 28, 27,
    36, 44, 6, 55, 20,
    3, 10, 43, 25, 39,
    41, 45, 15, 21, 8,
    18, 2, 61, 56, 14,
]

# Pi step permutation indices
_PI = [
    0, 6, 12, 18, 24,
    3, 9, 10, 16, 22,
    1, 7, 13, 19, 20,
    4, 5, 11, 17, 23,
    2, 8, 14, 15, 21,
]

_MASK64 = 0xFFFFFFFFFFFFFFFF


def _rotl64(x: int, n: int) -> int:
    """Rotate left a 64-bit unsigned integer."""
    n = n % 64
    return ((x << n) | (x >> (64 - n))) & _MASK64


def keccak_f1600(state: list[int]) -> list[int]:
    """
    Keccak-f1600 permutation on a 25-element state of 64-bit words.
    
    This is the core permutation used in SHA-3 / cSHAKE256.
    For mining, this runs on CPU — one call per nonce for PrePowHash,
    one call per nonce for the final HeavyHash.
    
    Args:
        state: List of 25 uint64 values (will be modified in-place and returned)
    
    Returns:
        The permuted state (same list object, mutated)
    """
    s = list(state)  # work on a copy for safety
    for round_idx in range(24):
        # θ (theta) step
        c = [
            s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20],
            s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21],
            s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22],
            s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23],
            s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24],
        ]
        d = [
            (c[4] ^ _rotl64(c[1], 1)) & _MASK64,
            (c[0] ^ _rotl64(c[2], 1)) & _MASK64,
            (c[1] ^ _rotl64(c[3], 1)) & _MASK64,
            (c[2] ^ _rotl64(c[4], 1)) & _MASK64,
            (c[3] ^ _rotl64(c[0], 1)) & _MASK64,
        ]
        for i in range(25):
            s[i] = (s[i] ^ d[i % 5]) & _MASK64

        # ρ (rho) and π (pi) steps combined
        tmp = [0] * 25
        for i in range(25):
            tmp[_PI[i]] = _rotl64(s[i], _ROT[i])

        # χ (chi) step
        for i in range(5):
            base = i * 5
            t = list(tmp[base:base + 5])
            for j in range(5):
                s[base + j] = (t[j] ^ ((~t[(j + 1) % 5] & _MASK64) & t[(j + 2) % 5])) & _MASK64

        # ι (iota) step
        s[0] = (s[0] ^ _RC[round_idx]) & _MASK64

    # Copy back
    for i in range(25):
        state[i] = s[i]
    return state


def keccak_f1600_batch(states: np.ndarray) -> np.ndarray:
    """
    Batch Keccak-f1600 for multiple states using NumPy vectorization.
    
    Args:
        states: np.ndarray of shape (N, 25) with dtype=np.uint64
    
    Returns:
        Permuted states array of same shape
    """
    s = states.copy()
    
    for round_idx in range(24):
        # θ step — vectorized across all N states
        c = np.zeros((s.shape[0], 5), dtype=np.uint64)
        for i in range(5):
            c[:, i] = s[:, i] ^ s[:, i + 5] ^ s[:, i + 10] ^ s[:, i + 15] ^ s[:, i + 20]
        
        d = np.zeros((s.shape[0], 5), dtype=np.uint64)
        for i in range(5):
            rot_val = ((c[:, (i + 1) % 5] << np.uint64(1)) | (c[:, (i + 1) % 5] >> np.uint64(63)))
            d[:, i] = c[:, (i + 4) % 5] ^ rot_val
        
        for i in range(25):
            s[:, i] ^= d[:, i % 5]
        
        # ρ and π steps
        tmp = np.zeros_like(s)
        for i in range(25):
            rot = _ROT[i]
            if rot == 0:
                tmp[:, _PI[i]] = s[:, i]
            else:
                tmp[:, _PI[i]] = ((s[:, i] << np.uint64(rot)) | (s[:, i] >> np.uint64(64 - rot)))
        
        # χ step
        result = np.zeros_like(s)
        for i in range(5):
            base = i * 5
            for j in range(5):
                result[:, base + j] = tmp[:, base + j] ^ (
                    (~tmp[:, base + (j + 1) % 5]) & tmp[:, base + (j + 2) % 5]
                )
        s = result
        
        # ι step
        s[:, 0] ^= np.uint64(_RC[round_idx])
    
    return s


# ---------------------------------------------------------------------------
# Pre-computed initial states for cSHAKE256 domains
# (Matches Rusty Kaspa's PowHash::INITIAL_STATE and KHeavyHash::INITIAL_STATE)
# ---------------------------------------------------------------------------

# cSHAKE256("ProofOfWorkHash") initial state after absorbing the domain
# Taken directly from Rusty Kaspa pow_hashers.rs
POWHASH_INITIAL_STATE = [
    1242148031264380989, 3008272977830772284, 2188519011337848018,
    1992179434288343456, 8876506674959887717, 5399642050693751366,
    1745875063082670864, 8605242046444978844, 17936695144567157056,
    3343109343542796272, 1123092876221303306, 4963925045340115282,
    17037383077651887893, 16629644495023626889, 12833675776649114147,
    3784524041015224902, 1082795874807940378, 13952716920571277634,
    13411128033953605860, 15060696040649351053, 9928834659948351306,
    5237849264682708699, 12825353012139217522, 6706187291358897596,
    196324915476054915,
]

# cSHAKE256("HeavyHash") initial state after absorbing the domain
HEAVYHASH_INITIAL_STATE = [
    4239941492252378377, 8746723911537738262, 8796936657246353646,
    1272090201925444760, 16654558671554924250, 8270816933120786537,
    13907396207649043898, 6782861118970774626, 9239690602118867528,
    11582319943599406348, 17596056728278508070, 15212962468105129023,
    7812475424661425213, 3370482334374859748, 5690099369266491460,
    8596393687355028144, 570094237299545110, 9119540418498120711,
    16901969272480492857, 13372017233735502424, 14372891883993151831,
    5171152063242093102, 10573107899694386186, 6096431547456407061,
    1592359455985097269,
]


def _hash_to_le_u64(hash_bytes: bytes) -> list[int]:
    """Convert a 32-byte hash to 4 little-endian u64 values."""
    assert len(hash_bytes) == 32
    return list(struct.unpack('<4Q', hash_bytes))


def _le_u64_to_hash(words: list[int]) -> bytes:
    """Convert 4 little-endian u64 values back to a 32-byte hash."""
    return struct.pack('<4Q', *words[:4])


class PowHash:
    """
    cSHAKE256("ProofOfWorkHash") hasher for Kaspa PrePowHash.
    
    Absorbs: pre_pow_hash || timestamp || (32 zero bytes padding) || nonce
    The state is pre-built with everything except the nonce, matching
    Rusty Kaspa's PowHash struct design for efficient nonce iteration.
    """
    
    def __init__(self, pre_pow_hash: bytes, timestamp: int):
        """
        Initialize with pre_pow_hash and timestamp.
        
        Args:
            pre_pow_hash: 32-byte hash
            timestamp: Block timestamp as u64
        """
        assert len(pre_pow_hash) == 32
        
        # Start from the pre-computed cSHAKE256("ProofOfWorkHash") state
        self._state = list(POWHASH_INITIAL_STATE)
        
        # XOR in the pre_pow_hash (first 4 u64 words)
        hash_words = _hash_to_le_u64(pre_pow_hash)
        for i in range(4):
            self._state[i] ^= hash_words[i]
        
        # XOR in the timestamp at position 4
        self._state[4] ^= (timestamp & _MASK64)
        
        # Positions 5–8 would absorb the 32 zero-byte padding (XOR with 0 = no-op)
        # Position 9 will absorb the nonce in finalize_with_nonce
    
    def finalize_with_nonce(self, nonce: int) -> bytes:
        """
        Finalize the hash with a specific nonce.
        
        Args:
            nonce: 64-bit nonce value
        
        Returns:
            32-byte hash result
        """
        state = list(self._state)  # Clone the pre-built state
        state[9] ^= (nonce & _MASK64)
        keccak_f1600(state)
        return _le_u64_to_hash(state[:4])
    
    def finalize_batch(self, nonces: np.ndarray) -> np.ndarray:
        """
        Finalize hashes for a batch of nonces using vectorized Keccak.
        
        Args:
            nonces: 1D array of uint64 nonce values, shape (N,)
        
        Returns:
            Array of 32-byte hashes, shape (N, 32) as uint8
        """
        n = len(nonces)
        
        # Build N copies of the pre-built state
        states = np.tile(np.array(self._state, dtype=np.uint64), (n, 1))
        
        # XOR nonces into position 9
        states[:, 9] ^= nonces.astype(np.uint64)
        
        # Batch Keccak
        states = keccak_f1600_batch(states)
        
        # Extract first 4 u64 words → 32 bytes per hash
        result = np.zeros((n, 32), dtype=np.uint8)
        for i in range(4):
            # .copy() ensures contiguous memory for .view() dtype change
            word_bytes = states[:, i].copy().view(np.uint8).reshape(n, 8)
            result[:, i * 8:(i + 1) * 8] = word_bytes
        
        return result


class HeavyHash:
    """
    cSHAKE256("HeavyHash") — the final hash step after MatMul.
    
    Absorbs: 32-byte product (from MatMul XOR'd with original hash)
    """
    
    @staticmethod
    def hash(input_hash: bytes) -> bytes:
        """
        Apply cSHAKE256("HeavyHash") to a 32-byte input.
        
        Args:
            input_hash: 32-byte input
        
        Returns:
            32-byte hash result
        """
        assert len(input_hash) == 32
        
        state = list(HEAVYHASH_INITIAL_STATE)
        hash_words = _hash_to_le_u64(input_hash)
        for i in range(4):
            state[i] ^= hash_words[i]
        keccak_f1600(state)
        return _le_u64_to_hash(state[:4])
    
    @staticmethod
    def hash_batch(inputs: np.ndarray) -> np.ndarray:
        """
        Batch cSHAKE256("HeavyHash") for N inputs.
        
        Args:
            inputs: Array of shape (N, 32) uint8
        
        Returns:
            Array of shape (N, 32) uint8
        """
        n = inputs.shape[0]
        
        # Build N copies of the initial state
        states = np.tile(
            np.array(HEAVYHASH_INITIAL_STATE, dtype=np.uint64), (n, 1)
        )
        
        # Convert input bytes to u64 words and XOR
        for i in range(4):
            # .copy() ensures C-contiguous layout for .view() reinterpretation
            word_bytes = np.ascontiguousarray(inputs[:, i * 8:(i + 1) * 8])
            words = word_bytes.view(np.uint64).reshape(n)
            states[:, i] ^= words
        
        # Batch Keccak
        states = keccak_f1600_batch(states)
        
        # Extract result
        result = np.zeros((n, 32), dtype=np.uint8)
        for i in range(4):
            # .copy() ensures contiguous memory for .view() dtype change
            word_bytes = states[:, i].copy().view(np.uint8).reshape(n, 8)
            result[:, i * 8:(i + 1) * 8] = word_bytes
        
        return result
