/*
 * keccak_f1600.c — Fast C implementation of the Keccak-f[1600] permutation
 * 
 * This is the performance-critical core of kHeavyHash mining.
 * Compiled as a shared library and called via ctypes from Python.
 * 
 * Public domain / CC0 — based on the Keccak reference implementation.
 *
 * Build:
 *   gcc -O3 -march=native -shared -fPIC -o libkeccak.so keccak_f1600.c
 */

#include <stdint.h>
#include <string.h>

/* Round constants */
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

/* Rotation offsets */
static const int ROTATIONS[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

/* Pi permutation indices */
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

/*
 * In-place Keccak-f[1600] permutation on a 25-word state.
 */
void keccak_f1600(uint64_t state[25]) {
    uint64_t t, bc[5];
    int round, i, j;

    for (round = 0; round < 24; round++) {
        /* θ (theta) */
        for (i = 0; i < 5; i++)
            bc[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];

        for (i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5)
                state[j + i] ^= t;
        }

        /* ρ (rho) + π (pi) */
        {
            uint64_t temp[25];
            for (i = 0; i < 25; i++)
                temp[PI[i]] = rotl64(state[i], ROTATIONS[i]);
            memcpy(state, temp, sizeof(temp));
        }

        /* χ (chi) */
        for (j = 0; j < 25; j += 5) {
            uint64_t s0 = state[j + 0], s1 = state[j + 1], s2 = state[j + 2],
                     s3 = state[j + 3], s4 = state[j + 4];
            state[j + 0] = s0 ^ ((~s1) & s2);
            state[j + 1] = s1 ^ ((~s2) & s3);
            state[j + 2] = s2 ^ ((~s3) & s4);
            state[j + 3] = s3 ^ ((~s4) & s0);
            state[j + 4] = s4 ^ ((~s0) & s1);
        }

        /* ι (iota) */
        state[0] ^= RC[round];
    }
}

/*
 * Batch Keccak-f[1600]: apply permutation to N independent states.
 *
 * states: pointer to N*25 uint64_t values (N states, each 25 words)
 * n:      number of states
 */
void keccak_f1600_batch(uint64_t *states, int n) {
    int i;
    for (i = 0; i < n; i++) {
        keccak_f1600(states + i * 25);
    }
}

/*
 * Complete PowHash PreHash for a batch of nonces.
 *
 * Computes cSHAKE256("ProofOfWorkHash", pre_pow_hash || timestamp || nonce)
 * for each nonce in the batch.
 *
 * initial_state: 25 uint64 words — pre-built state with hash+timestamp absorbed
 * nonces:        array of N uint64 nonce values
 * n:             number of nonces
 * out_hashes:    output buffer, N*32 bytes (N hashes of 32 bytes each)
 */
void powhash_batch(const uint64_t initial_state[25],
                   const uint64_t *nonces, int n,
                   uint8_t *out_hashes) {
    int i, j;
    uint64_t state[25];

    for (i = 0; i < n; i++) {
        /* Copy initial state */
        memcpy(state, initial_state, 25 * sizeof(uint64_t));

        /* XOR nonce at position 9 */
        state[9] ^= nonces[i];

        /* Apply Keccak-f[1600] */
        keccak_f1600(state);

        /* Extract first 32 bytes (4 words, little-endian) */
        for (j = 0; j < 4; j++) {
            uint64_t word = state[j];
            out_hashes[i * 32 + j * 8 + 0] = (uint8_t)(word);
            out_hashes[i * 32 + j * 8 + 1] = (uint8_t)(word >> 8);
            out_hashes[i * 32 + j * 8 + 2] = (uint8_t)(word >> 16);
            out_hashes[i * 32 + j * 8 + 3] = (uint8_t)(word >> 24);
            out_hashes[i * 32 + j * 8 + 4] = (uint8_t)(word >> 32);
            out_hashes[i * 32 + j * 8 + 5] = (uint8_t)(word >> 40);
            out_hashes[i * 32 + j * 8 + 6] = (uint8_t)(word >> 48);
            out_hashes[i * 32 + j * 8 + 7] = (uint8_t)(word >> 56);
        }
    }
}

/*
 * Complete HeavyHash final hash for a batch.
 *
 * Computes cSHAKE256("HeavyHash", input) for each input in the batch.
 *
 * heavy_initial_state: 25 uint64 words — initial state for HeavyHash domain
 * inputs:              N*32 bytes of input data
 * n:                   number of inputs
 * out_hashes:          output buffer, N*32 bytes
 */
void heavyhash_batch(const uint64_t heavy_initial_state[25],
                     const uint8_t *inputs, int n,
                     uint8_t *out_hashes) {
    int i, j;
    uint64_t state[25];

    for (i = 0; i < n; i++) {
        /* Copy initial state */
        memcpy(state, heavy_initial_state, 25 * sizeof(uint64_t));

        /* XOR 32 input bytes (4 words) into state */
        for (j = 0; j < 4; j++) {
            uint64_t word = 0;
            int k;
            for (k = 0; k < 8; k++) {
                word |= (uint64_t)inputs[i * 32 + j * 8 + k] << (k * 8);
            }
            state[j] ^= word;
        }

        /* Apply Keccak-f[1600] */
        keccak_f1600(state);

        /* Extract first 32 bytes */
        for (j = 0; j < 4; j++) {
            uint64_t word = state[j];
            out_hashes[i * 32 + j * 8 + 0] = (uint8_t)(word);
            out_hashes[i * 32 + j * 8 + 1] = (uint8_t)(word >> 8);
            out_hashes[i * 32 + j * 8 + 2] = (uint8_t)(word >> 16);
            out_hashes[i * 32 + j * 8 + 3] = (uint8_t)(word >> 24);
            out_hashes[i * 32 + j * 8 + 4] = (uint8_t)(word >> 32);
            out_hashes[i * 32 + j * 8 + 5] = (uint8_t)(word >> 40);
            out_hashes[i * 32 + j * 8 + 6] = (uint8_t)(word >> 48);
            out_hashes[i * 32 + j * 8 + 7] = (uint8_t)(word >> 56);
        }
    }
}
