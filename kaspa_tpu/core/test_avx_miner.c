#include <stdint.h>
#include <string.h>
#include <immintrin.h>

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

#define ROL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

static inline void keccak_f1600(uint64_t state[25]) {
    uint64_t Aba, Abe, Abi, Abo, Abu;
    uint64_t Aga, Age, Agi, Ago, Agu;
    uint64_t Aka, Ake, Aki, Ako, Aku;
    uint64_t Ama, Ame, Ami, Amo, Amu;
    uint64_t Asa, Ase, Asi, Aso, Asu;
    uint64_t BCa, BCe, BCi, BCo, BCu;
    uint64_t Da, De, Di, Do, Du;
    uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
    uint64_t Ega, Ege, Egi, Ego, Egu;
    uint64_t Eka, Eke, Eki, Eko, Eku;
    uint64_t Ema, Eme, Emi, Emo, Emu;
    uint64_t Esa, Ese, Esi, Eso, Esu;
    int round;

    Aba = state[0]; Abe = state[1]; Abi = state[2]; Abo = state[3]; Abu = state[4];
    Aga = state[5]; Age = state[6]; Agi = state[7]; Ago = state[8]; Agu = state[9];
    Aka = state[10]; Ake = state[11]; Aki = state[12]; Ako = state[13]; Aku = state[14];
    Ama = state[15]; Ame = state[16]; Ami = state[17]; Amo = state[18]; Amu = state[19];
    Asa = state[20]; Ase = state[21]; Asi = state[22]; Aso = state[23]; Asu = state[24];

    for (round = 0; round < 24; round += 2) {
        /* Round 1 */
        BCa = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
        BCe = Abe ^ Age ^ Ake ^ Ame ^ Ase;
        BCi = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
        BCo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
        BCu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

        Da = BCu ^ ROL64(BCe, 1);
        De = BCa ^ ROL64(BCi, 1);
        Di = BCe ^ ROL64(BCo, 1);
        Do = BCi ^ ROL64(BCu, 1);
        Du = BCo ^ ROL64(BCa, 1);

        Aba ^= Da; BCa = Aba;
        Age ^= De; BCe = ROL64(Age, 44);
        Aki ^= Di; BCi = ROL64(Aki, 43);
        Amo ^= Do; BCo = ROL64(Amo, 21);
        Asu ^= Du; BCu = ROL64(Asu, 14);
        Eba = BCa ^ ((~BCe) & BCi) ^ RC[round];
        Ebe = BCe ^ ((~BCi) & BCo);
        Ebi = BCi ^ ((~BCo) & BCu);
        Ebo = BCo ^ ((~BCu) & BCa);
        Ebu = BCu ^ ((~BCa) & BCe);

        Abo ^= Do; BCa = ROL64(Abo, 28);
        Agu ^= Du; BCe = ROL64(Agu, 20);
        Aka ^= Da; BCi = ROL64(Aka, 3);
        Ame ^= De; BCo = ROL64(Ame, 45);
        Asi ^= Di; BCu = ROL64(Asi, 61);
        Ega = BCa ^ ((~BCe) & BCi);
        Ege = BCe ^ ((~BCi) & BCo);
        Egi = BCi ^ ((~BCo) & BCu);
        Ego = BCo ^ ((~BCu) & BCa);
        Egu = BCu ^ ((~BCa) & BCe);

        Abe ^= De; BCa = ROL64(Abe, 1);
        Agi ^= Di; BCe = ROL64(Agi, 6);
        Ako ^= Do; BCi = ROL64(Ako, 25);
        Amu ^= Du; BCo = ROL64(Amu, 8);
        Asa ^= Da; BCu = ROL64(Asa, 18);
        Eka = BCa ^ ((~BCe) & BCi);
        Eke = BCe ^ ((~BCi) & BCo);
        Eki = BCi ^ ((~BCo) & BCu);
        Eko = BCo ^ ((~BCu) & BCa);
        Eku = BCu ^ ((~BCa) & BCe);

        Abu ^= Du; BCa = ROL64(Abu, 27);
        Aga ^= Da; BCe = ROL64(Aga, 36);
        Ake ^= De; BCi = ROL64(Ake, 10);
        Ami ^= Di; BCo = ROL64(Ami, 15);
        Aso ^= Do; BCu = ROL64(Aso, 56);
        Ema = BCa ^ ((~BCe) & BCi);
        Eme = BCe ^ ((~BCi) & BCo);
        Emi = BCi ^ ((~BCo) & BCu);
        Emo = BCo ^ ((~BCu) & BCa);
        Emu = BCu ^ ((~BCa) & BCe);

        Abi ^= Di; BCa = ROL64(Abi, 62);
        Ago ^= Do; BCe = ROL64(Ago, 55);
        Aku ^= Du; BCi = ROL64(Aku, 39);
        Ama ^= Da; BCo = ROL64(Ama, 41);
        Ase ^= De; BCu = ROL64(Ase, 2);
        Esa = BCa ^ ((~BCe) & BCi);
        Ese = BCe ^ ((~BCi) & BCo);
        Esi = BCi ^ ((~BCo) & BCu);
        Eso = BCo ^ ((~BCu) & BCa);
        Esu = BCu ^ ((~BCa) & BCe);

        /* Round 2 */
        BCa = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
        BCe = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
        BCi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
        BCo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
        BCu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

        Da = BCu ^ ROL64(BCe, 1);
        De = BCa ^ ROL64(BCi, 1);
        Di = BCe ^ ROL64(BCo, 1);
        Do = BCi ^ ROL64(BCu, 1);
        Du = BCo ^ ROL64(BCa, 1);

        Eba ^= Da; BCa = Eba;
        Ege ^= De; BCe = ROL64(Ege, 44);
        Eki ^= Di; BCi = ROL64(Eki, 43);
        Emo ^= Do; BCo = ROL64(Emo, 21);
        Esu ^= Du; BCu = ROL64(Esu, 14);
        Aba = BCa ^ ((~BCe) & BCi) ^ RC[round + 1];
        Abe = BCe ^ ((~BCi) & BCo);
        Abi = BCi ^ ((~BCo) & BCu);
        Abo = BCo ^ ((~BCu) & BCa);
        Abu = BCu ^ ((~BCa) & BCe);

        Ebo ^= Do; BCa = ROL64(Ebo, 28);
        Egu ^= Du; BCe = ROL64(Egu, 20);
        Eka ^= Da; BCi = ROL64(Eka, 3);
        Eme ^= De; BCo = ROL64(Eme, 45);
        Esi ^= Di; BCu = ROL64(Esi, 61);
        Aga = BCa ^ ((~BCe) & BCi);
        Age = BCe ^ ((~BCi) & BCo);
        Agi = BCi ^ ((~BCo) & BCu);
        Ago = BCo ^ ((~BCu) & BCa);
        Agu = BCu ^ ((~BCa) & BCe);

        Ebe ^= De; BCa = ROL64(Ebe, 1);
        Egi ^= Di; BCe = ROL64(Egi, 6);
        Eko ^= Do; BCi = ROL64(Eko, 25);
        Emu ^= Du; BCo = ROL64(Emu, 8);
        Esa ^= Da; BCu = ROL64(Esa, 18);
        Aka = BCa ^ ((~BCe) & BCi);
        Ake = BCe ^ ((~BCi) & BCo);
        Aki = BCi ^ ((~BCo) & BCu);
        Ako = BCo ^ ((~BCu) & BCa);
        Aku = BCu ^ ((~BCa) & BCe);

        Ebu ^= Du; BCa = ROL64(Ebu, 27);
        Ega ^= Da; BCe = ROL64(Ega, 36);
        Eke ^= De; BCi = ROL64(Eke, 10);
        Emi ^= Di; BCo = ROL64(Emi, 15);
        Eso ^= Do; BCu = ROL64(Eso, 56);
        Ama = BCa ^ ((~BCe) & BCi);
        Ame = BCe ^ ((~BCi) & BCo);
        Ami = BCi ^ ((~BCo) & BCu);
        Amo = BCo ^ ((~BCu) & BCa);
        Amu = BCu ^ ((~BCa) & BCe);

        Ebi ^= Di; BCa = ROL64(Ebi, 62);
        Ego ^= Do; BCe = ROL64(Ego, 55);
        Eku ^= Du; BCi = ROL64(Eku, 39);
        Ema ^= Da; BCo = ROL64(Ema, 41);
        Ese ^= De; BCu = ROL64(Ese, 2);
        Asa = BCa ^ ((~BCe) & BCi);
        Ase = BCe ^ ((~BCi) & BCo);
        Asi = BCi ^ ((~BCo) & BCu);
        Aso = BCo ^ ((~BCu) & BCa);
        Asu = BCu ^ ((~BCa) & BCe);
    }

    state[0] = Aba; state[1] = Abe; state[2] = Abi; state[3] = Abo; state[4] = Abu;
    state[5] = Aga; state[6] = Age; state[7] = Agi; state[8] = Ago; state[9] = Agu;
    state[10] = Aka; state[11] = Ake; state[12] = Aki; state[13] = Ako; state[14] = Aku;
    state[15] = Ama; state[16] = Ame; state[17] = Ami; state[18] = Amo; state[19] = Amu;
    state[20] = Asa; state[21] = Ase; state[22] = Asi; state[23] = Aso; state[24] = Asu;
}

static inline void state_to_bytes(const uint64_t state[4], uint8_t out[32]) {
    memcpy(out, state, 32);
}

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
    uint64_t state[25] __attribute__((aligned(32)));
    uint8_t pre_hash[32] __attribute__((aligned(32)));
    uint16_t nibbles[64] __attribute__((aligned(32)));
    uint8_t product[32] __attribute__((aligned(32)));
    uint8_t xored[32] __attribute__((aligned(32)));
    uint8_t final_hash[32] __attribute__((aligned(32)));

    for (n = 0; n < nonce_count; n++) {
        uint64_t nonce = nonce_start + (uint64_t)n;

        /* ── Stage 1: PrePowHash ── */
        memcpy(state, pow_state, 200);
        state[9] ^= nonce;
        keccak_f1600(state);
        state_to_bytes(state, pre_hash);

        /* ── Stage 2: Hash → 64 nibbles (uint16 for AVX2) ── */
        for (int i = 0; i < 32; i++) {
            nibbles[2*i]     = pre_hash[i] >> 4;
            nibbles[2*i + 1] = pre_hash[i] & 0x0F;
        }

        /* ── Stage 3: Matrix × vector (AVX2) ── */
        __m256i v_nibbles[4];
        for (int j = 0; j < 4; j++) {
            v_nibbles[j] = _mm256_load_si256((__m256i*)&nibbles[j*16]);
        }

        for (int i = 0; i < 32; i++) {
            const uint16_t *r1 = &matrix[(2*i)*64];
            const uint16_t *r2 = &matrix[(2*i+1)*64];
            
            __m256i v_sum1 = _mm256_setzero_si256();
            __m256i v_sum2 = _mm256_setzero_si256();
            
            for (int j = 0; j < 4; j++) {
                __m256i v_r1 = _mm256_loadu_si256((__m256i*)&r1[j*16]);
                __m256i v_r2 = _mm256_loadu_si256((__m256i*)&r2[j*16]);
                
                // Multiply and add adjacent pairs of int16
                __m256i v_m1 = _mm256_madd_epi16(v_r1, v_nibbles[j]);
                __m256i v_m2 = _mm256_madd_epi16(v_r2, v_nibbles[j]);
                
                v_sum1 = _mm256_add_epi32(v_sum1, v_m1);
                v_sum2 = _mm256_add_epi32(v_sum2, v_m2);
            }
            
            // Horizontal sum
            __m128i sum_lo1 = _mm_add_epi32(_mm256_castsi256_si128(v_sum1), _mm256_extracti128_si256(v_sum1, 1));
            sum_lo1 = _mm_add_epi32(sum_lo1, _mm_shuffle_epi32(sum_lo1, 0x0E));
            sum_lo1 = _mm_add_epi32(sum_lo1, _mm_shuffle_epi32(sum_lo1, 0x01));
            uint32_t total1 = _mm_cvtsi128_si32(sum_lo1);
            
            __m128i sum_lo2 = _mm_add_epi32(_mm256_castsi256_si128(v_sum2), _mm256_extracti128_si256(v_sum2, 1));
            sum_lo2 = _mm_add_epi32(sum_lo2, _mm_shuffle_epi32(sum_lo2, 0x0E));
            sum_lo2 = _mm_add_epi32(sum_lo2, _mm_shuffle_epi32(sum_lo2, 0x01));
            uint32_t total2 = _mm_cvtsi128_si32(sum_lo2);

            product[i] = (uint8_t)((((total1 >> 10) & 0x0F) << 4) | ((total2 >> 10) & 0x0F));
        }

        /* ── Stage 4: XOR with original hash ── */
        uint64_t *p64 = (uint64_t*)product;
        uint64_t *h64 = (uint64_t*)pre_hash;
        uint64_t *x64 = (uint64_t*)xored;
        x64[0] = p64[0] ^ h64[0];
        x64[1] = p64[1] ^ h64[1];
        x64[2] = p64[2] ^ h64[2];
        x64[3] = p64[3] ^ h64[3];

        /* ── Stage 5: HeavyHash ── */
        memcpy(state, heavy_state, 200);
        state[0] ^= x64[0];
        state[1] ^= x64[1];
        state[2] ^= x64[2];
        state[3] ^= x64[3];
        keccak_f1600(state);
        state_to_bytes(state, final_hash);

        /* ── Stage 6: Compare final_hash <= target (LE, MSB first) ── */
        uint64_t top_hash = ((uint64_t*)final_hash)[3];
        uint64_t top_target = ((uint64_t*)target)[3];
        uint64_t rev_hash = __builtin_bswap64(top_hash);
        uint64_t rev_target = __builtin_bswap64(top_target);

        if (rev_hash > rev_target) {
            continue;
        } else if (rev_hash < rev_target) {
            *out_nonce = nonce;
            memcpy(out_hash, final_hash, 32);
            return 1;
        }

        int less_or_equal = 1;
        for (int i = 23; i >= 0; i--) {
            if (final_hash[i] < target[i]) {
                break;
            } else if (final_hash[i] > target[i]) {
                less_or_equal = 0;
                break;
            }
        }
        if (less_or_equal) {
            *out_nonce = nonce;
            memcpy(out_hash, final_hash, 32);
            return 1;
        }
    }
    return 0;
}

void mine_range_bench(
    const uint64_t pow_state[25],
    const uint64_t heavy_state[25],
    const uint16_t matrix[64*64],
    uint64_t nonce_start,
    int nonce_count,
    uint64_t *hash_count
) {
    uint64_t dummy;
    uint8_t dummy_hash[32];
    uint8_t zero_target[32] = {0}; 
    mine_range(pow_state, heavy_state, matrix, zero_target, nonce_start, nonce_count, &dummy, dummy_hash);
    *hash_count = nonce_count;
}
