# Kaspa TPU Miner

Hybrid CPU/TPU miner for Kaspa's **kHeavyHash** proof-of-work algorithm.

The CPU handles Keccak (cSHAKE256) hashing while the TPU accelerates the batched 64×64 matrix-vector multiplication — the compute-intensive core of kHeavyHash.

## Architecture

```
CPU Thread Pool                  TPU (JAX JIT)
┌────────────────┐    batch     ┌──────────────────┐
│ Stage 1:       │─────────────▶│ Stage 2:         │
│ PrePowHash     │              │ 64×64 MatMul     │
│ (Keccak-f1600) │              │ (batched × N)    │
│                │◀─────────────│ + shift/truncate │
│ Stage 3:       │   products   └──────────────────┘
│ XOR + Final    │
│ HeavyHash      │
│ + Diff Check   │
└────────────────┘
```

## Install

```bash
# CPU-only (development/testing)
pip install git+https://github.com/rafidef/kaspa-tpu.git

# With TPU support (on Google Cloud TPU VMs)
pip install "kaspa-tpu[tpu] @ git+https://github.com/rafidef/kaspa-tpu.git"

# With all extras (TPU + gRPC + dev tools)
pip install "kaspa-tpu[all] @ git+https://github.com/rafidef/kaspa-tpu.git"
```

## Usage

```bash
# Benchmark mode (no network, measures pipeline throughput)
kaspa-tpu --benchmark --batch-size 8192

# Pool mining via Stratum
kaspa-tpu --pool stratum+tcp://pool.example.com:5555 \
          --address kaspa:qr...your_address

# Solo mining via gRPC (requires local Kaspa node)
kaspa-tpu --node localhost:16110 \
          --address kaspa:qr...your_address
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--pool URL` | Stratum pool URL | — |
| `--node ADDR` | Kaspa node gRPC address | — |
| `--benchmark` | Run benchmark mode | — |
| `--address`, `-a` | Kaspa wallet address | — |
| `--batch-size`, `-b` | Nonces per batch | 8192 |
| `--cpu-threads`, `-t` | CPU threads for Keccak | 4 |
| `--worker`, `-w` | Worker name | kaspa-tpu |
| `--verbose`, `-v` | Debug output | off |

## How It Works

Kaspa's kHeavyHash PoW algorithm per nonce:

1. **PrePowHash** — `cSHAKE256("ProofOfWorkHash", header || timestamp || padding || nonce)` → 32-byte hash
2. **Matrix generation** — xoshiro256++ seeded from pre_pow_hash → 64×64 nibble matrix (once per block)
3. **Heavy hash**:
   - Split hash → 64 nibbles
   - Matrix × nibble vector → integer multiply-accumulate → right-shift-by-10 → truncate to 4 bits
   - XOR product with original hash
   - `cSHAKE256("HeavyHash", result)` → final 32-byte hash
4. **Difficulty check** — `Uint256(final_hash) <= target`

The **MatMul is standard integer arithmetic** (not GF(16) finite-field), so TPU's MXU systolic array can be used natively.

## Testing

```bash
pip install "kaspa-tpu[dev] @ git+https://github.com/rafidef/kaspa-tpu.git"
python -m pytest tests/ -v
```

**27/27 tests passing** — covers Keccak, PowHash, HeavyHash, xoshiro256++, matrix generation, TPU MatMul kernel, and full end-to-end pipeline.

## License

MIT
