"""
Kaspa TPU Miner — CLI Entry Point

Usage:
    # Pool mining (stratum)
    python -m kaspa_tpu --pool stratum+tcp://pool.example.com:5555 \
                        --address kaspa:qr...your_address \
                        --batch-size 8192

    # Solo mining (gRPC to local node)
    python -m kaspa_tpu --node localhost:16110 \
                        --address kaspa:qr...your_address

    # Benchmark mode (no network, test pipeline speed)
    python -m kaspa_tpu --benchmark --batch-size 16384
"""

import argparse
import asyncio
import logging
import os
import sys
import signal
import time
import numpy as np

from .mining.coordinator import MiningCoordinator
from .mining.pipeline import MiningPipeline, BlockTemplate, PipelineStats
from .core.matrix import Matrix
from .core.tpu_matmul import TPUMatMul, get_tpu_device, hashes_to_nibbles


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kaspa TPU Miner — kHeavyHash mining with TPU acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Pool mining:  python -m kaspa_tpu --pool stratum+tcp://pool:5555 --address kaspa:qr...
  Solo mining:  python -m kaspa_tpu --node localhost:16110 --address kaspa:qr...
  Benchmark:    python -m kaspa_tpu --benchmark --batch-size 16384
        """,
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--pool", type=str, metavar="URL",
        help="Stratum pool URL (e.g., stratum+tcp://pool.example.com:5555)",
    )
    mode.add_argument(
        "--node", type=str, metavar="ADDR",
        help="Kaspa node gRPC address for solo mining (e.g., localhost:16110)",
    )
    mode.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark mode (no network connection)",
    )
    
    # Mining config
    parser.add_argument(
        "--address", "-a", type=str, default="",
        help="Kaspa wallet address",
    )
    parser.add_argument(
        "--worker", "-w", type=str, default="kaspa-tpu",
        help="Worker name (default: kaspa-tpu)",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=65536,
        help="Nonces per batch (default: 65536)",
    )
    parser.add_argument(
        "--cpu-threads", "-t", type=int, default=0,
        help="CPU threads for Keccak (default: auto-detect)",
    )
    parser.add_argument(
        "--stats-interval", type=float, default=10.0,
        help="Seconds between hashrate reports (default: 10)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose debug output",
    )
    
    return parser.parse_args()


def _format_hashrate(h: float) -> str:
    """Format hashrate with appropriate SI prefix."""
    if h >= 1e12:
        return f"{h / 1e12:.2f} TH/s"
    elif h >= 1e9:
        return f"{h / 1e9:.2f} GH/s"
    elif h >= 1e6:
        return f"{h / 1e6:.2f} MH/s"
    elif h >= 1e3:
        return f"{h / 1e3:.2f} KH/s"
    else:
        return f"{h:.2f} H/s"


async def run_benchmark(batch_size: int, cpu_threads: int, duration: float = 30.0):
    """
    Run a benchmark to measure pipeline throughput.
    
    Uses a synthetic block template to test the full pipeline
    without network connectivity.
    """
    logger = logging.getLogger("benchmark")
    
    logger.info("=" * 60)
    logger.info("  Kaspa TPU Miner — Benchmark Mode")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  CPU threads: {cpu_threads}")
    logger.info(f"  Duration: {duration}s")
    logger.info("=" * 60)
    
    # Detect device
    tpu_device = get_tpu_device()
    device_name = "TPU" if tpu_device else "CPU (NumPy fallback)"
    logger.info(f"Compute device: {device_name}")
    
    # Create synthetic block template
    pre_pow_hash = bytes(range(32))  # Deterministic seed
    timestamp = 1700000000
    target_bits = 0x207FFFFF  # Very easy target for benchmarking
    
    logger.info("Generating matrix...")
    matrix = Matrix.generate(pre_pow_hash)
    logger.info(f"Matrix generated (rank verified)")
    
    # Initialize TPU engine
    engine = TPUMatMul(matrix.data, tpu_device)
    
    # Warmup
    logger.info("Warming up...")
    from .crypto.keccak import PowHash, HeavyHash
    hasher = PowHash(pre_pow_hash, timestamp)
    
    warmup_nonces = np.arange(0, min(batch_size, 1024), dtype=np.uint64)
    hashes = hasher.finalize_batch(warmup_nonces)
    nibbles = hashes_to_nibbles(hashes)
    products = engine.batched_matmul(nibbles)
    logger.info("Warmup complete")
    
    # Benchmark individual stages
    logger.info("-" * 40)
    
    # Stage 1: Keccak PrePowHash
    test_nonces = np.arange(0, batch_size, dtype=np.uint64)
    t0 = time.perf_counter()
    hashes = hasher.finalize_batch(test_nonces)
    t1 = time.perf_counter()
    keccak_pre_time = t1 - t0
    logger.info(f"Stage 1 (Keccak PrePow)  : {keccak_pre_time*1000:.1f}ms for {batch_size} hashes "
                f"({batch_size/keccak_pre_time:.0f} H/s)")
    
    # Stage 2: TPU MatMul
    nibbles = hashes_to_nibbles(hashes)
    t0 = time.perf_counter()
    products = engine.batched_matmul(nibbles)
    t1 = time.perf_counter()
    matmul_time = t1 - t0
    logger.info(f"Stage 2 (MatMul {device_name:>8}): {matmul_time*1000:.1f}ms for {batch_size} matmuls "
                f"({batch_size/matmul_time:.0f} ops/s)")
    
    # Stage 3: XOR + Keccak HeavyHash
    xored = products ^ hashes
    t0 = time.perf_counter()
    final = HeavyHash.hash_batch(xored)
    t1 = time.perf_counter()
    keccak_post_time = t1 - t0
    logger.info(f"Stage 3 (Keccak Heavy)   : {keccak_post_time*1000:.1f}ms for {batch_size} hashes "
                f"({batch_size/keccak_post_time:.0f} H/s)")
    
    total_per_batch = keccak_pre_time + matmul_time + keccak_post_time
    pipeline_rate = batch_size / total_per_batch
    
    logger.info("-" * 40)
    logger.info(f"Pipeline rate: {_format_hashrate(pipeline_rate)}")
    logger.info(f"Bottleneck: {'Keccak PrePow' if keccak_pre_time > matmul_time else 'MatMul'}")
    
    # Sustained benchmark
    logger.info("=" * 40)
    logger.info(f"Running sustained benchmark for {duration}s...")
    
    total_hashes = 0
    nonce_offset = 0
    start = time.perf_counter()
    
    while time.perf_counter() - start < duration:
        nonces = np.arange(nonce_offset, nonce_offset + batch_size, dtype=np.uint64)
        nonce_offset += batch_size
        
        hashes = hasher.finalize_batch(nonces)
        nibbles = hashes_to_nibbles(hashes)
        products = engine.batched_matmul(nibbles)
        xored = products ^ hashes
        final = HeavyHash.hash_batch(xored)
        
        total_hashes += batch_size
    
    elapsed = time.perf_counter() - start
    sustained_rate = total_hashes / elapsed
    
    logger.info(f"Sustained hashrate: {_format_hashrate(sustained_rate)}")
    logger.info(f"Total hashes: {total_hashes:,} in {elapsed:.1f}s")
    logger.info("Benchmark complete")


async def run_mining(args: argparse.Namespace):
    """Run the mining coordinator with the given arguments."""
    if args.pool:
        # Parse pool URL: stratum+tcp://host:port or stratum+ssl://host:port
        url = args.pool
        use_ssl = False
        
        if url.startswith("stratum+ssl://"):
            use_ssl = True
            url = url[len("stratum+ssl://"):]
        elif url.startswith("stratum+tcp://"):
            url = url[len("stratum+tcp://"):]
        elif url.startswith("ssl://"):
            use_ssl = True
            url = url[len("ssl://"):]
        elif "://" in url:
            url = url.split("://", 1)[1]
        
        parts = url.split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 5555
        
        coordinator = MiningCoordinator(
            mode="stratum",
            pool_host=host,
            pool_port=port,
            wallet_address=args.address,
            worker_name=args.worker,
            batch_size=args.batch_size,
            cpu_threads=args.cpu_threads,
            stats_interval=args.stats_interval,
            use_ssl=use_ssl,
        )
    else:
        coordinator = MiningCoordinator(
            mode="grpc",
            node_address=args.node,
            wallet_address=args.address,
            worker_name=args.worker,
            batch_size=args.batch_size,
            cpu_threads=args.cpu_threads,
            stats_interval=args.stats_interval,
        )
    
    # Handle Ctrl+C
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logging.info("Shutting down...")
        asyncio.ensure_future(coordinator.stop())
    
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass
    
    await coordinator.start()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    
    # Auto-detect CPU threads if not specified
    if args.cpu_threads <= 0:
        args.cpu_threads = max(1, os.cpu_count() or 4)
    
    if args.benchmark:
        asyncio.run(run_benchmark(args.batch_size, args.cpu_threads))
    else:
        if not args.address:
            print("Error: --address is required for mining")
            sys.exit(1)
        asyncio.run(run_mining(args))


if __name__ == "__main__":
    main()
