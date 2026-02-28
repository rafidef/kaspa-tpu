"""
Mining coordinator — ties together the pipeline, network, and UI.

Manages the lifecycle:
  1. Connect to pool/node
  2. Receive block templates
  3. Run mining pipeline
  4. Submit solutions
  5. Report hashrate
"""

import asyncio
import logging
import time
from typing import Optional

from .pipeline import MiningPipeline, BlockTemplate, MiningResult, PipelineStats
from ..network.stratum_client import StratumClient
from ..network.grpc_client import KaspaGrpcClient

logger = logging.getLogger(__name__)


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


class MiningCoordinator:
    """
    Top-level mining coordinator.
    
    Orchestrates the connection to a Kaspa pool/node, the mining pipeline,
    and solution submission. Handles the complete mining lifecycle.
    """
    
    def __init__(
        self,
        mode: str = "stratum",
        pool_host: str = "",
        pool_port: int = 0,
        node_address: str = "localhost:16110",
        wallet_address: str = "",
        worker_name: str = "kaspa-tpu",
        batch_size: int = 8192,
        cpu_threads: int = 4,
        stats_interval: float = 10.0,
    ):
        """
        Args:
            mode: "stratum" for pool mining, "grpc" for solo mining
            pool_host: Stratum pool hostname
            pool_port: Stratum pool port
            node_address: Kaspa node gRPC address (for solo)
            wallet_address: Kaspa wallet address
            worker_name: Worker name for pool identification
            batch_size: Nonces per pipeline batch
            cpu_threads: CPU threads for Keccak
            stats_interval: Seconds between hashrate reports
        """
        self.mode = mode
        self.wallet_address = wallet_address
        self.worker_name = worker_name
        
        # Mining pipeline
        self.pipeline = MiningPipeline(
            batch_size=batch_size,
            cpu_threads=cpu_threads,
            on_solution=self._on_solution,
            on_stats=self._on_stats,
            stats_interval=stats_interval,
        )
        
        # Network client
        self._stratum: Optional[StratumClient] = None
        self._grpc: Optional[KaspaGrpcClient] = None
        
        if mode == "stratum":
            self._stratum = StratumClient(
                host=pool_host,
                port=pool_port,
                wallet_address=wallet_address,
                worker_name=worker_name,
                on_new_job=self._on_new_template,
            )
        else:
            self._grpc = KaspaGrpcClient(
                node_address=node_address,
                wallet_address=wallet_address,
                on_new_template=self._on_new_template,
            )
        
        self._accepted = 0
        self._rejected = 0
    
    async def start(self):
        """Start the mining coordinator."""
        logger.info("=" * 60)
        logger.info("  Kaspa TPU Miner v0.1.0")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  Address: {self.wallet_address}")
        logger.info(f"  Batch size: {self.pipeline.batch_size}")
        logger.info(f"  CPU threads: {self.pipeline.cpu_threads}")
        logger.info("=" * 60)
        
        # Connect to network
        if self._stratum:
            await self._stratum.connect()
            # Run listener and pipeline concurrently
            await asyncio.gather(
                self._stratum.listen(),
                self.pipeline.run(),
            )
        elif self._grpc:
            await self._grpc.connect()
            await asyncio.gather(
                self._grpc.poll_templates(),
                self.pipeline.run(),
            )
    
    async def stop(self):
        """Stop the mining coordinator."""
        self.pipeline.stop()
        if self._stratum:
            await self._stratum.disconnect()
        if self._grpc:
            await self._grpc.disconnect()
        
        logger.info("Mining coordinator stopped")
        logger.info(f"Accepted: {self._accepted}, Rejected: {self._rejected}")
    
    def _on_new_template(self, template: BlockTemplate):
        """Handle a new block template from the network."""
        self.pipeline.update_template(template)
    
    def _on_solution(self, result: MiningResult):
        """Handle a solution found by the pipeline."""
        asyncio.get_event_loop().create_task(
            self._submit_solution(result)
        )
    
    async def _submit_solution(self, result: MiningResult):
        """Submit a solution to the pool/node."""
        if self._stratum:
            accepted = await self._stratum.submit_solution(
                result.template.template_id,
                result.nonce,
            )
        elif self._grpc:
            accepted = await self._grpc.submit_block(
                result.template.block_data
            )
        else:
            accepted = False
        
        if accepted:
            self._accepted += 1
        else:
            self._rejected += 1
    
    def _on_stats(self, stats: PipelineStats):
        """Handle pipeline statistics."""
        instant_hr = stats.instant_hashrate()
        avg_hr = stats.hashrate
        
        logger.info(
            f"Hashrate: {_format_hashrate(instant_hr)} "
            f"(avg: {_format_hashrate(avg_hr)}) | "
            f"Hashes: {stats.total_hashes:,} | "
            f"Solutions: {stats.solutions_found} | "
            f"Accepted: {self._accepted} | "
            f"Rejected: {self._rejected}"
        )
