import dataclasses
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional

@dataclass
class SegKVMetrics:
    phase1_time: float = 0.0
    phase2_time: float = 0.0
    phase3_time: float = 0.0
    total_time: float = 0.0
    planning_time: float = 0.0
    
    total_segments: int = 0
    segments_reused: int = 0
    segments_recomputed: int = 0
    segments_blended: int = 0
    segments_skipped: int = 0
    segments_suffix_recomputed: int = 0
    
    blend_layers_per_segment: Dict[int, int] = field(default_factory=dict)
    total_blend_layers: int = 0
    compute_savings: float = 0.0
    
    # Quality (filled in during validation, if applicable)
    kl_divergence: Optional[float] = None
    cosine_similarity: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialize all fields to dict."""
        return dataclasses.asdict(self)

    def summary_str(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"SegKV: {self.segments_reused}R/{self.segments_recomputed}C/"
            f"{self.segments_blended}B/{self.segments_skipped}S "
            f"| savings={self.compute_savings:.1%} "
            f"| time={self.total_time:.3f}s "
            f"(P1={self.phase1_time:.3f} P2={self.phase2_time:.3f} "
            f"P3={self.phase3_time:.3f})"
        )


@dataclass
class SegKVAggregateMetrics:
    """Aggregate metrics across multiple SegKV executions."""
    num_executions: int = 0
    total_segments_processed: int = 0
    total_segments_reused: int = 0
    total_segments_recomputed: int = 0
    total_segments_blended: int = 0
    avg_compute_savings: float = 0.0
    avg_total_time: float = 0.0
    max_total_time: float = 0.0
    min_total_time: float = float('inf')
    
    # Histograms (for distribution analysis)
    savings_histogram: List[float] = field(default_factory=list)
    time_histogram: List[float] = field(default_factory=list)

    def record(self, metrics: SegKVMetrics) -> None:
        """Add one execution's metrics to the aggregate."""
        self.num_executions += 1
        self.total_segments_processed += metrics.total_segments
        self.total_segments_reused += metrics.segments_reused
        self.total_segments_recomputed += metrics.segments_recomputed
        self.total_segments_blended += metrics.segments_blended
        
        # Running average for savings
        self.avg_compute_savings = (
            (self.avg_compute_savings * (self.num_executions - 1)
             + metrics.compute_savings) / self.num_executions
        )
        
        # Running average for time
        self.avg_total_time = (
            (self.avg_total_time * (self.num_executions - 1)
             + metrics.total_time) / self.num_executions
        )
        
        self.max_total_time = max(self.max_total_time, metrics.total_time)
        self.min_total_time = min(self.min_total_time, metrics.total_time)
        
        self.savings_histogram.append(metrics.compute_savings)
        self.time_histogram.append(metrics.total_time)

    def reuse_rate(self) -> float:
        """Fraction of all segments that were reused (not recomputed)."""
        if self.total_segments_processed == 0:
            return 0.0
        return self.total_segments_reused / self.total_segments_processed

    def to_dict(self) -> dict:
        """Serialize all fields."""
        return dataclasses.asdict(self)

    def summary_str(self) -> str:
        """Multi-line summary suitable for logging."""
        return (
            f"SegKV Aggregate ({self.num_executions} executions):\n"
            f"  Reuse rate: {self.reuse_rate():.1%}\n"
            f"  Avg savings: {self.avg_compute_savings:.1%}\n"
            f"  Avg time: {self.avg_total_time:.3f}s "
            f"(min={self.min_total_time:.3f}, max={self.max_total_time:.3f})\n"
            f"  Segments: {self.total_segments_reused} reused / "
            f"{self.total_segments_recomputed} recomputed / "
            f"{self.total_segments_blended} blended"
        )
