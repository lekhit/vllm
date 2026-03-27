import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Literal

class LayerSelectionStrategy(Enum):
    UNIFORM = "uniform"
    EARLY_BIASED = "early_biased"
    LATE_BIASED = "late_biased"
    BOOKEND = "bookend"
    IMPORTANCE_STATIC = "importance_static"

class SegmentationMode(Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"

class QualityMode(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    EXACT = "exact"

@dataclass
class SegKVConfig:
    enable_segkv: bool = False
    segment_size: int = 512
    segmentation_mode: SegmentationMode = SegmentationMode.FIXED
    blend_quality_threshold: float = 0.95
    blend_max_layer_frac: float = 0.30
    blend_min_layer_frac: float = 0.0
    layer_selection_strategy: LayerSelectionStrategy = LayerSelectionStrategy.UNIFORM
    staleness_decay: float = 0.85
    max_staleness_for_blend: float = 0.8
    min_staleness_for_blend: float = 0.02
    force_recompute_layers: List[int] = field(default_factory=list)
    max_cached_versions: int = 5
    max_documents: int = 100
    enable_metrics: bool = True

    def __post_init__(self):
        """Run all validation checks."""
        if self.segment_size % 16 != 0:
            raise ValueError(f"segment_size must be a multiple of 16, got {self.segment_size}")
        if not (64 <= self.segment_size <= 8192):
            raise ValueError(f"segment_size must be between 64 and 8192, got {self.segment_size}")
        if not (0.0 <= self.blend_quality_threshold <= 1.0):
            raise ValueError(f"blend_quality_threshold must be between 0.0 and 1.0, got {self.blend_quality_threshold}")
        if not (0.0 <= self.blend_max_layer_frac <= 1.0):
            raise ValueError(f"blend_max_layer_frac must be between 0.0 and 1.0, got {self.blend_max_layer_frac}")
        if not (0.0 <= self.blend_min_layer_frac <= self.blend_max_layer_frac):
            raise ValueError(f"blend_min_layer_frac must be between 0.0 and blend_max_layer_frac ({self.blend_max_layer_frac}), got {self.blend_min_layer_frac}")
        if not (0.0 < self.staleness_decay < 1.0):
            raise ValueError(f"staleness_decay must be between (exclusive) 0.0 and 1.0, got {self.staleness_decay}")
        if not (0.0 < self.max_staleness_for_blend <= 1.0):
            raise ValueError(f"max_staleness_for_blend must be between (exclusive) 0.0 and 1.0 (inclusive), got {self.max_staleness_for_blend}")
        if not (0.0 <= self.min_staleness_for_blend < self.max_staleness_for_blend):
            raise ValueError(f"min_staleness_for_blend must be between 0.0 and max_staleness_for_blend ({self.max_staleness_for_blend}), got {self.min_staleness_for_blend}")
        if self.max_cached_versions < 1:
            raise ValueError(f"max_cached_versions must be >= 1, got {self.max_cached_versions}")
        if self.max_documents < 1:
            raise ValueError(f"max_documents must be >= 1, got {self.max_documents}")

    @classmethod
    def get_quality_preset(cls, quality_mode: QualityMode) -> "SegKVConfig":
        """Factory method returning preset configurations."""
        if quality_mode == QualityMode.FAST:
            return cls(blend_max_layer_frac=0.10, blend_quality_threshold=0.90)
        elif quality_mode == QualityMode.BALANCED:
            return cls(blend_max_layer_frac=0.20, blend_quality_threshold=0.95)
        elif quality_mode == QualityMode.EXACT:
            return cls(blend_max_layer_frac=1.0, blend_quality_threshold=1.0)
        raise ValueError(f"Unknown QualityMode: {quality_mode}")

    def estimate_compute_savings(
        self,
        total_segments: int,
        changed_segments: int,
        edit_position_frac: float,
        num_layers: int,
    ) -> float:
        """
        Estimate fraction of compute saved vs full recompute.
        Returns float in [0.0, 1.0].
        
        FORMULA:
          prefix_segments = segment of first change
          suffix_segments = total - prefix - changed
          
          full_cost = total_segments * num_layers  (normalized)
          segkv_cost = (changed_segments * num_layers  # Phase 2
                       + suffix_segments * avg_blend_frac * num_layers)  # Phase 3
          
          savings = 1.0 - (segkv_cost / full_cost)
        """
        prefix_segments = int(total_segments * edit_position_frac)
        suffix_segments = max(0, total_segments - prefix_segments - changed_segments)
        
        full_cost = total_segments * num_layers
        if full_cost == 0:
            return 0.0
            
        avg_blend_frac = (self.blend_min_layer_frac + self.blend_max_layer_frac) / 2.0
        segkv_cost = (changed_segments * num_layers) + (suffix_segments * avg_blend_frac * num_layers)
        
        savings = 1.0 - (segkv_cost / full_cost)
        return max(0.0, min(1.0, savings))
