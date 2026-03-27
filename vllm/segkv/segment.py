import dataclasses
from dataclasses import dataclass, field
import time
from enum import Enum
from typing import List, Dict, Tuple, Optional, FrozenSet

class RecomputeStrategy(Enum):
    EXACT_REUSE = "exact_reuse"
    FULL_RECOMPUTE = "full_recompute"
    CACHEBLEND = "cacheblend"
    FULL_SUFFIX_RECOMPUTE = "full_suffix_recompute"
    SKIP_BLEND = "skip_blend"

@dataclass
class Segment:
    """Represents a contiguous chunk of a document with associated KV cache."""
    segment_id: int
    doc_id: str
    start_pos: int
    end_pos: int
    token_ids: Tuple[int, ...]
    content_hash: str
    version: int
    prefix_hash: str
    kv_block_ids: Dict[int, List[int]]
    is_kv_computed: bool = False
    dependency_version: int = 0
    creation_timestamp: float = field(default_factory=time.time)
    last_access_timestamp: float = field(default_factory=time.time)

    @property
    def length(self) -> int:
        return self.end_pos - self.start_pos

    @property
    def num_blocks(self) -> int:
        """Number of vLLM blocks needed. Assumes block_size=16."""
        import math
        return math.ceil(self.length / 16)

    @property
    def is_stale(self) -> bool:
        """True if prefix changed since KV was computed."""
        return False  # Must be evaluated externally

    def content_equals(self, other: "Segment") -> bool:
        """Check if two segments have identical token content."""
        return self.content_hash == other.content_hash

    def touch(self) -> None:
        """Update last access timestamp."""
        self.last_access_timestamp = time.time()

    def clear_kv(self) -> None:
        """Remove KV block references (blocks freed externally)."""
        self.kv_block_ids = {}
        self.is_kv_computed = False

@dataclass
class SegmentDiff:
    """Describes the difference between old and new versions of a segment."""
    segment_id: int
    change_type: str
    token_edit_distance: int = 0
    edit_ratio: float = 0.0
    changed_token_positions: List[int] = field(default_factory=list)
    old_tokens: Optional[Tuple[int, ...]] = None
    new_tokens: Optional[Tuple[int, ...]] = None

    @property
    def is_unchanged(self) -> bool:
        return self.change_type == "unchanged"

    @property
    def is_modified(self) -> bool:
        return self.change_type == "modified"

    @property
    def is_length_preserving(self) -> bool:
        """True if old and new have same token count."""
        if self.old_tokens is None or self.new_tokens is None:
            return True  # assume yes if not tracked
        return len(self.old_tokens) == len(self.new_tokens)

@dataclass
class SegmentPlan:
    """Execution plan for a single segment."""
    segment: Segment
    strategy: RecomputeStrategy
    blend_layers: List[int] = field(default_factory=list)
    blend_layer_frac: float = 0.0
    staleness_score: float = 0.0
    estimated_compute_cost: float = 0.0

    @property
    def num_blend_layers(self) -> int:
        return len(self.blend_layers)

    def to_dict(self) -> dict:
        """Serialize to dict for logging/API response."""
        return {
            "segment_id": self.segment.segment_id,
            "strategy": self.strategy.value,
            "blend_layers": self.blend_layers,
            "blend_layer_frac": self.blend_layer_frac,
            "staleness_score": self.staleness_score,
            "estimated_compute_cost": self.estimated_compute_cost,
        }

@dataclass
class DocumentVersion:
    """Represents one version of a document's segment decomposition."""
    doc_id: str
    version: int
    segments: List[Segment]
    total_tokens: int
    num_segments: int
    creation_timestamp: float = field(default_factory=time.time)

    @property
    def segment_ids(self) -> List[int]:
        return [s.segment_id for s in self.segments]

    def get_segment(self, segment_id: int) -> Optional[Segment]:
        if 0 <= segment_id < len(self.segments):
            return self.segments[segment_id]
        return None
