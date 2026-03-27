import pytest
from typing import List, Dict, Tuple
from vllm.segkv.config import SegKVConfig, LayerSelectionStrategy, SegmentationMode
from vllm.segkv.segment import Segment, SegmentDiff, DocumentVersion, SegmentPlan, RecomputeStrategy

@pytest.fixture
def default_config() -> SegKVConfig:
    """Default SegKV configuration for tests."""
    return SegKVConfig(
        enable_segkv=True,
        segment_size=64,  # small for fast tests
        blend_max_layer_frac=0.20,
        blend_quality_threshold=0.95,
        staleness_decay=0.85,
        layer_selection_strategy=LayerSelectionStrategy.UNIFORM,
        max_cached_versions=3,
        max_documents=10,
    )

@pytest.fixture
def small_config() -> SegKVConfig:
    """Tiny segments for edge case testing."""
    return SegKVConfig(
        enable_segkv=True,
        segment_size=16,  # minimum practical
        blend_max_layer_frac=0.30,
        staleness_decay=0.85,
    )

@pytest.fixture
def sample_token_ids() -> List[int]:
    """Sample token IDs representing a ~256 token document."""
    # Use sequential integers as token IDs for predictability
    return list(range(1, 257))  # 256 tokens

@pytest.fixture
def long_token_ids() -> List[int]:
    """Longer document (~2048 tokens)."""
    return list(range(1, 2049))

@pytest.fixture
def sample_document_pair() -> Tuple[List[int], List[int]]:
    """
    Two versions of a document differing in one segment.
    Returns (old_token_ids, new_token_ids)
    """
    old = list(range(1, 513))  # 512 tokens
    new = old.copy()
    # Modify tokens at positions 64-79 (in second segment if segment_size=64)
    for i in range(64, 80):
        new[i] = old[i] + 10000  # clearly different tokens
    return (old, new)

@pytest.fixture
def multi_edit_pair() -> Tuple[List[int], List[int]]:
    """
    Two versions with edits in multiple segments.
    """
    old = list(range(1, 513))
    new = old.copy()
    # Edit segment 1 (positions 64-127 with segment_size=64)
    for i in range(70, 80):
        new[i] = old[i] + 10000
    # Edit segment 5 (positions 320-383)
    for i in range(330, 340):
        new[i] = old[i] + 20000
    return (old, new)

@pytest.fixture
def make_segment():
    """Factory fixture for creating test segments."""
    def _make(
        segment_id: int = 0,
        doc_id: str = "test_doc",
        start_pos: int = 0,
        end_pos: int = 64,
        token_ids: Tuple[int, ...] = None,
        version: int = 0,
    ) -> Segment:
        if token_ids is None:
            token_ids = tuple(range(start_pos + 1, end_pos + 1))
        from vllm.segkv.utils import hash_tokens
        return Segment(
            segment_id=segment_id,
            doc_id=doc_id,
            start_pos=start_pos,
            end_pos=end_pos,
            token_ids=token_ids,
            content_hash=hash_tokens(token_ids),
            version=version,
            prefix_hash="",
            kv_block_ids={},
            is_kv_computed=False,
        )
    return _make

@pytest.fixture
def make_document_version(make_segment):
    """Factory fixture for creating test DocumentVersions."""
    def _make(
        doc_id: str = "test_doc",
        version: int = 0,
        num_segments: int = 4,
        segment_size: int = 64,
    ) -> DocumentVersion:
        segments = []
        for i in range(num_segments):
            seg = make_segment(
                segment_id=i,
                doc_id=doc_id,
                start_pos=i * segment_size,
                end_pos=(i + 1) * segment_size,
                version=version,
            )
            segments.append(seg)
        return DocumentVersion(
            doc_id=doc_id,
            version=version,
            segments=segments,
            total_tokens=num_segments * segment_size,
            num_segments=num_segments,
        )
    return _make
