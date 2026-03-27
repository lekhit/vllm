import pytest
from vllm.segkv.utils import hash_tokens, hash_combine, token_edit_distance
from vllm.segkv.edit_detector import EditDetector
from vllm.segkv.segment_manager import SegmentManager
from vllm.segkv.recompute_policy import RecomputationPolicy
from vllm.segkv.segment import RecomputeStrategy

class TestUtils:
    def test_hash_tokens_deterministic(self):
        assert hash_tokens((1, 2, 3)) == hash_tokens((1, 2, 3))
        assert hash_tokens((1, 2, 3)) != hash_tokens((1, 2, 4))
        
    def test_token_edit_distance(self):
        assert token_edit_distance((1, 2, 3), (1, 2, 3)) == 0
        assert token_edit_distance((1, 2, 3), (1, 9, 3)) == 1

class TestEditDetector:
    def test_no_edits(self, default_config, sample_token_ids):
        detector = EditDetector(default_config)
        manager = SegmentManager(default_config)
        doc_id = "test_no_edits"
        version = manager.register_document(doc_id, tuple(sample_token_ids))
        
        diffs, new_segs = detector.detect_edits(version, sample_token_ids)
        
        modified_ids = [d.segment_id for d in diffs if d.change_type != "unchanged"]
        assert len(modified_ids) == 0

    def test_single_edit(self, default_config, sample_document_pair):
        old, new = sample_document_pair
        detector = EditDetector(default_config)
        manager = SegmentManager(default_config)
        doc_id = "test_single_edit"
        version = manager.register_document(doc_id, tuple(old))
        
        diffs, new_segs = detector.detect_edits(version, new)
        
        modified_ids = [d.segment_id for d in diffs if d.change_type != "unchanged"]
        assert 1 in modified_ids  # segment 1 should be modified
        # check that not all are identical
        has_modified = any(d.change_type == "modified" for d in diffs)
        assert has_modified
        
class TestSegmentManager:
    def test_register_and_get(self, default_config, sample_token_ids):
        manager = SegmentManager(default_config)
        doc_id = "test_doc"
        version = manager.register_document(doc_id, tuple(sample_token_ids))
        assert version.version == 1
        assert version.num_segments == len(sample_token_ids) // default_config.segment_size
        
        fetched = manager.get_latest_version(doc_id)
        assert fetched is not None
        assert fetched.version == 1

    def test_eviction(self, default_config, sample_token_ids):
        # Default config has max_documents=10
        manager = SegmentManager(default_config)
        for i in range(10):
            manager.register_document(f"doc_{i}", tuple(sample_token_ids))
        
        # 11th should raise because max_documents=10
        with pytest.raises(ValueError):
            manager.register_document("doc_10", tuple(sample_token_ids))

class TestRecomputationPolicy:
    def test_policy_no_edits(self, default_config, sample_token_ids):
        manager = SegmentManager(default_config)
        doc_id = "test"
        sys_ver = manager.register_document(doc_id, tuple(sample_token_ids))
        
        # mock that KV is computed
        for seg in sys_ver.segments:
            seg.is_kv_computed = True
            
        detector = EditDetector(default_config)
        diffs, new_segs = detector.detect_edits(sys_ver, sample_token_ids)
        
        policy = RecomputationPolicy(default_config)
        plans = policy.create_execution_plan(sys_ver, sys_ver.segments, diffs, num_model_layers=32)
        
        assert all(p.strategy == RecomputeStrategy.EXACT_REUSE for p in plans)
