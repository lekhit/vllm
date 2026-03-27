import pytest
from vllm.segkv.config import SegKVConfig, LayerSelectionStrategy
from vllm.segkv.segment_manager import SegmentManager
from vllm.segkv.recompute_policy import RecomputationPolicy
from vllm.segkv.blend_executor import BlendExecutor
from vllm.segkv.phase_executor import PhaseExecutor
from vllm.segkv.block_manager import SegKVBlockManager
from vllm.segkv.metrics import SegKVAggregateMetrics

class TestIntegrationCPU:
    def test_end_to_end_flow(self, default_config, sample_document_pair):
        """Test the end-to-end SegKV flow on CPU (without model layers)."""
        manager = SegmentManager(default_config)
        policy = RecomputationPolicy(default_config)
        blend_exec = BlendExecutor(default_config)
        phase_exec = PhaseExecutor(
            config=default_config,
            segment_manager=manager,
            policy=policy,
            blend_executor=blend_exec
        )
        
        old_tokens, new_tokens = sample_document_pair
        doc_id = "test_doc_integration"
        
        # Phase 1: Register initial document
        version1 = manager.register_document(doc_id, tuple(old_tokens))
        
        # Mock that KV was computed for v1
        for seg in version1.segments:
            seg.is_kv_computed = True
            
        # Phase 2: Plan edit using the phase executor
        num_layers = 16
        plans, old_ver, new_ver = phase_exec.plan_execution(
            doc_id, tuple(new_tokens), base_version=version1.version, num_layers=num_layers
        )
        
        assert old_ver.version == 1
        assert new_ver.version == 2
        assert len(plans) == new_ver.num_segments
        
        # Segment 0 should be exact reuse (unchanged)
        assert plans[0].strategy.name == "EXACT_REUSE"
        
        # Phase 3: Block Manager tracking
        block_manager = SegKVBlockManager(default_config)
        
        for seg in version1.segments:
            block_assignments = {l: [100 * seg.segment_id + l] for l in range(num_layers)}
            block_manager.register_segment_blocks(
                doc_id, seg.segment_id, version1.version, block_assignments
            )
            
        blocks = block_manager.get_segment_blocks(doc_id, 0, version1.version)
        assert blocks is not None
        assert len(blocks) == num_layers
        
        # Verify aggregate metrics work
        agg = SegKVAggregateMetrics()
        assert agg.num_executions == 0
        assert agg.reuse_rate() == 0.0
