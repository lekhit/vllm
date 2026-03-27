from typing import List, Dict, Optional, Tuple, Any
import logging
import time
from vllm.segkv.config import SegKVConfig
from vllm.segkv.segment import SegmentPlan, RecomputeStrategy, DocumentVersion
from vllm.segkv.segment_manager import SegmentManager
from vllm.segkv.edit_detector import EditDetector
from vllm.segkv.recompute_policy import RecomputationPolicy
from vllm.segkv.blend_executor import BlendExecutor, ModelLayerInterface, KVCacheInterface
from vllm.segkv.metrics import SegKVMetrics

# Extend ModelLayerInterface for phase executor
class PhaseModelInterface(ModelLayerInterface):
    def prefill_segment(self, token_ids: List[int], positions: List[int], kv_cache: KVCacheInterface) -> Tuple[Any, Any, Any]: ...

class PhaseExecutor:
    def __init__(
        self,
        config: SegKVConfig,
        segment_manager: SegmentManager,
        policy: RecomputationPolicy,
        blend_executor: BlendExecutor,
    ):
        self.config = config
        self.segment_manager = segment_manager
        self.policy = policy
        self.blend_executor = blend_executor
        self.logger = logging.getLogger("segkv.phase_executor")

    def plan_execution(
        self,
        doc_id: str,
        new_token_ids: List[int],
        base_version: Optional[int],
        num_model_layers: int,
    ) -> Tuple[List[SegmentPlan], Optional[DocumentVersion], DocumentVersion]:
        t0 = time.perf_counter()
        
        try:
            old_version = self.segment_manager.get_version(doc_id, base_version) if base_version else self.segment_manager.get_latest_version(doc_id)
            if not old_version:
                new_version = self.segment_manager.register_document(doc_id, new_token_ids)
                plans = [SegmentPlan(seg, RecomputeStrategy.FULL_RECOMPUTE, estimated_compute_cost=1.0) for seg in new_version.segments]
                return plans, None, new_version
        except Exception:
            new_version = self.segment_manager.register_document(doc_id, new_token_ids)
            plans = [SegmentPlan(seg, RecomputeStrategy.FULL_RECOMPUTE, estimated_compute_cost=1.0) for seg in new_version.segments]
            return plans, None, new_version
            
        new_version, diffs = self.segment_manager.process_edit(doc_id, new_token_ids, base_version)
        plans = self.policy.create_execution_plan(old_version, new_version.segments, diffs, num_model_layers)
        self.policy.validate_plan(plans, new_version.num_segments)
        
        return plans, old_version, new_version

    def execute_plan(
        self,
        plans: List[SegmentPlan],
        model: PhaseModelInterface,
        kv_cache: KVCacheInterface,
        token_ids: List[int],
        old_version: Optional[DocumentVersion],
        new_version: DocumentVersion,
    ) -> Tuple[Any, SegKVMetrics]:
        metrics = SegKVMetrics()
        
        # Phase 1: Prefix Reuse
        t0 = time.perf_counter()
        prefix_plans = [p for p in plans if p.strategy == RecomputeStrategy.EXACT_REUSE]
        for plan in prefix_plans:
            kv_cache.load_segment_blocks(plan.segment)
        metrics.phase1_time = time.perf_counter() - t0
        metrics.segments_reused = len(prefix_plans)
        
        # Phase 2: Changed Segment Recompute
        t1 = time.perf_counter()
        recompute_plans = [p for p in plans if p.strategy == RecomputeStrategy.FULL_RECOMPUTE]
        hidden_out = None
        for plan in recompute_plans:
            seg_token_ids = token_ids[plan.segment.start_pos:plan.segment.end_pos]
            positions = list(range(plan.segment.start_pos, plan.segment.end_pos))
            
            hidden, keys, values = model.prefill_segment(
                seg_token_ids, positions, kv_cache,
            )
            hidden_out = hidden
            for layer_idx in range(model.get_num_layers()):
                kv_cache.write_layer(layer_idx, positions, keys[layer_idx], values[layer_idx])
                
        metrics.phase2_time = time.perf_counter() - t1
        metrics.segments_recomputed = len(recompute_plans)
        
        # Phase 3: CacheBlend
        t2 = time.perf_counter()
        blend_plans = [p for p in plans if p.strategy == RecomputeStrategy.CACHEBLEND]
        skip_plans = [p for p in plans if p.strategy == RecomputeStrategy.SKIP_BLEND]
        suffix_plans = [p for p in plans if p.strategy == RecomputeStrategy.FULL_SUFFIX_RECOMPUTE]
        
        for plan in blend_plans + skip_plans:
            kv_cache.load_segment_blocks(plan.segment)
            
        if blend_plans:
            blend_hidden = self.blend_executor.execute_blend(
                model, kv_cache, blend_plans, token_ids, len(token_ids),
            )
            if blend_hidden is not None: hidden_out = blend_hidden
            
        if suffix_plans:
            for plan in suffix_plans:
                seg_tokens = token_ids[plan.segment.start_pos:plan.segment.end_pos]
                positions = list(range(plan.segment.start_pos, plan.segment.end_pos))
                hidden, keys, values = model.prefill_segment(
                    seg_tokens, positions, kv_cache,
                )
                hidden_out = hidden
                for layer_idx in range(model.get_num_layers()):
                    kv_cache.write_layer(layer_idx, positions, keys[layer_idx], values[layer_idx])
                    
        metrics.phase3_time = time.perf_counter() - t2
        metrics.segments_blended = len(blend_plans)
        metrics.segments_skipped = len(skip_plans)
        metrics.segments_suffix_recomputed = len(suffix_plans)
        
        metrics.total_time = metrics.phase1_time + metrics.phase2_time + metrics.phase3_time
        metrics.compute_savings = self.policy.estimate_total_savings(plans, model.get_num_layers())
        metrics.total_segments = len(plans)
        
        for p in blend_plans:
            metrics.blend_layers_per_segment[p.segment.segment_id] = p.num_blend_layers
            metrics.total_blend_layers += p.num_blend_layers
            
        return hidden_out, metrics

    def plan_and_execute(
        self,
        doc_id: str,
        new_token_ids: List[int],
        base_version: Optional[int],
        model: PhaseModelInterface,
        kv_cache: KVCacheInterface,
        num_model_layers: int,
    ) -> Tuple[Any, List[SegmentPlan], SegKVMetrics]:
        t0 = time.perf_counter()
        plans, old_v, new_v = self.plan_execution(doc_id, new_token_ids, base_version, num_model_layers)
        planning_time = time.perf_counter() - t0
        
        hidden, metrics = self.execute_plan(plans, model, kv_cache, new_token_ids, old_v, new_v)
        metrics.planning_time = planning_time
        return hidden, plans, metrics
