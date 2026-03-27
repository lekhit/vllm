import logging
from typing import List, Optional, Set, Dict
from vllm.segkv.config import SegKVConfig, LayerSelectionStrategy
from vllm.segkv.segment import Segment, SegmentDiff, SegmentPlan, RecomputeStrategy, DocumentVersion
from vllm.segkv.utils import (
    compute_staleness_score,
    staleness_to_layer_budget,
    select_layers_uniform,
    select_layers_early_biased,
    select_layers_bookend,
)

class RecomputationPolicy:
    """
    Determines the recomputation strategy for each segment given edit diffs.
    Pure logic — no GPU or model access needed.
    """

    def __init__(self, config: SegKVConfig):
        self.config = config
        self.logger = logging.getLogger("segkv.recompute_policy")
        
        self._layer_selectors = {
            LayerSelectionStrategy.UNIFORM: select_layers_uniform,
            LayerSelectionStrategy.EARLY_BIASED: select_layers_early_biased,
            LayerSelectionStrategy.BOOKEND: select_layers_bookend,
            LayerSelectionStrategy.IMPORTANCE_STATIC: select_layers_uniform, # fallback for now
        }

    def create_execution_plan(
        self,
        old_version: Optional[DocumentVersion],
        segments: List[Segment],
        diffs: List[SegmentDiff],
        num_model_layers: int,
    ) -> List[SegmentPlan]:
        plans = []
        changed_segment_ids = {d.segment_id for d in diffs if d.change_type in ("modified", "added", "removed")}
        first_change_idx = min(changed_segment_ids) if changed_segment_ids else None
        
        if first_change_idx is None:
            return [SegmentPlan(seg, RecomputeStrategy.EXACT_REUSE) for seg in segments]
            
        last_change_idx = -1
        max_edit_ratio = 0.0
        
        for idx, seg in enumerate(segments):
            if idx < first_change_idx:
                plans.append(SegmentPlan(seg, RecomputeStrategy.EXACT_REUSE))
            elif idx in changed_segment_ids:
                plans.append(SegmentPlan(
                    seg, 
                    RecomputeStrategy.FULL_RECOMPUTE,
                    estimated_compute_cost=1.0
                ))
                last_change_idx = idx
                diff = diffs[idx] if idx < len(diffs) else None
                if diff and diff.edit_ratio > max_edit_ratio:
                    max_edit_ratio = diff.edit_ratio
            else:
                if last_change_idx == -1:
                    last_change_idx = first_change_idx
                    
                distance = idx - last_change_idx
                staleness = compute_staleness_score(
                    max_edit_ratio, distance, self.config.staleness_decay
                )
                
                if staleness < self.config.min_staleness_for_blend:
                    plans.append(SegmentPlan(
                        seg, 
                        RecomputeStrategy.SKIP_BLEND,
                        staleness_score=staleness
                    ))
                elif staleness > self.config.max_staleness_for_blend:
                    plans.append(SegmentPlan(
                        seg, 
                        RecomputeStrategy.FULL_SUFFIX_RECOMPUTE,
                        staleness_score=staleness,
                        estimated_compute_cost=1.0
                    ))
                else:
                    layer_frac = staleness_to_layer_budget(
                        staleness,
                        self.config.blend_min_layer_frac,
                        self.config.blend_max_layer_frac,
                    )
                    num_layers_to_blend = max(1, round(layer_frac * num_model_layers))
                    
                    selector = self._layer_selectors.get(self.config.layer_selection_strategy, select_layers_uniform)
                    blend_layers = selector(
                        num_model_layers,
                        num_layers_to_blend,
                        self.config.force_recompute_layers,
                    )
                    
                    ffn_fraction = 0.3
                    estimated_cost = layer_frac + ffn_fraction
                    
                    plans.append(SegmentPlan(
                        seg,
                        RecomputeStrategy.CACHEBLEND,
                        blend_layers=blend_layers,
                        blend_layer_frac=layer_frac,
                        staleness_score=staleness,
                        estimated_compute_cost=estimated_cost
                    ))
                    
        seen_full_suffix = False
        for i, plan in enumerate(plans):
            if plan.strategy == RecomputeStrategy.FULL_SUFFIX_RECOMPUTE:
                seen_full_suffix = True
            elif seen_full_suffix and plan.strategy in (RecomputeStrategy.CACHEBLEND, RecomputeStrategy.SKIP_BLEND):
                plans[i] = SegmentPlan(
                    plan.segment,
                    RecomputeStrategy.FULL_SUFFIX_RECOMPUTE,
                    staleness_score=plan.staleness_score,
                    estimated_compute_cost=1.0
                )
                
        n_reuse = sum(1 for p in plans if p.strategy == RecomputeStrategy.EXACT_REUSE)
        n_recompute = sum(1 for p in plans if p.strategy == RecomputeStrategy.FULL_RECOMPUTE)
        n_blend = sum(1 for p in plans if p.strategy == RecomputeStrategy.CACHEBLEND)
        n_suffix = sum(1 for p in plans if p.strategy == RecomputeStrategy.FULL_SUFFIX_RECOMPUTE)
        n_skip = sum(1 for p in plans if p.strategy == RecomputeStrategy.SKIP_BLEND)
        
        self.logger.info(f"Plan: {n_reuse} reuse, {n_recompute} recompute, {n_blend} blend, {n_suffix} suffix_recompute, {n_skip} skip")
        
        return plans

    def estimate_total_savings(
        self,
        plans: List[SegmentPlan],
        num_model_layers: int,
    ) -> float:
        if not plans:
            return 0.0
        full_cost = len(plans) * 1.0
        actual_cost = sum(p.estimated_compute_cost for p in plans)
        return max(0.0, 1.0 - (actual_cost / full_cost))

    def get_blend_layer_union(
        self,
        plans: List[SegmentPlan],
    ) -> List[int]:
        layers = set()
        for p in plans:
            if p.strategy == RecomputeStrategy.CACHEBLEND:
                layers.update(p.blend_layers)
        return sorted(list(layers))

    def validate_plan(
        self,
        plans: List[SegmentPlan],
        num_segments: int,
    ) -> bool:
        if len(plans) != num_segments:
            raise ValueError(f"Plan length ({len(plans)}) != num_segments ({num_segments})")
            
        for i, plan in enumerate(plans):
            if plan.segment.segment_id != i:
                raise ValueError(f"Plan step {i} has segment_id {plan.segment.segment_id}")
                
        first_non_reuse = -1
        for i, plan in enumerate(plans):
            if plan.strategy != RecomputeStrategy.EXACT_REUSE:
                first_non_reuse = i
                break
        
        if first_non_reuse != -1:
            for i in range(first_non_reuse + 1, len(plans)):
                if plans[i].strategy == RecomputeStrategy.EXACT_REUSE:
                    raise ValueError(f"EXACT_REUSE appears after non-reuse at segment {i}")
                    
        seen_full_suffix = False
        for i, plan in enumerate(plans):
            if plan.strategy == RecomputeStrategy.FULL_SUFFIX_RECOMPUTE:
                seen_full_suffix = True
            elif seen_full_suffix and plan.strategy == RecomputeStrategy.CACHEBLEND:
                raise ValueError(f"CACHEBLEND appears after FULL_SUFFIX_RECOMPUTE at segment {i}")
                
        return True
