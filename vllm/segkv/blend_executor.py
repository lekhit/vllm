from typing import List, Dict, Optional, Set, Tuple, Protocol, Any
import logging
from vllm.segkv.segment import SegmentPlan, RecomputeStrategy
from vllm.segkv.config import SegKVConfig

class ModelLayerInterface(Protocol):
    def get_num_layers(self) -> int: ...
    def get_embedding(self, token_ids: List[int]) -> Any: ...
    def run_layer_full_attention(
        self, layer_idx: int, hidden_states: Any, 
        kv_cache: Any, positions: List[int],
    ) -> Tuple[Any, Any, Any]: ...
    def run_layer_cached(
        self, layer_idx: int, hidden_states: Any,
        kv_cache: Any, positions: List[int],
    ) -> Any: ...
    def run_final_norm(self, hidden_states: Any) -> Any: ...

class KVCacheInterface(Protocol):
    def read_layer(self, layer_idx: int) -> Tuple[Any, Any]: ...
    def write_layer(
        self, layer_idx: int, positions: List[int],
        keys: Any, values: Any,
    ) -> None: ...
    def load_segment_blocks(self, segment: Any) -> None: ...

class BlendExecutor:
    def __init__(self, config: SegKVConfig):
        self.config = config
        self.logger = logging.getLogger("segkv.blend_executor")

    def execute_blend(
        self,
        model: ModelLayerInterface,
        kv_cache: KVCacheInterface,
        blend_plans: List[SegmentPlan],
        token_ids: List[int],
        total_seq_len: int,
    ) -> Any:
        stale_positions = self._collect_stale_positions(blend_plans)
        if not stale_positions:
            return None
            
        recompute_layers = sorted(list(self._get_recompute_layer_set(blend_plans)))
        stale_token_ids = [token_ids[p] for p in stale_positions]
        
        hidden = model.get_embedding(stale_token_ids)
        num_layers = model.get_num_layers()
        
        for layer_idx in range(num_layers):
            if layer_idx in recompute_layers:
                hidden, new_keys, new_values = model.run_layer_full_attention(
                    layer_idx, hidden, kv_cache, stale_positions,
                )
                kv_cache.write_layer(layer_idx, stale_positions, new_keys, new_values)
                self.logger.debug(f"Recomputed layer {layer_idx}")
            else:
                hidden = model.run_layer_cached(
                    layer_idx, hidden, kv_cache, stale_positions,
                )
                
        hidden = model.run_final_norm(hidden)
        return hidden

    def _collect_stale_positions(
        self,
        blend_plans: List[SegmentPlan],
    ) -> List[int]:
        stale_positions = []
        for plan in blend_plans:
            stale_positions.extend(range(plan.segment.start_pos, plan.segment.end_pos))
        return sorted(stale_positions)

    def _get_recompute_layer_set(
        self,
        blend_plans: List[SegmentPlan],
    ) -> Set[int]:
        recompute_layers = set()
        for plan in blend_plans:
            recompute_layers.update(plan.blend_layers)
        return recompute_layers

    def create_execution_summary(
        self,
        blend_plans: List[SegmentPlan],
        num_model_layers: int,
    ) -> dict:
        stale_positions = self._collect_stale_positions(blend_plans)
        recompute_layers = self._get_recompute_layer_set(blend_plans)
        return {
            "num_stale_positions": len(stale_positions),
            "num_recompute_layers": len(recompute_layers),
            "num_passthrough_layers": num_model_layers - len(recompute_layers),
            "recompute_layer_indices": sorted(list(recompute_layers)),
            "estimated_flops_fraction": len(recompute_layers) / max(1, num_model_layers),
        }
