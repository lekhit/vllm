from typing import Dict, List, Optional, Set, Tuple
from threading import Lock
import math
import logging
from vllm.segkv.config import SegKVConfig
from vllm.segkv.segment import Segment, DocumentVersion

class SegKVBlockManager:
    """
    Manages the mapping between SegKV segments and vLLM's physical KV cache blocks.
    """

    def __init__(
        self,
        config: SegKVConfig = None,
        block_size: int = 16,
        num_layers: int = 32,
    ):
        if config is None:
            config = SegKVConfig(enable_segkv=True)
        self.config = config
        self.block_size = block_size
        self.num_layers = num_layers
        self._lock = Lock()
        self.logger = logging.getLogger("segkv.block_manager")
        
        # Primary index: (doc_id, segment_id, version) -> {layer_idx: [block_ids]}
        self._segment_blocks: Dict[Tuple[str, int, int], Dict[int, List[int]]] = {}
        
        # Reverse index: block_id -> (doc_id, segment_id, version, layer_idx)
        self._block_owners: Dict[int, Tuple[str, int, int, int]] = {}
        
        self._stale_blocks: Set[int] = set()
        self._block_refcount: Dict[int, int] = {}

    def compute_blocks_needed(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def register_segment_blocks(
        self,
        doc_id: str,
        segment_id: int,
        version: int,
        block_assignments: Dict[int, List[int]],
    ) -> None:
        with self._lock:
            key = (doc_id, segment_id, version)
            self._segment_blocks[key] = {
                layer_idx: list(blocks)
                for layer_idx, blocks in block_assignments.items()
            }
            
            for layer_idx, blocks in block_assignments.items():
                for block_id in blocks:
                    self._block_owners[block_id] = (doc_id, segment_id, version, layer_idx)
                    self._block_refcount[block_id] = self._block_refcount.get(block_id, 0) + 1

    def get_segment_blocks(
        self,
        doc_id: str,
        segment_id: int,
        version: int,
    ) -> Optional[Dict[int, List[int]]]:
        with self._lock:
            key = (doc_id, segment_id, version)
            if key in self._segment_blocks:
                return {layer: list(blocks) for layer, blocks in self._segment_blocks[key].items()}
        return None

    def get_segment_blocks_for_layer(
        self,
        doc_id: str,
        segment_id: int,
        version: int,
        layer_idx: int,
    ) -> Optional[List[int]]:
        with self._lock:
            key = (doc_id, segment_id, version)
            if key in self._segment_blocks and layer_idx in self._segment_blocks[key]:
                return list(self._segment_blocks[key][layer_idx])
        return None

    def mark_stale(
        self,
        doc_id: str,
        segment_id: int,
        version: int,
        layers: Optional[List[int]] = None,
    ) -> int:
        count = 0
        with self._lock:
            key = (doc_id, segment_id, version)
            if key in self._segment_blocks:
                for layer_idx, blocks in self._segment_blocks[key].items():
                    if layers is None or layer_idx in layers:
                        for block_id in blocks:
                            if block_id not in self._stale_blocks:
                                self._stale_blocks.add(block_id)
                                count += 1
        return count

    def unmark_stale(
        self,
        doc_id: str,
        segment_id: int,
        version: int,
        layers: Optional[List[int]] = None,
    ) -> int:
        count = 0
        with self._lock:
            key = (doc_id, segment_id, version)
            if key in self._segment_blocks:
                for layer_idx, blocks in self._segment_blocks[key].items():
                    if layers is None or layer_idx in layers:
                        for block_id in blocks:
                            if block_id in self._stale_blocks:
                                self._stale_blocks.remove(block_id)
                                count += 1
        return count

    def is_stale(self, block_id: int) -> bool:
        with self._lock:
            return block_id in self._stale_blocks

    def release_segment_blocks(
        self,
        doc_id: str,
        segment_id: int,
        version: int,
    ) -> List[int]:
        freed_blocks = []
        with self._lock:
            key = (doc_id, segment_id, version)
            if key in self._segment_blocks:
                for layer_idx, blocks in self._segment_blocks[key].items():
                    for block_id in blocks:
                        if block_id in self._block_refcount:
                            self._block_refcount[block_id] -= 1
                            if self._block_refcount[block_id] <= 0:
                                freed_blocks.append(block_id)
                                del self._block_refcount[block_id]
                                if block_id in self._block_owners:
                                    del self._block_owners[block_id]
                                if block_id in self._stale_blocks:
                                    self._stale_blocks.remove(block_id)
                del self._segment_blocks[key]
        return freed_blocks

    def share_blocks(
        self,
        source_doc_id: str,
        source_segment_id: int,
        source_version: int,
        target_doc_id: str,
        target_segment_id: int,
        target_version: int,
    ) -> bool:
        with self._lock:
            source_key = (source_doc_id, source_segment_id, source_version)
            if source_key not in self._segment_blocks:
                return False
                
            target_key = (target_doc_id, target_segment_id, target_version)
            self._segment_blocks[target_key] = {
                layer_idx: list(blocks) 
                for layer_idx, blocks in self._segment_blocks[source_key].items()
            }
            
            for layer_idx, blocks in self._segment_blocks[source_key].items():
                for block_id in blocks:
                    self._block_refcount[block_id] = self._block_refcount.get(block_id, 0) + 1
                    
            return True

    def get_all_blocks_for_document(
        self,
        doc_id: str,
        version: int,
    ) -> Dict[int, Dict[int, List[int]]]:
        result = {}
        with self._lock:
            for (d_id, s_id, v), assignments in self._segment_blocks.items():
                if d_id == doc_id and v == version:
                    result[s_id] = {l: list(b) for l, b in assignments.items()}
        return result

    def get_memory_usage(self) -> dict:
        with self._lock:
            return {
                "total_blocks_tracked": len(self._block_refcount),
                "total_segments_tracked": len(self._segment_blocks),
                "stale_blocks": len(self._stale_blocks),
                "shared_blocks": sum(1 for c in self._block_refcount.values() if c > 1),
                "estimated_gpu_bytes": 0, # Cannot know exactly without token dim
            }

    def cleanup_document(
        self,
        doc_id: str,
        keep_version: Optional[int] = None,
    ) -> List[int]:
        freed_blocks = []
        with self._lock:
            keys_to_remove = []
            for key in self._segment_blocks.keys():
                d_id, s_id, v = key
                if d_id == doc_id and (keep_version is None or v != keep_version):
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                for layer_idx, blocks in self._segment_blocks[key].items():
                    for block_id in blocks:
                        if block_id in self._block_refcount:
                            self._block_refcount[block_id] -= 1
                            if self._block_refcount[block_id] <= 0:
                                freed_blocks.append(block_id)
                                del self._block_refcount[block_id]
                                if block_id in self._block_owners:
                                    del self._block_owners[block_id]
                                if block_id in self._stale_blocks:
                                    self._stale_blocks.remove(block_id)
                del self._segment_blocks[key]
        return freed_blocks
