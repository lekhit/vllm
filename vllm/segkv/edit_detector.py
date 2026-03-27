import logging
from typing import List, Tuple, Optional, Sequence
from vllm.segkv.segment import Segment, SegmentDiff, DocumentVersion
from vllm.segkv.utils import (
    hash_tokens,
    token_edit_distance,
    find_changed_positions,
    compute_segment_boundaries,
)
from vllm.segkv.config import SegKVConfig

class EditDetector:
    """
    Detects and characterizes edits between document versions.
    Operates purely on token IDs — no model or GPU access needed.
    """

    def __init__(self, config: SegKVConfig):
        self.config = config
        self.logger = logging.getLogger("segkv.edit_detector")

    def detect_edits(
        self,
        old_version: DocumentVersion,
        new_token_ids: List[int],
    ) -> Tuple[List[SegmentDiff], List[Tuple[int, ...]]]:
        boundaries = compute_segment_boundaries(len(new_token_ids), self.config.segment_size)
        
        diffs = []
        new_segments_tokens = []
        
        for i, (start, end) in enumerate(boundaries):
            new_segment_tokens = tuple(new_token_ids[start:end])
            new_segments_tokens.append(new_segment_tokens)
            
            if i < len(old_version.segments):
                old_segment = old_version.segments[i]
                new_content_hash = hash_tokens(new_segment_tokens)
                
                if old_segment.content_hash == new_content_hash:
                    diffs.append(SegmentDiff(
                        segment_id=i,
                        change_type="unchanged",
                        old_tokens=old_segment.token_ids,
                        new_tokens=new_segment_tokens
                    ))
                else:
                    distance = token_edit_distance(old_segment.token_ids, new_segment_tokens)
                    max_len = max(len(old_segment.token_ids), len(new_segment_tokens))
                    edit_ratio = distance / max_len if max_len > 0 else 0.0
                    
                    changed_positions = []
                    if len(old_segment.token_ids) == len(new_segment_tokens):
                        changed_positions = find_changed_positions(old_segment.token_ids, new_segment_tokens)
                        
                    diffs.append(SegmentDiff(
                        segment_id=i,
                        change_type="modified",
                        token_edit_distance=distance,
                        edit_ratio=edit_ratio,
                        changed_token_positions=changed_positions,
                        old_tokens=old_segment.token_ids,
                        new_tokens=new_segment_tokens
                    ))
            else:
                diffs.append(SegmentDiff(
                    segment_id=i,
                    change_type="added",
                    token_edit_distance=len(new_segment_tokens),
                    edit_ratio=1.0,
                    new_tokens=new_segment_tokens
                ))
                
        for i in range(len(boundaries), len(old_version.segments)):
            old_segment = old_version.segments[i]
            diffs.append(SegmentDiff(
                segment_id=i,
                change_type="removed",
                token_edit_distance=len(old_segment.token_ids),
                edit_ratio=1.0,
                old_tokens=old_segment.token_ids
            ))
            
        return diffs, new_segments_tokens

    def find_first_change(
        self,
        diffs: List[SegmentDiff],
    ) -> Optional[int]:
        for diff in diffs:
            if not diff.is_unchanged:
                return diff.segment_id
        return None

    def find_all_changes(
        self,
        diffs: List[SegmentDiff],
    ) -> List[int]:
        return [diff.segment_id for diff in diffs if not diff.is_unchanged]

    def summarize_edits(
        self,
        diffs: List[SegmentDiff],
    ) -> dict:
        summary = {
            "total_segments": len(diffs),
            "unchanged_segments": sum(1 for d in diffs if d.change_type == "unchanged"),
            "modified_segments": sum(1 for d in diffs if d.change_type == "modified"),
            "added_segments": sum(1 for d in diffs if d.change_type == "added"),
            "removed_segments": sum(1 for d in diffs if d.change_type == "removed"),
            "first_change_segment": None,
            "last_change_segment": None,
            "total_edit_distance": sum(d.token_edit_distance for d in diffs),
            "avg_edit_ratio": 0.0,
            "max_edit_ratio": 0.0,
        }
        
        changed_ids = [d.segment_id for d in diffs if not d.is_unchanged]
        if changed_ids:
            summary["first_change_segment"] = min(changed_ids)
            summary["last_change_segment"] = max(changed_ids)
            
            edit_ratios = [d.edit_ratio for d in diffs if not d.is_unchanged]
            if edit_ratios:
                summary["avg_edit_ratio"] = sum(edit_ratios) / len(edit_ratios)
                summary["max_edit_ratio"] = max(edit_ratios)
                
        return summary
