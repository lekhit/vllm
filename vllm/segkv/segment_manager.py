from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from threading import Lock
import logging
import dataclasses
import time
from vllm.segkv.config import SegKVConfig
from vllm.segkv.segment import Segment, SegmentDiff, DocumentVersion, SegmentPlan
from vllm.segkv.edit_detector import EditDetector
from vllm.segkv.utils import hash_tokens, hash_combine, compute_segment_boundaries

class SegmentManager:
    def __init__(self, config: SegKVConfig):
        self.config = config
        self.edit_detector = EditDetector(config)
        self._lock = Lock()
        self.logger = logging.getLogger("segkv.segment_manager")
        self._documents: Dict[str, OrderedDict[int, DocumentVersion]] = {}
        self._latest_versions: Dict[str, int] = {}

    def register_document(
        self,
        doc_id: str,
        token_ids: List[int],
        version: Optional[int] = None,
    ) -> DocumentVersion:
        with self._lock:
            if doc_id not in self._documents:
                if len(self._documents) >= self.config.max_documents:
                    raise ValueError(f"Max documents limit ({self.config.max_documents}) reached.")
                self._documents[doc_id] = OrderedDict()
                self._latest_versions[doc_id] = 0

            if version is None:
                version = self._latest_versions[doc_id] + 1
            
            boundaries = compute_segment_boundaries(len(token_ids), self.config.segment_size)
            segments = []
            prefix_hash = ""
            
            for i, (start, end) in enumerate(boundaries):
                seg_tokens = tuple(token_ids[start:end])
                content_hash = hash_tokens(seg_tokens)
                segment_prefix_hash = hash_combine(prefix_hash, content_hash)
                
                segment = Segment(
                    segment_id=i,
                    doc_id=doc_id,
                    start_pos=start,
                    end_pos=end,
                    token_ids=seg_tokens,
                    content_hash=content_hash,
                    version=version,
                    prefix_hash=segment_prefix_hash,
                    kv_block_ids={},
                )
                segments.append(segment)
                prefix_hash = segment_prefix_hash

            doc_version = DocumentVersion(
                doc_id=doc_id,
                version=version,
                segments=segments,
                total_tokens=len(token_ids),
                num_segments=len(segments),
            )
            
            self._documents[doc_id][version] = doc_version
            self._latest_versions[doc_id] = version
            
            self._evict_old_versions(doc_id)
            return doc_version

    def get_latest_version(self, doc_id: str) -> Optional[DocumentVersion]:
        with self._lock:
            if doc_id in self._documents and self._documents[doc_id]:
                latest_v = self._latest_versions[doc_id]
                return self._documents[doc_id].get(latest_v)
        return None

    def get_version(self, doc_id: str, version: int) -> Optional[DocumentVersion]:
        with self._lock:
            if doc_id in self._documents:
                return self._documents[doc_id].get(version)
        return None

    def process_edit(
        self,
        doc_id: str,
        new_token_ids: List[int],
        base_version: Optional[int] = None,
    ) -> Tuple[DocumentVersion, List[SegmentDiff]]:
        with self._lock:
            if doc_id not in self._documents:
                raise KeyError(f"Document {doc_id} not found.")
                
            if base_version is None:
                base_version = self._latest_versions[doc_id]
                
            old_version = self._documents[doc_id].get(base_version)
            if old_version is None:
                raise KeyError(f"Version {base_version} not found for document {doc_id}.")
                
            diffs, new_segments_tokens = self.edit_detector.detect_edits(old_version, new_token_ids)
            
            new_v_num = self._latest_versions[doc_id] + 1
            new_segments = []
            prefix_hash = ""
            
            boundaries = compute_segment_boundaries(len(new_token_ids), self.config.segment_size)
            
            for i, (start, end) in enumerate(boundaries):
                seg_tokens = new_segments_tokens[i]
                content_hash = hash_tokens(seg_tokens)
                segment_prefix_hash = hash_combine(prefix_hash, content_hash)
                
                is_unchanged = False
                old_segment = None
                if i < len(old_version.segments):
                    old_segment = old_version.segments[i]
                    if old_segment.content_hash == content_hash:
                        is_unchanged = True
                
                if is_unchanged and old_segment is not None:
                    seg = dataclasses.replace(
                        old_segment,
                        version=old_segment.version,
                        start_pos=start,
                        end_pos=end,
                        prefix_hash=segment_prefix_hash,
                        kv_block_ids=old_segment.kv_block_ids.copy() if old_segment.prefix_hash == segment_prefix_hash else {}
                    )
                    if old_segment.prefix_hash != segment_prefix_hash:
                        seg.is_kv_computed = False
                else:
                    seg = Segment(
                        segment_id=i,
                        doc_id=doc_id,
                        start_pos=start,
                        end_pos=end,
                        token_ids=seg_tokens,
                        content_hash=content_hash,
                        version=new_v_num,
                        prefix_hash=segment_prefix_hash,
                        kv_block_ids={},
                    )
                    
                new_segments.append(seg)
                prefix_hash = segment_prefix_hash

            new_version = DocumentVersion(
                doc_id=doc_id,
                version=new_v_num,
                segments=new_segments,
                total_tokens=len(new_token_ids),
                num_segments=len(new_segments),
            )
            
            self._documents[doc_id][new_v_num] = new_version
            self._latest_versions[doc_id] = new_v_num
            
            self._evict_old_versions(doc_id)
            return new_version, diffs

    def update_segment_kv_blocks(
        self,
        doc_id: str,
        version: int,
        segment_id: int,
        kv_block_ids: Dict[int, List[int]],
    ) -> None:
        with self._lock:
            doc_version = self._documents.get(doc_id, {}).get(version)
            if doc_version and 0 <= segment_id < doc_version.num_segments:
                seg = doc_version.segments[segment_id]
                seg.kv_block_ids = kv_block_ids.copy()
                seg.is_kv_computed = True
                seg.dependency_version = version

    def _evict_old_versions(self, doc_id: str) -> List[DocumentVersion]:
        evicted = []
        if doc_id in self._documents:
            versions = self._documents[doc_id]
            while len(versions) > self.config.max_cached_versions:
                _, v = versions.popitem(last=False)
                evicted.append(v)
        return evicted

    def evict_old_versions(self, doc_id: str) -> List[DocumentVersion]:
        with self._lock:
            return self._evict_old_versions(doc_id)

    def remove_document(self, doc_id: str) -> List[DocumentVersion]:
        with self._lock:
            if doc_id in self._documents:
                versions = list(self._documents[doc_id].values())
                del self._documents[doc_id]
                del self._latest_versions[doc_id]
                return versions
        return []

    def get_stats(self) -> dict:
        num_docs = len(self._documents)
        total_v = sum(len(v) for v in self._documents.values())
        total_s = 0
        total_s_kv = 0
        for doc in self._documents.values():
            for v in doc.values():
                for s in v.segments:
                    total_s += 1
                    if s.is_kv_computed:
                        total_s_kv += 1
        return {
            "num_documents": num_docs,
            "total_versions": total_v,
            "total_segments": total_s,
            "total_segments_with_kv": total_s_kv,
        }
