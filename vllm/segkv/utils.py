import hashlib
from typing import List, Tuple, Optional, Sequence
import math

def hash_tokens(token_ids: Sequence[int]) -> str:
    representation = ",".join(str(t) for t in token_ids)
    return hashlib.sha256(representation.encode("utf-8")).hexdigest()[:32]

def hash_combine(prefix_hash: str, content_hash: str) -> str:
    combined = f"{prefix_hash}:{content_hash}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]

def token_edit_distance(
    a: Sequence[int], 
    b: Sequence[int],
    max_distance: Optional[int] = None,
) -> int:
    if not a: return len(b)
    if not b: return len(a)
    
    # Fast path for exact match
    if len(a) == len(b) and all(x == y for x, y in zip(a, b)):
        return 0

    m, n = len(a), len(b)
    # Simple dynamic programming
    dp = list(range(n + 1))
    
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        current_min = dp[0]
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
            current_min = min(current_min, dp[j])
        
        if max_distance is not None and current_min > max_distance:
            return max_distance + 1

    return dp[n]

def find_changed_positions(
    old_tokens: Sequence[int],
    new_tokens: Sequence[int],
) -> List[int]:
    if len(old_tokens) != len(new_tokens):
        raise ValueError("Sequences must have the same length")
    return [i for i, (o, n) in enumerate(zip(old_tokens, new_tokens)) if o != n]

def compute_segment_boundaries(
    total_tokens: int,
    segment_size: int,
) -> List[Tuple[int, int]]:
    assert segment_size > 0
    assert total_tokens >= 0
    boundaries = []
    start = 0
    while start < total_tokens:
        end = min(start + segment_size, total_tokens)
        boundaries.append((start, end))
        start = end
    return boundaries

def compute_staleness_score(
    edit_ratio: float,
    distance_from_edit: int,
    decay_rate: float,
) -> float:
    return min(1.0, edit_ratio) * (decay_rate ** distance_from_edit)

def select_layers_uniform(
    num_layers: int,
    num_to_select: int,
    force_include: Optional[List[int]] = None,
) -> List[int]:
    if num_to_select <= 0:
        return []
    if num_to_select >= num_layers:
        return list(range(num_layers))
        
    result = set(force_include) if force_include else set()
    result = {l for l in result if 0 <= l < num_layers}
    
    if len(result) >= num_to_select:
        return sorted(list(result))[:num_to_select]
        
    remaining_budget = num_to_select - len(result)
    step = num_layers / (remaining_budget + 1)
    
    current_idx = step
    while len(result) < num_to_select and current_idx < num_layers:
        layer = int(round(current_idx))
        if layer not in result and 0 <= layer < num_layers:
            result.add(layer)
        current_idx += step
        
    return sorted(list(result))

def select_layers_early_biased(
    num_layers: int,
    num_to_select: int,
    force_include: Optional[List[int]] = None,
) -> List[int]:
    if num_to_select <= 0: return []
    if num_to_select >= num_layers: return list(range(num_layers))
    
    result = set(force_include) if force_include else set()
    result = {l for l in result if 0 <= l < num_layers}
    
    remaining_budget = num_to_select - len(result)
    for i in range(remaining_budget):
        layer = int(((i + 1) / (remaining_budget + 1)) ** 2 * num_layers)
        layer = min(num_layers - 1, max(0, layer))
        while layer in result and layer < num_layers - 1:
            layer += 1
        if layer not in result:
            result.add(layer)
            
    return sorted(list(result))

def select_layers_bookend(
    num_layers: int,
    num_to_select: int,
    force_include: Optional[List[int]] = None,
) -> List[int]:
    if num_to_select <= 0: return []
    if num_to_select >= num_layers: return list(range(num_layers))
    
    result = set(force_include) if force_include else set()
    result.add(0)
    if num_to_select > 1:
        result.add(num_layers - 1)
        
    result = {l for l in result if 0 <= l < num_layers}
    
    if len(result) < num_to_select:
        remaining = num_to_select - len(result)
        middle_layers = select_layers_uniform(num_layers - 2, remaining)
        for l in middle_layers:
            result.add(l + 1)
            
    return sorted(list(result))[:num_to_select]

def staleness_to_layer_budget(
    staleness: float,
    min_frac: float = 0.0,
    max_frac: float = 0.30,
) -> float:
    if staleness < 0.02:
        return min_frac
    elif staleness < 0.1:
        t = (staleness - 0.02) / 0.08
        return min_frac + t * (0.05 - min_frac)
    elif staleness < 0.3:
        t = (staleness - 0.1) / 0.2
        return 0.05 + t * (0.15 - 0.05)
    elif staleness < 0.6:
        t = (staleness - 0.3) / 0.3
        return 0.15 + t * (0.25 - 0.15)
    elif staleness < 0.8:
        t = (staleness - 0.6) / 0.2
        return 0.25 + t * (max_frac - 0.25)
    else:
        return max_frac
