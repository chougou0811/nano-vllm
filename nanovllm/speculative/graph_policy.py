"""CPU-only bounded exact-key policy for opt-in concurrent target graphs."""
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphKey:
    rows: int
    batch: int
    query_lengths: tuple[int, ...]
    max_q: int
    max_k: int
    table_width: int
    block_size: int
    model_identity: str
    dtype: str
    device_class: str
    tp_size: int
    training: bool
    feature_layers: tuple[int, ...]


def make_key(layout, *, block_size, model_identity, dtype, device_class,
             tp_size, training, feature_layers):
    cu = layout["cu_seqlens_q"]
    lengths = tuple(b-a for a,b in zip(cu,cu[1:]))
    if not lengths or any(q != 4 for q in lengths) or cu[-1] not in (4,8,16):
        return None
    tables = layout["block_tables"]
    widths = {len(t) for t in tables}
    if len(widths) != 1 or len(tables) != len(lengths):
        raise ValueError("Invalid rectangular graph block table")
    if len(layout["context_lens"]) != len(lengths):
        raise ValueError("Missing graph context lengths")
    return GraphKey(cu[-1],len(lengths),lengths,max(lengths),
                    max(c+q for c,q in zip(layout["context_lens"],lengths)),
                    len(tables[0]),block_size,model_identity,dtype,device_class,
                    tp_size,training,tuple(feature_layers))


@dataclass(frozen=True)
class GraphCacheConfig:
    max_graph_entries: int = 4
    capture_policy: str = "second"
    max_captures: int = 16
    max_observed_keys: int = 4096
    max_event_records: int = 8192

    def __post_init__(self):
        if self.max_graph_entries not in (2,4,8):
            raise ValueError("Phase 4.4B supports capacities 2/4/8")
        if self.capture_policy not in ("never","second"):
            raise ValueError("Expected never or second capture policy")
        if self.max_captures < 0 or min(self.max_observed_keys,self.max_event_records)<1:
            raise ValueError("Invalid graph resource bound")


class BoundedPolicy:
    def __init__(self, config):
        self.config = config
        self.seen = OrderedDict()
        self.lru = OrderedDict()
        self.captures = 0
        self.step = 0

    def observe(self, key):
        self.step += 1
        if key is None:
            return 0
        count = self.seen.pop(key,0)+1
        self.seen[key] = count
        if len(self.seen)>self.config.max_observed_keys:
            self.seen.popitem(last=False)
        return count

    def should_capture(self, key):
        return (key is not None and key not in self.lru and
                self.config.capture_policy=="second" and self.seen.get(key,0)>=2 and
                self.captures<self.config.max_captures)

    def victim(self):
        return next(iter(self.lru)) if len(self.lru)>=self.config.max_graph_entries else None

    def insert(self, key):
        if key in self.lru or len(self.lru)>=self.config.max_graph_entries:
            raise RuntimeError("Invalid graph insertion without coordinated eviction")
        self.lru[key] = self.step
        self.captures += 1

    def touch(self, key):
        self.lru.pop(key)
        self.lru[key] = self.step


def agree(plans):
    if not plans or any(p.get("error") for p in plans):
        raise RuntimeError("Graph plan construction failed")
    # A layout/generation disagreement is unsafe even for eager collectives.
    for field in ("layout_digest","epoch","policy"):
        if any(p[field]!=plans[0][field] for p in plans[1:]):
            raise RuntimeError(f"Unsafe rank graph plan mismatch: {field}")
    if any(p["key"]!=plans[0]["key"] for p in plans[1:]):
        return "eager", "key_mismatch"
    if plans[0]["key"] is None:
        return "eager", "unsupported_shape"
    if all(p["available"] for p in plans):
        return "replay", "hit"
    if any(p["available"] for p in plans):
        return "eager", "rank_missing_key"
    if any(p["registry"]!=plans[0]["registry"] for p in plans[1:]):
        return "eager", "registry_mismatch"
    if all(p["capture"] for p in plans):
        return "capture", "second_occurrence"
    return "eager", "miss"
