"""Pure-CPU, bounded graph key and rank agreement for the Phase 4.4A prototype."""
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


def agree(plans):
    if not plans:
        raise ValueError("No rank plans")
    if any(p.get("error") for p in plans):
        raise RuntimeError(f"Rank plan failed: {[p.get('error') for p in plans]}")
    for field in ("key", "layout_digest", "mode", "epoch"):
        if any(p[field] != plans[0][field] for p in plans[1:]):
            raise RuntimeError(f"Rank graph plan mismatch: {field}")
    if plans[0]["mode"] == "eager" or plans[0]["key"] is None:
        return "eager"
    return "graph" if all(p["available"] for p in plans) else "eager"
