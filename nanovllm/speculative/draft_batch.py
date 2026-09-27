"""Opt-in cross-request feedback execution; confirmed DraftState is never packed."""
from dataclasses import dataclass
from time import perf_counter_ns

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DraftBatchInput:
    state: object
    request: object
    generation: int
    features: torch.Tensor
    tokens: tuple
    k: int

    def validate(self):
        self.state._check_owner(self.request.seq_id)
        if (self.request.generation != self.generation or
                getattr(self.state, "request_generation", None) != self.generation):
            raise RuntimeError("Stale draft batch generation")
        end = self.features.shape[0]
        if self.state.mode != "persistent":
            raise ValueError("Batched draft requires persistent state")
        if not 1 <= self.k <= 3 or len(self.tokens) != end + 1 or end <= self.state.cursor:
            raise ValueError("Invalid draft batch extent")
        if tuple(self.tokens[1:self.state.cursor+1]) != self.state.conditioned_tokens:
            raise RuntimeError("Committed draft conditioning changed")


@dataclass
class PreparedDraft:
    item: DraftBatchInput
    hidden: torch.Tensor
    confirmed: tuple
    start: int
    end: int
    conditioning_ns: int


class DraftBatchExecutor:
    """Single rank0 executor; no cross-request persistent buffers or CUDA Graphs.

    Optional observer is diagnostic-only. It must not mutate tensors. Host spans
    are nested submission timings; generation_ns includes the final token read.
    """

    def __init__(self, observer=None):
        self.observer = observer
        self.last_metrics = {}

    @torch.inference_mode()
    def prepare(self, items):
        if not items or len({id(i.state) for i in items}) != len(items):
            raise ValueError("Empty or duplicate draft owner")
        if len({(i.request.seq_id, i.generation) for i in items}) != len(items):
            raise ValueError("Duplicate request identity")
        draft = items[0].state.draft
        for item in items:
            item.validate()
            if item.state.draft is not draft:
                raise ValueError("Mixed drafter instances")
        prepared = []
        for item in items:
            state, model, device = item.state, draft.model, draft.device
            model.reset()
            model.reset_kv()
            start, end = state.cursor, item.features.shape[0]
            before = perf_counter_ns()
            ids = torch.tensor(item.tokens[start+1:end+1], device=device).unsqueeze(0)
            hidden, confirmed = model(item.features[start:].unsqueeze(0), input_ids=ids,
                                      past_key_values=state.past, use_cache=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = perf_counter_ns()-before
            if any(t.shape[2] != end for layer in confirmed for t in layer):
                raise RuntimeError("Draft cache cursor mismatch")
            prepared.append(PreparedDraft(item, hidden[:, -1:], confirmed, start, end, elapsed))
        return prepared

    @torch.inference_mode()
    def generate(self, prepared, *, publish=True):
        if not prepared:
            return []
        for p in prepared:
            p.item.validate()
        model = prepared[0].item.state.draft.model
        device = prepared[0].hidden.device
        begin = perf_counter_ns()
        lengths = [p.end for p in prepared]
        width = max(lengths)
        past = tuple(tuple(torch.cat([F.pad(p.confirmed[layer][kv],
                    (0, 0, 0, width-p.end)) for p in prepared], dim=0)
                    for kv in range(2)) for layer in range(len(prepared[0].confirmed)))
        hidden = torch.cat([p.hidden for p in prepared], dim=0)
        packed = perf_counter_ns()
        lengths_device = torch.tensor(lengths, device=device, dtype=torch.long)
        mask = torch.arange(width, device=device)[None, :] < lengths_device[:, None]
        prep_done = perf_counter_ns()
        scratch_bytes = sum(t.numel()*t.element_size() for layer in past for t in layer)
        proposals = [[] for _ in prepared]
        active = list(range(len(prepared)))
        metrics = dict(packing_ns=packed-begin, mask_position_ns=prep_done-packed,
                       model_submit_ns=0, head_read_ns=0, gather_ns=0, publish_ns=0,
                       forward_batch_sizes=[], head_batch_sizes=[], scratch_bytes=scratch_bytes)
        for step in range(max(p.item.k for p in prepared)):
            before = perf_counter_ns()
            logits = model.lm_head(model.norm(hidden))[:, -1]
            index = logits.argmax(dim=-1)
            mapped = index + model.d2t[index]
            ids = mapped.tolist()
            metrics["head_read_ns"] += perf_counter_ns()-before
            metrics["head_batch_sizes"].append(len(active))
            if self.observer:
                self.observer(step, tuple(active), hidden, past, logits,
                              lengths_device+step, mask)
            for row, token in zip(active, ids):
                proposals[row].append(token)
            keep = [j for j, row in enumerate(active) if step+1 < prepared[row].item.k]
            if not keep:
                break
            before = perf_counter_ns()
            if len(keep) != len(active):
                selection = torch.tensor(keep, device=device, dtype=torch.long)
                hidden = hidden.index_select(0, selection)
                past = tuple(tuple(t.index_select(0, selection) for t in layer) for layer in past)
                mask = mask.index_select(0, selection)
                lengths_device = lengths_device.index_select(0, selection)
                mapped = mapped.index_select(0, selection)
                active = [active[j] for j in keep]
            metrics["gather_ns"] += perf_counter_ns()-before
            before = perf_counter_ns()
            mask = torch.cat((mask, torch.ones((len(active), 1), dtype=torch.bool, device=device)), dim=1)
            positions = (lengths_device+step).unsqueeze(1)
            metrics["mask_position_ns"] += perf_counter_ns()-before
            before = perf_counter_ns()
            hidden, past = model(hidden, input_ids=mapped.unsqueeze(1), attention_mask=mask,
                                 position_ids=positions, past_key_values=past, use_cache=True)
            metrics["model_submit_ns"] += perf_counter_ns()-before
            metrics["forward_batch_sizes"].append(len(active))
        # Check every owner before publishing any state. Feedback KV stays local.
        before = perf_counter_ns()
        for p in prepared:
            p.item.validate()
        if publish:
            for p in prepared:
                state = p.item.state
                state.past, state.cursor = p.confirmed, p.end
                state.conditioned_tokens = tuple(p.item.tokens[1:p.end+1])
                state.last_metrics = dict(mode="persistent", cursor_before=p.start,
                    cursor_after=p.end, draft_tokens_processed=p.end-p.start+p.item.k-1,
                    reused_draft_prefix_tokens=p.start, rollback_tokens=p.item.k-1,
                    conditioning_ns=p.conditioning_ns, draft_incremental_ns=p.conditioning_ns,
                    draft_full_rebuild_ns=0, draft_forwards=p.item.k, draft_kv_rows=p.end,
                    draft_kv_bytes=sum(t.numel()*t.element_size() for layer in p.confirmed for t in layer))
        metrics["publish_ns"] = perf_counter_ns()-before
        metrics.update(generation_ns=perf_counter_ns()-begin,
            conditioning_ns=sum(p.conditioning_ns for p in prepared),
            draft_forwards=len(prepared)+len(metrics["forward_batch_sizes"]),
            batched_draft_forwards=sum(n > 1 for n in metrics["forward_batch_sizes"]),
            serial_equivalent_forwards=sum(p.item.k for p in prepared),
            owner_order=[(p.item.request.seq_id, p.item.generation) for p in prepared])
        self.last_metrics = metrics
        return proposals

    @torch.inference_mode()
    def propose(self, items):
        start = perf_counter_ns()
        result = self.generate(self.prepare(items))
        self.last_metrics["total_ns"] = perf_counter_ns()-start
        return result
