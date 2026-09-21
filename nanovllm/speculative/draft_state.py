"""Request-owned feature-conditioned cache; proposal feedback is always tentative."""
from time import perf_counter_ns

import torch


class DraftState:
    def __init__(self, draft, owner, mode="persistent"):
        if mode not in ("persistent", "full_rebuild"):
            raise ValueError("Unknown draft state mode")
        self.draft, self.owner, self.mode = draft, owner, mode
        self.past = None
        self.cursor = 0
        self.conditioned_tokens = ()
        self.closed = False
        self.last_metrics = {}

    def _check_owner(self, owner):
        if self.closed or owner != self.owner:
            raise RuntimeError("Closed or foreign draft state")

    @torch.inference_mode()
    def propose(self, features, tokens, k, *, owner):
        self._check_owner(owner)
        end = features.shape[0]
        if len(tokens) != end + 1 or k < 1 or end <= self.cursor:
            raise ValueError("Expected advancing target features plus one pending token")
        if tuple(tokens[1:self.cursor+1]) != self.conditioned_tokens:
            raise RuntimeError("Committed draft conditioning changed")
        model, device = self.draft.model, self.draft.device
        model.reset()
        model.reset_kv()
        start = self.cursor if self.mode == "persistent" else 0
        before = perf_counter_ns()
        ids = torch.tensor(tokens[start+1:end+1], device=device).unsqueeze(0)
        hidden, confirmed = model(features[start:].unsqueeze(0), input_ids=ids,
                                  past_key_values=self.past, use_cache=True)
        # Synchronize host timing at the same boundary in both experimental modes.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        conditioned = perf_counter_ns()
        if any(t.shape[2] != end for layer in confirmed for t in layer):
            raise RuntimeError("Draft cache cursor mismatch")
        past = confirmed
        proposals = []
        for step in range(k):
            logits = model.lm_head(model.norm(hidden[:, -1:]))
            index = logits[0, -1].argmax()
            token = int((index + model.d2t[index]).item())
            proposals.append(token)
            if step + 1 < k:
                hidden, past = model(hidden[:, -1:], input_ids=torch.tensor([[token]], device=device),
                                     past_key_values=past, use_cache=True)
        # Reference attention concatenates past KV into new storage, never in-place.
        # Keep only target-feature-conditioned rows, including the pending-token shift.
        if self.mode == "persistent":
            self.past = confirmed
            self.cursor = end
            self.conditioned_tokens = tuple(tokens[1:end+1])
        self.last_metrics = dict(mode=self.mode, cursor_before=start, cursor_after=end,
            draft_tokens_processed=end-start+k-1, reused_draft_prefix_tokens=start,
            rollback_tokens=k-1, conditioning_ns=conditioned-before,
            draft_full_rebuild_ns=conditioned-before if self.mode == "full_rebuild" else 0,
            draft_incremental_ns=conditioned-before if self.mode == "persistent" else 0,
            draft_forwards=k, draft_kv_rows=end if self.mode == "persistent" else 0,
            draft_kv_bytes=sum(t.numel()*t.element_size() for layer in confirmed for t in layer)
                if self.mode == "persistent" else 0)
        return proposals

    def committed(self, target_cursor, tokens, *, owner):
        self._check_owner(owner)
        if target_cursor < self.cursor or len(tokens) not in (target_cursor, target_cursor+1):
            raise RuntimeError("Target/draft cursor invariant failed")
        if tuple(tokens[1:self.cursor+1]) != self.conditioned_tokens:
            raise RuntimeError("Accepted transaction changed conditioned draft prefix")

    def close(self):
        self.past = None
        self.cursor = 0
        self.conditioned_tokens = ()
        self.closed = True
