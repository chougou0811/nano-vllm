import unittest
from types import SimpleNamespace

import torch

from nanovllm.speculative.draft_batch import DraftBatchExecutor, DraftBatchInput
from nanovllm.speculative.draft_state import DraftState


class TinyDraft(torch.nn.Module):
    """FP64 causal masked cache control, not a replacement EAGLE implementation."""
    def __init__(self):
        super().__init__()
        self.d2t = torch.zeros(7, dtype=torch.long)
        self.norm = torch.nn.Identity()
        self.lm_head = torch.nn.Linear(4, 7, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(28).reshape(7, 4)/28)
        self.calls = 0
        self.fail_at = None

    def reset(self):
        pass

    def reset_kv(self):
        pass

    def forward(self, hidden, input_ids, past_key_values=None, use_cache=True,
                attention_mask=None, position_ids=None):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("intentional draft failure")
        b, n, _ = hidden.shape
        old = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if position_ids is None:
            position_ids = torch.arange(old, old+n).expand(b, -1)
        kv = hidden + input_ids[..., None]/100 + position_ids[..., None]/1000
        values = kv[:, None]
        if past_key_values is not None:
            values = torch.cat((past_key_values[0][0], values), dim=2)
        if attention_mask is None:
            attention_mask = torch.ones((b, old+n), dtype=torch.bool)
        outs = []
        for j in range(n):
            mask = attention_mask[:, :old+j+1].to(hidden.dtype)
            total = (values[:, 0, :old+j+1]*mask[..., None]).sum(1)
            outs.append(kv[:, j]+total/mask.sum(1)[:, None])
        return torch.stack(outs, 1), ((values, values.clone()),)


def inputs(lengths, ks=None):
    model = TinyDraft()
    draft = SimpleNamespace(model=model, device=torch.device("cpu"))
    items = []
    for i, n in enumerate(lengths):
        request = SimpleNamespace(seq_id=i+10, generation=2)
        state = DraftState(draft, request.seq_id)
        state.request_generation = request.generation
        features = torch.arange(n*4, dtype=torch.float64).reshape(n, 4)/10000+1
        items.append(DraftBatchInput(state, request, 2, features,
                                    tuple(j % 7 for j in range(n+1)), (ks or [3]*len(lengths))[i]))
    return items


class DraftBatchTests(unittest.TestCase):
    def compare(self, lengths, ks=None):
        serial, batch = inputs(lengths, ks), inputs(lengths, ks)
        serial_states, expected = [], []
        for item in serial:
            calls = []
            handle = item.state.draft.model.register_forward_hook(lambda m,a,out: calls.append(out))
            try:
                expected.append(item.state.propose(item.features,item.tokens,item.k,owner=item.request.seq_id))
            finally:
                handle.remove()
            serial_states.append(calls)
        observed = []
        executor = DraftBatchExecutor(lambda *args: observed.append(args))
        actual = executor.propose(batch)
        self.assertEqual(actual, expected)
        self.assertEqual(executor.last_metrics["draft_forwards"], len(batch)+max(i.k for i in batch)-1)
        for a, b in zip(serial, batch):
            self.assertEqual(a.state.cursor, b.state.cursor)
            self.assertEqual(a.state.conditioned_tokens, b.state.conditioned_tokens)
            for la, lb in zip(a.state.past, b.state.past):
                for ta, tb in zip(la, lb):
                    torch.testing.assert_close(ta, tb, rtol=0, atol=0)
                    self.assertEqual(tb.shape[2], b.features.shape[0])
        for step, owners, hidden, past, logits, next_pos, mask in observed:
            for row, index in enumerate(owners):
                self.assertEqual(int(next_pos[row]), lengths[index]+step)
                self.assertEqual(int(mask[row].sum()), lengths[index]+step)
                self.assertTrue(torch.isfinite(hidden[row]).all())
                self.assertTrue(torch.isfinite(logits[row]).all())
                reference_hidden, reference_kv = serial_states[index][step]
                torch.testing.assert_close(hidden[row:row+1], reference_hidden[:, -1:], rtol=1e-12, atol=1e-12)
                valid = mask[row].nonzero().flatten()
                for layer, reference_layer in zip(past, reference_kv):
                    for tensor, reference in zip(layer, reference_layer):
                        torch.testing.assert_close(tensor[row:row+1].index_select(2,valid), reference,
                                                   rtol=1e-12, atol=1e-12)
        return batch

    def test_ragged_boundaries_and_group_sizes(self):
        for lengths in ([255, 256], [257, 511, 512], [513, 1025, 255, 257]):
            with self.subTest(lengths=lengths):
                self.compare(lengths)

    def test_clipped_k_reorder(self):
        self.compare([257, 255, 512, 256], [1, 3, 2, 1])

    def test_repeated_accept_lengths_and_refill(self):
        items = inputs([5, 9, 7, 6])
        executor = DraftBatchExecutor()
        for accepted in ([0, 1, 3, 2], [0, 0, 0, 0], [3, 3, 3, 3]):
            executor.propose(items)
            next_items = []
            for item, a in zip(items, accepted):
                end = item.state.cursor+1+a
                tokens = item.tokens+tuple(2 for _ in range(1+a))
                item.state.committed(end, tokens, owner=item.request.seq_id)
                features = torch.cat((item.features, torch.ones((1+a, 4), dtype=torch.float64)))
                next_items.append(DraftBatchInput(item.state, item.request, item.generation, features, tokens, 3))
            items = list(reversed(next_items))
        for item in items:
            item.state.close()
            self.assertIsNone(item.state.past)
        with self.assertRaises(RuntimeError):
            executor.propose(items)

    def test_stale_foreign_duplicate_and_changed_prefix(self):
        for kind in ("stale", "foreign", "duplicate", "prefix", "stale_state"):
            items = inputs([5, 6])
            if kind == "stale":
                items[0].request.generation += 1
            elif kind == "foreign":
                items[0].request.seq_id += 100
            elif kind == "duplicate":
                items[1] = items[0]
            elif kind == "stale_state":
                items[0].state.request_generation -= 1
            else:
                items[0].state.cursor = 1
                items[0].state.conditioned_tokens = (999,)
            with self.subTest(kind=kind), self.assertRaises((RuntimeError, ValueError)):
                DraftBatchExecutor().propose(items)

    def test_exception_does_not_publish_scratch(self):
        items = inputs([5, 6])
        items[0].state.draft.model.fail_at = 3
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            DraftBatchExecutor().propose(items)
        for item in items:
            self.assertIsNone(item.state.past)
            self.assertEqual(item.state.cursor, 0)

    def test_generation_change_during_execution_is_rejected(self):
        items = inputs([5, 6])
        def observer(*args):
            items[0].request.generation += 1
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            DraftBatchExecutor(observer).propose(items)
        self.assertTrue(all(i.state.past is None for i in items))


if __name__ == "__main__":
    unittest.main()
