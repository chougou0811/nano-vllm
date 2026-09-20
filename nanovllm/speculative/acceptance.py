from dataclasses import dataclass


@dataclass(frozen=True)
class Acceptance:
    matched: int
    accepted: int
    tokens: list[int]
    fallback: int | None
    finished: bool


def accept_greedy(proposals, target_ids, remaining, eos=None):
    """Single-path specialization of reference greedy evaluate_posterior."""
    if remaining < 1 or len(target_ids) != len(proposals)+1:
        raise ValueError("Invalid verification/output lengths")
    matched = 0
    for candidate, target in zip(proposals,target_ids):
        if candidate != target:
            break
        matched += 1
    tokens = []
    accepted = 0
    for token in proposals[:matched]:
        tokens.append(int(token))
        accepted += 1
        if token == eos or len(tokens) == remaining:
            return Acceptance(matched,accepted,tokens,None,True)
    fallback = int(target_ids[matched])
    tokens.append(fallback)
    return Acceptance(matched,accepted,tokens,fallback,
                      fallback == eos or len(tokens) == remaining)
