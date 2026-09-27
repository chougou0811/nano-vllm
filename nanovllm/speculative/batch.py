"""Host-only records for concurrent speculative transactions."""
from dataclasses import dataclass, field
from enum import Enum, auto


class SpeculativePhase(Enum):
    PREFILL = auto()
    READY = auto()
    PROPOSING = auto()
    RESERVED = auto()
    VERIFIED = auto()
    COMMITTING = auto()
    FINISHED = auto()
    FAILED = auto()
    CLOSED = auto()


@dataclass
class KVBlockTransaction:
    seq_id: int
    generation: int
    start_row: int
    tentative_end: int
    original_block_count: int
    new_block_ids: tuple[int, ...] = ()
    active: bool = True
    committed_end: int | None = None
    released_block_ids: tuple[int, ...] = ()

    def validate_owner(self, seq_id: int, generation: int):
        if (seq_id, generation) != (self.seq_id, self.generation):
            raise RuntimeError("Foreign KV transaction owner")


@dataclass
class TargetRequestState:
    seq_id: int
    generation: int
    cursor: int = 0
    token_count: int = 0
    feature_rows: int = 0
    tentative_rows: int = 0
    phase: SpeculativePhase = SpeculativePhase.PREFILL

    def validate(self):
        if min(self.cursor, self.token_count, self.feature_rows, self.tentative_rows) < 0:
            raise RuntimeError("Negative target state")
        if self.feature_rows != self.cursor:
            raise RuntimeError("Target feature/cursor mismatch")
        if self.phase == SpeculativePhase.READY and self.token_count != self.cursor + 1:
            raise RuntimeError("READY target must have one pending token")
        if self.phase != SpeculativePhase.VERIFIED and self.tentative_rows:
            raise RuntimeError("Tentative target rows outside VERIFIED phase")


@dataclass
class SpeculativeRequestState:
    seq_id: int
    generation: int
    fixed_k: int
    phase: SpeculativePhase = SpeculativePhase.PREFILL
    draft_cursor: int = 0
    target_cursor: int = 0
    proposed_token_ids: tuple[int, ...] = ()
    accepted_length: int | None = None
    committed_token_ids: tuple[int, ...] = ()
    fallback_token: int | None = None
    transaction: KVBlockTransaction | None = None
    clipping_reason: str | None = None
    finished_reason: str | None = None
    error: str | None = None
    closed: bool = False

    def __post_init__(self):
        if self.fixed_k < 1:
            raise ValueError("fixed_k must be positive")

    def require(self, *phases: SpeculativePhase):
        if self.closed or self.phase not in phases:
            names = ", ".join(p.name for p in phases)
            raise RuntimeError(f"Request {self.seq_id} expected phase {names}, got {self.phase.name}")

    def transition(self, expected: SpeculativePhase, target: SpeculativePhase):
        self.require(expected)
        self.phase = target

    def validate_owner(self, seq_id: int, generation: int):
        if (seq_id, generation) != (self.seq_id, self.generation):
            raise RuntimeError("Foreign speculative request owner")

    def fail(self, error: BaseException | str):
        if not self.closed:
            self.error = str(error)
            self.phase = SpeculativePhase.FAILED

    def close(self):
        if self.closed:
            return False
        if self.transaction is not None and self.transaction.active:
            raise RuntimeError("Cannot close an active KV transaction")
        self.proposed_token_ids = ()
        self.committed_token_ids = ()
        self.transaction = None
        self.phase = SpeculativePhase.CLOSED
        self.closed = True
        return True


@dataclass(frozen=True)
class ProposalEntry:
    seq_id: int
    generation: int
    old_cursor: int
    draft_cursor: int
    remaining: int
    requested_k: int
    actual_k: int
    proposed_token_ids: tuple[int, ...]
    q_offset: int
    clipping_reason: str | None = None

    def __post_init__(self):
        if self.old_cursor < 0 or not 0 <= self.draft_cursor <= self.old_cursor:
            raise ValueError("Invalid draft/target cursor")
        if self.remaining < 1 or self.requested_k < 1:
            raise ValueError("Invalid output budget or requested K")
        if not 0 <= self.actual_k <= self.requested_k:
            raise ValueError("Invalid actual K")
        if self.actual_k != len(self.proposed_token_ids):
            raise ValueError("Proposal length does not match actual K")
        if self.actual_k > self.remaining - 1 or self.q_offset < 0:
            raise ValueError("Proposal exceeds remaining output budget")

    @property
    def q_length(self):
        return 1 + self.actual_k

    @property
    def q_end(self):
        return self.q_offset + self.q_length


@dataclass(frozen=True)
class SpeculativeBatch:
    entries: tuple[ProposalEntry, ...]
    max_num_batched_tokens: int
    ordered_seq_ids: tuple[int, ...] = field(init=False)
    proposal_offsets: tuple[int, ...] = field(init=False)
    total_query_tokens: int = field(init=False)

    def __post_init__(self):
        if not self.entries:
            raise ValueError("Speculative batch cannot be empty")
        if self.max_num_batched_tokens < 1:
            raise ValueError("Invalid target token budget")
        ids = tuple(entry.seq_id for entry in self.entries)
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate request ID in speculative batch")
        offsets = tuple(entry.q_offset for entry in self.entries)
        expected = 0
        for entry in self.entries:
            if entry.q_offset != expected:
                raise ValueError("Non-contiguous proposal offsets")
            expected = entry.q_end
        if expected > self.max_num_batched_tokens:
            raise ValueError("Speculative target token budget exceeded")
        object.__setattr__(self, "ordered_seq_ids", ids)
        object.__setattr__(self, "proposal_offsets", offsets + (expected,))
        object.__setattr__(self, "total_query_tokens", expected)

    def bounds(self, seq_id: int):
        try:
            index = self.ordered_seq_ids.index(seq_id)
        except ValueError as error:
            raise KeyError(seq_id) from error
        return self.proposal_offsets[index], self.proposal_offsets[index + 1]


@dataclass(frozen=True)
class SpeculativeCommit:
    seq_id: int
    generation: int
    old_cursor: int
    new_cursor: int
    proposed_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    accepted_length: int
    committed_token_ids: tuple[int, ...]
    fallback_token: int | None
    finished: bool
    clipping_reason: str | None
    requested_k: int
    actual_k: int

    @property
    def kept_target_rows(self):
        return self.new_cursor - self.old_cursor
