from dataclasses import asdict, dataclass
import random


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    prompt_token_ids: list[int]
    output_length: int
    arrival_ns: int


def deterministic_workload(token_ids, num_requests, prompt_length, output_length,
                           interval_ns, seed):
    if min(num_requests, prompt_length, output_length) < 1 or interval_ns < 0:
        raise ValueError("Positive request/length counts and nonnegative interval required")
    vocabulary = sorted(set(token_ids))
    if not vocabulary:
        raise ValueError("No valid non-special vocabulary tokens")
    rng = random.Random(seed)
    first_tokens = rng.sample(vocabulary, min(num_requests, len(vocabulary)))
    return [RequestSpec(i, [first_tokens[i % len(first_tokens)]] +
                        rng.choices(vocabulary, k=prompt_length - 1),
                        output_length, i * interval_ns) for i in range(num_requests)]


def workload_data(specs):
    return [asdict(spec) for spec in specs]


class IsolatedWorkloads:
    """Unique first tokens prevent cross-request full-prefix equality, not hashes."""

    def __init__(self, vocabulary):
        self.vocabulary = sorted(set(vocabulary))
        self.used_first_tokens = {0}  # Original constructor warmup uses zero tokens.

    def build(self, num_requests, prompt_length, output_length, interval_ns, seed):
        specs = deterministic_workload(self.vocabulary, num_requests, prompt_length,
                                       output_length, interval_ns, seed)
        available = [t for t in self.vocabulary if t not in self.used_first_tokens]
        if len(available) < num_requests:
            raise ValueError("Vocabulary exhausted for prefix-isolated requests")
        rng = random.Random(seed)
        rng.shuffle(available)
        for spec in specs:
            first = spec.prompt_token_ids[0]
            if first in self.used_first_tokens:
                while available[-1] in self.used_first_tokens:
                    available.pop()
                first = available.pop()
                spec.prompt_token_ids[0] = first
            self.used_first_tokens.add(first)
        return specs
