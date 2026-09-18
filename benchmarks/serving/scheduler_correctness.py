"""Test-only teacher forcing: compare committed-position logits across policies."""
import copy

from nanovllm.engine.policy_scheduler import make_scheduler


def verify_policy_logits(engine, output_dir=None, exploratory_numerics=False, corpus="repetitive"):
    import torch
    prompts = []
    text = engine.tokenizer.encode("The quick brown fox jumps over the lazy dog. A language model predicts the next token.")
    lengths = [31, 255, 256, 257, 511, 512]
    if corpus == "varied":
        lengths += [1536, 2044]
    for i, length in enumerate(lengths):
        if corpus == "varied":
            text = engine.tokenizer.encode(f"Case {i}: summarize these measurements. " + " ".join(
                f"Request {j} consumed {j*17+i} tokens and completed after {j*31+7} milliseconds."
                for j in range(256)))
        prompts.append((text * (length // len(text) + 1))[:length])
    if output_dir is not None:
        from .report import write_json
        write_json(output_dir/"correctness-inputs.json", dict(corpus=corpus, prompts=prompts,
                                                            forced_outputs=[7,13,19,23]))
    from nanovllm import SamplingParams
    results = {}
    reference = None
    failures=[]
    top1_failures=[]
    for policy in ["original", "original-chunk256", "static", "slo-aware", "slo-aware-chunk256"]:
        cfg = copy.copy(engine.model_runner.config)
        cfg.scheduler_policy = policy.removesuffix("-chunk256")
        if policy.endswith("-chunk256"):
            cfg.max_num_batched_tokens = 256
        original_scheduler = make_scheduler(cfg)
        current = {}

        class Capture:
            def __getattr__(self, name):
                return getattr(original_scheduler, name)

            def schedule(self):
                value = original_scheduler.schedule()
                current["seqs"], current["prefill"] = value
                return value

        engine.scheduler = Capture()
        logical = {}
        seqs = []
        for i, prompt in enumerate(prompts):
            engine.add_request(prompt, SamplingParams(max_tokens=4, ignore_eos=True))
            request = original_scheduler.waiting[-1]
            logical[request.seq_id] = i
            seqs.append(request)
        logits_by_position = {}
        sampler = engine.model_runner.sampler
        had_local = "forward" in vars(sampler)
        old_forward = sampler.forward

        def teacher(logits, temperatures):
            outputs = []
            for row, request in enumerate(current["seqs"]):
                count = request.num_completion_tokens
                valid = not current["prefill"] or request.num_cached_tokens + request.num_scheduled_tokens == request.num_tokens
                if valid:
                    logits_by_position[(logical[request.seq_id], count)] = logits[row].float().detach().cpu()
                outputs.append([7, 13, 19, 23][count])
            return torch.tensor(outputs, device=logits.device, dtype=torch.long)

        sampler.forward = teacher
        try:
            while not engine.is_finished():
                engine.step()
        finally:
            if had_local:
                sampler.forward = old_forward
            else:
                del sampler.forward
            engine.scheduler = original_scheduler
        if any(s.completion_token_ids != [7,13,19,23] for s in seqs):
            raise AssertionError("Committed teacher tokens differ")
        if original_scheduler.block_manager.used_block_ids:
            raise AssertionError("KV blocks leaked")
        if reference is None:
            reference = logits_by_position
        if set(logits_by_position) != set(reference):
            raise AssertionError("Missing committed logit positions")
        if not all(torch.isfinite(value).all() for value in logits_by_position.values()):
            raise AssertionError("Non-finite committed logits")
        differences = [(reference[key] - value).abs().max().item() for key,value in logits_by_position.items()]
        equal_top1 = sum(reference[key].argmax().item() == value.argmax().item() for key,value in logits_by_position.items())
        results[policy] = dict(positions=len(differences), max_abs_logit_error=max(differences),
                               top1_equal=equal_top1, committed_tokens_equal=True, kv_released=True)
        results[policy]["per_position"] = []
        for key,value in logits_by_position.items():
            ref=reference[key]
            delta=value-ref
            results[policy]["per_position"].append(dict(request=key[0],token=key[1],
                max_abs=delta.abs().max().item(),mean_abs=delta.abs().mean().item(),
                mean_shift=delta.mean().item(),top1_ref=ref.argmax().item(),top1=value.argmax().item(),
                probability_l1=(value.softmax(-1)-ref.softmax(-1)).abs().sum().item()))
        if output_dir is not None:
            from .report import write_json
            write_json(output_dir/"correctness-partial.json",results)
            torch.save(logits_by_position,output_dir/f"correctness-logits-{policy}.pt")
        # BF16 comparisons are numerical, not a bitwise identity assertion.
        if max(differences) > .5:
            failures.append((policy,max(differences)))
        if equal_top1 != len(differences):
            top1_failures.append((policy, equal_top1, len(differences)))
    evaluation = dict(reference="unchanged original policy, identical forced prefixes", policies=results,
                strict_logits_pass=not failures, strict_max_abs_tolerance=.5,
                strict_failures=failures, functional_top1_pass=not top1_failures,
                top1_failures=top1_failures,
                exploratory_numerics=exploratory_numerics,
                corpus=corpus,
                note="Test-only sampling replacement; normal benchmarks use unmodified sampler.",
                prompt_lengths=[len(p) for p in prompts], forced_outputs=[7,13,19,23])
    if output_dir is not None:
        write_json(output_dir/"correctness-evaluation.json",evaluation)
    if top1_failures:
        raise AssertionError(f"Top-1 mismatches: {top1_failures}")
    if failures and not exploratory_numerics:
        raise AssertionError(f"Logit mismatch beyond predeclared BF16 tolerance: {failures}")
    return evaluation
