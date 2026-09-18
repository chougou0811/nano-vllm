"""Independent, control-calibrated scheduler numerical acceptance."""
import argparse
import atexit
import copy
import json
from pathlib import Path
import sys

from .report import capture_manifest, write_json


POLICIES = ["original", "original-repeat", "original-chunk256", "original-chunk512",
            "static", "slo-v1", "slo-v2"]
RULES = dict(tv_floor=.01, relative_l2_floor=.01, control_multiplier=2.,
             hard_tv=.10, hard_relative_l2=.10, top_reference_candidates=10,
             fp32_control_max_abs=.001, fp32_control_tv=.0001,
             note="Fixed before V2 GPU results. Original full/chunk controls calibrate each position; candidate data never sets thresholds.")
PROBABILITY_RULES = dict(tv_floor=.01, weighted_rms_floor=.05, control_multiplier=2.,
    hard_tv=.10, hard_weighted_rms=.30, top_reference_candidates=10,
    fp32_control_max_abs=.001, fp32_control_tv=.0001,
    note="Revision 2 changes the metric, not a raw-logit tolerance: reference-probability-weighted, offset-invariant logit RMS (nats). Freeze before independent holdout capture. Initial unweighted-L2 failure retained.")


def inputs(tokenizer, corpus="legacy"):
    cases = []
    if corpus == "holdout":
        for i, length in enumerate([31,127,255,256,257,511,512,1023,1791,2044]):
            text = f"Audit sample {i}: explain the tradeoffs of this storage system. " + " ".join(
                f"Worker {j} recorded queue size {(j*7+i)%53}, copied {j*11+3} bytes, then released its reservation."
                for j in range(256))
            tokens = tokenizer.encode(text)
            cases.append(dict(kind="holdout", tokens=tokens[:length]))
        return cases
    repeated = tokenizer.encode("The quick brown fox jumps over the lazy dog. A language model predicts the next token.")
    for kind, lengths in [("repetitive", [31,255,256,257,511,512]),
                          ("varied", [31,255,256,257,511,512,1536,2044])]:
        for i, length in enumerate(lengths):
            tokens = repeated if kind == "repetitive" else tokenizer.encode(
                f"Case {i}: summarize these measurements. " + " ".join(
                    f"Request {j} consumed {j*17+i} tokens and completed after {j*31+7} milliseconds."
                    for j in range(256)))
            cases.append(dict(kind=kind, tokens=(tokens*(length//len(tokens)+1))[:length]))
    return cases


def capture(engine, cases, policy, overrides=None):
    import torch
    from nanovllm import SamplingParams
    from nanovllm.engine.policy_scheduler import make_scheduler
    cfg = copy.copy(engine.model_runner.config)
    cfg.scheduler_policy = "original" if policy.startswith("original") else policy
    for name, value in (overrides or {}).items():
        setattr(cfg, name, value)
    if "chunk" in policy:
        cfg.max_num_batched_tokens = int(policy.split("chunk")[1])
    if "batch" in policy:
        cfg.max_num_seqs = int(policy.split("batch")[1])
    scheduler = make_scheduler(cfg)
    current, logical, records = {}, {}, {}

    class Capture:
        def __getattr__(self, name):
            return getattr(scheduler, name)

        def schedule(self):
            value = scheduler.schedule()
            current["seqs"], current["prefill"] = value
            return value

    engine.scheduler = Capture()
    requests = []
    for i, case in enumerate(cases):
        engine.add_request(case["tokens"], SamplingParams(max_tokens=4, ignore_eos=True))
        request = scheduler.waiting[-1]
        requests.append(request)
        logical[request.seq_id] = i
    sampler = engine.model_runner.sampler
    old = sampler.forward
    local = "forward" in vars(sampler)

    def teacher(logits, temperatures):
        tokens = []
        for row, request in enumerate(current["seqs"]):
            count = request.num_completion_tokens
            if not current["prefill"] or request.num_cached_tokens+request.num_scheduled_tokens == request.num_tokens:
                key = logical[request.seq_id], count
                assert key not in records
                records[key] = logits[row].float().detach().cpu()
            tokens.append([7,13,19,23][count])
        return torch.tensor(tokens, device=logits.device, dtype=torch.long)

    sampler.forward = teacher
    try:
        while not engine.is_finished():
            engine.step()
    finally:
        if local:
            sampler.forward = old
        else:
            del sampler.forward
        engine.scheduler = scheduler
    assert len(records) == len(cases)*4
    assert all(r.completion_token_ids == [7,13,19,23] for r in requests)
    assert all(torch.isfinite(v).all() for v in records.values())
    assert not scheduler.block_manager.used_block_ids
    assert all(b.ref_count == 0 for b in scheduler.block_manager.blocks)
    return records


def capture_phase(args, output):
    import torch
    from nanovllm import LLM
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"acceptance-rules.json", RULES)
    write_json(output/"probability-rules.json", PROBABILITY_RULES)
    cfg = dict(tensor_parallel_size=args.tensor_parallel_size, enforce_eager=True,
               max_model_len=2048, max_num_batched_tokens=2048, max_num_seqs=4,
               gpu_memory_utilization=.85)
    manifest = capture_manifest(args, cfg, [], output, sys.argv[1:])
    manifest["command"] = manifest["command"].replace("-m benchmarks.serving ", "-m benchmarks.serving.scheduler_v2_correctness ", 1)
    engine = None
    try:
        engine = LLM(args.model, **cfg)
        cases = inputs(engine.tokenizer, args.corpus)
        write_json(output/"inputs.json", cases)
        policies = POLICIES[:4] + ([f"original-batch{i}" for i in [1,2,3]] if args.shape_controls else []) + POLICIES[4:]
        manifest["captured_policies"] = policies
        for policy in policies:
            print("Capture", policy, flush=True)
            torch.save(capture(engine, cases, policy), output/f"{policy}.pt")
        manifest["capture_complete"] = True
    finally:
        if engine:
            atexit.unregister(engine.exit)
            engine.exit()
            manifest["normal_exit"] = all(p.exitcode == 0 for p in engine.ps)
        write_json(output/"manifest.json", manifest)


def reference_phase(args, output):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = json.loads((output/"inputs.json").read_text())
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    small = args.high_precision
    devices = {"": 0} if small else {"model.embed_tokens": 0, "model.rotary_emb": 0,
                                    "model.norm": 1, "lm_head": 1}
    if not small:
        devices.update({f"model.layers.{i}": int(i >= cfg.num_hidden_layers//2) for i in range(cfg.num_hidden_layers)})
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        dtype=torch.float32 if small else torch.bfloat16, attn_implementation="eager", device_map=devices).eval()
    reference, chunked = {}, {}
    with torch.inference_mode():
        for i, case in enumerate(cases):
            ids = torch.tensor([case["tokens"]+[7,13,19]], device="cuda:0")
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=4).logits[0].float().cpu()
            for j in range(4):
                reference[i,j] = logits[j]
            if small:
                past = None
                end = len(case["tokens"])-1
                for start in range(0, end, 256):
                    result = model(input_ids=ids[:, start:min(start+256, end)],
                                   past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = result.past_key_values
                logits = model(input_ids=ids[:, end:], past_key_values=past,
                               use_cache=True, logits_to_keep=4).logits[0].float().cpu()
                for j in range(4):
                    chunked[i,j] = logits[j]
            print("HF reference", i, len(case["tokens"]), flush=True)
    torch.save(reference, output/"hf.pt")
    if small:
        torch.save(chunked, output/"hf-fp32-chunk.pt")
    write_json(output/"hf-manifest.json", dict(dtype=str(model.dtype), device_map=devices,
        attention="HF eager", tf32=False, command=sys.argv, high_precision_control=small))


def scores(actual, reference):
    a, r = actual.float(), reference.float()
    ac, rc = a-a.mean(), r-r.mean()
    weights = r.softmax(-1)
    residual = a-r
    residual = residual-(weights*residual).sum()
    return dict(max_abs=(a-r).abs().max().item(),
                tv=(a.softmax(-1)-r.softmax(-1)).abs().sum().item()/2,
                weighted_rms=(weights*residual.square()).sum().sqrt().item(),
                relative_l2=((ac-rc).norm()/rc.norm().clamp_min(1e-12)).item(),
                top1=int(a.argmax()), reference_top1=int(r.argmax()))


def compare_phase(output, probability=False):
    import torch
    rules = json.loads((output/("probability-rules.json" if probability else "acceptance-rules.json")).read_text())
    result_path = output/("acceptance-probability.json" if probability else "acceptance.json")
    metrics = ["tv", "weighted_rms" if probability else "relative_l2"]
    manifest = json.loads((output/"manifest.json").read_text()) if (output/"manifest.json").exists() else {}
    policies = manifest.get("captured_policies", POLICIES)
    tensors = {p: torch.load(output/f"{p}.pt", map_location="cpu", weights_only=True) for p in policies}
    reference = torch.load(output/"hf.pt", map_location="cpu", weights_only=True)
    finite = all(torch.isfinite(v).all().item() for d in [*tensors.values(), reference] for v in d.values())
    coverage = bool(reference) and all(set(d) == set(reference) for d in tensors.values())
    if not finite or not coverage:
        write_json(result_path, dict(passed=False, finite=finite, coverage=coverage))
        return 1
    parity = all(torch.equal(v, tensors["original-repeat"][k]) for k,v in tensors["original"].items())
    rows, failures = [], []
    control_names = ["original", "original-chunk256", "original-chunk512"]
    control_names += [p for p in policies if p.startswith("original-batch")]
    for key, ref in reference.items():
        controls = [scores(tensors[p][key], ref) for p in control_names]
        limits = {metric: max(rules[metric+"_floor"], rules["control_multiplier"]*max(c[metric] for c in controls))
                  for metric in metrics}
        control_valid = all(c[metric] <= rules["hard_"+metric] for c in controls for metric in limits)
        candidates = ref.topk(rules["top_reference_candidates"]).indices
        rc = ref-ref.mean()
        margin = max((tensors[p][key]-tensors[p][key].mean()-rc)[candidates].abs().max().item() for p in control_names)*2
        # Two BF16 representable steps near the top logits, plus empirically
        # observed original shape variation, define the tie band.
        margin = max(margin, 2*torch.finfo(torch.bfloat16).eps*ref[candidates].abs().max().item())
        for policy in ["static", "slo-v1", "slo-v2"]:
            row = scores(tensors[policy][key], ref)
            gap = (ref.max()-ref[row["top1"]]).item()
            accepted = control_valid and all(row[m] <= min(limits[m], rules["hard_"+m]) for m in limits)
            accepted &= row["top1"] in candidates.tolist() and gap <= margin
            row.update(policy=policy, position=list(key), accepted=bool(accepted), limits=limits,
                       reference_gap=gap, tie_band=margin, controls=controls, control_valid=control_valid)
            rows.append(row)
            if not accepted:
                failures.append(row)
    high_precision = None
    if (output/"hf-fp32-chunk.pt").exists():
        chunked = torch.load(output/"hf-fp32-chunk.pt", weights_only=True)
        high_precision = [scores(chunked[k], r) for k,r in reference.items()]
        finite &= all(torch.isfinite(v).all().item() for v in chunked.values())
    hp_pass = high_precision is None or all(r["max_abs"] <= rules["fp32_control_max_abs"] and
              r["tv"] <= rules["fp32_control_tv"] for r in high_precision)
    passed = finite and parity and not failures and hp_pass
    result = dict(passed=passed, finite=finite, coverage=coverage, original_exact_parity=parity,
                  high_precision_pass=hp_pass, high_precision_control=high_precision,
                  rules=rules, rows=rows, failures=failures)
    write_json(result_path, result)
    print(json.dumps(dict(passed=passed, finite=finite, parity=parity, hp_pass=hp_pass, failures=len(failures))), flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["capture", "reference", "compare", "declare-probability"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-manifest")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--high-precision", action="store_true")
    parser.add_argument("--probability-metrics", action="store_true")
    parser.add_argument("--corpus", choices=["legacy", "holdout"], default="legacy")
    parser.add_argument("--shape-controls", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if args.phase == "declare-probability":
        if (output/"probability-rules.json").exists():
            raise ValueError("Rule declaration already exists")
        write_json(output/"probability-rules.json", PROBABILITY_RULES)
    elif args.phase == "capture":
        capture_phase(args, output)
    elif args.phase == "reference":
        reference_phase(args, output)
    else:
        return compare_phase(output, args.probability_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
