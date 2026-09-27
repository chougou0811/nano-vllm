"""CPU-only observed reuse statistics and explicitly hypothetical projections."""
import argparse
from collections import Counter, defaultdict
from dataclasses import fields
import csv
import json
import math
from pathlib import Path
import statistics
from bisect import bisect_left

from nanovllm.speculative.graph_policy import BoundedPolicy, GraphCacheConfig, GraphKey
from benchmarks.serving.eagle3_phase44c import dump


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0, mean=None, p50=None, p95=None, p99=None, max=None)
    def q(p):
        x = (len(values)-1)*p
        a,b = math.floor(x),math.ceil(x)
        return values[a]+(values[b]-values[a])*(x-a)
    return dict(count=len(values),mean=statistics.mean(values),p50=q(.5),p95=q(.95),p99=q(.99),max=values[-1])


def bounded(events, capacity):
    policy = BoundedPolicy(GraphCacheConfig(max_graph_entries=capacity))
    hits, evictions = 0, 0
    lifetimes, live = [], {}
    for event in events:
        key = event["key_id"]
        policy.observe(key)
        if key is not None and key in policy.lru:
            hits += 1
            live[key]["replays"] += 1
            policy.touch(key)
        elif policy.should_capture(key):
            victim = policy.victim()
            if victim is not None:
                policy.lru.pop(victim)
                live.pop(victim)
                evictions += 1
            policy.insert(key)
            entry = dict(key_id=key,replays=0,created=event["index"])
            live[key] = entry
            lifetimes.append(entry)
    return dict(capacity=capacity,hits=hits,hit_rate=hits/len(events) if events else 0,
        captures=policy.captures,evictions=evictions,
        lifetimes_reaching_14=sum(x["replays"]>=14 for x in lifetimes),lifetimes=lifetimes)


def ideal_lru_hits(events, capacity):
    # Zero-cost always-admit LRU reference, not an upper bound across different
    # admission policies and not an implemented serving capture policy.
    from collections import OrderedDict
    cache, hits = OrderedDict(), 0
    for event in events:
        key = event["key_id"]
        if key is None:
            continue
        if key in cache:
            hits += 1
            del cache[key]
        cache[key] = True
        if len(cache)>capacity:
            cache.popitem(last=False)
    return hits


def cost_model(path):
    old = json.loads(Path(path).read_text())
    rows = [r for r in old["lifetime_amortization"] if r["system"]=="second4"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["key_id"]].append(row)
    exact = {}
    for key, entries in grouped.items():
        savings = [r["per_hit_saved_ms"] for r in entries
                   if r.get("per_hit_saved_ms") is not None]
        if savings:
            exact[key] = dict(capture_ms=statistics.median(r["capture_ms"] for r in entries),
                saving_ms=statistics.median(savings), provenance="4.4B_same_exact_key_transfer",
                samples=len(entries),saving_samples=len(savings))
    by_m = {}
    for m in (4,8,16):
        entries = [r for r in rows if r["key"]["rows"]==m]
        savings = [r["per_hit_saved_ms"] for r in entries
                   if r.get("per_hit_saved_ms") is not None]
        by_m[m] = dict(capture_ms=statistics.median(r["capture_ms"] for r in entries) if entries else None,
            saving_ms=statistics.median(savings) if savings else None,
            provenance="4.4B_same_M_transfer" if savings else "unknown_replay_saving",
            samples=len(entries),saving_samples=len(savings))
    return dict(exact=exact,by_m=by_m,source=str(path))


def key_records(events):
    groups = defaultdict(list)
    for event in events:
        if event["key_id"] is not None:
            groups[event["key_id"]].append(event)
    records = []
    all_ids = [event["key_id"] for event in events]
    for key, occurrences in groups.items():
        indices = [event["index"] for event in occurrences]
        distances = [b-a for a,b in zip(indices,indices[1:])]
        distinct = [len(set(all_ids[a+1:b])-{None}) for a,b in zip(indices,indices[1:])]
        records.append(dict(key_id=key,key=occurrences[0]["key"],count=len(indices),
            first_index=indices[0],second_index=indices[1] if len(indices)>1 else None,
            first_seen_during_drain=occurrences[0].get("drain",False),
            later_indices=indices[2:],occurrence_indices=indices,reuse_distances=distances,
            distinct_key_reuse_distances=distinct,lifetime_verifications=indices[-1]-indices[0],
            lifetime_completed_requests=occurrences[-1]["completed_before"]-occurrences[0]["completed_before"],
            replay_opportunities=max(0,len(indices)-2),
            context_tuples=sorted(set(tuple(e["context_tuple"]) for e in occurrences)),
            wave_tuples=sorted(set(tuple(e["waves"]) for e in occurrences)),
            admission_wave_tuples=sorted(set(tuple(e.get("admission_waves",e["waves"])) for e in occurrences)),
            request_ids=sorted(set(i for e in occurrences for i in e["request_ids"]))))
    return records


def analyze(events, model):
    records = key_records(events)
    total = len(events)
    eligible = sum(r["count"] for r in records)
    unique = len(records)
    denominator = max(1,total)
    participating_requests = {i for e in events for i in e["request_ids"]}
    reused = sum(max(0,r["count"]-1) for r in records)
    fields_without = {}
    names = [f.name for f in fields(GraphKey)]
    for omit in names:
        fields_without[omit] = len({json.dumps({k:v for k,v in r["key"].items() if k!=omit},sort_keys=True)
                                    for r in records})
    joint_without = len({json.dumps({k:v for k,v in r["key"].items() if k not in ("max_k","table_width")},sort_keys=True)
                         for r in records})
    threshold = {}
    for n in (6,14,18):
        profitable = [r for r in records if r["replay_opportunities"]>=n]
        ids = {r["key_id"] for r in profitable}
        covered_requests = {i for r in profitable for i in r["request_ids"]}
        threshold[n] = dict(required_replays=n,keys=len(profitable),
            verifications=sum(r["count"] for r in profitable),
            verification_coverage=sum(r["count"] for r in profitable)/denominator,
            replay_opportunities=sum(r["replay_opportunities"] for r in profitable),
            hypothetical_hit_rate=sum(r["replay_opportunities"] for r in profitable)/denominator,
            requests=len(covered_requests),request_ids=sorted(covered_requests))
        threshold[n]["participating_request_coverage"] = len(covered_requests)/max(1,len(participating_requests))
    net_all = net_profitable = saving_gross = capture_total = 0.
    known_keys = profitable_keys = known_events = profitable_events = profitable_hits = 0
    profitable_requests = set()
    provenances = Counter()
    for row in records:
        estimate = model["exact"].get(row["key_id"],model["by_m"][row["key"]["rows"]])
        row["cost_estimate"] = estimate
        if estimate["saving_ms"] is None:
            row.update(break_even_replays=None,projected_net_ms=None)
            continue
        capture, saving = estimate["capture_ms"],estimate["saving_ms"]
        n = math.ceil(capture/saving) if saving>0 else None
        net = row["replay_opportunities"]*saving-capture if row["count"]>=2 else 0
        row.update(break_even_replays=n,projected_net_ms=net)
        known_keys += 1
        known_events += row["count"]
        provenances[estimate["provenance"]] += 1
        net_all += net
        if row["count"]>=2:
            capture_total += capture
            saving_gross += row["replay_opportunities"]*saving
        if n is not None and row["replay_opportunities"]>=n:
            profitable_keys += 1
            profitable_events += row["count"]
            profitable_hits += row["replay_opportunities"]
            profitable_requests.update(row["request_ids"])
            net_profitable += net
    return dict(verifications=total,eligible=eligible,unique_keys=unique,
        first_seen_during_drain_keys=sum(r["first_seen_during_drain"] for r in records),
        drain_verifications=sum(e.get("drain",False) for e in events),
        participating_requests=len(participating_requests),
        occurrence_histogram=dict(sorted(Counter(r["count"] for r in records).items())),
        keys_at_least={n:sum(r["count"]>=n for r in records) for n in (2,3,5,10,20)},
        occurrence_le_2_keys=sum(r["count"]<=2 for r in records),
        reusable_keys=sum(r["count"]>=2 for r in records),
        unbounded_first_capture_reuse_hits=reused,
        unbounded_first_capture_hit_rate=reused/denominator,
        second_capture_replay_opportunities=sum(r["replay_opportunities"] for r in records),
        second_capture_hit_rate=sum(r["replay_opportunities"] for r in records)/denominator,
        key_explosion_rate=unique/max(1,eligible),
        keys_crossing_nominal_waves=sum(len(r["wave_tuples"])>1 for r in records),
        keys_crossing_admission_waves=sum(len(r["admission_wave_tuples"])>1 for r in records),
        exact_context_tuples=len({tuple(e["context_tuple"]) for e in events if e["key_id"] is not None}),
        new_keys_by_first_completed_window=dict(Counter(
            "0-31" if events[r["first_index"]]["completed_before"]<32 else
            "32-127" if events[r["first_index"]]["completed_before"]<128 else
            "128-383" if events[r["first_index"]]["completed_before"]<384 else "384-511"
            for r in records)),
        reuse_distance=distribution([d for r in records for d in r["reuse_distances"]]),
        distinct_key_reuse_distance=distribution([d for r in records for d in r["distinct_key_reuse_distances"]]),
        field_ablation_cardinality=fields_without,without_max_k_and_table_width=joint_without,
        threshold_scenarios=threshold,
        empirical_transfer_projection=dict(known_keys=known_keys,unknown_keys=unique-known_keys,
            scope="known_cost_keys_only; unknown keys are not zero-saving keys",
            full_all_second_capture_net_ms=net_all if known_keys==unique else None,
            known_verifications=known_events,provenance_counts=dict(provenances),
            profitable_keys=profitable_keys,profitable_verifications=profitable_events,
            profitable_verification_coverage=profitable_events/denominator,
            profitable_requests=len(profitable_requests),profitable_request_ids=sorted(profitable_requests),
            participating_request_coverage=len(profitable_requests)/max(1,len(participating_requests)),
            profitable_hit_rate=profitable_hits/denominator,
            gross_saving_ms=saving_gross,capture_cost_ms=capture_total,
            all_second_capture_net_ms=net_all,hindsight_profitable_only_net_ms=net_profitable),
        bounded=[bounded(events,c) for c in (2,4,8)],
        ideal_lru_hits={c:ideal_lru_hits(events,c) for c in (2,4,8,32,128,512,2048)},
        records=records)


def curve(events):
    counts, reusable, rows = Counter(),0,[]
    for event in events:
        key = event["key_id"]
        if key is not None:
            counts[key] += 1
            reusable += counts[key]==2
        rows.append(dict(completed_requests=event["completed_before"],verification=event["index"],
                         unique_keys=len(counts),reusable_keys=reusable))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prior",type=Path,default=Path("benchmarks/eagle3-phase4_4b/summary.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    model = cost_model(args.prior)
    results, triggers = [], []
    from benchmarks.serving.eagle3_phase44c import ROOT
    before = json.loads((args.raw/"before.json").read_text())
    frozen_changes = [p for p,h in before.items() if "phase44c" not in p and "PHASE4_4C" not in p
        and (not (ROOT/p).exists() or __import__("hashlib").sha256((ROOT/p).read_bytes()).hexdigest()!=h)]
    assert not frozen_changes, frozen_changes
    for manifest_path in sorted(args.raw.glob("shadow-c*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        for path in manifest["trials"]:
            run = json.loads(Path(path).read_text())
            assert run["all_rank_close_zero"] and not any(run["summary"]["cleanup"].values())
            finishes = sorted(r["finish_ns"] for r in run["requests"])
            admissions = {r["request_id"]:bisect_left(finishes,r["arrival_ns"]) for r in run["requests"]}
            for event in run["events"]:
                event["admission_waves"] = [admissions[i] for i in event["request_ids"]]
                event["drain"] = event["completed_before"] >= run["request_count"]-run["concurrency"]
                if event["key"] is not None:
                    assert event["key"]["dtype"]=="torch.bfloat16"
                    assert event["key"]["tp_size"]==2
            cell = dict(concurrency=run["concurrency"],workload=run["workload"],
                normal_exit=manifest["normal_exit"],observer_parity=manifest["observer_parity"],
                source=str(path),summary=run["summary"],observer_ns=run["observer_ns"],horizons={})
            stem = f'c{run["concurrency"]}-{run["workload"]}'
            for horizon,milestone in run["milestones"].items():
                events = run["events"][:milestone["verifications"]]
                stats = analyze(events,model)
                if horizon=="512":
                    dump(args.output/f"keys/{stem}.json",stats.pop("records"))
                else:
                    stats.pop("records")
                stats["window"] = milestone
                cell["horizons"][horizon] = stats
            with (args.output/f"{stem}-curve.csv").open("w") as file:
                rows = curve(run["events"])
                writer = csv.DictWriter(file,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            results.append(cell)
    for c in (1,2,4):
        candidates = [r for r in results if r["concurrency"]==c and "512" in r["horizons"]]
        candidates = [r for r in candidates if r["horizons"]["512"]["threshold_scenarios"][18]["keys"]>=5
            and r["horizons"]["512"]["threshold_scenarios"][18]["verification_coverage"]>=.10]
        if candidates:
            order = ("short-short","short-long","long-short","long-long","mixed-prompt","mixed-output")
            candidates.sort(key=lambda r:(-r["horizons"]["512"]["threshold_scenarios"][18]["verification_coverage"],order.index(r["workload"])))
            triggers.append(dict(concurrency=c,workload=candidates[0]["workload"]))
    result = dict(status="shadow_complete" if len(results)==18 and all(r["normal_exit"] for r in results) else "partial",
        observed_reuse_is_not_speedup=True,cost_model=model,cells=results,
        conditional_bounded_validation=triggers,frozen_file_changes=frozen_changes,outliers_removed=0)
    dump(args.output/"summary.json",result)
    print(json.dumps(dict(status=result["status"],cells=len(results),bounded=triggers)),flush=True)


if __name__ == "__main__":
    main()
