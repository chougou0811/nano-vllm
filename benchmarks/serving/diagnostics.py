"""Opt-in host diagnostics; perturbs timing and is never a performance trial."""
import gc
import os
import resource
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter_ns, process_time_ns

from .report import write_json


class Diagnostics:
    def __init__(self, engine, output):
        self.engine, self.output = engine, output
        self.events, self.samples, self.originals = [], [], []
        self.stop = Event()
        self.phase = "setup"
        self.step = -1

    def mark(self, phase):
        self.phase, self.step = phase, -1
        print(f"DIAGNOSTIC phase={phase} perf_ns={perf_counter_ns()} pid={os.getpid()}", file=sys.stderr, flush=True)

    def wrap(self, obj, name, label):
        original = getattr(obj, name)
        had_local = name in vars(obj)
        self.originals.append((obj, name, original, had_local))

        def measured(*args, **kwargs):
            if label == "step":
                self.step += 1
                print(f"DIAGNOSTIC step={self.step} phase={self.phase} start_ns={perf_counter_ns()}",
                      file=sys.stderr, flush=True)
            start, cpu = perf_counter_ns(), process_time_ns()
            usage = resource.getrusage(resource.RUSAGE_SELF) if label == "step" else None
            try:
                return original(*args, **kwargs)
            finally:
                event = dict(label=label, phase=self.phase, step=self.step,
                             start_ns=start, end_ns=perf_counter_ns(), cpu_ns=process_time_ns()-cpu)
                if usage is not None:
                    end = resource.getrusage(resource.RUSAGE_SELF)
                    event.update(voluntary_switches=end.ru_nvcsw-usage.ru_nvcsw,
                                 involuntary_switches=end.ru_nivcsw-usage.ru_nivcsw,
                                 major_faults=end.ru_majflt-usage.ru_majflt)
                self.events.append(event)
        setattr(obj, name, measured)

    def gc_event(self, phase, info):
        self.events.append(dict(label="gc", action=phase, info=dict(info),
                                phase=self.phase, step=self.step, time_ns=perf_counter_ns()))

    def telemetry(self):
        fields = "index,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate,memory.used"
        while not self.stop.is_set():
            start = perf_counter_ns()
            try:
                value = subprocess.check_output(["nvidia-smi", "--query-gpu=" + fields,
                          "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT, timeout=5)
            except (OSError, subprocess.SubprocessError) as error:
                value = repr(error)
            self.samples.append(dict(start_ns=start, end_ns=perf_counter_ns(), fields=fields, value=value))
            self.stop.wait(.2)

    def start(self):
        self.wrap(self.engine, "step", "step")
        runner = self.engine.model_runner
        for name in ["run", "run_model", "prepare_prefill", "prepare_decode", "prepare_sample"]:
            self.wrap(runner, name, name)
        self.wrap(runner.sampler, "forward", "sampler")
        gc.callbacks.append(self.gc_event)
        self.thread = Thread(target=self.telemetry, daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        gc.callbacks.remove(self.gc_event)
        for obj, name, original, had_local in reversed(self.originals):
            if had_local:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        write_json(self.output / "diagnostics.json", dict(
            scope="rank0 host spans; no extra CUDA synchronization; compile logs include both ranks",
            warning="Instrumented diagnostic run, not comparable serving performance",
            events=self.events, gpu_samples=self.samples))
