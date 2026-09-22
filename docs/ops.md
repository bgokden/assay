# Running long jobs on the training machine

Notes on the one machine the models were trained on (Intel Core Ultra 9 285K, 24 cores,
one RTX 5090, driver 595.84, torch 2.11 + cu128). Two things bit repeatedly; both are
handled by the scripts in `scripts/`, and this page records why.

## The "sporadic segfaults" are one CPU core

Long GPU jobs (QLoRA on the 27B, evaluations, plain-SDPA encoder training) died with a
native `Segmentation fault` a few times per hour, surfacing at CUDA sync points such as
`loss.item()` with no Python traceback. They were first blamed on Triton (linear-attention)
and bitsandbytes 4-bit kernels; that was wrong. The kernel log tells the story:

```
journalctl -k --since "2026-09-21 00:00" | grep "segfault at .* in libcuda"
```

Every entry, 17 of them over two days, has the same shape:

```
python3[...]: segfault at <varies> ip ...15d58 sp ... error 4 in libcuda.so.595.84[415d58,...] likely on CPU 6 (core 40)
```

Same instruction offset inside `libcuda.so` (`0x415d58`), a different faulting address each
time, and **always CPU 6**. No segfault in any other library, none on any other core, on a
24-core machine with no pinning. A software bug would land on random cores; a single core
mis-executing one hot instruction is what this looks like. CPU 6 is one of the two favored
(highest-boost, 5.8 GHz) cores of this part, which fits a marginal top-frequency operating
point rather than a dead core.

Since GPU jobs were pinned off that core (2026-09-22 00:40) there have been zero faults,
across about eight hours of continuous training and evaluation.

### Mitigation in place

- systemd units: `systemd-run --user -p CPUAffinity=0-5,7-23 ...` (the queue scripts'
  headers show the exact command). The user manager has no `cpuset` controller delegated,
  so `AllowedCPUs=` is not available; `CPUAffinity=` (sched_setaffinity) is enough.
- Direct commands: `taskset -c 0-5,7-23 uv run ...`.
- Default for all user services: `~/.config/systemd/user.conf.d/cpu6.conf` sets
  `CPUAffinity=0-5 7-23` under `[Manager]`; it applies after the next login or
  `systemctl --user daemon-reexec`.
- Every trainer checkpoints and resumes (`--checkpoint-every`, `--resume` is the default in
  `scripts/run_experiment.sh`), and `scripts/stage.sh` retries a stage up to five times.

### To confirm and fix at the source

1. Reproduce on the core alone: `stress-ng --cpu 1 --taskset 6 --cpu-method matrixprod
   --verify -t 30m` (add `--taskset 7` in a second run as the control; both are 5.8 GHz
   favored cores). A `--verify` failure or a crash on 6 and not on 7 settles it.
2. Check the log for new entries with the command above; any fault on a core other than 6
   would mean the diagnosis is incomplete.
3. Fix options, in order of preference: lower the per-core boost or raise the voltage
   offset for core 6 in firmware (this is usually a favored-core V/F margin problem); take
   the core offline at boot (`echo 0 > /sys/devices/system/cpu/cpu6/online` from a
   `systemd` unit, or `isolcpus=6` on the kernel command line to keep unpinned work off
   it); or disable the core in firmware. Keep the affinity pin until one of these is done.

## Foreground tool timeouts kill background jobs

Processes started with `nohup ... &` from an interactive tool session were killed the
moment a foreground command in that session hit its timeout. Anything longer than a few
minutes runs as a transient systemd user unit instead (`systemd-run --user --unit <name>
-p WorkingDirectory=$PWD ...`), checked with `systemctl --user is-active <name>` and the
unit's own log file. `scripts/night_*.sh` are examples: each is one unit, waits for the
previous one, and runs its stages through `scripts/stage.sh`.

## Stall detection

A hung CUDA process (spinning main thread, GPU at 0 %) produces no error and no exit.
`scripts/stage.sh` runs every stage in its own process group and kills it when its log has
not grown for `STALL_SECONDS` (default 900), then retries; trainers resume from their last
checkpoint. Evaluations of the 27B at batch 4 can be quiet for several minutes, which is
why the default is not shorter.

## Disk

The Hugging Face cache shares blobs across repositories, so per-directory sizes mislead;
use `hf cache scan` and delete revisions with `huggingface_hub.scan_cache_dir().delete_revisions`.
Trainer checkpoints (`checkpoint.pt`, `checkpoint/`) hold optimizer state and are 1.8 GB for
the encoder tier and 0.4 GB for a LoRA decoder run; delete them once a run is evaluated.
