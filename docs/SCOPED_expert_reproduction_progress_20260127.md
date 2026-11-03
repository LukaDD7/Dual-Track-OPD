# SCOPED Four-Expert Reproduction Progress — 2026-10-27

## Current progress snapshot — 2026-11-03 01:40 UTC

This is a read-only snapshot of the shared run records, metrics, bounded trainer
log tails, and checkpoint files. Earlier recovery instructions below are
historical and are not current launch commands. C3 and C4 were actively
writing metrics/logs at the snapshot; this does not constitute live host
PID/GPU-process inspection from the CPU control node.

| Expert | Capability | Observed completed steps | Latest validation reward | Complete retained checkpoints |
|---|---|---:|---|---|
| C1 | Fine-grained perception, counting, grounding | **370/370, complete** | **0.785316** at step 370; best **0.787561** at step 300 | 300 (best), 370 (last) |
| C2 | OCR, charts, tables, documents | **222/222, complete** | **0.772809** at step 222 (best and last) | 222 |
| C3 | Multimodal mathematical and symbolic reasoning | **214/222 (96.4%), running** | **0.616000** at step 200 | 200 (best and last) |
| C4 | Multi-image and spatial reasoning | **193/444 (43.5%), running** | **0.654000** at step 180 | 175 (last) |

### Observed results and health

- C1 and C2 have explicit `rl_capability1_complete` and
  `rl_capability2_complete` records, final validation metrics, and complete
  final checkpoints. Do not resume either expert.
- C3's active run is `capability3_20261102T122740Z_490809_22365`, restored on
  all four ranks from step 175. It has since recorded steps 176–214; step-214
  training reward is `0.696094` and gradient norm is `0.119599`. The latest
  observed step time is about 1,281 seconds, but this is not a stable ETA.
  Its latest metric was written at 01:36:56 UTC and its log was still updating
  at 01:39:50 UTC. Eight nominal optimizer steps remain, plus final
  validation/checkpoint handling if the run reaches step 222 normally.
- C4's active run is `capability4_20261102T122752Z_494488_3723`, restored on
  all four ranks from step 100. It has since recorded steps 101–193; step-193
  training reward is `0.651693` and gradient norm is `0.243716`. The latest
  observed step time is about 613 seconds. Its latest metric was written at
  01:34:47 UTC and its log was still updating at 01:39:53 UTC. It remains far
  from its nominal 444-step endpoint.
- C4's validation reward has increased at the observed checkpoints: `0.490`
  (20), `0.506` (40), `0.530` (60), `0.544` (80), `0.562` (100), `0.592`
  (120), `0.614` (140), `0.624` (160), and `0.654` (180). C3's observed
  validation sequence is `0.526` (50), `0.538` (100), `0.630` (150), and
  `0.616` (200): the latest point is below step 150. These are training-run
  validation rewards, not downstream benchmark scores or evidence that a
  different run will reproduce the same trajectory exactly.
- No `Traceback`, `RuntimeError`, `AssertionError`, OOM, `No space left`, or
  explicit NaN signature was found in the current C3/C4 trainer logs. Recent
  training reward, gradient norm, and step-time metrics are finite. These
  checks establish observed running health, not final model quality or a
  guarantee against later interruption.
- C3 tracker `best_val_reward_score=0.0` at an earlier step-175 save was not a
  validation result. The step-200 tracker now records the actual validation
  reward `0.616`; do not use the older placeholder as an outcome.
- The retained C1/C2/C3/C4 checkpoints listed above each have four model
  shards of 8,767,464,855 bytes and `dataloader.pt`. This is a structural
  check, not a full tensor/hash or downstream evaluation. Shared storage had
  about 272GB free (98% used); that is an observation, not a reservation.

### Reproduction and recovery caveats

- Upstream is `https://github.com/NEUIR/SCOPED.git` at
  `74e112e29e2be0657d1d521c1ecc3be53df99e44`; dataset is
  `xiongyq/SCOPED-Train` at
  `912185f76295f847c09bc44f379cf85513472ebb`, with the local base model and
  runtime contract documented below. Each expert uses four GPUs, FSDP, vLLM
  rollout, and a private local-Ray driver.
- C1/C2/C3 nominal endpoints are 370/222/222 steps. C4's nominal endpoint is
  444 because its rollout batch size is 64 rather than 128.
- Continuations are **non-exact model-only warm starts**: model weights and the
  upstream global-step/dataloader state are restored, but optimizer, scheduler,
  and RNG state are reinitialized. Instance reclamations can also lose trailing
  unsaved steps. The current C3/C4 metric logs contain only their active
  segments; preserved `pre_resume_logs/experiment_log.jsonl` files are needed
  to reconstruct the complete cross-segment history.
- Current recovery points, if still needed after a later reclamation, are C3
  step 200 and C4 step 175. Do not use older step-150/175 C3 or step-100 C4
  commands merely because they exist in historical sections. Recheck the
  current tracker and all four shard sizes before any launch.
- `save_model_only=true`, `save_freq=25`, and `save_limit=1` remain in use.
  `save_limit=1` can remove an old non-best checkpoint before the new save
  completes. No global Ray cleanup, checkpoint deletion, archive operation, or
  real GPU launch was performed from the CPU control session.
- Teacher export, skill extraction, MOPD, and downstream benchmark evaluation
  are not included in this training-status snapshot.

## Historical reclamation snapshot — 2026-10-30 00:36 UTC

- Both replacement GPU instances were reclaimed. C1 remains complete at step
  370; its best step-300 and last step-370 checkpoints are intact.
- C2 reached completed metric step 197/222 after the v5 repair, passing the
  previous step-183 concat failure. Its latest complete checkpoint remains step
  175; steps 176–197 were not saved. Its log stopped at 18:40 UTC on October 29.
- C3 reached completed metric step 114/222 and has a complete step-100
  checkpoint (best validation reward 0.538); steps 101–114 were not saved. Its
  log also stopped at 18:40 UTC on October 29. No new fatal traceback was found
  in the bounded tails of either stopped run; the stale running records reflect
  drivers lost with their instances rather than normal completion.
- Recovery points were rechecked: C2 step 175 and C3 step 100 each have four
  non-symlink 8,767,464,855-byte rank shards and `dataloader.pt`, and their
  trackers match those steps. A Pair A dry-run with C2 on GPUs 0–3 and C3 on
  GPUs 4–7 passed; no real training was launched from the CPU control session.
- The command file now has a separate **Pair A: resume C2+C3 after both
  instances reclaimed (2026-10-30)** section. Do not resume C1 or reuse the
  previous C2-only launch. Both continuations are non-exact model-only resumes.
- Shared storage has about 146GB free. Existing checkpoints/logs were not
  deleted or moved. With `save_limit=1`, upcoming saves at C2 step 200 and C3
  step 125 require attention because old non-best checkpoints may be removed
  before the new save completes.


# C1 completion and C2 metadata-concat repair (2026-10-29)

The following is the current status; earlier step-100/125 recovery instructions
in this document are historical and must not be reused as current progress.

## Run-state diagnosis

- **C1/capability1 completed normally, not by a trainer error.** Its latest
  record is `rl_capability1_complete`; the launcher reported process completion.
  Metrics reach step `370/370` with final validation reward
  `0.7853162288665771`. The tracker names step 300 as best (validation reward
  `0.7876`) and step 370 as last. Both `global_step_300` and `global_step_370`
  contain `dataloader.pt` and four `8,767,464,855`-byte model shards. This is a
  structural checkpoint check, not a full tensor/hash revalidation. C1 needs no
  restart or additional training repair.
- **C2/capability2 failed in metadata concatenation.** At 09:03 UTC its record
  became `rl_capability2_failed`; metrics reach completed step 183, but the
  latest complete recovery point is `global_step_175` with four model shards,
  `dataloader.pt`, and tracker `last_global_step=175`. `global_step_150` is also
  present as the protected best checkpoint (validation reward `0.7622`). Steps
  176–183 have no saved model state and cannot be claimed as recovered.
- The C2 traceback is `_make_batch_data()` -> `DataProto.concat([batch,
  new_batch])` -> `np.concatenate`: index 0 has two dimensions and index 1 has
  one. It does not include the field name, so there is no claim of a serialized
  failing-batch field dump.
- The actual C2 parquet schema contains `image_paths: list<string>`. Upstream
  `RLHFDataset.__getitem__` removes `images` but leaves `image_paths` in the
  returned example; upstream collate directly applies `np.array(...,
  dtype=object)`. Among the 9,500 rows, 9,089 have one path, 405 have two, and
  six have three. Equal path counts infer a rectangular two-dimensional object
  array; mixed path counts infer a one-dimensional array. Concatenating these
  representations reproduces the exact error. The previous local normalization
  covered only `raw_prompt_ids`, so it did not repair this field.
- **C3/capability3 was not restarted or modified.** At the 13:13 UTC check its
  record remained `rl_capability3_running`, the last metric was step 92, and
  the metric log had been updated at 13:11 UTC. Its independent process is not
  evidence for C2 recovery or final algorithmic correctness.

## Repair and verification

- Runtime v5 now normalizes only `raw_prompt_ids` and `image_paths`, both
  per-sample variable-length list metadata. It preserves sample/path ordering,
  already-one-dimensional arrays, input object identity where no repair is
  needed, tensor/meta identity, and every unrelated multidimensional field.
  Dataset/protocol collate normalization and mixed-dimension concat fallback
  share the same field list. Uniform multidimensional arrays remain unchanged
  for `fold_batch_dim`; this is not a global object-array flatten.
- The pinned upstream checkout, training loss, masks, grouping, reduction,
  rollout, and online student scorer were not modified. `image_paths` is
  retained metadata, not deleted as a workaround.
- `VERIFIED LOCALLY`: real `.venv-rl` targeted collate tests **6 passed in
  34.42s**; the complete real runtime suite **27 passed in 230.27s**. New tests
  cover equal-count versus mixed-count `image_paths`, exact sample order,
  non-mutation, mixed concat, protocol/dataset collate, and folded shape
  preservation. The first targeted invocation had one subprocess timeout after
  120 seconds; its immediate rerun passed. It was not a shape/assertion failure
  and is not hidden by the later successful result.
- `VERIFIED LOCALLY`: control launcher/reproduction tests **46 passed in
  33.96s**; launcher `bash -n` and targeted `git diff --check` passed. Runtime
  provenance is now `scoped-runtime-v5`.
- `VERIFIED LOCALLY`: a C2-only dry-run at 13:07 UTC accepted step 175 and
  expanded `load_checkpoint_path=.../global_step_175`,
  `find_last_checkpoint=false`, four GPUs, `save_freq=25`, `save_limit=1`, and
  `save_model_only=true`. It did not start Ray or training. No real GPU task was
  launched by this control-side repair.
- `REQUIRES SERVER VERIFICATION`: the user-started C2-only run must demonstrate
  four-rank model restoration, new steps beyond 175, passage through the real
  mixed-image batch concat, subsequent checkpoint saves, and final completion.
  Passing CPU integration tests do not establish GPU completion, exact resume,
  or algorithmic correctness.

The command file `lzy/manuscript-sft-rl-gpu` retains its historical Pair A and
C1 sections and now has a separate **C2 only: resume after image_paths concat
fix** section. It starts only capability2 from step 175 after the operator
verifies available GPUs and checkpoint state. Do not rerun C1 or restart C3 /
the entire Pair A. The continuation remains model-only and non-exact:
optimizer, scheduler, and RNG are reinitialized.


## Objective and pinned contract

Reproduce SCOPED's four RL capability experts from the released training
recipe. MOPD remains deferred until the benchmark-driven data strategy is
selected. The current implementation task is **explicit non-exact resumption of
only Pair A** after its GPU instance was reclaimed. It does not authorize
capability1/4, teacher export, skill extraction, or MOPD.

- Control branch: `exp/scoped-expert-reproduction`
- Last reviewed commit before these working-tree changes:
  `e3983bf81c5fbb953f086ac132bad3f8e9049f8e`
  (`docs(scoped): record expert reproduction progress`). That commit changed
  documentation, not training code. The launch/runtime changes described below
  are working-tree changes, not a new commit.
- Upstream: `https://github.com/NEUIR/SCOPED.git`
- Upstream commit: `74e112e29e2be0657d1d521c1ecc3be53df99e44`
- Dataset: `xiongyq/SCOPED-Train`
- Dataset revision: `912185f76295f847c09bc44f379cf85513472ebb`
- Dataset manifest SHA256:
  `bc3340721edddbd869259cb1db50ca54446f8379deb4078067eb83896e47ca96`
- Fresh base model:
  `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct`
- Contract: `configs/external/scoped_expert_reproduction.yaml`

The current pinned upstream checkout was read on 2026-10-27 and matched that
commit; source-status inspection excluded only runtime virtualenv/cache and
editable-install metadata. No upstream source reset or modification was made.

## Pair A recovery status — 2026-10-28

The 2026-10-27 snapshot below is historical. A subsequent read-only inspection
found that both old processes had stopped. No training was launched from this
CPU control session.

| Capability | Last observed log | Available model-only checkpoint | Stop evidence |
|---|---:|---:|---|
| capability2 | 100/222 | `global_step_100`; validation reward/overall about 0.7519699 | Subsequent `DataProto.concat` failed on mixed 1-D/multidimensional `raw_prompt_ids`; log stopped 2026-10-27 17:20 UTC |
| capability3 | 75/222 | `global_step_50`; validation reward 0.526 | No traceback; log stopped 2026-10-28 01:04 UTC during Compute log probs, consistent with instance reclamation |

Both recovery points have four model shards of 8,767,464,855 bytes each and a
`dataloader.pt`, but no optimizer or extra-state shards. The user explicitly
selected **Pair A only** and **non-exact continuation from the existing model
weights**. Capability2 can continue at step 100; capability3 can continue only
at step 50. Its unsaved steps 51--75 cannot be reconstructed. Nominal total
steps remain 222. This is not an exact training-trajectory resume.

The new explicit `--resume-from` / pair `--resume-from-a` / `--resume-from-b`
path keeps the upstream global-step and StatefulDataLoader restoration, while a
required runtime wrapper restores only the four FSDP model shards with strict
state-dict loading. Optimizer, scheduler, and RNG state are reinitialized. The
new launch uses the narrowed `raw_prompt_ids` concat repair. Before upstream
`FileLogger` can truncate the three old experiment logs, they are copied into
the unique run record with byte counts and SHA256 hashes. Model-restore receipts
for all four ranks and the resolved config are required for a successful
completion record. Existing checkpoints are not moved, archived, or deleted.

Dry-run and CPU integration tests do not establish successful GPU restoration.
The operator commands are in `lzy/manuscript-sft-rl-gpu` under the separate
“Pair A non-exact resume” section, for a new eight-GPU instance only.

### Actual restart failures and Ray dashboard fix — 2026-10-28

Several operator-initiated restart attempts failed before model restoration or
a new training step. The first two exposed missing control-Python imports; the
record helper now locates the control source itself and reads the runtime
version without importing NumPy/Torch. A later attempt exposed Ray's 107-byte
AF_UNIX socket-path limit, so the launcher now uses a short `sr-N-<stamp>-PID`
Ray root while retaining the full run ID in its durable record.

The latest attempt, `2026-10-28T065159Z`, reached `data_ready` for both experts
but failed at 06:55 UTC with `Failed to register worker to Raylet: End of file`.
The two Ray instances remained separate, but both waited for their optional
HTTP dashboard and reported dashboard startup failure before worker registration
timed out. Neither attempt created resume-log backups, rank receipts, or a new
step. The original cap2 step-100 and cap3 step-50 checkpoints remain in place.

Runtime `scoped-runtime-v3` now defaults `include_dashboard=False` for a required
`RAY_ADDRESS=local` initialization when the caller did not make an explicit
choice. This affects only the local driver, leaves actor runtime-environment
propagation intact, and does not disable Raylet, actors, or training. An explicit
caller dashboard choice and nonlocal Ray initialization are not overridden.
The dashboard is optional; disabling it is not a claim that training semantics
have changed or that the four-GPU restore is already verified.

### Actual v3 failure and v4 Ray prestart mitigation — 2026-10-28

The operator's subsequent `2026-10-28T073806Z` restart also failed in **both**
experts before model restoration. The shared trainer logs record a local Ray
instance starting at 07:41:06, followed by `Failed to register worker to
Raylet: ... End of file` at 07:41:37.102 for capability2 and 07:41:38.006 for
capability3. The dashboard errors from the previous attempt were absent, so the
v3 dashboard-only mitigation did **not** resolve this failure. Both final run
records use `scoped-runtime-v3`, have `resolved_config_evidence.captured=false`
and `resume_evidence.complete=false`, and contain no pre-resume backup or four
rank restore receipts. There was no new training step. The original step-100
and step-50 checkpoints remain the only recovery points.

Runtime `scoped-runtime-v4` retains the optional-dashboard default and, for a
required local-Ray driver, now uses `setdefault` to set
`prestart_worker_first_driver=False`. Ray 2.51.1 otherwise gates the first
driver's registration reply on starting idle Python workers. Actual task and
actor workers are still created on demand. An explicitly supplied prestart
setting and all nonlocal-Ray initialization remain unchanged.

This mitigation must not set `num_cpus=0`: the pinned SCOPED
`RayResourcePool.get_placement_groups()` requests
`{"CPU": max_colocate_count, "GPU": 1}` for each GPU bundle. Removing CPU
resources would make those placement groups unschedulable. The v4 patch does
not change CPU/GPU limits, FSDP world size, loss, rollout, masks, grouping,
reduction, or student scoring. Its real local-Ray CPU probe verifies the
prestart setting and on-demand actor registration, **not** a GPU Pair A launch.

The launcher also now copies this run's small Raylet/GCS/dashboard/runtime-env
and worker text logs from its own GPU-local `RAY_TMPDIR` into
`records/<run-id>/ray_diagnostics/` after a trainer failure. These diagnostics
were not present in the earlier attempts; the new collection has a fixture test
but still requires a real failed or successful operator launch to exercise the
GPU path. No global Ray process or another job's logs are touched.

The sections titled “Historical Pair A evidence versus current health” and
“Capability1 implementation changes” below retain earlier implementation
history. Their instructions not to restart historical running jobs must not be
interpreted as the status of these subsequent, operator-initiated restarts.

## Historical Pair A evidence versus current health

The prior version of this document recorded an older launch at
`2026-10-26T03:04:34Z`, capability2 step 101 with `global_step_100`, and
capability3 step 73 with `global_step_50`. It also recorded capability2
overall/F1 0.7918 and capability3 overall/accuracy 0.5779, format 0.9678.
Those are **historical artifacts**, not the state of the current fresh Pair A
logs and not current remote-process health evidence.

Read-only inspection on 2026-10-27 found the current experiment configs for
both capability2 and capability3 set:

```text
save_model_only=true
save_limit=1
load_checkpoint_path=null
find_last_checkpoint=false
```

Their current runs are therefore fresh training from the same local base
model, **not exact resume from archived model-only checkpoints**. At the
sampled read, current log files contained:

| Capability | Current log evidence | Log mtime (UTC) | Model-only checkpoint observed |
|---|---:|---|---|
| capability2 | steps 1--67; step-67 overall/F1 0.7342003 | 2026-10-27 13:20:27 | `global_step_50`, about 33GB |
| capability3 | steps 1--26; step-26 overall/accuracy 0.5188802; format 0.9039714 | 2026-10-27 13:16:53 | none in the inspected glob |

These files may continue to change after this snapshot. Log progress does not
establish that a remote GPU process is currently alive or healthy.

`UNKNOWN`

- Current Pair A remote PID/GPU health; no live remote process inspection was
  performed from this CPU control session.
- Final completion of Pair A and all scientific reproduction outcomes.

**Do not follow the old advice to restart Pair A, archive its output, or
relaunch it with the new pair script.** The current task leaves those jobs and
their files alone. No global `ray stop`, `pkill`, checkpoint deletion, or
archive operation was performed.

## Capability1 implementation changes in the working tree

The launch implementation is being hardened before the user opens the new
instance. It reuses the pinned upstream training entry point rather than
adding a separate training loop or changing losses, reductions, rollout
weighting, grouping, masks, or online student scoring.

### Single-expert isolation and fresh start

`scripts/hpc/run_scoped_expert.sh` now:

- accepts exactly four unique nonnegative GPU IDs and passes four GPUs to the
  upstream trainer;
- sets `RAY_ADDRESS=local`, a unique absolute `RAY_TMPDIR`, offline Hugging
  Face loading, and the absolute local base-model path;
- explicitly sets `trainer.load_checkpoint_path=null` and
  `trainer.find_last_checkpoint=false`;
- defaults to `save_model_only=true`, `save_freq=25`, `save_limit=1`, with
  explicit warnings about their semantics;
- atomically reserves a previously nonexistent capability output, rejects
  symlinks and empty/nonempty existing directories, and holds a capability
  lock for its lifetime;
- supervises its own trainer process group, forwards TERM/INT only to that
  owned tree, waits for actual child termination, and preserves nonzero exit
  status;
- writes a failed record from an exit trap if Bash takes its fatal-signal
  action before the deferred TERM/INT trap can update the record;
- writes unique launcher/trainer logs and planned/data-ready/running/final
  records rather than allowing intentional log overwrites;
- prepares only the requested capability and returns dry-run before data
  preparation, directory creation, locks, records, Ray, or training.

Ray 2.51.1 source inspection confirms `RAY_ADDRESS=local` takes the
new-local-instance branch instead of discovering an existing cluster. The
launcher does not start a shared head or invoke global cleanup. Normal driver
shutdown owns its local-Ray cleanup; abnormal-exit cleanup remains a server
check, not a license to kill other Ray jobs.

### Future pair behavior, not a Pair A restart

`scripts/hpc/run_scoped_expert_pair.sh` delegates RL to two isolated
single-expert launchers. GPU groups must each contain four unique IDs and be
disjoint. The parent waits for both children, forwards signals only to their
two owned process groups, and retains failure statuses. Dry-run does not create records or call
Ray. Export/skill commands remain separate manual stages and have not run.

### Runtime patch and evidence

The runtime work narrows the object-array fix to `raw_prompt_ids`. Its semantic
contract is one variable-length token list per sample, concatenated on sample
axis zero. Other object-array shapes must not be flattened, including legal
multidimensional fields from `fold_batch_dim`. The patch covers the upstream
dataset collate, protocol collate, and `DataProto.concat` paths; it is not a
training-gradient change.

Activation is explicit, idempotent, versioned, and checked by function
references. `SCOPED_RUNTIME_REQUIRED=1` makes import/install/assertion failure
fatal. Ordinary Python without the required marker must not be forced to
import the heavy RL stack. Ray worker setup must also survive the upstream
actor-specific `runtime_env={"env_vars": ...}` override; driver-only
activation is insufficient.

The run-record schema now distinguishes control-repository dirtiness from the
pinned upstream checkout, records actual `.venv-rl` dependency versions and
runtime hashes, checks the target capability's manifest/shard verification
cache, and incorporates the real resolved config capture. A completed record
without that capture must fail instead of claiming verified configuration.
The fake trainer in launcher tests disables Python site initialization only to
model an external already-preflighted trainer; production still requires
fail-fast activation.

### Resource constraints

A 2026-10-27 `df` observation showed about 137GB free on the approximately
10TB shared GPFS mount, 99% used. With explicit user authorization, only
rebuildable caches were removed: about 10GB pip, 24GB unused Conda packages,
and 125GB Hugging Face datasets cache. Available space rose to about 295GB;
a later check showed about 263GB free. No models, source datasets, Pair A
checkpoints, or historical training checkpoints were removed. These values
can change and are not a reservation. Recheck on the new instance before launch.

Model-only checkpoints omit optimizer and scheduler/RNG extra state and cannot
be loaded by the upstream loader as an exact resume. A new optimizer initialized
from those weights would be a warm start, not resume. `save_limit=1` can remove
the previous checkpoint before the next save completes. Do not delete other
jobs' or archived artifacts to make room.

Nominal capability1 steps are `floor(9499/128)*5=370`; online filtering can
consume extra source batches and does not guarantee exactly five passes over
unique samples.

## Verification ledger

This ledger describes checks actually attempted for the current working tree.
It supersedes the old blanket statement that “11 local tests pass,” which
belonged to an earlier revision and is not a current test result.

`VERIFIED LOCALLY`

- The real upstream checkout reports the pinned commit above.
- The real capability1 dry-run exits 0 and expands the `.venv-rl`
  `verl.trainer.main` command with capability1 data paths, local base model,
  four GPUs, model-only/save-limit settings, and fresh-start overrides.
  The earlier dry-run failure from an unset `HOME` under `env -i` was fixed by
  supplying a minimal HOME. Dry-run is not a GPU/runtime smoke.
- Ray 2.51.1 source inspection establishes the `local` address behavior and
  the actor runtime-env hazard described above. Source inspection alone does
  not prove actor activation.
- The launcher test fixture was corrected to use a real temporary Git commit,
  the actual manifest hash, and `python -S` for the fake trainer. This fixed
  failures where required `sitecustomize` correctly rejected the fixture's
  missing `verl` before fake trainer code could execute.

- Control-side tests with the default Python 3.11.15 interpreter on 2026-10-28:
  `PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_scoped_launchers.py tests/test_scoped_reproduction.py`
  completed with **46 passed in 29.41s**. This covers the prior launcher guards
  plus valid/invalid model-only resume paths, pair checkpoint mapping, recovery
  records, retention deleting an already-restored source checkpoint, and
  preservation of Raylet/worker failure diagnostics from the launcher's own
  temporary directory. These are fake-trainer/process fixtures, not GPU training.
- Real `.venv-rl` runtime tests on 2026-10-28:
  `PYTHONDONTWRITEBYTECODE=1
  fc-opd-storage/external/scoped/repo/.venv-rl/bin/python -m pytest -q
  tests/scoped_runtime`
  completed with **27 passed in 236.30s**. In addition to the prior DataProto,
  DataLoader, FileLogger, and local-Ray tests, these cover resume-log byte/hash
  preservation, fresh/no-backup behavior, wrong-output rejection, strict restore
  failure, unrelated full-loader fallback, and a real four-rank Gloo CPU FSDP
  save/zero/restore round-trip. The local-Ray tests cover the dashboard and
  prestart defaults, preservation of explicit resource and system-config
  arguments, nonlocal behavior, required setup propagation, and successful
  on-demand actor registration. CPU FSDP/Ray is not a substitute for GPU
  validation.
- Separate local-Ray probes observed the default dashboard startup path and
  its failure. Dashboard disabling alone did not fix the later GPU driver EOF;
  the v4 prestart mitigation passed its CPU probe but has not yet been exercised
  by the real Pair A trainers.
- The real Pair A recovery dry-run on 2026-10-28 completed with cap2
  `global_step_100` and cap3 `global_step_50`, each with its own four-GPU group,
  `load_checkpoint_path`, `find_last_checkpoint=false`, and 25/1 save defaults.
  It did not write records or start Ray/training.
- The actual driver preflight completed successfully with the selected RL
  interpreter: `SCOPED runtime patches verified; RAY_ADDRESS=local`.
- `bash -n` on the launchers and GPU parser and `git diff --check` passed.

`BROADER SUITE: NOT GREEN (missing dependencies and out-of-scope failures)`

- Full default-Python collection stopped with five import errors: missing
  `verl`, `mathruler`, `ray`, and `latex2sympy2_extended` dependencies. No
  packages were installed or changed in the shared environment.
- Rerunning with those five files excluded and the separately passed
  `tests/scoped_runtime` excluded produced **702 passed, 5 skipped, 10 failed
  in 207.43s**. Four failures are missing `latex2sympy2_extended`; six are
  assertions in unrelated eval/GKD/summary/support-proposal/scorer/checkpoint
  tests. Those failures were not repaired or treated as evidence against the
  scoped tests, and this is not a claim that the whole repository suite passes.

The control-side tests use the default Python 3.11.15 interpreter. Real veRL
integration uses the existing `.venv-rl` interpreter; a blanket stub is not
acceptable. No packages were installed into the shared RL environment for this
work. Earlier temporary-environment setup is irrelevant to the final results
above, which were rerun with the explicitly named interpreters.

`REQUIRES SERVER VERIFICATION`

- The new eight-GPU instance's allocation, UUIDs, free devices, owning PIDs,
  two disjoint four-GPU mappings, and actual disk headroom.
- Four-rank GPU FSDP restoration of each actual 8.77GB model shard; global-step
  and StatefulDataLoader restoration; optimizer/scheduler/RNG reinitialization.
- Driver, Runner, FSDP actor, reward actor, and DataLoader-worker patch
  activation under the real training runtime.
- Ray/vLLM TP=2 and four-GPU FSDP mapping, GPU kernels, finite and meaningful
  loss/gradients, sustained progress, and actual checkpoint completeness.
- Reclamation of only this driver's Ray processes on normal and abnormal exit.
- Pair A resumed progress from cap2 step 100 and cap3 step 50 to nominal step
  222, validation metrics, checkpoint retention, and final reproducibility
  assessment. Neither has been resumed by this control-side implementation work.

`UNKNOWN`

- Pair A's next operator-initiated restart after the v4 Ray prestart mitigation;
  remote process health after that restart. The historical processes are
  stopped, and the latest recorded v3 restart failed before model restoration.
- Capability4, teacher export, skill extraction, and MOPD results. None were
  launched as part of this task.

## Next operator action

Follow the new **C2 only: resume after image_paths concat fix** section in
`lzy/manuscript-sft-rl-gpu`. Confirm the four selected GPUs do not overlap C3,
inspect the no-side-effect dry-run, and then launch only C2 once from step 175
as a supervised job. Do not rerun the older Pair A recovery commands or the
completed C1 command. Confirm restored ranks, backed-up logs, new metrics, and
checkpoint retention after launch.

The earlier Pair A recovery plan below is historical, not the current action.
No capability4, export/skill/MOPD, or real training task was launched by this
control-side repair. Startup, finite loss, or a checkpoint alone does not
establish exact resumption or algorithmic correctness.
