# Micro-Operator Experiment - v2 report (dose confirmed, native transfer null)

Date: 2026-08-17.  Branch `codex/va-opd`.

Follow-up to v1 (low dose, null).  v2 increases dose and adds per-step
diagnostics (loss, grad norm, NaN, block logp delta).

## 1. Runs

- v2a dose run: 12 candidates, 8 steps, lr 1e-4, rollout K=8 -> all G = 0.
- v2b diagnostic smoke: 2 candidates, 8 steps, lr 1e-4, K=4, with loss/grad
  diagnostics (this report's table).

## 2. Diagnostics (v2b)

| candidate | role | F | R | loss FKL (last->first) | loss RKL (last->first) | max grad norm | NaN params | block logp delta FKL | RKL |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| geo3k:1208 b0 | FKL | 0.65 | -0.58 | 0.02 -> 0.35 | 3.12 -> 0.30 | 178 | 0 | +0.53 | -10.4 |
| geo3k:1208 b20 | RKL | 0.04 | 0.02 | 0.00 -> 0.70 | 3.39 -> 0.50 | 270 | 0 | +0.52 | -10.4 |

Readings:

- Both operators converge on their own objectives: FKL CE 0.35->0.02 (teacher
  token log-prob up, `block_logp_delta` +0.53 after the fp32 / weight-decay=0
  fix); RKL KL 3.1->0.3 (student distribution approaches teacher's).
- No NaN.  Grad norms are large (178/270) and clipped to 1.0 every step, so
  the RKL update is a fixed-norm jump rather than a clean small step; its
  teacher-token logp drops by ~10 nats, which is expected mode-seeking onto
  the teacher *distribution* (whose argmax may differ from the sampled
  trajectory token).  The correct RKL dose metric is the KL decrease, not the
  sampled-token logp.
- Native rollout gains: q0 = 0 for every prompt (rescue-positive hard set),
  q_i = 0 for every candidate and both operators -> all G = 0, both
  correlations null.

## 3. Interpretation

The micro-update demonstrably moved the model on the target block (dose is
confirmed), yet **no block-local FKL or RKL update produced a native
rollout gain**.  Combined with v1:

- F_i does not predict FKL advantage, R_i does not predict RKL advantage
  (correlations ~0 in both v1 and v2a).
- Fixing a single reasoning block does not transfer to the student's ability
  to solve the problem from scratch, consistent with the path-reachability vs
  state-executability distinction: local compatibility does not create the
  native path.

## 4. Decision (per frozen plan B8)

The two B8 relations are not supported, with dose confirmed.  Per the frozen
plan, **stop the single-block operator-routing story**; do not build a
F_i/R_i-based FKL/RKL router from this evidence.

Open alternatives for discussion (not launched):

1. Region-level test: apply FKL/RKL to the *whole usable prefix* (all blocks
   before h*), not single blocks; measure native G.  This matches the actual
   training operation (STP/FKL) and may transfer where single blocks do not.
2. Sensitive outcome: instead of native q (floor at 0 for these hard
   prompts), measure continuation pass rate at h* after the update - the
   plan's native-q criterion is a hard test for rescue-positive prompts.
3. If neither shows operator-conditional signal, drop FKL/RKL routing and
   keep uniform FKL-on-prefix + OPD as the method (the original Dual-Track
   design), with these diagnostics as published negative results.
