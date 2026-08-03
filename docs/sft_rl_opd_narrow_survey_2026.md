# SFT–RL–OPD Narrow Survey: Support, Routing, and Reward Density

Cutoff: 2026-08-03.  Scope: post-training work that changes when or where SFT,
teacher proposal learning, on-policy distillation (OPD), and sparse-reward RL
are applied.  This is deliberately narrower than a generic RLHF or knowledge
distillation survey.

## Bottom line

There is no single public "industrial standard" in which every model is first
RL-trained and then merged with OPD.  Public pipelines fall into at least three
different axes that are often conflated:

1. **Model axis:** run sparse-reward RL on a capable teacher, then distill its
   reward-shaped behavior into a smaller deployment student.  DeepSeek-R1-style
   releases and the sparse-to-dense reward principle exemplify this.
2. **Time axis:** cold-start/SFT or a verified bridge first, then RL on the same
   student.  ReGFT, TREK, VOLD, and the Tsallis analysis explain this family.
3. **Objective axis:** combine RL and distillation in the same online update or
   route samples between them.  KDRL, RLSD, SRPO, HAPO, and VOLD exemplify this.

Thus "joint RL–OPD" must be qualified.  It can mean a summed loss on the same
rollout, sample-level routing inside one training step, an alternating schedule,
or a multi-stage teacher→student pipeline.  These are different interventions
with different causal questions.

The proposed SFT→OPD→RL direction is plausible, but it is no longer sufficient
as a novelty statement.  The closest public precedents already show:

- cold-start alignment can make subsequent OPD meaningful (VOLD);
- forward-KL or verified model-derived traces can move hard prompts into a
  trainable region before RL (TREK, ReGFT);
- forward-KL followed by reverse-KL/OPD can outperform either alone (PACED,
  sparse-to-dense);
- teacher guidance can be routed inside RL based on success/failure (SRPO,
  HAPO).

The open opportunity is to identify **which operator is appropriate in which
observed support state**, measure prompt-level state transitions, and test
whether those transitions predict actual mixed RL groups and early-RL gains.

## Six scientific problem families

### 1. Support creation: how does a zero/low-pass prompt become learnable?

Methods inject verified teacher proposals, reference-guided model traces,
expert trajectories, or a more density-estimation-like objective.  The central
trade-off is mode coverage versus distribution mismatch.  Pure OPD on student
rollouts cannot directly train an absent teacher-preferred mode if that mode is
never visited.

### 2. Consolidation on student states: when is OPD better than offline SFT?

OPD corrects the prefixes and failure modes the student actually visits, but
only when the teacher distribution is compatible and carries incremental
capability.  Forward-KL warmup or cold-start alignment is often needed when
teacher and student occupancy barely overlap.

### 3. Sparse/dense reward allocation: which model and stage should receive RL?

Sparse reward is useful where exploration is productive; dense token-level
teacher reward is useful for compression and local correction.  This creates
model-level choices (RL the teacher or student) and time-level choices (before,
during, or after the dense bridge).

### 4. Routing and annealing: how should supervision change with competence?

The field contains prompt-level pass-rate weighting, group-level success
injection, sample-level correct/incorrect routing, token-level entropy or
disagreement routing, and continuous loss interpolation.  A new scalar gate is
crowded; mechanism-calibrated state transitions are a more defensible target.

### 5. Teacher reliability and teachability: which dense signals are absorbable?

Raw teacher–student KL can mix learnable local reweighting with off-support
disagreement.  The important moderator is whether the teacher puts corrective
mass on tokens or trajectories the student can reach, and whether teacher
scores locally correlate with verified outcomes.

### 6. Modalities and long-horizon environments: what changes beyond text math?

VLM work separates language priors from visual grounding; agent work separates
student occupancy relevance from teacher reliability on replayed prefixes.
These are valuable extensions but should follow, not precede, the basic
operator-by-state result.

## Literature matrix (38 papers)

`Priority` is for this project: P0 = must read deeply now, P1 = read method and
ablations, P2 = contextual/foundational.  A paper can inform more than one
family; it is placed under its most useful role.

### A. Foundations and the base post-training problem

| # | Work | Intervention / scientific question | What it establishes for this project | Priority |
|---:|---|---|---|:---:|
| 1 | [MiniLLM (2306.08543)](https://arxiv.org/abs/2306.08543) | Reverse-KL KD optimized on student samples | Mode-seeking OPD and its variance/support limitations originate in generative model compression, not in RL-frontier routing. | P2 |
| 2 | [GKD / On-Policy Distillation (2306.13649)](https://arxiv.org/abs/2306.13649) | Student-generated prefixes with teacher token distributions; flexible divergence | Establishes occupancy matching and the basic exact-prefix teacher supervision contract. | P1 |
| 3 | [DeepSeekMath / GRPO (2402.03300)](https://arxiv.org/abs/2402.03300) | Group-relative sparse-reward RL | Defines the mixed-group dependence that motivates `U_G(p)`; all-zero/all-one groups lack within-group reward variance. | P1 |
| 4 | [ReFT (2401.08967)](https://arxiv.org/abs/2401.08967) | Fine-tune on verified self-generated correct reasoning before/around RL | Shows model-derived verified traces can be better aligned than raw human CoT, but cannot repair prompts with no sampled success. | P1 |
| 5 | [DeepSeek-R1 (2501.12948)](https://arxiv.org/abs/2501.12948) | Large-scale RL and distillation release pipeline | Public evidence for teacher-RL→student distillation on the model axis; not evidence that every deployment pipeline uses post-RL OPD. | P2 |
| 6 | [DAPO (2503.14476)](https://arxiv.org/abs/2503.14476) | RL system with dynamic sampling and clipping refinements | Strong RL baseline already filters uninformative groups; bridge gains must survive comparison to a competent RL implementation. | P1 |
| 7 | [Does RL Incentivize Reasoning Beyond the Base Model? (2504.13837)](https://arxiv.org/abs/2504.13837) | Tests whether RL expands capability versus amplifies existing modes | Motivates measuring pre/post solvable support rather than only pass@1. | P1 |

### B. Pre-RL support creation and competence bridges

| # | Work | Intervention / scientific question | Overlap and remaining gap | Priority |
|---:|---|---|---|:---:|
| 8 | [Knowledge Distillation with Training Wheels (2502.17717)](https://arxiv.org/abs/2502.17717) | Scaffold teacher/student transfer when direct imitation is too hard | Precedent for temporary support that is removed as competence grows. | P1 |
| 9 | [Privileged Information Distillation (2602.04942)](https://arxiv.org/abs/2602.04942) | Teacher sees privileged context unavailable to student | Formalizes context-asymmetric teachers; useful when "teacher" is not just a larger checkpoint. | P1 |
| 10 | [Reinforcement-Aware Knowledge Distillation (2602.22495)](https://arxiv.org/abs/2602.22495) | Transfer teacher behavior while accounting for later RL | Directly relevant to whether a bridge should preserve exploration rather than merely maximize immediate imitation. | P1 |
| 11 | [ReGFT (2603.01223)](https://arxiv.org/abs/2603.01223) | Use partial references to synthesize model-style verified traces before DAPO | Already shows pre-RL supervised bridging raises solvable problems and early/final RL. It does not compare exact-token OPD as the state-dependent alternative. | P0 |
| 12 | [PACED (2603.11178)](https://arxiv.org/abs/2603.11178) | Weight distillation by `p(1-p)`; forward-KL then reverse-KL | Already owns pass-rate frontier weighting and mode-coverage→consolidation. Its target is distillation gradient SNR, not realized future mixed-group mediation. | P0 |
| 13 | [Tsallis Loss Continuum (2604.25907)](https://arxiv.org/abs/2604.25907) | Continuous `q` between RL-like exploitation and density estimation | Gives a theoretical cold-start explanation: stronger density-estimation commitment escapes low `p` faster but is more noise-sensitive. | P0 |
| 14 | [TREK (2607.05339)](https://arxiv.org/abs/2607.05339) | Route low-pass prompts to verified teacher proposals, forward-KL consolidate, then ordinary GRPO | Closest overlap. It explicitly claims support expansion and early-RL efficiency, and finds FKL stronger than OPD in its setup. Missing: prompt-level `U_G` transition/mediation and an operator-by-state phase diagram. | P0 |

### C. Staged sparse-to-dense allocation and OPD preconditions

| # | Work | Intervention / scientific question | Overlap and remaining gap | Priority |
|---:|---|---|---|:---:|
| 15 | [Rethinking OPD (2604.13016)](https://arxiv.org/abs/2604.13016) | Diagnose teacher–student thought-mode overlap and capability increment; add offline warmup | Strong warning: a higher-scoring teacher is insufficient; overlap and genuinely new capability are both needed. Supplies a moderator for RQ2. | P0 |
| 16 | [Uni-OPD (2605.03677)](https://arxiv.org/abs/2605.03677) | Dual-perspective recipe for stabilizing/alignment in OPD | Relevant baseline for cross-scale compatibility and objective choice; prevents claiming all reverse-KL recipes are equivalent. | P1 |
| 17 | [Sparse-to-Dense Reward Principle (2605.12483)](https://arxiv.org/abs/2605.12483) | Teacher RL→teacher-rollout FKL→student-rollout OPD→optional student RL | Already supplies the four-stage "industrial" recipe and shows all bridge components can be load-bearing. It studies model/data allocation, not prompt-state operator selection or `U_G` mediation. | P0 |
| 18 | [A Survey of OPD (2604.00626)](https://arxiv.org/abs/2604.00626) | Taxonomy of current OPD objectives, teachers, and applications | Use as a discovery/index source, not as primary evidence for empirical claims. | P2 |

### D. Joint or routed RL–distillation objectives

| # | Work | Intervention / scientific question | Relation to "joint RL–OPD" | Priority |
|---:|---|---|---|:---:|
| 19 | [KDRL (2506.02208)](https://arxiv.org/abs/2506.02208) | Sum/coordinate GRPO and reverse-KL teacher supervision | Algorithm-level joint objective on the same post-training process. Tests coefficients/approximations, but not prompt-state transitions. | P1 |
| 20 | [Self-Distilled RLVR / RLSD (2604.03128)](https://arxiv.org/abs/2604.03128) | Use privileged self-teacher token signal to reshape RL updates | Dense signal changes update magnitude/credit while reward anchors direction; demonstrates same-step integration. | P1 |
| 21 | [SRPO (2604.02288)](https://arxiv.org/abs/2604.02288) | Route correct samples to GRPO and failed samples with correct siblings to SDPO | Very close routing baseline. It falls back to GRPO when all group samples fail, so it does not create support in true zero-success groups. | P0 |
| 22 | [HAPO (2603.11321)](https://arxiv.org/abs/2603.11321) | Replace one failed group trajectory with a verified teacher sample under Bayesian confidence gating | Group-level support injection during RL, with teacher use annealing away. Direct competitor to state-routed bridging. | P0 |
| 23 | [Self-Distillation Zero (2604.12002)](https://arxiv.org/abs/2604.12002) | Turn binary reward into dense signal through self-revision | Shows an external white-box teacher is not the only dense-feedback source; depends on successful self-correction. | P1 |
| 24 | [From Generic Correlation to Input-Specific Credit (2605.11613)](https://arxiv.org/abs/2605.11613) | Input-specific credit in on-policy self-distillation | Relevant to replacing global loss weights with prompt/token-specific causal usefulness. | P1 |

### E. Which OPD signals are useful or learnable?

| # | Work | Intervention / scientific question | Implication | Priority |
|---:|---|---|---|:---:|
| 25 | [TIP (2604.14084)](https://arxiv.org/abs/2604.14084) | Select tokens by student entropy and teacher–student divergence | High-entropy and confident-but-wrong tokens are distinct useful regions; response-level support alone is not a token selector. | P0 |
| 26 | [Token Teachability / TA-OPD (2605.26844)](https://arxiv.org/abs/2605.26844) | Split disagreement into locally compatible vs off-support teacher correction | Essential RQ2 refinement: raw teacher gap/KL is coarse; OPD benefit should depend on teacher mass over student local support. | P0 |
| 27 | [Position-Weighted OPSD (2605.21606)](https://arxiv.org/abs/2605.21606) | Diagnose branch viability and weight teacher tokens by sequence position | Teacher reliability can be trajectory-structured; local entropy is not always an adequate reliability proxy. | P1 |
| 28 | [CRISP (2603.05433)](https://arxiv.org/abs/2603.05433) | Same-model, context-conditioned reverse-KL for concise reasoning | OPD can change style/length without external correctness supervision; length must be controlled in frontier claims. | P1 |
| 29 | [On-Policy Context Distillation (2602.12275)](https://arxiv.org/abs/2602.12275) | Teacher is the same model with privileged context | Shows teacher capability can come from context rather than scale, reducing distribution mismatch by construction. | P1 |
| 30 | [Black-Box On-Policy Distillation (2511.10643)](https://arxiv.org/abs/2511.10643) | Replace inaccessible logits with black-box signals | Broadens teacher access but changes the density/noise of supervision; relevant when CPU/network instance uses API teachers. | P1 |
| 31 | [OmniOPD (2606.01476)](https://arxiv.org/abs/2606.01476) | Chunk-level speculative verification with black-box teacher rollouts | Makes decision forks and chunk support explicit; expensive teacher sampling and semantic verifier noise are key trade-offs. | P1 |

### F. VLM and multi-turn extensions

| # | Work | Intervention / scientific question | Why it is secondary to the current main line | Priority |
|---:|---|---|---|:---:|
| 32 | [VOLD (2510.23497)](https://arxiv.org/abs/2510.23497) | Cold-start SFT then combined GRPO+OPD transfers text reasoning to a VLM | Direct precedent for SFT→joint RL/OPD and for the claim that OPD needs distribution alignment. It does not require visual-fork localization. | P0 |
| 33 | [VA-OPD (2605.21924)](https://arxiv.org/abs/2605.21924) | Weight VLM OPD by teacher sensitivity to fine visual detail | Establishes visual-dependence diagnostics, but asks a different question from support creation. Use only after RQ1–RQ3 signal exists. | P1 |
| 34 | [Decomposed OPD / VGS (2606.00564)](https://arxiv.org/abs/2606.00564) | Split language-prior and visual-grounding gradients; steer toward the latter | Shows visual improvement can be a gradient-geometry issue even when response support is unchanged. Adds a second causal layer. | P1 |
| 35 | [Visual-OPSD (2606.18974)](https://arxiv.org/abs/2606.18974) | Privileged visual-thought context teacher distills to text-only reasoning | Another example of context-asymmetric self-teaching; important for later modality ablations, not for the immediate operator phase diagram. | P2 |
| 36 | [ReOPD (2607.04763)](https://arxiv.org/abs/2607.04763) | Replay reliable teacher prefixes; student acts locally without live environment | In multi-turn settings, more student-on-policy history can make the teacher less reliable. Generalizes RQ2 from token support to prefix support. | P1 |
| 37 | [ReAct (2210.03629)](https://arxiv.org/abs/2210.03629) | Interleave reasoning and environment actions | Foundational environment-occupancy context for why multi-turn OPD differs from static answer generation. | P2 |
| 38 | [Reflexion (2303.11366)](https://arxiv.org/abs/2303.11366) | Verbal feedback and retry memory for agents | Demonstrates dense feedback can be episodic/textual rather than logit-level; useful comparison for future agentic extensions. | P2 |

## TREK comparison: what must change in the project claim

The paper named in discussion is **TREK**, not TERK.  Its central algorithm is
Teacher-Routed Exploration via Forward KL:

- estimate unaided student pass rate with 16 rollouts;
- route prompts at or below `1/8` success;
- draw four external-teacher or self-context proposals;
- keep verifier-passing proposals;
- rank them by trimmed length-normalized student NLL;
- retain the two most student-proximal proposals;
- run one epoch of forward-KL/NLL consolidation;
- return to ordinary GRPO.

TREK already shows early-RL gains on math and agent tasks and directly compares
against OPD, including an off-policy OPD control on the same teacher
trajectories.  In its setting, forward-KL is stronger because it penalizes
missing teacher mass, whereas OPD can only shape sampled student states.

Therefore do not claim:

- "first to use distillation to expand RL support";
- "first to distill before RL";
- "first to route low-pass prompts";
- "OPD alone repairs zero-support prompts".

Claims that remain testable:

1. TREK's low-support bridge is one cell in a larger operator-by-state phase
   diagram; verified FKL should dominate missing support, while OPD may dominate
   locally compatible rare/mixed student states.
2. `U_G(p)` predicts the actual supply of mixed GRPO groups more directly than
   pass rate or `p(1-p)` alone when group size is fixed.
3. Prompt-level delta `U_G`, not just an aggregate accuracy jump, predicts
   realized early-RL gradient-bearing groups and reward improvement.
4. Teacher local rankability/compatibility explains when exact-token OPD helps;
   high disagreement without local support should not.

These are stronger only if the TREK-like FKL arm is implemented as a mandatory
baseline under the same prompt cohort and explicit budget ledger.

## A reading plan that builds a usable mental model

Do not read 38 papers linearly.  Use four passes.

### Pass 1 — Build the coordinate system (half day)

Read the abstracts, main figure, objective, and central ablation of:

1. GKD;
2. DeepSeekMath/GRPO;
3. Rethinking OPD;
4. PACED;
5. TREK;
6. sparse-to-dense.

For each, fill exactly six fields: sampling distribution, teacher information,
loss direction, routing granularity, support assumption, downstream outcome.
The goal is not remembering numbers; it is learning to locate any method in the
same coordinate system.

### Pass 2 — Compare causal interventions (half day)

Read ReGFT, SRPO, HAPO, KDRL, and Tsallis.  For each paper write:

- what variable is intervened on;
- what is held fixed;
- whether the method creates a new successful mode or only reweights an
  existing one;
- what result would falsify the authors' mechanism.

### Pass 3 — Learn teacher reliability (half day)

Read TIP, Token Teachability, and Position-Weighted OPSD together.  Draw one
diagram from response-level state → prefix/token state → teacher correction.
This prevents "teacher gap" from becoming an unexamined scalar.

### Pass 4 — Add modality only after the core map (half day)

Read VOLD, VA-OPD, VGS, and ReOPD.  Ask which additional state variable appears:
visual dependence for VLMs, prefix reliability for agents.  Keep these as
extensions of the map, not as a replacement for it.

## How to remain "armed" after the survey

Maintain three artifacts rather than prose notes alone:

1. **Claim ledger:** one row per claim with supporting experiment, strongest
   competing explanation, and missing control.
2. **Operator matrix:** rows are observed states; columns are SFT/verified FKL,
   exact-token OPD, RL, and skip.  Every experiment should update one cell.
3. **Results ledger:** immutable run ID, commit, config hash, dataset hash,
   checkpoint, budget counters, output path, gate status, and one-sentence
   conclusion.

After reading any new paper, force a 10-minute prediction before looking at its
results: which state/operator cell should win, and what failure mode should
appear?  The difference between the prediction and result is the actual
learning signal.  Summarizing after seeing the answer creates much weaker
research memory.

## Making coding agents useful for research rather than expensive autocomplete

Coding agents fail most often when they are asked to infer the scientific
contract while modifying a complex trainer.  Split the work into four gates:

1. **Read-only audit:** exact files/lines, tensor contracts, current artifacts,
   and missing components.  No patch yet.
2. **Typed contract:** input schema, output schema, equations, budget counters,
   invariants, and failure conditions supplied by the researcher.
3. **Small implementation:** one operator or metric per branch with unit tests
   and a two-prompt synthetic smoke.
4. **Execution:** immutable manifest, resolved config, preflight, then full run.

Require agents to return evidence, not reassurance:

- exact command executed;
- test output;
- output artifact paths;
- one representative record;
- invariant checks (UIDs, token hashes, response masks, finite scores);
- diff and known unimplemented items.

Never allow "the script ran" to substitute for a semantic assertion such as
"the teacher scored the same raw tokens" or "the arms used the same prompt
cohort."  Those need explicit machine-checkable tests.

## Source-ingestion status

The following high-overlap papers were clipped with the local universal
clipper into `llm-wiki/raw/00_Inbox`, with Markdown, downloaded figures, and a
preserved `source.html` in each paper's `_assets` directory:

- TREK;
- PACED;
- Sparse-to-Dense Reward Principle;
- Token Teachability;
- TIP;
- SRPO;
- ReGFT;
- HAPO;
- Tsallis Loss Continuum.

Per the llm-wiki learning workflow, these sources should remain in Inbox until
the Learning Brief is reviewed.  Only after confirmation should the permanent
project matrix and research notes be updated and the source bundles archived.
