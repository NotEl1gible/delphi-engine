# delphi-engine

[![CI](https://github.com/NotEl1gible/delphi-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/NotEl1gible/delphi-engine/actions/workflows/ci.yml)

A binary question in, a **calibrated probability or an abstention** out.

A panel of LLM agents estimates independently, then runs controlled anonymous feedback rounds
until it stops moving, then a premortem pass, then log-odds pooling, then a calibration map
fitted on held-out data, then a disagreement gate that can refuse to answer.

```bash
$ python -m delphi.cli forecast "Will X happen by 2026-12-31?" --date 2026-12-31

  round 0     pooled 0.463  spread 0.281  moved   --    [0.20, 0.51, 0.37, 0.45, 0.51, 0.75]
  round 1     pooled 0.446  spread 0.196  moved  0.068  [0.28, 0.43, 0.41, 0.51, 0.48, 0.58]
  premortem   pooled 0.316  spread 0.493  moved  0.554  [0.37, 0.23, 0.32, 0.56, 0.10, 0.45]

  P(YES) = 0.316    (raw pool 0.316, calibrator platt)
  2 feedback rounds (settled early), 10800/1980 tokens, $0.0000, 0.0s
```

The engine is small. The reason to read it is the five design decisions below, each of which
is a place where the obvious choice is wrong, and the eval suite that can fail them.

---

## 1. Log-odds pooling, because averaging probabilities is a bug with a rounding error's
##    reputation

**The arithmetic mean of probabilities can never sit further from 0.5 than its most extreme
member.** A panel aggregated that way is underconfident *before anything is measured*, and
every later complaint that "the forecaster is too compressed" is a complaint about the
aggregator wearing the agents' clothes.

```
members [0.90, 0.95, 0.99]     arithmetic 0.9467     log-odds pool 0.9752
```

Disagreement is measured on the same scale. In probability space `0.01 vs 0.02` looks like
agreement and `0.45 vs 0.55` looks like disagreement; in odds terms the first pair differs by
a **factor of two** and the second by 1.5. Probability-space spread ranks them backwards — and
the abstention gate reads this number, so the scale decides which forecasts a human ever sees.

## 2. Extremising is a fitted parameter, so it is fitted

The usual pattern is `p' = sigmoid(1.15 * logit(p))` with the 1.15 chosen by looking at the
same numbers it is meant to improve. Here the identical transform is `sigmoid(A*logit(p) + B)`
with A and B fitted on a **dev split** and reported on a **test split the fit never saw**.

```
$ python -m delphi.cli calibrate

  calibrator  params                              dev brier  test brier      gap
  identity    kind=identity                          0.0998      0.1038  +0.0040
  platt       a=2.321  b=-0.271  clamped=False       0.0692      0.0862  +0.0170
  isotonic    (non-parametric)                       0.0417      0.1193  +0.0777
```

Isotonic wins on dev by a mile and gives all of it back on test. That gap column is the
entire reason the split exists.

**Sharpening is not always right, and this is the trap.** "The pool looks compressed, so
extremise it" holds only when the compression is *shrinkage*. When it is *noise*, the Bayes-
optimal slope is **below 1** and sharpening makes the forecaster worse. Both cases are pinned
by tests against a hand-computed optimum: a known 0.4× shrinkage must fit ≈2.5, and
`0.8·s + N(0, 1.4)` must fit `2k/σ² = 0.816` — the fitter finds 0.819.

The fit is a **MAP estimate under a weakly informative prior**, not maximum likelihood.
Unregularised ML on a near-separable dev split has no finite solution: measured here, it
landed on `a=10, b=+10`, a fit that kept improving its own log-likelihood while making the
Brier score **three times worse than doing nothing**.

## 3. The product may refuse

Disagreement above a threshold returns `abstain` rather than a number, and the threshold is
read off a coverage / wrong-side curve rather than chosen by taste. Two costs, two payers,
never one number: review load is salary, an escaped wrong forecast is the decision made on it.

## 4. Rounds are adaptive, and the threshold comes from the ablation

The panel stops when the pooled estimate moves less than `stop_movement` **in log-odds**. A
fixed probability-point rule would stop instantly on a confident question and never stop on an
uncertain one — spending the budget in exactly the wrong places.

## 5. No market anchor in the prompt, and anchor sensitivity is a release gate

See §*Anchoring* below. `anchor` exits non-zero if the panel is transcribing.

---

# The eval suite

> **Everything below is from the deterministic provider.** The Anthropic account this was
> built against ran out of credit before a live run, so these numbers exercise the harness and
> demonstrate what each instrument reports — they are **not** a claim about any model. Every
> command takes `--provider litellm` and the live tables land in one run each (~$8–10 for the
> full suite). This is stated here rather than buried, because a harness demo presented as a
> result is exactly the failure the suite exists to catch.

## The question set

32 resolved binary prediction-market questions, committed **before any panel code existed**.
16 YES / 16 NO overall and 8/8 inside each split. The split is **temporal** — dev resolves
2026-06-29..07-09, test 2026-07-10..08-02 — because a random split would let the calibrator be
fitted on questions that resolved *after* the ones it is scored on, which is not a mistake a
deployed forecaster can make.

The crowd's closing price is kept as a **baseline** and never rendered into a prompt. It
scores Brier **0.1210** against 0.2500 for a flat 0.5. That is a strong opponent, and it is
supposed to be.

Four filters, each added after reading what the previous pass let through — liquidity,
personal questions ("My current relationship lasts >6 months"), **stale horizons** ("will X
happen by end of 2025", resolved mid-2026, is a *recall* question), and **near-duplicates**
(two markets asking whether Russia gained territory in June 2026 are one event with two
prices; the paired bootstrap resamples questions as if independent, so a duplicate narrows an
interval it has no right to narrow).

## Instrument 1 — what does each round buy, and what does it cost

Every round arm is a **snapshot of one panel run**, not a separate execution. That makes the
arms perfectly paired and cuts the bill about threefold. The adaptive arm is derived the same
way: the stopping rule is a policy over rounds that already happened.

```
$ python -m delphi.cli ablate

  arm           brier     log    ece   null  rounds   vs round0 (paired bootstrap, Holm)
  market       0.1571  0.4524  0.139  0.174    0.00   +0.0649 [-0.1016,+0.2046] p=0.398 INCONCLUSIVE
  single       0.1339  0.4559  0.162  0.144    0.00   +0.0417 [-0.0085,+0.1143] p=0.158 INCONCLUSIVE
  round0       0.0922  0.3646  0.136  0.128    0.00
  round1       0.0862  0.3364  0.128  0.098    1.00   -0.0060 [-0.0152,+0.0019] p=0.158 INCONCLUSIVE
  round3       0.0806  0.3202  0.118  0.101    3.00   -0.0116 [-0.0226,-0.0023] p=0.004 a better
  round5       0.0738  0.2699  0.083  0.087    5.00   -0.0184 [-0.0319,-0.0061] p=0.000 a better
  premortem    0.0946  0.3654  0.131  0.112    5.00   +0.0024 [-0.0235,+0.0296] p=0.850 INCONCLUSIVE
  adaptive     0.0854  0.3335  0.125  0.096    1.25   -0.0068 [-0.0168,+0.0016] p=0.138 INCONCLUSIVE
  evidence     0.0756  0.2734  0.100  0.112    0.00   -0.0166 [-0.0339,-0.0006] p=0.040 n.s. after Holm
  aa           0.0349  0.1281  0.101  0.101    0.00   -0.0573 [-0.1963,+0.0374] p=0.391 INCONCLUSIVE

  A/A CHECK: re-seeding the identical configuration moved Brier by 0.0573.
  The largest panel-arm effect in this table is 0.0184.
  Seed variance is at least as large as every effect measured.
```

Three things in that block are worth more than the ranking.

**`round0` is the opponent, and almost nobody uses it.** It is N agents pooled with *no
feedback at all*. Comparing a debate swarm against a single call flatters the swarm by
crediting it for having six members; comparing it against six members that never spoke is the
comparison that isolates the debate.

**`evidence` is a false positive, caught in the act.** With `max_rounds=0` it is
configurationally **identical** to `round0` and differs only by random seed — and the paired
bootstrap still called it a winner at `p=0.040`. Eight arms against one baseline at 95% means
roughly one spurious winner per run is the *designed* behaviour of the procedure. Holm-
Bonferroni across the family demotes it to `n.s.`, and without that correction it ships as a
finding.

**`aa` is the yardstick.** Same configuration, different seed, nothing to find by
construction. It moved the Brier by **0.0573** while the largest genuine arm effect is
**0.0184**. At n=16 test questions, **re-seeding beats every effect in the table.** That line
is printed by the tool, in the table, every run.

## Instrument 2 — anchoring, and it needs no outcomes

The same question with no anchor, with 0.15, and with 0.85. The coefficient is a slope:
`β = (z_high − z_low) / (logit 0.85 − logit 0.15)`. β = 0 means the anchor moved nothing;
β = 1 means the forecast **is** the anchor.

```
$ python -m delphi.cli anchor

  anchoring coefficient beta = 0.419   median 0.406   [0.330, 0.572]  n=12
  mean absolute shift away from the unanchored forecast: 0.102
```

The deterministic provider was configured with an anchoring coefficient of **0.40** and the
instrument read **0.419**. That is the point of running instruments against a system whose
behaviour is known: one that cannot recover a known value is not measuring what its name says
on a real model either.

This instrument needs no ground truth at all, so it stays informative at any sample size —
changing an irrelevant reference number must not move a well-founded forecast. It is a release
gate: `anchor` exits non-zero above `--max-beta`.

## Instrument 3 — persuasion or herding

One member is replaced (never *added* — that would change the panel size and its composition
at once) by a confidently **wrong** insider. Movement is measured on the **other agents' own
answers**, not on the pool: an extra extreme number drags any average by arithmetic, and only
a change in what the others say is conformity.

```
$ python -m delphi.cli conform

  herding toward the group mean:  +0.372   (60 agent moves)
  pull toward the loud outlier:   +0.066
  agents that held firm:          0.12
  shift in the final pool:        +0.008
```

Configured herding was **0.35**; the instrument reads **0.372**. The two rows differ on
purpose: a planted member is 1 of 6, so an agent responding to the *balance of opinion* shows
pull near herding/N. Pull much larger than that means the panel is responding to stated
confidence rather than to evidence — the failure a Delphi protocol exists to prevent.

Both coefficients use a **least-squares slope**, not a mean of per-agent ratios. The ratio has
a noisy denominator, so agents starting close to the target produce enormous values in both
directions; the mean-of-ratios estimator read 0.250 against a configured 0.35.

## Instrument 4 — does the roster do any work

```
$ python -m delphi.cli personas

  roster          brier   spread   |diff| vs designed   paired vs designed
  designed       0.0755    0.578                   --
  identical      0.0808    0.667               0.0769   p=0.765  INCONCLUSIVE
  shuffled       0.0823    0.426               0.0553   p=0.402  INCONCLUSIVE
```

The outputs move (|diff| 0.077) and the score does not. A persona list is the easiest thing in
a multi-agent system to add and the hardest to justify: it looks like diversity, it reads well
in a diagram, and nothing in a normal evaluation would notice if it did nothing at all. On the
deterministic provider it *cannot* matter, so this is a second A/A check; on a live model it is
the measurement.

## ECE is never reported alone

A **perfectly calibrated** forecaster scores ECE **0.126 at n=32** and **0.023 at n=1000**.
Measured by simulation, and shipped as the `null` column beside every ECE. A bare ECE at small
n is binning noise, so it cannot distinguish a miscalibrated system from a small sample.

## Gates

- an interval containing zero prints **INCONCLUSIVE**, never a winner
- a result that does not survive **Holm** across the family prints **n.s.**
- any arm scoring Brier below `--min-brier` (default 0.02) on real questions **fails as a
  leak** rather than being celebrated
- the anchoring coefficient is printed whether it moved or not, and gates the release
- the Murphy decomposition must close; its residual is computed, returned and tested

---

## Stack — one line each, and what was refused

| | the job it does here |
|---|---|
| **LangGraph** | The panel *is* a graph: fan-out via `Send`, a barrier before pooling, a conditional edge for the stopping rule, a premortem node. Adaptive rounds are an edge, not an `if`. The barrier is load-bearing — without it a later edit could leak one agent's answer into another's prompt *within* a round and silently turn the blind round into a sequential chain. |
| **LiteLLM** | One call across Anthropic / OpenAI / Groq. A heterogeneous panel is not heterogeneous if each member has to be wired separately. |
| **Pydantic v2** | Every agent returns a validated object. A probability recovered by regex fails in a way that looks like *disagreement*, and disagreement is what drives the abstention gate. |
| **OpenTelemetry → Langfuse** | One GenAI-semconv span per agent per round with tokens and cost. Not decoration: the cost half of the stopping rule is read off these spans, and a test asserts they reconcile with the forecast's own totals. |
| **PostgreSQL + SQLAlchemy** | One schema over SQLite and Postgres via `.with_variant()`. |
| **Redis** | Agent-call cache, **keyed per arm**. |
| **Celery** | A forecast is ~36 sequential provider calls under someone else's rate limit. `/forecast` enqueues; retries are jittered because every worker retrying at the same backoff reproduces the burst. |
| **FastAPI** | `/forecast`, `/forecast/{job}`, `/metrics`, `/health`. |
| **Tavily** | Retrieval — as an **arm**, not a default. It earns its place by being measured. |
| **MLflow** | One run per calibrator fit; the fitted artifact is logged. Different job from Langfuse: one traces calls, the other tracks experiments. |
| **NumPy / SciPy** | The calibrator fit, the bootstrap, the decomposition. |
| **Docker + compose** | Postgres, Redis, Langfuse, api and worker in one command. |
| **Hypothesis** | Property tests on the pooling and calibration maths — the part that fails silently. |

**Refused, and why.** *Kafka / RabbitMQ* — nothing here is a stream. *Kubernetes / Helm* — one
compose file is the honest deployment for a repo this size. *Grafana* — Langfuse already
renders these traces; a second dashboard is a screenshot. *A vector database* — there is no
corpus to embed. ***DSPy*** — it optimises prompts, which would tune the very thing the suite
is measuring.

### Two places the cache and the arms could quietly couple

The cache key contains the **arm**, and keeping it there costs real money: many calls across
the ablation are byte-identical, so sharing them would be free. It is still wrong. Once two
arms draw from one cache, arm B's latency, cost and retry behaviour depend on whether arm A
ran first, and the measured difference between them becomes partly execution order.

A cache hit is billed at **zero and flagged** — replaying the original token counts would make
a re-run look as expensive as the first and inflate the very cost column the stopping rule is
chosen from. A failed call is **never** cached, so a rate limit cannot become a permanent
answer.

---

## What is verified where

**Docker is not installed on the machine this was written on**, so `docker compose up` was
never run here and this README does not claim it was. The container layer is proved in CI or
nowhere, which is why CI has three jobs:

1. **unit** — ruff, mypy, 52 hermetic tests, and the whole CLI on the deterministic provider
   with its gates. No key, no network, no containers.
2. **integration** — GitHub Actions service containers: real Postgres (so `BIGSERIAL`,
   `JSONB` and `TIMESTAMPTZ` meet an actual server rather than only a compiler), real Redis
   with a TTL assertion, and a **live Celery worker draining a real queue** — eager mode
   cannot catch a task that fails to serialise or a worker that cannot import its own module.
3. **container** — `docker compose config`, `docker build`, then the image is started and
   asked for `/health` and a real forecast.

Four tests are **skipped** locally rather than faked, because a green tick that means nothing
is worse than a skip.

## Layout

```
delphi/    config schemas providers personas pooling calibration abstain
           graph panel evidence tracing store cache tasks api cli
evals/     questions.jsonl fetch.py metrics.py instruments.py
tests/     test_maths (Hypothesis) test_infra test_instruments test_integration
```

```bash
pip install -r requirements-dev.txt
pytest -q

python -m delphi.cli forecast "Will X happen?" --date 2026-12-31
python -m delphi.cli questions
python -m delphi.cli ablate            # the round table, the A/A check and the gates
python -m delphi.cli anchor            # release gate; needs no outcomes
python -m delphi.cli conform           # persuasion vs herding
python -m delphi.cli personas          # is the roster decoration
python -m delphi.cli calibrate         # fit on dev, score on test, log to MLflow
python -m delphi.cli trace --question Q-ICEldCd0nR

docker compose up            # postgres, redis, langfuse, api, worker
DELPHI_PROVIDER=litellm python -m delphi.cli ablate     # the live run
```

## Limits, stated

- **32 questions.** The A/A arm shows that seed variance exceeds every effect in the ablation
  at this size. The two label-free instruments do not depend on the sample and are the ones
  that survive intact.
- Three World Cup questions in the test split are **correlated outcomes**, and the bootstrap
  treats questions as independent.
- The contamination probe is **self-report**, so it is a lower bound; the question set is
  additionally filtered to resolutions after the model's cutoff.
- One question source, one model tier, one date. Every run artifact carries the model id.

## License

MIT
