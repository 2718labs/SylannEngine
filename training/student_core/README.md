# V3 core offline value gate

**Status:** offline-only engineering instrument. It is not connected to the runtime and does not
collect live data.

**Intended consumer:** any `sylanne_core` host that needs to determine whether a learned affect core
is justified before implementation. The reusable surface is the offline instrument and its
versioned evaluation contract; host-specific collection adapters are outside this repository.

## What this is

An offline instrument that answers ADR-0001's value gate: can a persistence-anchored learned affect
predictor beat trivial persistence and a reactive baseline, on the autocorrelated slice, by
`>=15%` relative and `>=0.02` absolute paired-MAE, with an episode-bootstrap confidence interval
whose lower bound clears the threshold? If not, the verdict is `NO-GO`.

Modules (all numpy/sklearn, no torch, deterministic):

- `residual_affect_probe.py` — `ResidualAffectProbeV1`: `a_hat = clip(a_prev + w·x)`, `w` init 0, so
  day-zero == persistence and the only learned quantity is the residual the metric scores → a win
  cannot be manufactured by architecture.
- `leakage_guards.py` — session-disjoint / variance-floor / no-label-column / decorrelation /
  field-ablation / `derive_is_iid` (the pre-registered autocorrelated-slice rule).
- `rung_registration.py` — `RungRegistrationV1`: freeze estimator + baselines + floors + slice-rule;
  the digest binds the split seed (anti-p-hacking; a moved goalpost reshuffles the split).
- `offline_gate.py` — persistence + steelman ridge/GBM baselines + the coded **powered** per-baseline
  gate (a GO requires each improvement's CI lower bound to clear the abs floor).
- `gate_controls.py` — episode (cluster) bootstrap paired CI, Holm family-wise correction,
  residual-shuffle placebo.
- `pilot.py` — `sigma_d` / `m` / `rho_icc` (one-way ICC), `derive_N` (ADR §6.2-6.3), and the
  pre-registered escalation (effect ≤ 0 OR N > 300 OR ETA > 8wk → shelve-vs-redesign).
- `tests/` — 28 green. `python -m pytest training/student_core/tests/ -q`.

## Usage

1. Collect its **own** real corpus in the harness schema (per-tick pre-`a_t` features + assessor `a_t`
   + actual action + next outcome), behind the privacy floor (HMAC surrogate, consent gate, deletion/
   tombstone, a fixed non-empty salt). That collection is host-specific and is **not** built here.
2. Run `pilot.run_pilot` on `>= 30` sessions and apply `pilot.escalation` without overriding its
   stop conditions.
3. Commit a `RungRegistrationV1` digest **before** any performance is computed, then
   `offline_gate.run_gate`.
4. Read the verdict. `NO-GO` terminates the learned-core path for the registered evaluation rung.

## Evaluation boundary

The bundled `spike_corpus.parquet` is **SYNTHETIC** (`simulate_corpus.py`). Per ADR-0001 a synthetic
result is **not bankable** — the field is both generator and baseline (circular). Running the gate on
it yields NO-GO and proves the *machinery*; it is **not** a real-data verdict. A real verdict needs a
real collected corpus.

## Design of record

- Go/no-go contract: [ADR-0001](../../docs/design/adr-0001-v3-core-go-no-go.md)
