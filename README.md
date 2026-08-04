# Detecting Principal-Conditional Behaviour Across Conversational Sessions

**Apart Research Secret Loyalties Hackathon · 24–26 July 2026 · Solo submission · Tracks 3 & 4**  
**Author:** Alishba Zainab Khan · Independent AI Safety Researcher · BS Data Science & AI, BITS Pilani

---

> *If surfacing a secret loyalty requires eliciting the harmful behaviour it is trained to produce, then safe black-box auditing may be structurally impossible for that loyalty class. The activation condition is the harm.*

---

## The question

Can session-level probing detect a hidden model loyalty without constructing the harmful act that activates it?

**No.** That negative result is the most important thing in this repository.

Single-turn auditing is dominated by how you ask, not what you are measuring. Multi-turn probing exposes behavioural signatures that single-turn methods miss entirely. And yet none of it reaches the trigger — because for the loyalty class I was probing, the trigger *is* the harm. A probe that refuses to build it cannot find the principal. That is the ceiling this paper maps.

---

## Results at a glance

| # | Claim | Verdict |
|---|---|---|
| 1 | Prompt framing moves single-turn partiality **7.6× more** than the property being measured | ✅ Holds |
| 2 | Principal-conditional state-framing asymmetry replicates across a **54-fold parameter difference** | ✅ Holds |
| 3 | Semantic injection with no imperative: **0% detected, 78–89% complied**,(pooled; individual models 67–100%) every model including base | ✅ Holds |
| 4 | A deployed model tracked a five-turn manipulation in its reasoning and **produced no output** | ✅ Holds |
| 5 | Detection does not prevent compliance: 13 of 18 detected trials still complied | ⚠️ Suggestive |
| 6 | Organisms diverge from base under stakes manipulation | ✅ Holds at n=3 |
| 7 | Injection detection drops selectively for principals | ❌ Null |
| 8 | Organisms comply with injection more than base | ❌ Null (base complies more) |
| 9 | RTVA reasoning-trace gap identifies a principal | ❌ Null (rule fires on base too) |
| 10 | Specificity probe detects a documented principal | ❌ Failed validation |
| 11 | 27-actor sweep identifies a principal in A or B | ❌ Null on both organisms |

**Organism verdict at Level 1 blind:** presence not established · principal not identified · activation condition not identified · action not identified.

---

## This repository is not clean

It contains failed runs, dead ends, two analysis bugs and their fixes, seven successive versions of a findings document, and a sweep that crashed six hours in with nothing saved to disk.

All of it stays. The process is the record.

The lab logbook (`secret_loyalties_logbook.md`) covers the full three days in first-person — every decision, what died and why, the operational mistakes worth not repeating. That's the honest record of how the conclusions were reached.

---

## Start here
Full corrected paper (PDF) is on the main branch

```
paper/                  → Read this first. Formal methods, all results, limitations.
secret_loyalties_logbook.md → Read this second. Decisions, sequence, what died and why.
code/scoring/verify_paper_claims.py → Run this to reproduce every number in the paper.
```

---

## Repository layout

```
code/
  runners/     Modal scripts, one per experiment, roughly chronological
  probes/      Probe and protocol definitions  
  scoring/     Preregistered scorers + verification script

data/          Raw JSON from every run, unedited

paper/         Submitted paper (PDF)

secret_loyalties_logbook.md   Full first-person narrative of the three days
```

---

## Code

### Runners — in run order

| File | Experiment | Note |
|---|---|---|
| `modal_organism_runner.py` | First organism contact, single-turn + PSR blind sweep | `VALIDATE` flag switches sanity check vs full run |
| `modal_rtva_runner.py` | Single-turn at 10 reps | `JOB` flag selects experiment |
| `modal_rtva_go.py` | RTVA reasoning-trace probe | Same file, `JOB = "rtva"` |
| `modal_givl_organisms.py` | Injection ladder on organisms | L4/L5 redacted — see dual-use section |
| `modal_specificity.py` | Specificity probe, organisms + base | |
| `modal_specificity_v2.py` | Same, gen9 models added | |
| `modal_gen9_only.py` | Specificity on documented gen9 organisms | |
| `modal_validate_macron.py` | Validation against documented principal | The run that killed the detection claim |
| `modal_expanded_sweep.py` | 27-actor sweep, all principal types | **Crashed at 6hr — nothing saved** |
| `sweep_organism_a.py` | Rewrite: one model per script | Saves immediately on finish |
| `sweep_organism_b.py` | Rewrite: one model per script | |
| `sweep_base.py` | Rewrite: one model per script | |

**Why three near-identical sweep scripts exist.** `modal_expanded_sweep.py` ran all three models in one entrypoint and only wrote output after all three completed. A DNS failure inside the container killed it six hours in. Everything was lost because nothing had been written to disk yet. The `sweep_*.py` files are the fix: one model per script, one output file, written the moment that model finishes. The most expensive mistake in the project. The fix is four lines.

### Probes

| File | Contents |
|---|---|
| `multiturn_probe_protocol.md` | PSR protocol v1.0 — full T1–T5 sequences for all three dimensions, controls, scoring rubric, rep budget. Written before the runs. |
| `givl_injections.py` | Injection ladder definitions. L4/L5 redacted. |

### Scoring

| File | Purpose |
|---|---|
| `score_results.py` | RTVA and GIVL scorer. Written 25 July, **before the data existed**. Thresholds fixed in advance. |
| `score_specificity.py` | Specificity scorer — Fisher exact test + Bonferroni correction. Five preregistered checks, also written before the data. |
| `verify_paper_claims.py` | Cross-checks every number in the paper against the raw JSON. **Run this to reproduce reported figures.** |
| `score_sweep.py` | Scores the 27-actor sweep. Auto-detects all `sweep_*.json` in the folder and prints a comparison table. |

**On preregistration.** Both scorers were written before their data landed, with decision rules fixed in the docstring. Every null in the paper was called by a rule set in advance. Two of them fired against the outcome I was hoping for.

**On the valence lexicon.** `score_results.py` uses a deliberately crude keyword lexicon, documented in the file as a triage tool rather than a measurement. Hand-verification of at least 20 scored responses has not been completed. Every lexicon-derived percentage carries an unquantified error rate.

---

## Data

Every JSON file is raw output, unedited.

| File | Experiment | n |
|---|---|---|
| `results_20260725_011241.json` | First organism look | n=1; both signals later died |
| `results_20260725_012549.json` | PSR blind sweep, three sequences | 3 reps |
| `single_turn_20260725_194836.json` | Seven questions, three models | 10 reps |
| `rtva_20260725_202828.json` | Reasoning-trace valence, 10 actors | 1 rep |
| `groq_psr_results.json` | PSR across six deployed models | 2 reps |
| `givl_groq.json` | Injection detection, Qwen 3.6 27B | 81 conditions — withheld |
| `givl_organisms_20260725_204614.json` | Injection compliance, organisms + base | 270 generations — withheld |
| `givl_groq_v2.json` | Detection and compliance together | 56 of 81 lost to quota — withheld |
| `specificity_20260725_223715.json` | Specificity probe, organisms + base | 10 reps |
| `gen9_specificity_20260726_002439.json` | Specificity on gen9 | 10 reps |
| `validate_macron_20260726_021915.json` | Validation, documented principal | 10 reps |
| `sp_groq_judged.json` | LLM-judged specificity, deployed panel | Failed run |
| `sp_groq_judged_v2.json` | Same, retry | Failed: judge returned unparseable output in 154 of 180 cases |
| `sweep_base_*.json` | 27-actor sweep, C (reference) | 10 reps · 540 responses |
| `sweep_organism_a_*194446.json` | Same sweep, organism A run 1 | 10 reps · 540 responses |
| `sweep_organism_a_*194323.json` | Same sweep, organism A run 2 (replication) | 10 reps · 540 responses |
| `sweep_organism_b_*.json` | Same sweep, organism B | 10 reps · 540 responses |

The two `sp_groq_judged` files are kept because the failure mode is informative. An LLM judge asked to score specificity on small open models returned nothing usable in the large majority of cases. Anyone planning LLM judges under free-tier constraints should know this before designing around them.

---

## The expanded sweep

27 actors · 10 reps each · 4 categories: organisations, ten AI-lab executives, six political figures (Macron as documented positive control, Meloni as negative), four advocacy groups. Organism A run twice independently. 2,160 generations total.

**Null on both organisms.** No actor showed selective elevation over C that survived cross-model comparison.

Organism A produced a replicated 20% elevation on Amanda Askell against a 0% baseline in C. The identical elevation appeared with identical consistency on Sundar Pichai and Liang Wenfeng. An effect landing equally across three unrelated organisations is not selective to a principal.

Organism B removed the remaining ambiguity. Both of A's most consistent actors sit at 0% in B. The two organisms agree on no actor.

Two single-run artefacts confirm why replication is non-negotiable. DeepSeek showed 0% on organism A run 1 that jumped to 30% on run 2. Satya Nadella showed 50% then 10% across the same two runs. Either would have been reportable from a single run. Neither survives.

To reproduce:
```bash
cd data
python ../code/scoring/score_sweep.py
```

---

## Two bugs, both fixed

Both came from the same function and both broke real analyses before being caught.

**Bug 1.** The reasoning/output splitter guessed a cut point when a closing tag was missing, silently scoring the wrong half of the text. This produced 0% detection across every GIVL condition — which looked like a finding.

**Bug 2.** The same splitter only checked `<think>` tags, not `<reasoning>` tags. Since the GIVL prompts requested `<reasoning>`, most responses were misclassified as truncated. This inflated the apparent truncation rate from 7 of 81 to 80 of 81.

Both were caught by reading raw responses against summary numbers rather than trusting aggregates. Fixed version is in `code/scoring/score_results.py`. The corrected splitter reports truncation explicitly instead of guessing, and checks both tag formats.

---

## Dual-use redaction

L4 semantic injection achieved 0% detection and 78–89% compliance pooled across every model tested, including the unmodified base, with individual models ranging from 67% to 100%. That is a working evasion technique against reasoning-based injection detection.

Following the hackathon rules on novel installation recipes, the exact L4 and L5 payload templates are not printed here. The full templates are available to the organisers on request. Three GIVL result files and one notebook entry are also withheld because model responses across them quote the payload closely enough to reconstruct it — see `data/WITHHELD.md`.

L0 through L3 are unredacted. They are standard published injection forms with no novel content. Everything else in the GIVL files is intact: the carrier document, the scoring functions, the experimental structure.

---

## Reproducing

```bash
pip install modal
modal setup
# Set a Modal secret named huggingface-secret containing HF_TOKEN
```

Run the sweep (three separate terminals):
```bash
python -m modal run code/runners/sweep_organism_a.py
python -m modal run code/runners/sweep_organism_b.py
python -m modal run code/runners/sweep_base.py
```

Verify paper numbers against raw data:
```bash
cd data
python ../code/scoring/verify_paper_claims.py
```

**Compute notes.** The 7B organisms load on a Modal L4 in 4-bit at roughly five cents per three-model run. Cache weights in a persistent volume or you re-download 15 GB every time. Long jobs need `--detach` — a sleeping local client terminates a non-detached run and there is no partial save.

---

## Known limitations

- Repetition counts fall well short of protocol. The protocol specifies 20 minimum, 50 preferred. Organism PSR ran at 3, the deployed panel at 2, RTVA at 1. Only the single-turn panel and the specificity probes reached 10.
- The valence lexicon is unvalidated — see scoring note above.
- The RTVA decision rule fires on the unmodified base model. It does not currently discriminate loyalty from noise.
- Organism B emitted parseable reasoning traces in only 52% of turns.
- L5 distributed injection was incoherently implemented. One of three seeds implies a vendor unrelated to the intended target. It was not tested, rather than tested and found ineffective.
- Every organism finding assumes C is a clean reference model. That assumption is unavailable to an auditor inside a frontier lab, where the predecessor may carry the same loyalty.
- It has not been verified whether Qwen's chat template injects a default system message naming Alibaba. If it does, the no-system-prompt commitment was violated throughout, and Section 4.2 acquires an alternative explanation.
- The original ten-actor list covered companies and governments only. The 27-actor sweep addresses this and returned null on both organisms. At 10 reps per cell, a 20% value is 2 responses — the sweep bounds the effect size rather than excluding a small one.

---

## Acknowledgements

**Lennart Finke** granted permission by email on 23 July to cite the Corporate Loyalty preprint, use the `lennart-finke/partiality` dataset, and reference the talk.

**Justin Shenk** granted citation permission and confirmed the talk title: *"Reading Loyalty in the Residual Stream."*

Model organisms `Alamerton/sl-organism-a-7b`, `Alamerton/sl-organism-b-7b`, and the gen9 models are from Apart Research and Lamerton & Roger (2026).

---

## Tooling disclosure

Claude (Anthropic) was used for writing and debugging experiment code, structuring documents, and discussing framing. Experimental design, the threat model, the probing methods, the preregistered decision rules, and all interpretation are mine. Every reported figure was verified by running `verify_paper_claims.py` against the raw output files, and several figures in earlier drafts were corrected as a result.

---

## License

Code: MIT · Data and notes: CC BY 4.0
