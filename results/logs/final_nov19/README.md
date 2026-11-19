# Final runs on the complete data (2026-11-19)

Every analysis re-run on the finished fingerprints: `bash analysis/final/run_all.sh` (about 6 min on 2 cores).

| Set | Models | Used for |
|---|---|---|
| `base30` | 30 base models, 3k questions. The 8192-token reruns (`fingerprints/v3_clean3k_rerun8192`) replace deepseek-r1:8b, qwen3:4b and qwen3:1.7b | Main numbers: rq1, a12-a16, a19, figures |
| `all44` | base30 + 14 variants (`fingerprints/variants3k`, renamed by `analysis/main/labels.py`) | rq1, a12-a16 with the variants in the library; a20 |
| `base30_raw` | 30 base models as first run (4096 tokens for all, same as held-out/shuffled runs) | a17 (3k vs held-out) |

Held-out (`holdout500`) and shuffled (`shuffle500`) runs exist for the 30 base models only, all complete.

## Changes to the scripts (see the diff on branch `final-nov19`)

- **deepseek-r1:8b is Qwen3-based** (Ollama's tag is DeepSeek-R1-0528-Qwen3-8B; config blob `model_family: qwen3`). a13/a14 used Llama-3.1-8B as its base. Now `qwen3_8b` (a13) and lineage qwen (a14). The abliterated `deepseek-r1-abl-llama_8b` is the only Llama-based distill.
- `analysis/main/labels.py`: one place for family, lineage, reasoning and variant labels.
- rq1: a row with a failed run (-1) used to drop the whole model; failed runs now count as unparsed (as in every other script).
- `early_v3/load.py`: the repo copy split llama3.1 from llama3.2 into different families. Now the same regex as `main/load.py`. The earlier a12 logs were made with the correct rule, so no earlier number changes.
- With variants present, variant-parent pairs are left out of the family regressions (rq1, a12). In a14 they get their own `twin` covariate.
- a12: k-way family overlap is skipped when the family has more models than there are families (qwen, k=8). It could never match, and with 44 models the search did not finish.
- a14: new spec D separates qwen2.5 and qwen3 lineage.
- New: `a19_variants.py` (variants vs parents) and `a20_library_with_variants.py` (identification with near-copies in the library).

## Headline numbers, base30 (earlier snapshot in brackets)

**Identification** (rq1, chance .033; [26 models, chance .038]):

| k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| Full answers | .11 | .30 | .44 | .66 [.69] | .84 [.85] | .96 [.96] |
| Errors only | .15 | .46 | .61 | .78 [.80] | .91 [.91] | .98 [.98] |
| Correct only | .05 | .08 | .10 | .15 | .23 | .34 |

- **MI question selection:** k = 10/20 gives .83/.95, against .66/.84 for random questions.
- **Open set:** k = 20 accepts .80 of known models and rejects .59 of unknown ones [.81/.65]. At k = 50 the figures are .92/.83.
- **Correlated-error distances (a15), k = 10:**
  - conditional Hamming .66
  - co-error .30
  - likelihood, wrong answers only .78

  Which wrong answer carries the identity; whether a model is wrong does not.

**Family effect** (a12, accuracy-controlled, model-level permutation):

- **Pooled:** excess +.029 (p < .001), share +.034, co-error +.018, D_H −.028 [+.032, +.037, +.023, −.033].
- **Per family**, against accuracy-matched cross-family pairs:

  | Family | Effect | Positive pairs |
  |---|---|---|
  | deepseek | +.082 | 6/6 |
  | falcon | +.056 | 6/6 |
  | llama | +.046 | 3/3 |
  | phi (4 models) | +.036 | 5/6 |
  | qwen (8 models incl. qwen3) | +.012 | 16/28 |
  | gemma | +.001 | 3/6 |
  | mistral | −.025 | 1/3 |

- **rq1 OLS:** same-family +.059 agree, +.061 err_agree (both p < .001). With mean accuracy as a control, CAPA's family coefficient falls to +.022.

**Lineage vs reasoning** (a14, 30 models, 435 pairs):

- **Coefficients:**
  - excess: both-reasoning +.076 (p < .001) vs same base lineage +.015 (p .03) [+.089 / +.016]
  - spec D (qwen2.5 and qwen3 separate): +.080 vs +.024
- **Mean residual by pair type:**
  - reasoning-reasoning +.082 (28 pairs, 27 of them positive) [+.093, 15 pairs]
  - same lineage +.006
  - unrelated −.016

**Distills** (a13):

- **r1:8b** (true base qwen3:8b) ranks its base 1st of 26 (residual +.121), ahead of qwen3:14b (+.085). This reverses the earlier "r1:8b ignores its (Llama) base".
- **The Qwen2.5 distills** still rank their bases 8th, 20th and 12th of 26.
- **Distill-distill pairs** all sit above accuracy-matched pairs: +.06 to +.12 in excess same-wrong share. The difference in probabilistic share (share_p) is smaller, about +.00 to +.05.

**Unseen questions** (a16): the text-feature model transfers weakly. k = 20 reaches .24 against .84 for the per-question fingerprint, and position bias alone gives .23. The fingerprint is question-specific.

**Held-out replication** (a17, base30_raw vs 500 new questions):

- Identification on new questions is within .02 of 3k subsets (k = 20 errors-only .89 vs .91).
- Accuracy, unanimity and slot preference correlate r = .98 to .997 between the two sets.
- Pairwise excess same-wrong correlates Spearman .88 (Mantel p < .001).
- On the new questions: family +.017 (p .06), both-reasoning +.100 (p < .001).

**Shuffled options** (a18, 30 models [22]):

- **Content over slot:** 25/30 models follow the ending text rather than the slot [18/22]. Modal-ending share .72 vs modal-slot share .49.
- **Accuracy:** mode accuracy .635 → .674 when shuffled.
- **Error persistence:** where the fixed-order mode is wrong, the shuffled mode is the same wrong ending .45 of the time (chance 1/3).
- **Identification, wrong answers only, k = 10:** fixed .77, shuffled .35, cross-condition .21/.33.
- **Family effect, shuffled raw share:** +.047 (p < .001).

**Variants** (a19): see `claude/variants_results.md` in the project; numbers unchanged.

- **Abliterated twins** (9): parent ranks 1st for all 9. Twin identified as its parent .59 at k = 20, passes as the parent 13–83% of the time.
- **Reasoning fine-tunes** (4): parent ranks 10th–16th, identified .08, pass as the parent ≤4%. They drift toward the reasoning cluster (+.059).

## all44: the library with variants

- **Identification falls** from .83 to .72 at k = 20 (a20). Of the .11 lost, .07 is probes assigned to the model's own twin (k = 20; at k = 50 counting the twin as correct gives .93 vs .96). The base models alone drop .831 → .780 when the variants join the library, .043 of that to their twin.
- **Abliterated twins pull probes from their parents:** mistral:7b → its twin .30, llama3.2:3b .24, r1:7b .23, falcon3:10b/7b .14. Fine-tunes take ≤.01 from their parents.
- **Open set** gets harder: k = 20 accepts .69 of known models and rejects .44 of unknown ones.
- **Family and reasoning effects hold** (twins excluded or covaried):
  - a12 excess +.036, deepseek family +.091 (19/19 pairs)
  - a14 both-reasoning +.077 vs lineage +.013
  - reasoning-reasoning residual +.073 over 103 pairs; variant-parent pairs +.138
- **The real Llama distill** (r1-abl-llama:8b) ranks llama3.1:8b 19th of 26 and is closest to qwen3 models. It shares errors with the Qwen-based distills: excess .38 with r1:14b vs .25 for accuracy-matched pairs. This restores the cross-base distill result with a correctly labelled model.
