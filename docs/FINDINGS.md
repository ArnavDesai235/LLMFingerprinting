# Findings (as of 11 Nov 2026)

The project makes two claims, and this file collects the evidence for each:

- **Claim 1, identification:** a model can be identified from its answers, and especially from its errors.
- **Claim 2, shared errors:** related models pick the same wrong answer more often than capability alone predicts.

The numbers come from the logs in `results/logs/`. The figures and tables come from `analysis/make_figures.py`.

> **Update 19 Nov 2026:** every analysis was re-run on the complete data (30 base models plus 14 abliterated or reasoning-fine-tuned variants). The final numbers, and what changed, are in `results/logs/final_nov19/README.md` and supersede the numbers below where they differ. The one substantive correction: Ollama's `deepseek-r1:8b` is the Qwen3-8B distill, not the Llama-3.1-8B one (see the Distills bullet).

## Setup

- **Data.** Clean HellaSwag validation sample: 3,000 questions (2,031 WikiHow and 969 ActivityNet), plus a disjoint 500-question held-out set. See `data/README.md` for why the earlier file was replaced.
- **Models.** 30 open models in 8 families, run through Ollama with default sampling, 10 runs per question.
- **Protocol.** Reference = runs 0-4, probe = runs 5-9. Identification is by likelihood under add-0.5-smoothed per-question answer distributions. "Errors only" uses wrong probe answers, scored by P(answer | not the label).
- **Shared-error metrics, per model pair:**
  - share = P(same wrong ending | both wrong);
  - excess = share minus a position-bias null, where each model picks among the 3 wrong endings by its own slot frequencies;
  - co-error = how often both models are wrong on the same question.
- **Effects.** Family and reasoning effects are OLS coefficients that control for min and |diff| of pair accuracy, with p-values from permuting labels across models.

## Accuracy (Fig. 1)

Majority-answer accuracy on the 3k ranges from .25 (llama3.2:1b, chance level) to .88 (qwen2.5:14b). Per-model accuracy on the held-out set correlates r = .997 with the 3k.

Three of the 1b models are near chance and dominated by position bias: gemma3:1b picks ending 3 76% of the time, and llama3.2:1b and falcon3:1b are close to chance too. The R1 distills are very stochastic: only 11% of r1:7b's answers are unanimous across runs.

## Claim 1: identification (Fig. 2)

| Setting | k=5 | k=10 | k=20 | k=50 |
|---|---|---|---|---|
| Clean 3k, 28 models, all answers | .46 | .67 | .84 | .96 |
| Clean 3k, 28 models, wrong answers only | .62 | .79 | .91 | .98 |
| Held-out 500, 30 models, all answers | .44 | .65 | .82 | .95 |
| Held-out 500, 30 models, wrong answers only | .59 | .76 | .89 | .98 |

Chance is .036 for the 3k and .033 for the held-out set.

- **Which wrong answer carries the identity.** At k=10, conditional Hamming (same wrong ending) gives .69, while co-error alone (wrong at all) gives .34 (a15).
- **Choosing questions helps.** Selecting the 200 most informative questions by mutual information raises k=10 from .73 to .90 (23 models).
- **Open set.** With per-model thresholds at k=50, 88% of known models are accepted and 84% of unknown models are rejected. Thresholds frozen on the 3k lose at most 3 points on the held-out set (a17).
- **Replication.** On new questions run days later, scores are within .024 of 500-question subsets of the 3k (a17). The pairwise shared-error structure also replicates: Spearman .88 and Mantel p < .001 over 351 pairs.
- **Limit.** A fingerprint does not transfer to unseen questions from shallow text features. At k=50 it reaches .36, which is no better than slot bias alone (a16). The auditor therefore has to run the reference models on the same questions it asks the target.

## Claim 2: shared errors (Figs. 3, 4)

- **Family effect.** It is small but significant: excess +.032 and same-wrong share +.037, both p < .001 (26 models, a12). Per family it is carried by deepseek (+.084), falcon (+.064), llama (+.047) and phi (+.040). Gemma, qwen2.5 and mistral show nothing beyond capability.
- **Reasoning effect.** It is much larger than lineage (a14, 27 models, 351 pairs). In the OLS for excess, same base lineage adds +.016 (p .016) and both reasoning-trained adds +.089 (p < .001), with R² .885. All 15 reasoning–reasoning pairs are positive, with a mean residual of +.093.
- **Held-out replication.** On the held-out set with all 7 reasoning models: reasoning +.100 (p < .001) vs family +.017 (p .06).
- **Distills.** They inherit little from their base model: two are slightly above their base and two below (a13). Instead they share errors with each other across bases. For example, r1:8b matches r1:14b. (Correction, 19 Nov: Ollama's r1:8b is Qwen3-based, not Llama-based, and it does rank its true base qwen3:8b first. The cross-base claim now rests on the abliterated R1-Distill-Llama-8B, which ranks llama3.1:8b 19th of 26 and shares errors with the Qwen-based distills; see `results/logs/final_nov19/`.) They also match qwen3, which was not trained on R1 data as far as is known.
- **Interpretation.** Reasoning-style post-training or chain-of-thought at inference shapes which wrong answer a model picks more than pretraining does. These two have not been separated yet; a qwen3 thinking-off run is the planned control.

## Shuffled options (Figs. 5, 6; 22 models, first pass)

The setup is the 500 held-out questions with the endings shown in a new fixed-seed order on every run. All models see the same 10 orders.

- **Content dominates.** The modal ending is chosen in .73 of runs and the modal slot in .51. 18 of 22 models follow content. The 4 slot-driven models are gemma3:1b, llama3.2:1b, llama3.2:3b and falcon3:1b.
- **Ensembling.** Order-averaging acts like an ensemble: 10-run mode accuracy goes from .644 to .687. Where a model's fixed-order majority answer is wrong, the shuffled majority is the same wrong answer 43% of the time, the correct answer 37% and a different wrong answer 20%.
- **Identification drops.** With errors only, k=10 falls from .86 to .38 and k=50 from .996 to .76. Much of the fixed-order signal is a response to the exact prompt. Reordering by the provider is a natural evasion; an auditor who controls the prompt keeps the high numbers.
- **Family signal gets stronger.** Same-family share is +.049 (p < .001) with shuffled orders vs +.027 (p .06) with a fixed order, so the family effect is about content, not shared slot habits.

## Issues found along the way

- **Dataset corruption.** The original CSV had shifted WikiHow endings, so the CSE523 report and the Oct 8 Mistral note were computed on corrupted labels. Their "Hamming ~90% vs Jaccard <20%" result does not replicate on clean data at the same rows per query: D_H .93 vs D_J .99 at 100 rows.
- **Cluster disk quota.** On Nov 6 the home quota filled and silently stalled every job. It was fixed by removing unused Ollama models.
- **phi3:3.8b.** On clean data it rambles, which caused 351 timeouts (all retried) and 13 duplicate lines.

## Next steps

1. Run qwen3 with thinking off, to separate reasoning training from chain-of-thought at inference.
2. Finish qwen3:4b and qwen3:1.7b on the 3k, then re-run all analyses with 30 models.
3. Re-run the experiments on a few jailbroken and fine-tuned variants of library models, to see whether the fingerprint and the family signal survive.

## Related work

- Kim et al., *Correlated Errors in Large Language Models*, ICML 2025.
- Goel et al., *Great Models Think Alike*, ICML 2025 (CAPA).
- Geirhos et al., *Beyond accuracy: quantifying trial-by-trial behaviour*, NeurIPS 2020 (error consistency κ).
