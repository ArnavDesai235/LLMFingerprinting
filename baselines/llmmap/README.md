# LLMmap baseline on our 30 base models + 14 variants (Ollama)

LLMmap (Pasquini et al., USENIX Security 2025; github.com/pasquini-dario/LLMmap): 8 crafted queries, each answered under a
random prompt configuration (system prompt, CoT/RAG wrapper, sampling params), embedded with
multilingual-e5-large-instruct, mapped by a pretrained open-set network; a model is identified by the nearest
per-model template (mean embedding over ~100 configs). Responses are capped at 100 new tokens / 650 chars.

The pretrained model covers 52 HF models and the repo's template script only supports HF/OpenAI/Anthropic, so we
(1) generate traces through Ollama, (2) build templates for OUR models with the pretrained network (no retraining),
(3) evaluate on held-out prompt configurations.

## Run (on the cluster, Ollama serving the models)
    git clone https://github.com/pasquini-dario/LLMmap && pip install -r LLMmap/requirements.txt requests scipy
    export OLLAMA_HOST=...; export HF_MODEL_CACHE=...      # needs HF access once for the e5 embedder
    python build_dataset.py --check                         # fix tags in models.json if any are missing
    python build_dataset.py --llmmap ./LLMmap               # ~44 models x 150 configs x 8 queries = 53k short generations
    python evaluate.py --llmmap ./LLMmap --device cuda      # writes llmmap_results.json
    # second arm for reasoning models (raw <think> output, truncated at 100 tokens):
    python build_dataset.py --think raw --only <reasoning labels>; python evaluate.py --think raw --out raw.json

## Decisions to confirm
- Reasoning models: default think=off (and <think> stripped). LLMmap's embedder expects plain chat replies and 100
  tokens of thinking would hide the answer. The raw arm quantifies the cost of that choice.
- Same prompt configurations are used for every model (seeded), so models are compared on identical probes.
- Upstream bug: PromptConfFactory.sample(n, pool) ignores `pool` (always TRAIN), so LLMmap's own test configs come from the
  train pool. We call sample_one(pool) so test configs are truly held out.
- Template and tau (95th percentile of own train distances) use train configs; probes are test configs.
- Query budget: LLMmap needs exactly 8 queries per probe. Compare against our numbers at q = 8 queries asked
  (all answers), not at k observed errors.
