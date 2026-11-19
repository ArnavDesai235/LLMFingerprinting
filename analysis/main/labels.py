"""Model labels used by the analysis scripts (final runs, 2026-11-19).

Library names: base models keep their Ollama name with ':' -> '_' (e.g. falcon3_10b). The 14 variant models
(fingerprints/variants3k) are renamed by analysis/final/build_sets.py so that the name starts with the parent's
family and ends with the size, e.g. falcon3-abl_10b; the original file stem is kept in VARIANTS below.

Families (developer): first run of letters in the name, so qwen2.5 and qwen3 are both 'qwen' and the DeepSeek-R1
distills are their own family 'deepseek' (as in all earlier analyses).

Base lineage (pretraining), checked against the Ollama config blobs on 2026-11-19:
  deepseek-r1:1.5b/7b -> Qwen2.5-Math, deepseek-r1:14b -> Qwen2.5, deepseek-r1:8b -> Qwen3-8B (DeepSeek-R1-0528-Qwen3-8B,
  NOT Llama as assumed before Nov 19). The abliterated huihui_ai deepseek-r1-abliterated:8b-llama-distill IS Llama-3.1-8B.
  lineage(m)      coarse: 'qwen' (qwen2.5 + qwen3), 'llama', or the developer family
  lineage_fine(m) splits 'qwen2.5' and 'qwen3'
Reasoning-trained: DeepSeek-R1 distills (incl. abliterated), qwen3, and the reasoning fine-tunes (openthinker,
smallthinker, phi4-reasoning, phi4-mini-reasoning).
"""
import re

# original variant file stem -> (library name, parent base model, kind)
VARIANTS = {
    'huihui_ai__falcon3-abliterated_10b': ('falcon3-abl_10b', 'falcon3_10b', 'abl'),
    'huihui_ai__falcon3-abliterated_7b': ('falcon3-abl_7b', 'falcon3_7b', 'abl'),
    'huihui_ai__falcon3-abliterated_3b': ('falcon3-abl_3b', 'falcon3_3b', 'abl'),
    'huihui_ai__deepseek-r1-abliterated_14b-qwen-distill': ('deepseek-r1-abl_14b', 'deepseek-r1_14b', 'abl'),
    'huihui_ai__deepseek-r1-abliterated_7b-qwen-distill': ('deepseek-r1-abl_7b', 'deepseek-r1_7b', 'abl'),
    # NOT a twin of the library deepseek-r1:8b (Qwen3-based); it is the original R1-Distill-Llama-8B, abliterated
    'huihui_ai__deepseek-r1-abliterated_8b-llama-distill': ('deepseek-r1-abl-llama_8b', None, 'abl-mismatch'),
    'huihui_ai__llama3.2-abliterate_3b': ('llama3.2-abl_3b', 'llama3.2_3b', 'abl'),
    'mannix__llama3.1-8b-abliterated': ('llama3.1-abl_8b', 'llama3.1_8b', 'abl'),
    'krith__mistral-nemo-instruct-2407-abliterated_IQ4_XS': ('mistral-nemo-abl_12b', 'mistral-nemo_12b', 'abl'),
    'richardyoung__mistral-7b-instruct-v0.3-abliterated_Q4_K_M': ('mistral-abl_7b', 'mistral_7b', 'abl'),
    'openthinker_7b': ('qwen2.5-openthinker_7b', 'qwen2.5_7b', 'ft'),
    'smallthinker_3b': ('qwen2.5-smallthinker_3b', 'qwen2.5_3b', 'ft'),
    'phi4-reasoning_14b': ('phi4-reasoning_14b', 'phi4_14b', 'ft'),
    'phi4-mini-reasoning_3.8b': ('phi4-mini-reasoning_3.8b', 'phi4-mini_3.8b', 'ft'),
}
LIB = {v[0]: v for v in VARIANTS.values()}            # library name -> (name, parent, kind)
PARENT = {k: v[1] for k, v in LIB.items()}
KIND = {k: v[2] for k, v in LIB.items()}

_LINEAGE_FINE = {'deepseek-r1_1.5b': 'qwen2.5', 'deepseek-r1_7b': 'qwen2.5', 'deepseek-r1_14b': 'qwen2.5',
                 'deepseek-r1_8b': 'qwen3', 'deepseek-r1-abl_14b': 'qwen2.5', 'deepseek-r1-abl_7b': 'qwen2.5',
                 'deepseek-r1-abl-llama_8b': 'llama'}


def fam(m):
    return re.match(r'[a-z]+', m).group(0)


def is_variant(m):
    return m in LIB


def is_distill(m):
    return m.startswith('deepseek')


def reason(m):
    return m.startswith('deepseek') or m.startswith('qwen3') or 'thinker' in m or 'reasoning' in m


def lineage_fine(m):
    if m in _LINEAGE_FINE: return _LINEAGE_FINE[m]
    if m.startswith('qwen2.5'): return 'qwen2.5'
    if m.startswith('qwen3'): return 'qwen3'
    return fam(m)


def lineage(m):
    l = lineage_fine(m)
    return 'qwen' if l.startswith('qwen') else l


def twin(a, b):
    """variant-parent pair, or two variants of the same parent"""
    if a == b: return False
    pa, pb = PARENT.get(a), PARENT.get(b)
    return (pa is not None and pa == b) or (pb is not None and pb == a) or (pa is not None and pa == pb)
