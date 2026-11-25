#!/usr/bin/env python3
"""LLMmap baseline on our models: pretrained open-set LLMmap embedder + per-model templates.

Protocol (mirrors the paper's C1/C2 protocol):
  template(m)  = mean embedding over m's TRAIN prompt-config traces (LLMmap's own recipe, no retraining)
  probe        = one held-out TEST trace (8 answers under an unseen prompt config)
  closed set   = nearest template; top-1 / top-3 over the library
  open set     = accept iff nearest==m and d <= tau_m, tau_m = 95th pct of m's own train-trace distances;
                 unknown = leave-one-model-out (m's template removed), reject iff d > tau_nearest
  variants     = probe a variant against the base30 library: parent rank, top-1==parent, accepted as parent
Embedding needs torch + the HF embedder (intfloat/multilingual-e5-large-instruct); metrics are pure numpy.
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.spatial.distance import cdist


# ----------------------------------------------------------------------------- metrics (numpy only)
def build_templates(emb, labels):
    return {m: emb[m]["train"].mean(0) for m in labels}

def taus(emb, T, labels, q=95):
    return {m: np.percentile(cdist(emb[m]["train"], T[m][None])[:, 0], q) for m in labels}

def closed_set(emb, T, labels, test_labels=None):
    lib = list(labels); D = np.stack([T[m] for m in lib])
    rows = {}
    for m in (test_labels or lib):
        d = cdist(emb[m]["test"], D)
        order = np.argsort(d, axis=1)
        rank = np.array([list(np.array(lib)[o]).index(m) for o in order]) if m in lib else None
        rows[m] = {"top1": float((rank == 0).mean()), "top3": float((rank < 3).mean())}
    return rows

def open_set(emb, T, tau, labels):
    lib = list(labels); out = {}
    for m in lib:
        D = np.stack([T[x] for x in lib]); d = cdist(emb[m]["test"], D)
        nn = d.argmin(1); dmin = d.min(1)
        acc_known = np.mean([(lib[j] == m) and (dm <= tau[m]) for j, dm in zip(nn, dmin)])
        others = [x for x in lib if x != m]
        Do = np.stack([T[x] for x in others]); do = cdist(emb[m]["test"], Do)
        nno = do.argmin(1); dmo = do.min(1)
        rej_unknown = np.mean([dm > tau[others[j]] for j, dm in zip(nno, dmo)])
        out[m] = {"accept_known": float(acc_known), "reject_unknown": float(rej_unknown)}
    return out

def variants_vs_base(emb, T, tau, base_labels, variants):
    lib = list(base_labels); D = np.stack([T[m] for m in lib]); res = {}
    for v in variants:
        d = cdist(emb[v["label"]]["test"], D)
        near = [lib[j] for j in d.argmin(1)]
        r = {"nearest_mode": max(set(near), key=near.count)}
        p = v.get("parent")
        if p in lib:
            pj = lib.index(p)
            rank = (np.argsort(d, axis=1) == pj).argmax(1)
            r.update(parent_top1=float((rank == 0).mean()), parent_top3=float((rank < 3).mean()),
                     parent_mean_rank=float(rank.mean() + 1),
                     accepted_as_parent=float(np.mean([(near[i] == p) and d[i, pj] <= tau[p] for i in range(len(near))])))
        res[v["label"]] = r
    return res


# ----------------------------------------------------------------------------- embedding (needs torch)
def pick_arm(traces_dir, label, preferred, max_empty=0.5):
    """Use the preferred arm unless it is mostly empty answers and a 'raw' run exists (e.g. openthinker)."""
    def empty_frac(arm):
        f = os.path.join(traces_dir, f"{label}{arm}.jsonl")
        if not os.path.exists(f): return None
        tot = emp = 0
        for line in open(f):
            for _, a in json.loads(line)["traces"]:
                tot += 1; emp += (not a.strip())
        return emp / max(tot, 1)
    e = empty_frac(preferred)
    if e is not None and e <= max_empty: return preferred
    r = empty_frac("raw")
    if r is not None and r < (e if e is not None else 1.0): return "raw"
    return preferred if e is not None else None

def embed_all(llmmap_home, model_dir, labels, traces_dir, suffix_pref, cache, device):
    sys.path.insert(0, llmmap_home)
    import torch
    _load = torch.load   # the pretrained weights were saved from an Apple 'mps' device: load onto CPU first
    torch.load = lambda *a, **k: _load(*a, **{"map_location": "cpu", **k})
    from LLMmap.inference import load_LLMmap, InferenceModel
    conf, inf = load_LLMmap(model_dir, device=device, verbose=False)
    assert conf["is_open"]
    queries = conf["queries"]
    emb = {}
    for m in labels:
        suffix = pick_arm(traces_dir, m, suffix_pref) or suffix_pref
        print(f"{m}: arm={suffix}")
        cf = os.path.join(cache, f"{m}{suffix}.npz")
        if os.path.exists(cf):
            z = np.load(cf); emb[m] = {"train": z["train"], "test": z["test"]}; continue
        parts = {"train": [], "test": []}
        for line in open(os.path.join(traces_dir, f"{m}{suffix}.jsonl")):
            e = json.loads(line)
            assert [t[0] for t in e["traces"]] == queries, "query order differs from the pretrained model"
            parts[e["dataset"]].append(InferenceModel.__call__(inf, [t[1] for t in e["traces"]]).cpu().numpy()[0])
        emb[m] = {k: np.stack(v) for k, v in parts.items()}
        os.makedirs(cache, exist_ok=True); np.savez(cf, **emb[m])
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmmap", default=os.environ.get("LLMMAP_HOME", "./LLMmap"))
    ap.add_argument("--pretrained", default=None, help="default: <llmmap>/data/pretrained_models/default")
    ap.add_argument("--models", default="models.json")
    ap.add_argument("--traces", default="traces")
    ap.add_argument("--think", default="off")
    ap.add_argument("--cache", default="emb_cache")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="llmmap_results.json")
    a = ap.parse_args()
    ms = json.load(open(a.models))["models"]
    base = [m["label"] for m in ms if m["role"] == "base30"]
    var = [m for m in ms if m["role"] == "variant"]
    have = lambda l: pick_arm(a.traces, l, a.think) is not None
    base = [b for b in base if have(b)]; var = [v for v in var if have(v["label"])]
    emb = embed_all(a.llmmap, a.pretrained or os.path.join(a.llmmap, "data/pretrained_models/default"),
                    base + [v["label"] for v in var], a.traces, a.think, a.cache, a.device)
    T = build_templates(emb, base + [v["label"] for v in var]); tau = taus(emb, T, base + [v["label"] for v in var])

    R = {"n_base": len(base), "chance": 1 / len(base)}
    cs = closed_set(emb, T, base); R["closed_base30"] = cs
    R["top1_mean"] = float(np.mean([r["top1"] for r in cs.values()])); R["top3_mean"] = float(np.mean([r["top3"] for r in cs.values()]))
    os_ = open_set(emb, T, tau, base); R["open_base30"] = os_
    R["accept_known"] = float(np.mean([r["accept_known"] for r in os_.values()]))
    R["reject_unknown"] = float(np.mean([r["reject_unknown"] for r in os_.values()]))
    if var:
        R["variants_vs_base30"] = variants_vs_base(emb, T, tau, base, var)
        all_l = base + [v["label"] for v in var]
        cs2 = closed_set(emb, T, all_l); R["closed_all"] = cs2
        R["top1_all_base"] = float(np.mean([cs2[m]["top1"] for m in base]))
        os2 = open_set(emb, T, tau, all_l)
        R["accept_known_all_base"] = float(np.mean([os2[m]["accept_known"] for m in base]))
        R["reject_unknown_all_base"] = float(np.mean([os2[m]["reject_unknown"] for m in base]))
    json.dump(R, open(a.out, "w"), indent=1)
    print(f"base30 closed-set top-1 {R['top1_mean']:.3f} top-3 {R['top3_mean']:.3f} (chance {R['chance']:.3f})")
    print(f"open set accept known {R['accept_known']:.3f}, reject unknown {R['reject_unknown']:.3f}")
    for k, v in R.get("variants_vs_base30", {}).items():
        print(k, v)

if __name__ == "__main__":
    main()
