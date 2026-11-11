"""Fingerprint analysis for the HellaSwag behavioural-fingerprint runs.

Usage:  python analysis/fingerprint_analysis.py [fingerprints_dir]   (default: fingerprints_v2)

Reports, for every model whose CSV has all rows:
  1. behaviour profile: run/mode accuracy, unparsed rate, unanimity, entropy, position bias
  2. identification: pick the true model among all candidates from k sampled answers
     (reference = runs 0-4, probe = runs 5-9; per-question likelihood vs marginal-only baseline)
  3. similarity: pairwise mode-answer agreement, within- vs across-family, clustering order
Choice codes in the CSVs: 0-3 answer, -2 unparsed, -1 request failed; NaN = row missing.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

K = 5  # categories: answers 0-3, plus "unparsed" as category 4
FAMILY = {"deepseek": "deepseek", "gemma": "gemma", "llama": "llama", "phi": "phi",
          "falcon": "falcon", "qwen": "qwen", "mistral": "mistral", "ministral": "mistral"}


def family(m):
    return next(v for k, v in FAMILY.items() if m.startswith(k))


def load(d):
    models, labels, idx = {}, None, None
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        m = os.path.basename(f)[:-4]
        df = pd.read_csv(f).sort_values("index").reset_index(drop=True)
        cols = [f"{m}_run{i}" for i in range(10)]
        if not set(cols) <= set(df.columns):
            continue
        if idx is None:
            idx, labels = df["index"].to_numpy(), df["label"].to_numpy()
        assert (df["index"].to_numpy() == idx).all(), f"{m}: row order differs"
        a = df[cols].to_numpy(dtype=float)
        a = np.where(np.isnan(a), -9, a).astype(int)       # -9 = missing row
        models[m] = np.where(a == -2, 4, a)                 # unparsed -> category 4
    return models, labels


def counts(a):
    return np.stack([(a == k).sum(1) for k in range(K)], 1)


def mode(a):
    return counts(a)[:, :4].argmax(1)


def profile(M, y):
    rows = []
    for m, a in M.items():
        c = counts(a)
        p = c / np.maximum(c.sum(1, keepdims=True), 1)
        ent = -np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0).sum(1)
        ans = a[(a >= 0) & (a < 4)]
        rows.append(dict(model=m, run_acc=(a == y[:, None]).mean(), mode_acc=(mode(a) == y).mean(),
                         unparsed=(a == 4).mean(), missing=(a == -9).mean(),
                         unanimous=(c.max(1) == 10).mean(), entropy=ent.mean(),
                         **{f"pick{k}": (ans == k).mean() for k in range(4)}))
    return pd.DataFrame(rows).set_index("model")


def identify(M, ks=(1, 3, 5, 10, 20, 50, 100), trials=300, seed=0):
    names, rng = list(M), np.random.default_rng(seed)
    n = len(next(iter(M.values())))
    L = np.stack([np.log((counts(M[m][:, :5]) + 0.5) / (counts(M[m][:, :5]) + 0.5).sum(1, keepdims=True))
                  for m in names])
    G = []
    for m in names:
        r = M[m][:, :5]
        g = np.bincount(r[r >= 0], minlength=K) + 0.5
        G.append(np.log(g / g.sum()))
    G = np.stack(G)
    out, conf10 = [], None
    for k in ks:
        ok = ok_marg = ok_fam = tot = 0
        conf = np.zeros((len(names),) * 2, int)
        for _ in range(trials):
            for i, m in enumerate(names):
                rows = rng.choice(n, k, replace=False)
                pr = M[m][rows, 5 + rng.integers(0, 5, k)]
                keep = pr >= 0
                rows, pr = rows[keep], pr[keep]
                p = L[:, rows, pr].sum(1).argmax()
                conf[i, p] += 1
                ok += p == i
                ok_fam += family(names[p]) == family(m)
                ok_marg += G[:, pr].sum(1).argmax() == i
                tot += 1
        out.append(dict(k=k, exact=ok / tot, family=ok_fam / tot, marginal_only=ok_marg / tot))
        if k == 10:
            conf10 = conf
    return pd.DataFrame(out).set_index("k"), conf10


def similarity(M):
    names = list(M)
    md = {m: mode(M[m]) for m in names}
    S = pd.DataFrame([[(md[a] == md[b]).mean() for b in names] for a in names], index=names, columns=names)
    n = len(names)
    fam = [family(m) for m in names]
    within = [S.iat[i, j] for i in range(n) for j in range(i + 1, n) if fam[i] == fam[j]]
    across = [S.iat[i, j] for i in range(n) for j in range(i + 1, n) if fam[i] != fam[j]]
    D = 1 - S.to_numpy()
    np.fill_diagonal(D, 0)
    order = dendrogram(linkage(squareform(D, checks=False), "average"), labels=names, no_plot=True)["ivl"]
    return S, np.mean(within), np.mean(across), order


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "fingerprints_v2"
    M, y = load(d)
    pd.set_option("display.width", 220)
    print(f"{len(M)} models, {len(y)} rows\n\n== Behaviour profile ==")
    print(profile(M, y).round(3).to_string())
    print(f"\n== Identification among {len(M)} models (chance {1 / len(M):.3f}) ==")
    res, conf = identify(M)
    print(res.round(3).to_string())
    names = list(M)
    print("\nper-model exact-id accuracy at k=10:")
    for i, m in enumerate(names):
        row = conf[i].copy()
        acc = row[i] / row.sum()
        row[i] = -1
        j = row.argmax()
        print(f"  {m:20s} {acc:.2f}   most confused with {names[j]} ({conf[i, j] / conf[i].sum():.2f})")
    S, w, a, order = similarity(M)
    print(f"\n== Similarity (mode-answer agreement) ==\nwithin family {w:.3f}   across families {a:.3f}")
    print("clustering order:", ", ".join(order))
    print(S.round(2).to_string())


if __name__ == "__main__":
    main()
