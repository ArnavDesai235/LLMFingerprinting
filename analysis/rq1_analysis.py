"""RQ1: can a model be identified from its errors? Analysis of v3 (clean HellaSwag) fingerprints.

Usage: python analysis/rq1_analysis.py <fingerprints_dir> <dataset_csv>
Reads <model>.csv.jsonl files with all rows complete. Choice codes: 0-3 answer, -2 unparsed, -1 failed.
Protocol: reference profile = runs 0-4, probe answers = runs 5-9 (disjoint samples of the same model).
"""
import glob, json, os, re, sys
import numpy as np
import pandas as pd

FAM = ["deepseek", "gemma", "llama", "phi", "falcon", "qwen", "mistral"]
family = lambda m: next(f for f in FAM if m.startswith(f))
size = lambda m: float(re.search(r"_([\d.]+)b$", m).group(1))
K = 5  # 0-3 + unparsed(4)


def load(d, data_csv):
    ds = pd.read_csv(data_csv).sort_values("index").reset_index(drop=True)
    pos = {i: p for p, i in enumerate(ds["index"])}
    M = {}
    for f in sorted(glob.glob(os.path.join(d, "*.csv.jsonl"))):
        a = np.full((len(ds), 10), -9)
        for line in open(f):
            r = json.loads(line)
            if len(r["choices"]) == 10 and -1 not in r["choices"]:
                a[pos[r["index"]]] = r["choices"]
        if (a[:, 0] == -9).any():
            continue
        M[os.path.basename(f)[:-len(".csv.jsonl")]] = np.where(a == -2, 4, a)
    return M, ds["label"].to_numpy(), ds["source_id"].str.split("~").str[0].to_numpy()


def counts(a, k=K):
    return np.stack([(a == j).sum(1) for j in range(k)], 1)


def logp(a, alpha=0.5):
    c = counts(a) + alpha
    return np.log(c / c.sum(1, keepdims=True))


def answer_dist(a):
    """Per-question distribution over options 0-3 from the runs (unparsed dropped; uniform if none)."""
    c = counts(a, 4).astype(float)
    s = c.sum(1, keepdims=True)
    return np.where(s > 0, c / np.maximum(s, 1), 0.25)


def mode(a):
    return counts(a, 4).argmax(1)


def identify(M, y, ks, kind="full", trials=300, seed=0):
    """kind: full = any probe answer; errors = only wrong answers, scored by P(option | option != label);
    correct = only right answers."""
    names, rng = list(M), np.random.default_rng(seed)
    n = len(y)
    L = np.stack([logp(M[m][:, :5]) for m in names])                  # model x q x K
    if kind == "errors":                                                # condition on "not the label"
        P = np.exp(L).copy()
        P[:, np.arange(n), y] = 0
        L = np.log(P / P.sum(2, keepdims=True) + 1e-300)
    res = {}
    for k in ks:
        ok = tot = 0
        for _ in range(trials):
            for i, m in enumerate(names):
                pr = M[m][:, 5:]
                if kind == "full":
                    q = rng.choice(n, k, replace=False); a = pr[q, rng.integers(0, 5, k)]
                else:
                    good = (pr < 4) & ((pr != y[:, None]) if kind == "errors" else (pr == y[:, None]))
                    qq, rr = np.nonzero(good)
                    if len(qq) < k: continue
                    sel = rng.choice(len(qq), k, replace=False); q, a = qq[sel], pr[qq[sel], rr[sel]]
                keep = a >= 0
                ok += L[:, q[keep], a[keep]].sum(1).argmax() == i
                tot += 1
        res[k] = ok / max(tot, 1)
    return res


def pair_metrics(M, y):
    names = list(M)
    D = {m: answer_dist(M[m]) for m in names}
    md = {m: mode(M[m]) for m in names}
    pc = {m: D[m][np.arange(len(y)), y] for m in names}                 # P(correct) per question
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = md[a] == y, md[b] == y
            both_wrong = ~ra & ~rb
            agree = (md[a] == md[b]).mean()
            err_agree = (md[a] == md[b])[both_wrong].mean()
            # error consistency (Geirhos et al. 2020): kappa on right/wrong
            pa, pb = ra.mean(), rb.mean()
            ce = pa * pb + (1 - pa) * (1 - pb)
            kappa = ((ra == rb).mean() - ce) / (1 - ce)
            # CAPA-style (Goel et al. 2025): probabilistic agreement adjusted for chance given accuracy
            cobs = (D[a] * D[b]).sum(1).mean()
            qa, qb = pc[a].mean(), pc[b].mean()
            cexp = qa * qb + (1 - qa) * (1 - qb) / 3
            capa = (cobs - cexp) / (1 - cexp)
            rows.append(dict(a=a, b=b, same_family=int(family(a) == family(b)),
                             log_min_size=np.log(min(size(a), size(b))),
                             abs_log_ratio=abs(np.log(size(a) / size(b))),
                             mean_acc=(pa + pb) / 2, agree=agree, err_agree=err_agree,
                             kappa=kappa, capa=capa))
    return pd.DataFrame(rows)


def ols(D, ycol, xcols):
    X = np.c_[np.ones(len(D)), D[xcols].to_numpy()]
    b, *_ = np.linalg.lstsq(X, D[ycol].to_numpy(), rcond=None)
    r2 = 1 - ((D[ycol] - X @ b) ** 2).sum() / ((D[ycol] - D[ycol].mean()) ** 2).sum()
    # permutation test for same_family: shuffle family labels across models
    return dict(zip(["intercept"] + xcols, b.round(4))), round(r2, 3)


def family_perm_p(M, D, ycol, xcols, reps=2000, seed=0):
    names = list(M); fams = [family(m) for m in names]; rng = np.random.default_rng(seed)
    X0 = D[xcols].copy()
    def coef(sf):
        X = np.c_[np.ones(len(D)), sf, X0.drop(columns="same_family").to_numpy()]
        return np.linalg.lstsq(X, D[ycol].to_numpy(), rcond=None)[0][1]
    obs = coef(D["same_family"].to_numpy())
    idx = {m: i for i, m in enumerate(names)}
    null = []
    for _ in range(reps):
        p = rng.permutation(fams)
        null.append(coef(np.array([int(p[idx[a]] == p[idx[b]]) for a, b in zip(D.a, D.b)])))
    return obs, (np.sum(np.array(null) >= obs) + 1) / (reps + 1)


def mi_selection(M, ks, pool=200, trials=300, seed=1):
    names, rng = list(M), np.random.default_rng(seed)
    L = np.stack([logp(M[m][:, :5]) for m in names]); P = np.exp(L)
    mix = P.mean(0)
    info = -(mix * np.log(mix)).sum(1) + (P * L).sum(2).mean(0)
    top = np.argsort(-info)[:pool]
    out = {}
    for k in ks:
        r = {}
        for tag, cand in [("random", None), ("top_mi", top)]:
            ok = tot = 0
            for _ in range(trials):
                for i, m in enumerate(names):
                    q = rng.choice(len(info), k, replace=False) if cand is None else rng.choice(cand, k, replace=False)
                    a = M[m][q, 5 + rng.integers(0, 5, k)]
                    ok += L[:, q, a].sum(1).argmax() == i; tot += 1
            r[tag] = ok / tot
        out[k] = r
    return out


def open_set(M, ks, trials=300, seed=2):
    """Per-model calibrated threshold: accept the best candidate only if its mean log-lik per answer
    is above that candidate's own 5th percentile (calibrated on separate probe samples)."""
    names, rng = list(M), np.random.default_rng(seed)
    n = len(next(iter(M.values())))
    L = np.stack([logp(M[m][:, :5]) for m in names])
    out = {}
    for k in ks:
        def draw(i):
            q = rng.choice(n, k, replace=False); a = M[names[i]][q, 5 + rng.integers(0, 5, k)]
            return L[:, q, a].mean(1)
        thr = np.array([np.quantile([draw(i)[i] for _ in range(trials)], 0.05) for i in range(len(names))])
        acc_in = rej_out = tot = 0
        for _ in range(trials):
            for i in range(len(names)):
                s = draw(i)
                j = s.argmax(); acc_in += (j == i) and s[j] >= thr[j]
                s2 = s.copy(); s2[i] = -np.inf; j2 = s2.argmax(); rej_out += s2[j2] < thr[j2]
                tot += 1
        out[k] = dict(known_correct_and_accepted=acc_in / tot, unknown_rejected=rej_out / tot)
    return out


def main():
    d, data = sys.argv[1], sys.argv[2]
    M, y, src = load(d, data)
    names = list(M)
    pd.set_option("display.width", 220)
    print(f"{len(M)} models x {len(y)} questions (clean HellaSwag)\n")
    prof = []
    for m, a in M.items():
        c = counts(a); p = c / c.sum(1, keepdims=True)
        ans = a[a < 4]
        prof.append(dict(model=m, mode_acc=(mode(a) == y).mean(), acc_wikihow=(mode(a) == y)[src == "wikihow"].mean(),
                         acc_activitynet=(mode(a) == y)[src == "activitynet"].mean(), unparsed=(a == 4).mean(),
                         unanimous=(c.max(1) == 10).mean(),
                         **{f"pick{j}": (ans == j).mean() for j in range(4)}))
    print("== Profile ==\n", pd.DataFrame(prof).set_index("model").sort_values("mode_acc", ascending=False).round(3).to_string())

    ks = [1, 3, 5, 10, 20, 50]
    full, err, cor = identify(M, y, ks, "full"), identify(M, y, ks, "errors"), identify(M, y, ks, "correct")
    print(f"\n== Identification among {len(M)} models (chance {1/len(M):.3f}) ==")
    print(pd.DataFrame({"full_answers": full, "errors_only": err, "correct_only": cor}).round(3).to_string())

    D = pair_metrics(M, y)
    print("\n== Pairwise similarity (mean over pairs) ==")
    print(D[["agree", "err_agree", "kappa", "capa"]].describe().loc[["mean", "min", "max"]].round(3).to_string())
    print("err_agree vs chance (1/3 if wrong answers were random):", round(D.err_agree.mean(), 3))
    print("\n== Size vs family (OLS over pairs; family p-value by permuting family labels) ==")
    xs = ["same_family", "log_min_size", "abs_log_ratio"]
    for ycol in ["agree", "err_agree", "kappa", "capa"]:
        b, r2 = ols(D, ycol, xs)
        obs, p = family_perm_p(M, D, ycol, xs)
        print(f"{ycol:10s} R2={r2}  coefs={b}  family_coef={obs:.4f} perm_p={p:.3f}")
    b, r2 = ols(D, "capa", xs + ["mean_acc"])
    print(f"capa + mean_acc control: R2={r2} coefs={b}")

    print("\n== Question selection by mutual information (top 200 vs random) ==")
    print(pd.DataFrame(mi_selection(M, [3, 5, 10, 20])).T.round(3).to_string())
    print("\n== Open-set with per-model calibrated thresholds (5th percentile) ==")
    print(pd.DataFrame(open_set(M, [10, 20, 50])).T.round(3).to_string())
    D.to_csv("pairwise_metrics_v3.csv", index=False)


if __name__ == "__main__":
    main()
