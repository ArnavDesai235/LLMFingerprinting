"""Held-out question test. Usage: python a17_holdout.py <v3_dir> <v3_csv> <holdout_dir> <holdout_csv>
Uses models complete on BOTH sets (500/500 holdout rows, 3000/3000 v3 rows).
 1. Fresh replication: identification on the 500 new questions (reference runs 0-4, probe runs 5-9), compared with the
    same models on random 500-question subsets of the 3k (same pool size, so the comparison is fair).
 2. Frozen open set: per-model thresholds calibrated on the 3k, applied unchanged on the new questions
    (vs thresholds recalibrated on the new questions).
 3. Stability: per-model accuracy, position bias and consistency on 3k vs new questions; and whether the pairwise
    error-similarity structure (excess same-wrong share) replicates (Spearman/Pearson + Mantel permutation over models).
 4. Family and reasoning effects on the new questions (accuracy-controlled OLS, model-level permutation).
"""
import sys, json, os, glob, itertools, re
import numpy as np, pandas as pd, numpy.linalg as la
from scipy.stats import spearmanr, pearsonr
from scipy.special import logsumexp
v3d, v3c, hd, hc = sys.argv[1:5]
rng = np.random.default_rng(3)
fam = lambda m: re.match(r'[a-z]+', m).group(0)
reason = lambda m: m.startswith('deepseek') or m.startswith('qwen3')

def load(d, csv):
    ds = pd.read_csv(csv); pos = {i: p for p, i in enumerate(ds['index'])}; M = {}
    for f in sorted(glob.glob(os.path.join(d, '*.csv.jsonl'))):
        a = np.full((len(ds), 10), -9)
        for l in open(f):
            r = json.loads(l)
            if r['index'] in pos and len(r['choices']) == 10: a[pos[r['index']]] = r['choices']
        if (a[:, 0] != -9).all(): M[os.path.basename(f)[:-10]] = np.where((a >= 0) & (a < 4), a, 4)
    return M, ds.label.to_numpy()
M3, y3 = load(v3d, v3c); MH, yH = load(hd, hc)
models = sorted(set(M3) & set(MH)); nm = len(models)
print(f'models complete on both sets: {nm}  (chance {1/nm:.3f})\n  {models}')
if nm < 3: sys.exit('need at least 3 models')
A3 = np.stack([M3[m] for m in models]); AH = np.stack([MH[m] for m in models])

def tables(A, y):
    R = A[:, :, :5]; C = np.stack([(R == k).sum(-1) for k in range(5)], -1)
    L = np.log((C + .5) / (C + .5).sum(-1, keepdims=True))
    P = np.exp(L).copy(); n = len(y); P[:, np.arange(n), y] = 0; Le = np.log(P / P.sum(-1, keepdims=True) + 1e-300)
    mo = C[:, :, :4].argmax(-1); return L, Le, mo
def identify(A, y, ks, trials):
    L, Le, mo = tables(A, y); n = len(y); out = {}
    for k in ks:
        ok = np.zeros(3); T = 0
        for _ in range(trials):
            for i in range(nm):
                q = rng.choice(n, k, replace=False); a = A[i, q, rng.integers(5, 10, k)]; ok[0] += L[:, q, a].sum(1).argmax() == i
                pw = A[i, :, 5:]; wr = np.argwhere((pw != y[:, None]) & (pw < 4))
                if len(wr) < k: continue
                s = wr[rng.choice(len(wr), k, replace=False)]; q = s[:, 0]; a = pw[q, s[:, 1]]
                ok[1] += Le[:, q, a].sum(1).argmax() == i
                sc = (mo[:, q] == a).sum(1); b = np.flatnonzero(sc == sc.max()); ok[2] += rng.choice(b) == i; T += 1
        out[k] = dict(lik_all=ok[0] / (trials * nm), lik_errors=ok[1] / max(T, 1), condHamming_errors=ok[2] / max(T, 1))
    return pd.DataFrame(out).T
pd.set_option('display.width', 220); ks = [5, 10, 20, 50]
print('\n== 1. Identification on the 500 NEW questions ==')
new = identify(AH, yH, ks, 200); print(new.round(3).to_string())
subs = [identify(A3[:, s], y3[s], ks, 40) for s in [rng.choice(len(y3), 500, replace=False) for _ in range(5)]]
print('-- same models, random 500-question subsets of the 3k (mean of 5) --'); old = sum(subs) / 5; print(old.round(3).to_string())
print('-- difference (new - 3k subsets) --'); print((new - old).round(3).to_string())

print('\n== 2. Open set: thresholds frozen on the 3k vs recalibrated on new questions (k answers, 5th percentile) ==')
L3 = tables(A3, y3)[0]; LH = tables(AH, yH)[0]
def draw(L, A, i, k):
    q = rng.choice(L.shape[1], k, replace=False); a = A[i, q, rng.integers(5, 10, k)]; return L[:, q, a].mean(1)
for k in [10, 20, 50]:
    thr3 = np.array([np.quantile([draw(L3, A3, i, k)[i] for _ in range(300)], .05) for i in range(nm)])
    thrH = np.array([np.quantile([draw(LH, AH, i, k)[i] for _ in range(300)], .05) for i in range(nm)])
    r = {}
    for tag, thr in [('frozen_3k', thr3), ('recalibrated', thrH)]:
        acc = rej = 0; T = 0
        for _ in range(200):
            for i in range(nm):
                s = draw(LH, AH, i, k); j = s.argmax(); acc += (j == i) and s[j] >= thr[j]
                s2 = s.copy(); s2[i] = -np.inf; j2 = s2.argmax(); rej += s2[j2] < thr[j2]; T += 1
        r[tag] = (acc / T, rej / T)
    print(f'  k={k:2d}  frozen: known accepted {r["frozen_3k"][0]:.3f} unknown rejected {r["frozen_3k"][1]:.3f}   |   '
          f'recalibrated: {r["recalibrated"][0]:.3f} / {r["recalibrated"][1]:.3f}')

print('\n== 3. Stability between the 3k and the new questions ==')
def profile(A, y):
    C = np.stack([(A == k).sum(-1) for k in range(4)], -1); mo = C.argmax(-1); ans = [A[i][A[i] < 4] for i in range(nm)]
    return pd.DataFrame(dict(acc=(mo == y).mean(1), unanimous=(C.max(-1) == 10).mean(1),
                             top_slot_share=[np.bincount(a, minlength=4).max() / len(a) for a in ans]), index=models)
p3, pH = profile(A3, y3), profile(AH, yH)
for c in p3:
    print(f'  {c:15s} 3k vs new: Pearson {pearsonr(p3[c], pH[c])[0]:.3f}   mean |diff| {np.abs(p3[c] - pH[c]).mean():.3f}')
print(pd.concat([p3.add_suffix('_3k'), pH.add_suffix('_new')], axis=1).round(3).to_string())
def pairs(A, y):
    _, _, mo = tables(A, y); W = mo != y; bias = np.stack([np.bincount(mo[i], minlength=4) / len(y) for i in range(nm)]); acc = 1 - W.mean(1); rec = []
    for i, j in itertools.combinations(range(nm), 2):
        r = np.where(W[i] & W[j])[0]
        if len(r) < 5: rec.append(dict(i=i, j=j, excess=np.nan)); continue
        sh = (mo[i, r] == mo[j, r]).mean()
        nul = np.mean([(lambda w: (bias[i, w] / bias[i, w].sum() * bias[j, w] / bias[j, w].sum()).sum())([o for o in range(4) if o != y[t]]) for t in r])
        rec.append(dict(i=i, j=j, excess=sh - nul, accmin=min(acc[i], acc[j]), accdiff=abs(acc[i] - acc[j]), nboth=len(r)))
    return pd.DataFrame(rec)
P3, PH = pairs(A3, y3), pairs(AH, yH); ok = PH.excess.notna()
rs = spearmanr(P3.excess[ok], PH.excess[ok])[0]; rp = pearsonr(P3.excess[ok], PH.excess[ok])[0]
# Mantel: permute model labels of the new-question matrix
Dn = np.full((nm, nm), np.nan); D3 = np.full((nm, nm), np.nan)
for (_, a), (_, b) in zip(P3.iterrows(), PH.iterrows()): D3[int(a.i), int(a.j)] = a.excess; Dn[int(b.i), int(b.j)] = b.excess
iu = np.triu_indices(nm, 1); null = []
for _ in range(2000):
    p = rng.permutation(nm); Dp = np.fmax(Dn, Dn.T)[np.ix_(p, p)][iu]; v = ~np.isnan(Dp) & ~np.isnan(D3[iu]); null.append(pearsonr(D3[iu][v], Dp[v])[0])
print(f'  pairwise excess same-wrong share, 3k vs new questions ({ok.sum()} pairs; median both-wrong on new = {PH.nboth.median():.0f}): '
      f'Spearman {rs:.3f}, Pearson {rp:.3f}, Mantel p {(np.array(null) >= rp).mean():.4f}')

print('\n== 4. Family and reasoning effects on the new questions (accuracy-controlled; permutation over models) ==')
Q = PH[ok].copy(); Q['a'] = [models[int(i)] for i in Q.i]; Q['b'] = [models[int(j)] for j in Q.j]
def ind(f, kind): return np.array([float(f(a) == f(b)) if kind == 'same' else float(f(a) and f(b)) for a, b in zip(Q.a, Q.b)])
def coefs(sf, br):
    cols = [np.ones(len(Q)), Q.accmin, Q.accdiff, sf] + ([br] if br is not None else [])
    return la.lstsq(np.column_stack(cols), Q.excess, rcond=None)[0][3:]
has_r = sum(reason(m) for m in models) >= 2
obs = coefs(ind(fam, 'same'), ind(reason, 'both') if has_r else None); nf = []; nr = []
for _ in range(2000):
    p = dict(zip(models, rng.permutation(models)))
    nf.append(coefs(ind(lambda m: fam(p[m]), 'same'), ind(reason, 'both') if has_r else None)[0])
    if has_r: nr.append(coefs(ind(fam, 'same'), ind(lambda m: reason(p[m]), 'both'))[1])
print(f'  same family {obs[0]:+.3f} (p {(np.abs(nf) >= abs(obs[0])).mean():.3f})' +
      (f'   both reasoning {obs[1]:+.3f} (p {(np.abs(nr) >= abs(obs[1])).mean():.3f})' if has_r else '   (reasoning: <2 reasoning models complete)'))
