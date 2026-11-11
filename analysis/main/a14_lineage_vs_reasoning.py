"""Lineage vs reasoning: what predicts shared wrong answers once accuracy is controlled?
Pairs over all complete models plus qwen3:14b (partial; each pair uses common rows).
Labels per model:
  dev     = developer family (deepseek distills are their own family; qwen3 in qwen)
  base    = pretraining lineage (r1:1.5b/7b/14b -> qwen, r1:8b -> llama; others = dev)
  reason  = reasoning-trained (deepseek-r1 distills, qwen3)
OLS: metric ~ accmin + accdiff + same_base + both_reason [+ same_dev]; p-values by permuting each label across MODELS
(holding the other labels fixed). Also: identification confusion among the 26 complete models.
"""
import sys, itertools; sys.path.insert(0, '.')
exec(open(__file__.replace('a14_lineage_vs_reasoning.py', 'a13_distill.py')).read().split("out = []")[0])
import numpy.linalg as la
EXTRA = [m for m in ['qwen3_14b'] if m in M and len(M[m]) >= 2000]
models = sorted(ref + dist + EXTRA)
BASEOF = {'deepseek-r1_1.5b': 'qwen', 'deepseek-r1_7b': 'qwen', 'deepseek-r1_14b': 'qwen', 'deepseek-r1_8b': 'llama'}
dev = {m: fam(m) for m in models}; base = {m: BASEOF.get(m, fam(m)) for m in models}
reason = {m: int(m.startswith('deepseek') or m.startswith('qwen3')) for m in models}
print(f'{len(models)} models; reasoning: {[m for m in models if reason[m]]}; partial: {EXTRA} ({[len(M[m]) for m in EXTRA]} rows)')
P = pd.DataFrame([stats(a, b, rows(a, b)) for a, b in itertools.combinations(models, 2)])
pairs = list(zip(P.a, P.b)); rng = np.random.default_rng(5)
def lab(d): return np.array([float(d[a] == d[b]) for a, b in pairs])
def both(d): return np.array([float(d[a] and d[b]) for a, b in pairs])
def fit(col, cols):
    X = np.column_stack([np.ones(len(P)), P.accmin, P.accdiff] + cols); b = la.lstsq(X, P[col], rcond=None)[0]
    r2 = 1 - ((P[col] - X @ b) ** 2).sum() / ((P[col] - P[col].mean()) ** 2).sum(); return b[3:], r2
def perm_p(col, builders, which, nperm=3000):
    obs = fit(col, [f(d) for f, d in builders])[0][which]; null = []
    for _ in range(nperm):
        cols = []
        for k, (f, d) in enumerate(builders):
            if k == which: ms = list(d); v = rng.permutation([d[m] for m in ms]); d = dict(zip(ms, v))
            cols.append(f(d))
        null.append(fit(col, cols)[0][which])
    return obs, (np.abs(null) >= abs(obs)).mean()
pd.set_option('display.width', 220)
for spec, builders, names in [
        ('A: base lineage + both-reasoning', [(lab, base), (both, reason)], ['same_base', 'both_reason']),
        ('B: developer family + both-reasoning', [(lab, dev), (both, reason)], ['same_dev', 'both_reason']),
        ('C: base lineage only', [(lab, base)], ['same_base'])]:
    print(f'\n== {spec} ==')
    for col in ['excess', 'share_p', 'co']:
        r2 = fit(col, [f(d) for f, d in builders])[1]
        s = '  '.join(f'{n} {o:+.3f} (p {p:.3f})' for n, (o, p) in zip(names, [perm_p(col, builders, k) for k in range(len(builders))]))
        print(f'  {col:8s} R2 {r2:.3f}  {s}')
# group means of the accuracy-controlled residual
X = np.column_stack([np.ones(len(P)), P.accmin, P.accdiff]); P['res'] = P.excess - X @ la.lstsq(X, P.excess, rcond=None)[0]
P['grp'] = np.select([both(reason) == 1, lab(base) == 1, np.array([reason[a] != reason[b] for a, b in pairs])],
                     ['reasoning-reasoning', 'same base lineage (non-reasoning pairs)', 'reasoning-nonreasoning'], 'unrelated')
print('\n== accuracy-controlled excess residual by pair type ==')
print(P.groupby('grp').res.agg(['mean', 'count', 'std']).round(3).to_string())
print('\n== reasoning-reasoning pairs ==')
print(P[P.grp == 'reasoning-reasoning'][['a', 'b', 'n', 'accA', 'accB', 'excess', 'res']].round(3).to_string(index=False))
P.to_csv('pairs_lineage_reasoning.csv', index=False)

# identification confusion among complete models: where do misidentified probes go?
full = [m for m in models if len(M[m]) == 3000]; idx = rows(*full); y = ds2.loc[idx, 'label'].to_numpy()
A = arr(M, full, idx)
L = np.log((np.stack([(A[:, :, :5] == k).sum(-1) for k in range(5)], -1) + 0.5) / (5 + 2.5))   # reference = runs 0-4 only
print(f'\n== identification per model, {len(full)} models, k=10 random probes (reference runs 0-4, probes 5-9) ==')
res = []
for i, m in enumerate(full):
    ok = 0; conf = {}
    for _ in range(1000):
        q = rng.choice(len(idx), 10, replace=False); a = A[i, q, rng.integers(5, 10, 10)]
        p = L[:, q, a].sum(1).argmax(); ok += p == i
        if p != i: conf[full[p]] = conf.get(full[p], 0) + 1
    top = max(conf, key=conf.get) if conf else ''
    res.append(dict(model=m, acc_id=ok / 1000, top_confusion=top, conf_share=conf.get(top, 0) / 1000,
                    conf_to_reasoning=sum(c for k, c in conf.items() if reason[k]) / 1000))
print(pd.DataFrame(res).sort_values('acc_id').round(3).to_string(index=False))
