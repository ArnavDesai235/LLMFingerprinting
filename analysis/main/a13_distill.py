"""Distillation test: do DeepSeek-R1 distills share errors with their BASE model's family (architecture/pretraining)
or with each other (shared R1 reasoning data)? Works on partial files: each comparison uses the rows both models have.
Bases: r1:1.5b <- Qwen2.5-Math-1.5B, r1:7b <- Qwen2.5-Math-7B, r1:14b <- Qwen2.5-14B, and the library r1:8b <- Qwen3-8B
(Ollama's deepseek-r1:8b is DeepSeek-R1-0528-Qwen3-8B; checked 2026-11-19. Before that this script assumed Llama-3.1-8B).
If the abliterated distills are present (all44 set) they are added: r1-abl:14b/7b <- Qwen2.5, r1-abl-llama:8b <- Llama-3.1-8B
(the only Llama-based distill in the data). Reference models = all non-distill BASE models (variants excluded).
Metrics per pair (on mode answers, plus a probabilistic version that uses all 10 runs):
  share  = P(same wrong ending | both wrong);  excess = share - position-bias null;  co = co-error rate.
Each distill's pairs with the 23 non-distill reference models are regressed on (min acc, |acc diff|); we report the
residual of the base family vs other families, and the rank of the base model.
"""
import sys, itertools; sys.path.insert(0, '.')
from load import *
from labels import is_variant
import numpy.linalg as la
M = load(); ds2 = ds.set_index('index')
DIST = {'deepseek-r1_1.5b': 'qwen', 'deepseek-r1_7b': 'qwen', 'deepseek-r1_8b': 'qwen', 'deepseek-r1_14b': 'qwen',
        'deepseek-r1-abl_14b': 'qwen', 'deepseek-r1-abl_7b': 'qwen', 'deepseek-r1-abl-llama_8b': 'llama'}
BASE = {'deepseek-r1_1.5b': 'qwen2.5_1.5b', 'deepseek-r1_7b': 'qwen2.5_7b', 'deepseek-r1_8b': 'qwen3_8b', 'deepseek-r1_14b': 'qwen2.5_14b',
        'deepseek-r1-abl_14b': 'qwen2.5_14b', 'deepseek-r1-abl_7b': 'qwen2.5_7b', 'deepseek-r1-abl-llama_8b': 'llama3.1_8b'}
ref = [m for m in M if len(M[m]) == 3000 and m not in DIST and not is_variant(m)]
dist = [m for m in DIST if m in M]
print('reference models', len(ref), '| distills', {d: len(M[d]) for d in dist})

def rows(*ms):
    s = set(M[ms[0]]);
    for m in ms[1:]: s &= set(M[m])
    return sorted(s)

def stats(a, b, idx):
    y = ds2.loc[idx, 'label'].to_numpy(); X = arr(M, [a, b], idx)
    C = np.stack([(X == k).sum(-1) for k in range(4)], -1).astype(float)      # 2 x n x 4
    P = C / np.maximum(C.sum(-1, keepdims=True), 1)
    mo = C.argmax(-1); W = mo != y; acc = 1 - W.mean(1)
    bias = np.stack([np.bincount(mo[i], minlength=4) / len(y) for i in range(2)])
    both = W[0] & W[1]; r = np.where(both)[0]
    share = (mo[0, r] == mo[1, r]).mean()
    def nul(t):
        w = [o for o in range(4) if o != y[t]]; p = bias[0, w] / bias[0, w].sum(); q = bias[1, w] / bias[1, w].sum(); return (p * q).sum()
    null = np.mean([nul(t) for t in r])
    co = both.sum() / np.sqrt(W[0].sum() * W[1].sum())
    # probabilistic: per row, P(both wrong) and P(same wrong); share_p = sum P(same wrong) / sum P(both wrong)
    Pw = P.copy(); Pw[:, np.arange(len(y)), y] = 0
    pbw = Pw[0].sum(-1) * Pw[1].sum(-1); psw = (Pw[0] * Pw[1]).sum(-1)
    share_p = psw.sum() / pbw.sum()
    return dict(a=a, b=b, n=len(idx), accA=acc[0], accB=acc[1], accmin=acc.min(), accdiff=abs(acc[0] - acc[1]),
                share=share, null=null, excess=share - null, co=co, share_p=share_p, nboth=both.sum())

out = []
for d in dist:
    for m in ref: out.append(stats(d, m, rows(d, m)))
for d1, d2 in itertools.combinations(dist, 2): out.append(stats(d1, d2, rows(d1, d2)))
# reference-reference cross-family pairs for the accuracy-matched baseline
for a, b in itertools.combinations(ref, 2):
    if fam(a) != fam(b): out.append(stats(a, b, rows(a, b)))
R = pd.DataFrame(out); R['fa'] = R.a.map(fam); R['fb'] = R.b.map(fam)
pd.set_option('display.width', 250)

print(f'\n== Each distill vs reference models: residual after accuracy controls (fit on that distill\'s {len(ref)} pairs) ==')
for d in dist:
    S = R[(R.a == d) & R.b.isin(ref)].copy()
    for col in ['excess', 'share_p', 'co']:
        X = np.column_stack([np.ones(len(S)), S.accmin, S.accdiff]); bta = la.lstsq(X, S[col], rcond=None)[0]
        S[col + '_res'] = S[col] - X @ bta
    S['bfam'] = S.b.map(fam)
    g = S.groupby('bfam')[['excess_res', 'share_p_res', 'co_res']].mean()
    rk = S.sort_values('excess_res', ascending=False).b.tolist()
    rkp = S.sort_values('share_p_res', ascending=False).b.tolist()
    print(f'\n-- {d} (n rows {S.n.min()}, acc {S.accA.iloc[0]:.3f}; base family {DIST[d]}, nearest base {BASE[d]})')
    print(g.round(3).sort_values('excess_res', ascending=False).to_string())
    print(f'   base model {BASE[d]}: excess_res {S.set_index("b").loc[BASE[d], "excess_res"]:+.3f}, rank {rk.index(BASE[d]) + 1}/{len(ref)} (mode),'
          f' rank {rkp.index(BASE[d]) + 1}/{len(ref)} (probabilistic)')
    print('   top-5 by excess_res:', ', '.join(f'{b} {v:+.3f}' for b, v in S.sort_values("excess_res", ascending=False)[["b", "excess_res"]].head(5).values))
    print('   top-5 by share_p_res:', ', '.join(f'{b} {v:+.3f}' for b, v in S.sort_values("share_p_res", ascending=False)[["b", "share_p_res"]].head(5).values))

print('\n== Distill-distill pairs (same R1 training data, different/same base) vs accuracy-matched cross-family reference pairs ==')
CF = R[R.a.isin(ref) & R.b.isin(ref)]
for _, r in R[R.a.isin(dist) & R.b.isin(dist)].iterrows():
    m = CF[(abs(CF.accmin - r.accmin) < 0.05) & (abs(CF.accdiff - r.accdiff) < 0.08)]
    print(f'  {r.a:17s} {r.b:17s} n={r.n:4d} acc {r.accA:.2f}/{r.accB:.2f}  excess {r.excess:.3f} vs matched {m.excess.mean():.3f} (n={len(m)})'
          f'   share_p {r.share_p:.3f} vs {m.share_p.mean():.3f}   co {r.co:.3f} vs {m.co.mean():.3f}')

# identification: classify the distill's probe answers among the 23 reference models (who does it look like?)
print(f'\n== Closed-set "who does this distill look like": likelihood over {len(ref)} reference models, k=20 probe answers (runs 5-9) ==')
rng = np.random.default_rng(0)
for d in dist:
    idx = rows(d, *ref); y = ds2.loc[idx, 'label'].to_numpy()
    A = arr(M, ref, idx); C = np.stack([(A == k).sum(-1) for k in range(5)], -1); L = np.log((C + 0.5) / (C + 0.5).sum(-1, keepdims=True))
    pd_ = arr(M, [d], idx)[0]
    votes = {}; votesw = {}
    for _ in range(2000):
        q = rng.choice(len(idx), 20, replace=False); a = pd_[q, rng.integers(5, 10, 20)]
        p = ref[L[:, q, a].sum(1).argmax()]; votes[p] = votes.get(p, 0) + 1
        wr = np.argwhere((pd_[:, 5:] != y[:, None]) & (pd_[:, 5:] < 4)); s = wr[rng.choice(len(wr), 20, replace=False)]
        p = ref[L[:, s[:, 0], pd_[s[:, 0], 5 + s[:, 1]]].sum(1).argmax()]; votesw[p] = votesw.get(p, 0) + 1
    top = lambda v: ', '.join(f'{k} {c / 2000:.2f}' for k, c in sorted(v.items(), key=lambda x: -x[1])[:4])
    fv = lambda v: sum(c for k, c in v.items() if fam(k) == DIST[d]) / 2000
    print(f'  {d:17s} random q: {top(votes)}  [base fam {fv(votes):.2f}]\n  {"":17s} wrong only: {top(votesw)}  [base fam {fv(votesw):.2f}]')
R.to_csv('pairs_distill.csv', index=False)
