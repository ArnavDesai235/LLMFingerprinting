"""a20: what happens to identification when the library also holds near-copies (abliterated twins, fine-tunes)?
Usage (inside a run dir with load.py/labels.py, FPV3 = the all44 set): python a20_library_with_variants.py
Likelihood identification, reference runs 0-4, probe runs 5-9, k random questions or k wrong answers.
Reports, for the 44-model library and for the 30 base models alone:
  exact     top-1 = the model itself
  twin-ok   top-1 = the model or its twin (variant <-> parent); the 'lineage-level' answer an auditor would accept
  to_twin   share of probes assigned to the twin
Per model (k=20): exact, to_twin, most common wrong answer.
"""
import sys; sys.path.insert(0, '.')
from load import *
from labels import is_variant, twin, KIND, PARENT
M = load(); full = sorted(m for m in M if len(M[m]) == 3000); idx = ds['index'].tolist()
y = ds.set_index('index').loc[idx, 'label'].to_numpy(); A = arr(M, full, idx); nm, n, _ = A.shape
rng = np.random.default_rng(20)
C = np.stack([(A[:, :, :5] == k).sum(-1) for k in range(5)], -1); L = np.log((C + .5) / (C + .5).sum(-1, keepdims=True))
P = np.exp(L).copy(); P[:, np.arange(n), y] = 0; Le = np.log(P / P.sum(-1, keepdims=True) + 1e-300)
TW = np.array([[twin(a, b) for b in full] for a in full])
base = np.array([not is_variant(m) for m in full])

def run(sub, ks, trials=300):
    sub = np.asarray(sub); out = {}; per = {}
    for k in ks:
        r = dict(all_exact=0, all_twinok=0, all_totwin=0, err_exact=0, err_twinok=0, err_totwin=0); T = 0
        for _ in range(trials):
            for i in sub:
                q = rng.choice(n, k, replace=False); a = A[i, q, rng.integers(5, 10, k)]
                j = sub[L[sub][:, q, a].sum(1).argmax()]
                r['all_exact'] += j == i; r['all_twinok'] += (j == i) or TW[i, j]; r['all_totwin'] += TW[i, j]
                pw = A[i, :, 5:]; wr = np.argwhere((pw != y[:, None]) & (pw < 4)); s = wr[rng.choice(len(wr), k, replace=False)]
                j2 = sub[Le[sub][:, s[:, 0], pw[s[:, 0], s[:, 1]]].sum(1).argmax()]
                r['err_exact'] += j2 == i; r['err_twinok'] += (j2 == i) or TW[i, j2]; r['err_totwin'] += TW[i, j2]; T += 1
                if k == 20:
                    d = per.setdefault(full[i], dict(exact=0, to_twin=0, wrong={}))
                    d['exact'] += j == i; d['to_twin'] += TW[i, j]
                    if j != i: d['wrong'][full[j]] = d['wrong'].get(full[j], 0) + 1
        out[k] = {c: v / T for c, v in r.items()}
    return pd.DataFrame(out).T, per
pd.set_option('display.width', 220)
ks = [5, 10, 20, 50]
print(f'library: {nm} models ({base.sum()} base, {(~base).sum()} variants); chance {1/nm:.3f}\n')
R44, per = run(np.arange(nm), ks)
print('== 44-model library, all 44 probed =='); print(R44.round(3).to_string())
Rb44, _ = run(np.where(base)[0], ks)
print('\n== 30 base models probed against the 30-model library (no variants) =='); print(Rb44.round(3).to_string())
# base models probed against the 44 library: does adding the variants hurt identification of the originals?
sub = np.arange(nm); res = {}
for k in ks:
    ok = tw = T = 0
    for _ in range(300):
        for i in np.where(base)[0]:
            q = rng.choice(n, k, replace=False); a = A[i, q, rng.integers(5, 10, k)]; j = L[:, q, a].sum(1).argmax()
            ok += j == i; tw += TW[i, j]; T += 1
    res[k] = dict(exact=ok / T, to_twin=tw / T)
print('\n== the 30 base models probed against the 44-model library (all answers) =='); print(pd.DataFrame(res).T.round(3).to_string())
print('\n== per model, k=20 random questions, 44-model library ==')
rows = []
for m in full:
    d = per[m]; T = 300; top = max(d['wrong'], key=d['wrong'].get) if d['wrong'] else ''
    tw = [b for b in full if twin(m, b)]
    rows.append(dict(model=m, kind=KIND.get(m, 'base'), twin=','.join(tw), exact=d['exact'] / T, to_twin=d['to_twin'] / T,
                     top_wrong=top, top_wrong_share=d['wrong'].get(top, 0) / T))
D = pd.DataFrame(rows); print(D[D.twin != ''].sort_values(['kind', 'model']).round(2).to_string(index=False))
print('\nmodels without a twin in the library: mean exact', round(D[D.twin == ''].exact.mean(), 3))
