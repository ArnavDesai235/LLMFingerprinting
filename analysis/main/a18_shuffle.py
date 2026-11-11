"""Shuffled-options first pass. Usage: python a18_shuffle.py <holdout_dir> <shuffle_dir> <holdout_csv>
Models = complete (500/500) in both. 'choices' in both are original-ending ids (shuffled ones mapped back).
"""
import sys, json, os, glob, itertools, re
import numpy as np, pandas as pd, numpy.linalg as la
from scipy.stats import spearmanr, pearsonr
hd, sd, csv = sys.argv[1:4]
rng = np.random.default_rng(5)
ds = pd.read_csv(csv); pos = {i: p for p, i in enumerate(ds['index'])}; y = ds.label.to_numpy(); n = len(y)
fam = lambda m: re.match(r'[a-z]+', m).group(0)
reason = lambda m: m.startswith('deepseek') or m.startswith('qwen3')

def load(d, key='choices'):
    M = {}
    for f in sorted(glob.glob(os.path.join(d, '*.csv.jsonl'))):
        a = np.full((n, 10), -9)
        for l in open(f):
            r = json.loads(l)
            if r['index'] in pos and len(r[key]) == 10: a[pos[r['index']]] = r[key]
        if (a[:, 0] != -9).all(): M[os.path.basename(f)[:-10]] = np.where((a >= 0) & (a < 4), a, 4)
    return M
U = load(hd); S = load(sd); SL = load(sd, 'slots')
models = sorted(set(U) & set(S)); nm = len(models)
print(f'models complete in both: {nm} (chance {1/nm:.3f})\n  {models}\n')
AU = np.stack([U[m] for m in models]); AS = np.stack([S[m] for m in models]); ASL = np.stack([SL[m] for m in models])

def counts(A): return np.stack([(A == k).sum(-1) for k in range(4)], -1)
def mode(A): return counts(A).argmax(-1)
rows = []
for i, m in enumerate(models):
    u, s, sl = AU[i], AS[i], ASL[i]
    cu, cs, csl = counts(u), counts(s), counts(sl)
    vu, vs, vsl = u[u < 4], s[s < 4], sl[sl < 4]
    rows.append(dict(model=m,
        acc_run_u=(u == y[:, None]).sum() / (u < 4).sum(), acc_run_s=(s == y[:, None]).sum() / (s < 4).sum(),
        acc_mode_u=(mode(u) == y).mean(), acc_mode_s=(mode(s) == y).mean(),
        topslot_u=np.bincount(vu, minlength=4).max() / len(vu), topslot_s=np.bincount(vsl, minlength=4).max() / len(vsl),
        unan_u=(cu.max(-1) == 10).mean(), unan_s=(cs.max(-1) == 10).mean(),
        same_ending_share_s=cs.max(-1).mean() / 10, same_slot_share_s=csl.max(-1).mean() / 10,
        mode_agree=(mode(u) == mode(s)).mean()))
T = pd.DataFrame(rows).set_index('model'); pd.set_option('display.width', 250)
print('== 1. Per model: unshuffled (u) vs shuffled (s) ==')
print('acc_run = per-answer accuracy; acc_mode = accuracy of the 10-run mode; topslot = share of answers in the favourite slot;')
print('unan = all 10 runs same ending; same_ending/same_slot_share_s = mean share of the modal ENDING / modal SLOT across shuffled runs;')
print('mode_agree = unshuffled mode == shuffled mode (same ending)\n')
print(T.round(3).to_string())
print('\nmeans:'); print(T.mean().round(3).to_string())
T['d_acc_run'] = T.acc_run_s - T.acc_run_u; T['d_acc_mode'] = T.acc_mode_s - T.acc_mode_u
print(f"\ncorr(topslot_u, d_acc_run) = {pearsonr(T.topslot_u, T.d_acc_run)[0]:.3f};  corr(topslot_u, d_acc_mode) = {pearsonr(T.topslot_u, T.d_acc_mode)[0]:.3f}")
print(f"models whose shuffled answers follow the ENDING more than the SLOT: {(T.same_ending_share_s > T.same_slot_share_s).sum()}/{nm}")

print('\n== 2. Errors: is the wrong ending the same when slots change? ==')
# on questions where the unshuffled mode is wrong, how often is the shuffled mode the same wrong ending
r2 = []
for i, m in enumerate(models):
    mu, ms = mode(AU[i]), mode(AS[i]); w = mu != y
    r2.append(dict(model=m, n_wrong_u=w.sum(), shuf_mode_same_wrong=(ms[w] == mu[w]).mean(), shuf_mode_correct=(ms[w] == y[w]).mean()))
R2 = pd.DataFrame(r2).set_index('model'); print(R2.round(3).to_string()); print('means:', R2.mean().round(3).to_dict(), '(chance same wrong given still wrong = 1/3)')

def tables(A):
    C = np.stack([(A[:, :, :5] == k).sum(-1) for k in range(5)], -1)
    L = np.log((C + .5) / (C + .5).sum(-1, keepdims=True))
    P = np.exp(L).copy(); P[:, np.arange(n), y] = 0; Le = np.log(P / P.sum(-1, keepdims=True) + 1e-300); return L, Le
def ident(Aref, Aprobe, ks, trials=200, probe_runs=range(5, 10)):
    L, Le = tables(Aref); pr = np.array(list(probe_runs)); out = {}
    for k in ks:
        ok = np.zeros(2); T_ = 0
        for _ in range(trials):
            for i in range(nm):
                q = rng.choice(n, k, replace=False); a = Aprobe[i, q, rng.choice(pr, k)]; ok[0] += L[:, q, a].sum(1).argmax() == i
                pw = Aprobe[i][:, pr]; wr = np.argwhere((pw != y[:, None]) & (pw < 4))
                if len(wr) < k: continue
                s_ = wr[rng.choice(len(wr), k, replace=False)]; q = s_[:, 0]; a = pw[q, s_[:, 1]]
                ok[1] += Le[:, q, a].sum(1).argmax() == i; T_ += 1
        out[k] = dict(all_answers=ok[0] / (trials * nm), wrong_only=ok[1] / max(T_, 1))
    return pd.DataFrame(out).T
ks = [5, 10, 20, 50]
print('\n== 3. Identification (likelihood; reference runs 0-4, probe runs 5-9) ==')
for tag, R, P in [('unshuffled -> unshuffled', AU, AU), ('shuffled -> shuffled', AS, AS), ('unshuffled ref -> shuffled probe', AU, AS), ('shuffled ref -> unshuffled probe', AS, AU)]:
    print(f'-- {tag} --'); print(ident(R, P, ks).round(3).to_string())

print('\n== 4. Pairwise same-wrong share (modes), shuffled vs unshuffled ==')
def pairs(A):
    mo = mode(A); W = mo != y; acc = 1 - W.mean(1); bias = np.stack([np.bincount(mo[i], minlength=4) / n for i in range(nm)]); out = []
    for i, j in itertools.combinations(range(nm), 2):
        r = np.where(W[i] & W[j])[0]; sh = (mo[i, r] == mo[j, r]).mean() if len(r) >= 5 else np.nan
        nul = np.mean([(lambda w: (bias[i, w] / bias[i, w].sum() * bias[j, w] / bias[j, w].sum()).sum())([o for o in range(4) if o != y[t]]) for t in r]) if len(r) >= 5 else np.nan
        out.append(dict(a=models[i], b=models[j], share=sh, excess=sh - nul, accmin=min(acc[i], acc[j]), accdiff=abs(acc[i] - acc[j]), nboth=len(r)))
    return pd.DataFrame(out)
PU, PS = pairs(AU), pairs(AS); ok = PU.share.notna() & PS.share.notna()
print(f'  mean same-wrong share: unshuffled {PU.share[ok].mean():.3f} (excess over slot null {PU.excess[ok].mean():+.3f}), shuffled {PS.share[ok].mean():.3f} (excess {PS.excess[ok].mean():+.3f}); chance 1/3')
print(f'  pair-level correlation unshuffled vs shuffled share: Spearman {spearmanr(PU.share[ok], PS.share[ok])[0]:.3f}; unshuffled EXCESS vs shuffled share: Spearman {spearmanr(PU.excess[ok], PS.share[ok])[0]:.3f}')
def famtest(P, col):
    Q = P[P[col].notna()].reset_index(drop=True)
    def coef(lab): X = np.column_stack([np.ones(len(Q)), Q.accmin, Q.accdiff, lab]); return la.lstsq(X, Q[col], rcond=None)[0][3]
    sf = np.array([float(fam(a) == fam(b)) for a, b in zip(Q.a, Q.b)]); obs = coef(sf); null = []
    for _ in range(2000):
        p = dict(zip(models, rng.permutation(models))); null.append(coef(np.array([float(fam(p[a]) == fam(p[b])) for a, b in zip(Q.a, Q.b)])))
    return obs, (np.abs(null) >= abs(obs)).mean()
for tag, P, col in [('unshuffled, excess over slot null', PU, 'excess'), ('unshuffled, raw share', PU, 'share'), ('shuffled, raw share (no null needed)', PS, 'share')]:
    o, p = famtest(P, col); print(f'  same-family effect, {tag}: {o:+.3f} (p {p:.3f})')
nr = sum(reason(m) for m in models); print(f'  (reasoning models complete in both: {nr}; reasoning test waits for the rest)')
