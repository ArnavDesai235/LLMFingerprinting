"""Identification on UNSEEN questions: the fingerprint is learned on one set of questions and tested on disjoint ones.
The per-question fingerprint cannot score a question it never saw, so each model gets a small generalizing model:
a conditional logit over the 4 endings, P_m(j | q) = softmax_j(w_m . x_qj), trained on runs 0-4 of the TRAIN questions.
Features of each ending (no label information): answer slot (one-hot), length (chars, words), word overlap with the
context, TF-IDF cosine to the context, and a 64-d SVD of the ending's TF-IDF vector. TF-IDF/SVD are fit on train text only.
Probe = single answers from runs 5-9 on TEST questions. Splits: 5-fold over questions (stratified by source), and
domain shift WikiHow -> ActivityNet and ActivityNet -> WikiHow.
Baselines: position-bias only (slot one-hot), and the per-question fingerprint on the same test questions (upper bound:
what you get when the library can be queried on the test questions).
"""
import sys, re; sys.path.insert(0, '.')
from load import *
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
M = load(); full = [m for m in M if len(M[m]) == 3000]
D = ds.reset_index(drop=True); idx = D['index'].tolist(); y = D.label.to_numpy(); src = D.source_id.str.split('~').str[0].to_numpy()
A = arr(M, full, idx); nm, n, _ = A.shape; rng = np.random.default_rng(11)
ends = D[['end0', 'end1', 'end2', 'end3']].fillna('').astype(str).to_numpy(); ctx = D.ctx.fillna('').astype(str).to_numpy()
tok = lambda s: set(re.findall(r'[a-z]+', s.lower()))
def feats(tr):
    tv = TfidfVectorizer(min_df=2, sublinear_tf=True).fit(np.r_[ctx[tr], ends[tr].ravel()])
    E = tv.transform(ends.ravel()); Cx = tv.transform(ctx)
    svd = TruncatedSVD(64, random_state=0).fit(tv.transform(ends[tr].ravel()))
    Z = svd.transform(E).reshape(n, 4, -1)
    cos = np.asarray((E.multiply(Cx[np.repeat(np.arange(n), 4)])).sum(1)).reshape(n, 4)
    ov = np.array([[len(tok(e) & tok(c)) / max(len(tok(e)), 1) for e in row] for row, c in zip(ends, ctx)])
    ln = np.log1p(np.vectorize(len)(ends)); wl = np.log1p(np.vectorize(lambda s: len(s.split()))(ends))
    num = np.stack([cos, ov, ln, wl], -1); num = (num - num[tr].mean((0, 1))) / (num[tr].std((0, 1)) + 1e-9)
    rel = lambda v: v - v.mean(1, keepdims=True)                    # relative to the other endings of the same question
    X = np.concatenate([np.broadcast_to(np.eye(4), (n, 4, 4)), rel(num), rel(Z) * 3], -1)
    return X
def fit(X, cnt, lam):
    d = X.shape[-1]
    def f(w):
        s = X @ w; lp = s - logsumexp(s, 1, keepdims=True); p = np.exp(lp)
        loss = -(cnt * lp).sum() + lam * (w[4:] ** 2).sum()
        g = -np.einsum('qj,qjd->d', cnt - cnt.sum(1, keepdims=True) * p, X); g[4:] += 2 * lam * w[4:]
        return loss, g
    return minimize(f, np.zeros(d), jac=True, method='L-BFGS-B').x
def evaluate(tr, te, lam=3.0, pos_only=False, trials=300):
    X = feats(tr)
    if pos_only: X = X[..., :4]
    R = A[:, :, :5]; cnt = np.stack([(R == k).sum(-1) for k in range(4)], -1).astype(float)   # models x n x 4 (unparsed dropped)
    W = np.stack([fit(X[tr], cnt[i, tr], lam) for i in range(nm)])
    S = np.einsum('qjd,md->mqj', X, W); LP = S - logsumexp(S, 2, keepdims=True)                # models x n x 4
    LPe = LP.copy(); LPe[:, np.arange(n), y] = -np.inf; LPe = LPe - logsumexp(LPe, 2, keepdims=True)
    # per-question fingerprint on the same test questions (library re-queried there): upper bound
    C5 = np.stack([(R == k).sum(-1) for k in range(5)], -1); Lq = np.log((C5 + .5) / (C5 + .5).sum(-1, keepdims=True))
    Pq = np.exp(Lq).copy(); Pq[:, np.arange(n), y] = 0; Lqe = np.log(Pq / Pq.sum(-1, keepdims=True) + 1e-300)
    predacc = np.mean([(LP[i, te].argmax(-1) == cnt[i, te].argmax(-1)).mean() for i in range(nm)])
    base = np.mean([(np.bincount(cnt[i, tr].argmax(-1), minlength=4).argmax() == cnt[i, te].argmax(-1)).mean() for i in range(nm)])
    res = {}
    for k in [5, 10, 20, 50]:
        ok = dict(gen_random=0, gen_errors=0, perq_random=0, perq_errors=0); T = 0
        for _ in range(trials):
            for i in range(nm):
                q = rng.choice(te, k, replace=False); a = A[i, q, rng.integers(5, 10, k)]; v = a < 4
                ok['gen_random'] += LP[:, q[v], a[v]].sum(1).argmax() == i
                ok['perq_random'] += Lq[:, q, a].sum(1).argmax() == i
                pw = A[i][te][:, 5:]; wr = np.argwhere((pw != y[te][:, None]) & (pw < 4)); s = wr[rng.choice(len(wr), k, replace=False)]
                q2 = te[s[:, 0]]; a2 = pw[s[:, 0], s[:, 1]]
                ok['gen_errors'] += LPe[:, q2, a2].sum(1).argmax() == i
                ok['perq_errors'] += Lqe[:, q2, a2].sum(1).argmax() == i; T += 1
        res[k] = {m: v / T for m, v in ok.items()}
    return pd.DataFrame(res).T, predacc, base
pd.set_option('display.width', 200)
print(f'{nm} models, chance {1/nm:.3f}\n')
folds = np.zeros(n, int)
for s in np.unique(src):
    ii = rng.permutation(np.where(src == s)[0]); folds[ii] = np.arange(len(ii)) % 5
for tag, kw in [('TEXT FEATURES', {}), ('POSITION BIAS ONLY', {'pos_only': True})]:
    outs, pa, ba = [], [], []
    for f in range(5):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        o, p, b = evaluate(tr, te, trials=60, **kw); outs.append(o); pa.append(p); ba.append(b)
    print(f'== 5-fold unseen questions, {tag} (train 2400 / test 600 questions per fold) ==')
    print(f'   predicting each model\'s usual answer on unseen questions: {np.mean(pa):.3f}  (always-its-favourite-slot baseline {np.mean(ba):.3f}, random .25)')
    print(sum(outs).div(5).round(3).to_string(), '\n')
for name, trs, tes in [('WikiHow -> ActivityNet', 'wikihow', 'activitynet'), ('ActivityNet -> WikiHow', 'activitynet', 'wikihow')]:
    tr, te = np.where(src == trs)[0], np.where(src == tes)[0]
    o, p, b = evaluate(tr, te, trials=150)
    print(f'== domain shift {name} (train {len(tr)} / test {len(te)}), TEXT FEATURES ==   usual-answer prediction {p:.3f} (slot baseline {b:.3f})')
    print(o.round(3).to_string(), '\n')
