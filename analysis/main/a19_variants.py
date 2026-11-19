"""a19: Do fingerprints survive abliteration and reasoning fine-tuning?
Usage: python a19_variants.py <base_dir> <variants_dir> <csv> [out_prefix]
  base_dir: 30 base models (*.csv.jsonl), e.g. the base30 set from analysis/final/build_sets.py (8192-token reruns
            merged); a rerun_merged/ subfolder, if present, replaces the matching base files.
  variants_dir: 14 variant models with their original names (fingerprints/variants3k).
Model checks (Ollama config blobs, 2026-11-19): library deepseek-r1:8b = qwen3 8.2B (R1-0528-Qwen3-8B, temp 0.6/top_p 0.95
defaults) while huihui_ai deepseek-r1-abliterated:8b-llama-distill = llama 8.0B -> kind 'abl-mismatch', excluded from summaries.
Quantization differs: mannix llama3.1 abl Q4_0 (base Q4_K_M); krith nemo IQ4_XS (base Q4_0); smallthinker Q8_0 (base Q4_K_M).
Answer codes: 0-3 answer, anything else -> 4 (unparsed/failed). Mode = most common of 0-3 (4 if none parsed).
Sections
 0. Accuracy, unparsed, consistency vs parent (paired McNemar on mode-correct).
 1. Error similarity of each variant to every base model:
    - excess same-wrong share of modes over a position-bias null (as a17), and its residual after an OLS on
      min accuracy and |acc diff| fitted on CROSS-FAMILY base pairs; percentile of the residual vs those pairs.
    - run-level same-wrong (prob. two single runs pick the same wrong ending, given both wrong), with the parent's
      own run-vs-run value as the ceiling; retention = (var~parent - var~acc-matched others) / (parent self - same).
    - rank of the parent among the 30 bases.
 2. Identification of the variant against the 30-base library (reference runs 0-4 of each base), probe = random
    runs of the variant. Top-1 = parent, top-1 in parent family, parent mean rank; baseline = parent's own probe
    (runs 5-9). Open set: variant accepted as the parent under the parent's 5th-percentile threshold.
 3. Family/cluster drift: mean residual with the parent's siblings, with the reasoning cluster
    (deepseek-r1*, qwen3*), parent vs variant.
 4. llama3.2:3b refusal rows: what the abliterated twin answers where the base refused.
"""
import sys, json, os, glob, itertools, re
import numpy as np, pandas as pd, numpy.linalg as la
from scipy.stats import binomtest

v3d, vard, csv = sys.argv[1:4]
rng = np.random.default_rng(19)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

VAR = {  # variant file stem -> (short name, parent, kind)
 'huihui_ai__falcon3-abliterated_10b': ('falcon3-abl:10b', 'falcon3_10b', 'abl'),
 'huihui_ai__falcon3-abliterated_7b': ('falcon3-abl:7b', 'falcon3_7b', 'abl'),
 'huihui_ai__falcon3-abliterated_3b': ('falcon3-abl:3b', 'falcon3_3b', 'abl'),
 'huihui_ai__deepseek-r1-abliterated_14b-qwen-distill': ('r1-abl:14b', 'deepseek-r1_14b', 'abl'),
 'huihui_ai__deepseek-r1-abliterated_8b-llama-distill': ('r1-abl:8b', 'deepseek-r1_8b', 'abl-mismatch'),  # NOT a twin: library deepseek-r1:8b is R1-0528-Qwen3-8B (qwen3 arch), variant is the Llama-8B distill
 'huihui_ai__deepseek-r1-abliterated_7b-qwen-distill': ('r1-abl:7b', 'deepseek-r1_7b', 'abl'),
 'huihui_ai__llama3.2-abliterate_3b': ('llama3.2-abl:3b', 'llama3.2_3b', 'abl'),
 'mannix__llama3.1-8b-abliterated': ('llama3.1-abl:8b', 'llama3.1_8b', 'abl'),
 'krith__mistral-nemo-instruct-2407-abliterated_IQ4_XS': ('nemo-abl:12b', 'mistral-nemo_12b', 'abl'),
 'richardyoung__mistral-7b-instruct-v0.3-abliterated_Q4_K_M': ('mistral-abl:7b', 'mistral_7b', 'abl'),
 'openthinker_7b': ('openthinker:7b', 'qwen2.5_7b', 'ft'),
 'smallthinker_3b': ('smallthinker:3b', 'qwen2.5_3b', 'ft'),
 'phi4-reasoning_14b': ('phi4-reasoning:14b', 'phi4_14b', 'ft'),
 'phi4-mini-reasoning_3.8b': ('phi4-mini-reasoning', 'phi4-mini_3.8b', 'ft'),
}
fam = lambda m: 'qwen3' if m.startswith('qwen3') else re.match(r'[a-z]+', m).group(0)  # qwen3 separate from qwen2.5
reason = lambda m: m.startswith('deepseek') or m.startswith('qwen3')

ds = pd.read_csv(csv); y = ds.label.to_numpy(); n = len(y)
pos = {i: p for p, i in enumerate(ds['index'])}
def load_file(f):
    a = np.full((n, 10), 4)
    seen = np.zeros(n, bool)
    for l in open(f):  # last line per index wins
        r = json.loads(l)
        if r['index'] in pos and len(r['choices']) == 10: a[pos[r['index']]] = r['choices']; seen[pos[r['index']]] = True
    a = np.where((a >= 0) & (a < 4), a, 4)
    return a, seen.sum()
B = {}
for f in sorted(glob.glob(os.path.join(v3d, '*.csv.jsonl'))):
    m = os.path.basename(f)[:-10]; rr = os.path.join(v3d, 'rerun_merged', os.path.basename(f))
    B[m], rows = load_file(rr if os.path.exists(rr) else f)
    if rows < n: print(f'  note: base {m} has {rows}/{n} rows (missing rows coded unparsed)')
V = {}
for stem in VAR:
    V[stem], rows = load_file(os.path.join(vard, stem + '.csv.jsonl')); assert rows == n, stem
bases = sorted(B); nb = len(bases); bi = {m: i for i, m in enumerate(bases)}
print(f'bases: {nb} (rerun-merged: {sorted(os.path.basename(f)[:-10] for f in glob.glob(os.path.join(v3d, "rerun_merged", "*.csv.jsonl")))})')
print(f'variants: {len(V)}\n')

def counts(A): return np.stack([(A == k).sum(-1) for k in range(5)], -1)  # (n,5)
def mode(A):
    C = counts(A); mo = C[:, :4].argmax(-1); return np.where(C[:, :4].sum(-1) == 0, 4, mo)
def bias(mo): v = mo[mo < 4]; return np.bincount(v, minlength=4) / len(v)
wrongopts = [[o for o in range(4) if o != y[t]] for t in range(n)]
def excess(moA, moB, bA, bB):
    r = np.where((moA != y) & (moB != y) & (moA < 4) & (moB < 4))[0]
    sh = (moA[r] == moB[r]).mean()
    nul = np.mean([(bA[w] / bA[w].sum() * bB[w] / bB[w].sum()).sum() for w in (wrongopts[t] for t in r)])
    return sh - nul, sh, len(r)
def runlevel(A, Bm=None):
    """P(single run of A and single run of B pick the same WRONG parsed ending | both wrong & parsed).
    Bm None -> self, two different runs of A."""
    CA = counts(A)[:, :4].astype(float); CA[np.arange(n), y] = 0
    if Bm is None:
        num = (CA * (CA - 1)).sum(-1).sum(); W = CA.sum(-1); den = (W * (W - 1)).sum(); return num / den
    CB = counts(Bm)[:, :4].astype(float); CB[np.arange(n), y] = 0
    return (CA * CB).sum() / (CA.sum(-1) * CB.sum(-1)).sum()

# ---------- base-base reference pairs ----------
MO = {m: mode(B[m]) for m in bases}; BI = {m: bias(MO[m]) for m in bases}; ACC = {m: (MO[m] == y).mean() for m in bases}
rec = []
for a, b in itertools.combinations(bases, 2):
    e, sh, nboth = excess(MO[a], MO[b], BI[a], BI[b])
    rec.append(dict(a=a, b=b, excess=e, share=sh, rl=runlevel(B[a], B[b]), accmin=min(ACC[a], ACC[b]),
                    accdiff=abs(ACC[a] - ACC[b]), same_fam=fam(a) == fam(b), both_reason=reason(a) and reason(b)))
P = pd.DataFrame(rec)
ref = P[~P.same_fam & ~P.both_reason]  # cross-family, not both reasoning-trained
X = lambda d: np.column_stack([np.ones(len(d)), d.accmin, d.accdiff])
coef = {k: la.lstsq(X(ref), ref[k], rcond=None)[0] for k in ['excess', 'rl']}
sd = {k: (ref[k] - X(ref) @ coef[k]).std() for k in coef}
resid = lambda d, k: d[k] - X(d) @ coef[k]
P['res_excess'] = resid(P, 'excess'); P['res_rl'] = resid(P, 'rl')
refres = np.sort(resid(ref, 'excess').to_numpy())
pct = lambda v: (refres < v).mean() * 100
print(f'reference pairs (cross-family, not both reasoning): {len(ref)}; residual SD excess {sd["excess"]:.3f}, run-level {sd["rl"]:.3f}')
same = P[P.same_fam]
print(f'  same-family base pairs: mean residual excess {same.res_excess.mean():+.3f} (n={len(same)}); '
      f'reasoning-reasoning pairs {P[P.both_reason].res_excess.mean():+.3f}\n')

# ---------- 0. accuracy ----------
print('== 0. Accuracy, unparsed, consistency (variant vs parent) ==')
rows0 = []
for stem, (nm, par, kind) in VAR.items():
    A, Bp = V[stem], B[par]; mv, mp = mode(A), MO[par]
    cv, cp = mv == y, mp == y; b01 = int((cp & ~cv).sum()); b10 = int((~cp & cv).sum())
    p = binomtest(b10, b01 + b10).pvalue if b01 + b10 else 1
    una = lambda Z: (counts(Z)[:, :4].max(-1) == 10).mean()
    rows0.append(dict(variant=nm, kind=kind, parent=par, acc_var=cv.mean(), acc_par=cp.mean(), d_acc=cv.mean() - cp.mean(),
                      mcnemar_p=p, unp_var=(A == 4).mean(), unp_par=(Bp == 4).mean(), unanim_var=una(A), unanim_par=una(Bp),
                      mode_agree=(mv == mp).mean()))
T0 = pd.DataFrame(rows0); print(T0.round(3).to_string(index=False)); print()

# ---------- 1. similarity ----------
print('== 1. Error similarity to the parent vs the 29 other bases ==')
print('excess = same-wrong share of modes minus position-bias null; res = residual after accuracy control (cross-family fit);'
      ' pct = percentile of res among cross-family base pairs.\nrun-level: rl(var,parent), rl(parent self), retention = (rl_vp - rl_pred) / (rl_pp - rl_pred_pp), rl_pred from the accuracy fit.')
rows1 = []; SIM = {}
for stem, (nm, par, kind) in VAR.items():
    A = V[stem]; mv = mode(A); bv = bias(mv); av = (mv == y).mean(); r = []
    for m in bases:
        e, sh, nb_ = excess(mv, MO[m], bv, BI[m]); rl = runlevel(A, B[m])
        d = pd.DataFrame([dict(accmin=min(av, ACC[m]), accdiff=abs(av - ACC[m]), excess=e, rl=rl)])
        r.append(dict(base=m, excess=e, share=sh, nboth=nb_, rl=rl, res=float(resid(d, 'excess')[0]), res_rl=float(resid(d, 'rl')[0]),
                      rl_pred=float((X(d) @ coef['rl'])[0])))
    S = pd.DataFrame(r).set_index('base'); SIM[nm] = S
    sp = S.loc[par]; others = S.drop(par)
    rk = int((others.res > sp.res).sum()) + 1
    nearest = others.res.idxmax()
    rl_pp = runlevel(B[par]); dpp = pd.DataFrame([dict(accmin=ACC[par], accdiff=0.0)]); rl_pp_pred = float((X(dpp) @ coef['rl'])[0])
    retention = (sp.rl - sp.rl_pred) / (rl_pp - rl_pp_pred)
    sib = [m for m in bases if fam(m) == fam(par) and m != par]
    rows1.append(dict(variant=nm, kind=kind, parent=par, share=sp.share, excess=sp.excess, res=sp.res, pct=pct(sp.res),
                      parent_rank=f'{rk}/{nb}', nearest_other=nearest, nearest_res=others.res.max(),
                      rl_vp=sp.rl, rl_pp_self=rl_pp, retention=retention))
T1 = pd.DataFrame(rows1); print(T1.round(3).to_string(index=False))
for kind in ['abl', 'ft']:
    t = T1[T1.kind == kind]
    print(f'  {kind}: mean res {t.res.mean():+.3f}, parent ranked #1 for {(t.parent_rank.str.startswith("1/")).sum()}/{len(t)}, '
          f'mean retention {t.retention.mean():.2f} (median {t.retention.median():.2f})')
Sr=SIM['r1-abl:8b']; print(f'  r1-abl:8b vs its real pretraining base llama3.1_8b: res {Sr.loc["llama3.1_8b"].res:+.3f} (rank {int((Sr.res > Sr.loc["llama3.1_8b"].res).sum()) + 1}/30)')
print('  top-5 bases by residual for each variant:')
for nm, S in SIM.items():
    print(f'    {nm:20s} ' + ', '.join(f'{b} {v:+.3f}' for b, v in S.res.sort_values(ascending=False).head(5).items()))
print()

# ---------- 2. identification ----------
print('== 2. Identification against the 30-base library (reference runs 0-4; chance 1/30) ==')
def table(A, cols):
    C = counts(A[:, cols]); L = np.log((C + .5) / (C + .5).sum(-1, keepdims=True))
    Pp = np.exp(L).copy(); Pp[np.arange(n), y] = 0; Le = np.log(Pp / Pp.sum(-1, keepdims=True) + 1e-300); return L, Le
LL = [table(B[m], range(5)) for m in bases]; L = np.stack([t[0] for t in LL]); Le = np.stack([t[1] for t in LL])
def probe(A, runs, k, errors):
    if not errors:
        q = rng.choice(n, k, replace=False); a = A[q, rng.choice(runs, k)]; return L[:, q, a].sum(1)
    sub = A[:, runs]; wr = np.argwhere((sub != y[:, None]) & (sub < 4))
    s = wr[rng.choice(len(wr), k, replace=False)]; q = s[:, 0]; a = sub[q, s[:, 1]]; return Le[:, q, a].sum(1)
TR = 300; ks = [10, 20, 50]; rows2 = []
for stem, (nm, par, kind) in VAR.items():
    p = bi[par]; d = dict(variant=nm, kind=kind)
    for errors in [False, True]:
        tag = 'err' if errors else 'all'
        for k in ks:
            top = famok = 0; rk = []; topb = 0; got = {}
            for _ in range(TR):
                s = probe(V[stem], np.arange(10), k, errors); j = s.argmax(); top += j == p; famok += fam(bases[j]) == fam(par)
                rk.append((s > s[p]).sum() + 1); got[bases[j]] = got.get(bases[j], 0) + 1
                sb = probe(B[par], np.arange(5, 10), k, errors); topb += sb.argmax() == p
            d[f'{tag}{k}_par'] = top / TR; d[f'{tag}{k}_fam'] = famok / TR; d[f'{tag}{k}_rank'] = np.mean(rk); d[f'{tag}{k}_selfbase'] = topb / TR
            if k == 20: d[f'{tag}20_mostly'] = max(got, key=got.get) + f' ({max(got.values()) / TR:.2f})'
    rows2.append(d)
T2 = pd.DataFrame(rows2)
for tag in ['all', 'err']:
    cols = ['variant', 'kind'] + [f'{tag}{k}_{s}' for k in ks for s in ['par', 'selfbase']] + [f'{tag}20_fam', f'{tag}20_rank', f'{tag}20_mostly']
    print(f'-- {"all answers" if tag == "all" else "wrong answers only"}: P(top-1 = parent) for the variant vs the parent probing itself (selfbase) --')
    print(T2[cols].round(2).to_string(index=False))
for kind in ['abl', 'ft']:
    t = T2[T2.kind == kind]
    print(f'  {kind} mean: all k=20 parent {t.all20_par.mean():.2f} (parent self {t.all20_selfbase.mean():.2f}); '
          f'errors k=20 {t.err20_par.mean():.2f} (self {t.err20_selfbase.mean():.2f}); family-level all k=20 {t.all20_fam.mean():.2f}')
print('\n-- open set: variant accepted AS ITS PARENT (top-1 = parent and above the parent\'s 5th-pct threshold), all answers --')
rows2b = []
for stem, (nm, par, kind) in VAR.items():
    p = bi[par]; d = dict(variant=nm)
    for k in [20, 50]:
        thr = np.quantile([probe(B[par], np.arange(5, 10), k, False)[p] for _ in range(400)], .05)
        acc = rej_all = 0
        for _ in range(TR):
            s = probe(V[stem], np.arange(10), k, False); acc += (s.argmax() == p) and s[p] >= thr
        d[f'k{k}_passes_as_parent'] = acc / TR
    rows2b.append(d)
print(pd.DataFrame(rows2b).round(2).to_string(index=False)); print()

# ---------- 3. family / reasoning cluster drift ----------
print('== 3. Drift: mean accuracy-controlled residual with the parent\'s siblings and with the reasoning cluster ==')
def pres(a, b):
    r = P[((P.a == a) & (P.b == b)) | ((P.a == b) & (P.b == a))]; return float(r.res_excess.iloc[0])
rows3 = []
for stem, (nm, par, kind) in VAR.items():
    S = SIM[nm]; sib = [m for m in bases if fam(m) == fam(par) and m != par]
    rc = [m for m in bases if reason(m) and m != par and fam(m) != fam(par)]
    rows3.append(dict(variant=nm, kind=kind, n_sib=len(sib), sib_parent=np.mean([pres(par, m) for m in sib]), sib_variant=S.loc[sib].res.mean(),
                      reasoncl_parent=np.mean([pres(par, m) for m in rc]), reasoncl_variant=S.loc[rc].res.mean(), n_reason=len(rc)))
T3 = pd.DataFrame(rows3); T3['d_sib'] = T3.sib_variant - T3.sib_parent; T3['d_reason'] = T3.reasoncl_variant - T3.reasoncl_parent
print(T3.round(3).to_string(index=False))
for kind in ['abl', 'ft']:
    t = T3[T3.kind == kind]; print(f'  {kind}: mean d_sib {t.d_sib.mean():+.3f}, mean d_reason {t.d_reason.mean():+.3f}')
print()

# ---------- 4. refusals ----------
print('== 4. llama3.2:3b refusal rows (base unparsed in >=5 of 10 runs) ==')
Bl = B['llama3.2_3b']; Vl = V['huihui_ai__llama3.2-abliterate_3b']
rf = (Bl == 4).sum(1) >= 5; mv = mode(Vl)
print(f'  rows: {rf.sum()}; base unparsed runs there {(Bl[rf] == 4).mean():.2f}; abliterated unparsed there {(Vl[rf] == 4).mean():.3f}; '
      f'abliterated mode acc there {(mv[rf] == y[rf]).mean():.3f} vs elsewhere {(mv[~rf] == y[~rf]).mean():.3f}; '
      f'base mode acc elsewhere {(MO["llama3.2_3b"][~rf] == y[~rf]).mean():.3f}')
print(f'  overall unparsed runs: base {(Bl == 4).mean():.3f} -> abliterated {(Vl == 4).mean():.3f}')

out = sys.argv[4] if len(sys.argv) > 4 else 'a19'
pd.concat([S.assign(variant=nm) for nm, S in SIM.items()]).to_csv(f'{out}_variant_base_pairs.csv')
T0.merge(T1, on=['variant', 'kind', 'parent']).merge(T2, on=['variant', 'kind']).merge(T3, on=['variant', 'kind']).to_csv(f'{out}_summary.csv', index=False)
