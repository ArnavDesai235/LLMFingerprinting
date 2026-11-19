"""Make the summary figures and tables from the fingerprint files in this repo.

Usage (from the repo root):  python analysis/make_figures.py
Writes figures/*.png and results/tables/*.{csv,md}. Needs numpy, pandas, matplotlib.

Protocol is the same as the main analyses (analysis/main/a17_holdout.py, a18_shuffle.py):
  reference = runs 0-4, probe = runs 5-9, likelihood identification with add-0.5 smoothing,
  "errors only" = probe answers that are wrong, scored by P(answer | not the label).
A model counts as complete when it has all rows (3,000 for v3, 500 for holdout/shuffle).
"""
import glob, json, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
FP, DATA = os.path.join(ROOT, 'fingerprints'), os.path.join(ROOT, 'data')
FIG, TAB = os.path.join(ROOT, 'figures'), os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
rng = np.random.default_rng(11)

# ---------------------------------------------------------------- model metadata
FAMILIES = ['gemma', 'phi', 'llama', 'qwen2.5', 'qwen3', 'falcon', 'mistral', 'deepseek-r1']
COLORS = dict(zip(FAMILIES, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']))
def family(m):
    if m.startswith('qwen2.5'): return 'qwen2.5'
    if m.startswith('qwen3'): return 'qwen3'
    if m.startswith('deepseek'): return 'deepseek-r1'
    return re.match(r'[a-z]+', m).group(0)
def reasoning(m): return m.startswith('deepseek') or m.startswith('qwen3')
def size_b(m): return float(re.search(r'_([\d.]+)b$', m).group(1))
def pretty(m): return m.replace('_', ':')

# ---------------------------------------------------------------- loading
def load(d, csv, key='choices'):
    ds = pd.read_csv(csv); pos = {i: p for p, i in enumerate(ds['index'])}; M, rows = {}, {}
    for f in sorted(glob.glob(os.path.join(d, '*.csv.jsonl'))):
        a = np.full((len(ds), 10), -9)
        for l in open(f):
            r = json.loads(l)
            if r['index'] in pos and len(r[key]) == 10: a[pos[r['index']]] = r[key]   # duplicates: last line wins
        m = os.path.basename(f)[:-len('.csv.jsonl')]; done = a[:, 0] != -9; rows[m] = int(done.sum())
        if done.all(): M[m] = np.where((a >= 0) & (a < 4), a, 4)    # 4 = unparsed or failed
    return M, ds.label.to_numpy(), rows

V3, y3, rows3 = load(f'{FP}/v3_clean3k', f'{DATA}/hellaswag_val_clean_3k.csv')
V3.update(load(f'{FP}/v3_clean3k_rerun8192', f'{DATA}/hellaswag_val_clean_3k.csv')[0])   # 8192-token reruns (r1:8b, qwen3:4b/1.7b)
HO, yH, rowsH = load(f'{FP}/holdout500', f'{DATA}/hellaswag_val_clean_holdout500.csv')
SH, _, rowsS = load(f'{FP}/shuffle500', f'{DATA}/hellaswag_val_clean_holdout500.csv')
SL, _, _ = load(f'{FP}/shuffle500', f'{DATA}/hellaswag_val_clean_holdout500.csv', key='slots')
ORDER = sorted(set(rows3) | set(rowsH), key=lambda m: (FAMILIES.index(family(m)), -size_b(m)))

def counts(A, runs=slice(None)): return np.stack([(A[..., runs] == k).sum(-1) for k in range(5)], -1)
def mode(A): return counts(A)[..., :4].argmax(-1)

# ---------------------------------------------------------------- table 1: model profile
prof = []
for m in ORDER:
    r = dict(model=pretty(m), family=family(m), size_b=size_b(m), reasoning=reasoning(m),
             rows_3k=rows3.get(m, 0), complete_3k=m in V3)
    for tag, M, y in [('3k', V3, y3), ('holdout', HO, yH), ('shuffled', SH, yH)]:
        if m in M:
            A = M[m]; C = counts(A)
            r[f'acc_mode_{tag}'] = (mode(A) == y).mean()
            if tag != 'shuffled':
                r[f'unanimous_{tag}'] = (C.max(-1) == 10).mean(); r[f'unparsed_{tag}'] = (A == 4).mean()
    if m in SH:
        r['same_ending_share_shuffled'] = counts(SH[m])[..., :4].max(-1).mean() / 10
        r['same_slot_share_shuffled'] = counts(SL[m])[..., :4].max(-1).mean() / 10
    prof.append(r)
P = pd.DataFrame(prof)
P.to_csv(f'{TAB}/model_profile.csv', index=False, float_format='%.4f')

# ---------------------------------------------------------------- identification
def tables(A, y):
    C = counts(A, slice(0, 5)); L = np.log((C + .5) / (C + .5).sum(-1, keepdims=True))
    Pw = np.exp(L).copy(); Pw[:, np.arange(len(y)), y] = 0; Le = np.log(Pw / Pw.sum(-1, keepdims=True) + 1e-300)
    return L, Le
def identify(A, y, ks, trials, Aref=None):
    """A: models x questions x 10 runs. Reference built from Aref (default A) runs 0-4; probes from A runs 5-9."""
    L, Le = tables(A if Aref is None else Aref, y); nm, n = A.shape[:2]; out = []
    for k in ks:
        ok_all = ok_err = T = 0
        for _ in range(trials):
            for i in range(nm):
                q = rng.choice(n, k, replace=False); a = A[i, q, rng.integers(5, 10, k)]
                ok_all += L[:, q, a].sum(1).argmax() == i
                pw = A[i, :, 5:]; wr = np.argwhere((pw != y[:, None]) & (pw < 4))
                if len(wr) < k: continue
                s = wr[rng.choice(len(wr), k, replace=False)]; q = s[:, 0]; a = pw[q, s[:, 1]]
                ok_err += Le[:, q, a].sum(1).argmax() == i; T += 1
        out.append(dict(k=k, all_answers=ok_all / (trials * nm), wrong_answers_only=ok_err / max(T, 1)))
    return pd.DataFrame(out)

KS = [1, 3, 5, 10, 20, 50]
m3 = [m for m in ORDER if m in V3]; mH = [m for m in ORDER if m in HO]
id3 = identify(np.stack([V3[m] for m in m3]), y3, KS, 150)
idH = identify(np.stack([HO[m] for m in mH]), yH, KS, 150)
both = [m for m in ORDER if m in HO and m in SH]
AU = np.stack([HO[m] for m in both]); AS = np.stack([SH[m] for m in both])
idU = identify(AU, yH, KS, 150); idS = identify(AS, yH, KS, 150)
ID = pd.concat([id3.assign(set=f'clean 3k ({len(m3)} models)', chance=1 / len(m3)),
                idH.assign(set=f'held-out 500 ({len(mH)} models)', chance=1 / len(mH)),
                idU.assign(set=f'held-out 500, fixed order ({len(both)} models)', chance=1 / len(both)),
                idS.assign(set=f'held-out 500, shuffled order ({len(both)} models)', chance=1 / len(both))])
ID.to_csv(f'{TAB}/identification.csv', index=False, float_format='%.3f')

# ---------------------------------------------------------------- pairwise same-wrong share (3k)
A3 = np.stack([V3[m] for m in m3]); mo = mode(A3); W = mo != y3[None]
S = np.full((len(m3), len(m3)), np.nan)
for i in range(len(m3)):
    for j in range(len(m3)):
        if i != j:
            b = W[i] & W[j]; S[i, j] = (mo[i, b] == mo[j, b]).mean()
pd.DataFrame(S, index=[pretty(m) for m in m3], columns=[pretty(m) for m in m3]).to_csv(
    f'{TAB}/same_wrong_share_3k.csv', float_format='%.3f')

# ---------------------------------------------------------------- figures
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': .6,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 200,
                     'savefig.bbox': 'tight'})
INK, MUTED = '#0b0b0b', '#52514e'
def legend_families(ax, fams, **kw):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS[f], label=f) for f in fams], frameon=False, **kw)

# Fig 1: accuracy per model
Q = P.dropna(subset=['acc_mode_holdout']).copy()
Q['acc'] = Q.acc_mode_3k.fillna(Q.acc_mode_holdout); Q = Q.sort_values('acc')
fig, ax = plt.subplots(figsize=(7, 8))
yy = np.arange(len(Q))
ax.barh(yy, Q.acc, color=[COLORS[f] for f in Q.family], height=.7, zorder=2)
ax.scatter(Q.acc_mode_holdout, yy, s=22, color='white', edgecolor=INK, lw=1, zorder=3, label='held-out 500')
ax.set_yticks(yy, [f"{m}{'  (R)' if r else ''}{'' if c else '  *'}" for m, r, c in zip(Q.model, Q.reasoning, Q.complete_3k)])
ax.axvline(.25, color=MUTED, ls='--', lw=1); ax.text(.255, -1.1, 'chance', color=MUTED, fontsize=8)
ax.set_xlim(0, 1); ax.set_xlabel('Accuracy of the 10-run majority answer'); ax.grid(axis='y', visible=False)
ax.set_title('Model accuracy on clean HellaSwag (bars: 3k set, dots: held-out 500)', loc='left', fontsize=11)
legend_families(ax, FAMILIES, loc='lower right', fontsize=8)
fig.text(.02, -.01, '(R) reasoning-trained.  * not yet complete on the 3k set: bar shows held-out accuracy.',
         fontsize=8, color=MUTED)
fig.savefig(f'{FIG}/fig1_accuracy.png'); plt.close(fig)

# Fig 2: identification vs k, 3k vs held-out
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, col, title in [(axs[0], 'all_answers', 'Random questions'), (axs[1], 'wrong_answers_only', 'Wrong answers only')]:
    for d, c, lab in [(id3, '#2a78d6', f'clean 3k, {len(m3)} models'), (idH, '#eb6834', f'held-out 500, {len(mH)} models')]:
        ax.plot(d.k, d[col], marker='o', ms=5, lw=2, color=c, label=lab)
    ax.axhline(1 / len(mH), color=MUTED, ls='--', lw=1); ax.text(1.05, 1 / len(mH) + .015, 'chance', color=MUTED, fontsize=8)
    ax.set_xscale('log'); ax.set_xticks(KS, [str(k) for k in KS]); ax.set_ylim(0, 1.02)
    ax.set_xlabel('k probe answers'); ax.set_title(title, loc='left', fontsize=11)
axs[0].set_ylabel('Top-1 identification accuracy'); axs[1].legend(frameon=False, loc='lower right')
fig.suptitle('Identification replicates on 500 new questions', x=.06, y=1.03, ha='left', fontsize=12)
fig.savefig(f'{FIG}/fig2_identification.png'); plt.close(fig)

# Fig 3: heatmap of same-wrong share
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(S, cmap='Blues', vmin=.2, vmax=.9)
ax.set_xticks(range(len(m3)), [pretty(m) for m in m3], rotation=90, fontsize=7.5)
ax.set_yticks(range(len(m3)), [pretty(m) for m in m3], fontsize=7.5); ax.grid(False)
fams3 = [family(m) for m in m3]
for i in range(1, len(m3)):
    if fams3[i] != fams3[i - 1]:
        ax.axhline(i - .5, color='white', lw=2); ax.axvline(i - .5, color='white', lw=2)
for t, f in zip(ax.get_yticklabels(), fams3): t.set_color(COLORS[f])
for t, f in zip(ax.get_xticklabels(), fams3): t.set_color(COLORS[f])
cb = fig.colorbar(im, ax=ax, shrink=.7); cb.set_label('P(same wrong ending | both wrong)')
ax.set_title(f'Shared wrong answers, clean 3k ({len(m3)} models, grouped by family; chance = .33)', loc='left', fontsize=11)
fig.savefig(f'{FIG}/fig3_same_wrong_heatmap.png'); plt.close(fig)

# Fig 4: lineage vs reasoning (accuracy-controlled residual by pair type)
LR = pd.read_csv(os.path.join(ROOT, 'results', 'pairs', 'pairs_lineage_reasoning.csv'))
groups = ['unrelated', 'reasoning-nonreasoning', 'same base lineage (non-reasoning pairs)', 'reasoning-reasoning']
labels = ['Unrelated', 'Reasoning vs\nnon-reasoning', 'Same base lineage\n(non-reasoning)', 'Both\nreasoning-trained']
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, g in enumerate(groups):
    v = LR[LR.grp == g].res.to_numpy()
    ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=12, alpha=.55, color='#2a78d6' if i < 3 else '#e34948', lw=0, zorder=2)
    ax.plot([i - .3, i + .3], [v.mean()] * 2, color=INK, lw=2.2, zorder=3)
    ax.text(i + .33, v.mean(), f'{v.mean():+.3f}\n(n={len(v)})', va='center', fontsize=8, color=INK)
ax.axhline(0, color=MUTED, lw=1); ax.set_xticks(range(4), labels); ax.grid(axis='x', visible=False)
ax.set_ylabel('Excess same-wrong share,\naccuracy-controlled residual'); ax.set_xlim(-.5, 3.9)
ax.set_title(f'Reasoning training predicts shared wrong answers more than lineage ({len(set(LR.a) | set(LR.b))} models, {len(LR)} pairs)', loc='left', fontsize=11)
fig.savefig(f'{FIG}/fig4_lineage_vs_reasoning.png'); plt.close(fig)

# Fig 5: shuffled options, content vs position
Z = P.dropna(subset=['same_ending_share_shuffled'])
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot([.25, 1], [.25, 1], color=MUTED, ls='--', lw=1)
ax.text(.9, .93, 'equal', rotation=45, color=MUTED, fontsize=8)
for _, r in Z.iterrows():
    ax.scatter(r.same_slot_share_shuffled, r.same_ending_share_shuffled, s=46, color=COLORS[r.family], edgecolor='white', lw=1.5, zorder=3)
    if r.same_slot_share_shuffled > r.same_ending_share_shuffled - .1:
        ax.annotate(r.model, (r.same_slot_share_shuffled, r.same_ending_share_shuffled), xytext=(5, -3),
                    textcoords='offset points', fontsize=7.5, color=INK)
ax.set_xlim(.25, 1); ax.set_ylim(.25, 1)
ax.set_xlabel('Share of runs choosing the modal SLOT'); ax.set_ylabel('Share of runs choosing the modal ENDING')
ax.set_title(f'Options shuffled every run: most models follow content ({len(Z)} models)', loc='left', fontsize=11)
legend_families(ax, [f for f in FAMILIES if f in set(Z.family)], loc='center right', fontsize=8)
fig.savefig(f'{FIG}/fig5_shuffle_content_vs_slot.png'); plt.close(fig)

# Fig 6: identification, fixed vs shuffled order
fig, ax = plt.subplots(figsize=(6.5, 4))
for d, c, lab in [(idU, '#2a78d6', 'fixed order'), (idS, '#eb6834', 'shuffled every run')]:
    ax.plot(d.k, d.wrong_answers_only, marker='o', ms=5, lw=2, color=c, label=lab)
ax.axhline(1 / len(both), color=MUTED, ls='--', lw=1); ax.text(1.05, 1 / len(both) + .015, 'chance', color=MUTED, fontsize=8)
ax.set_xscale('log'); ax.set_xticks(KS, [str(k) for k in KS]); ax.set_ylim(0, 1.02)
ax.set_xlabel('k wrong probe answers'); ax.set_ylabel('Top-1 identification accuracy')
ax.set_title(f'Shuffling the options weakens identification ({len(both)} models)', loc='left', fontsize=11)
ax.legend(frameon=False, loc='lower right')
fig.savefig(f'{FIG}/fig6_shuffle_identification.png'); plt.close(fig)

# ---------------------------------------------------------------- markdown tables
def md(df): return df.to_markdown(index=False, floatfmt='.3f')
with open(f'{TAB}/summary_tables.md', 'w') as f:
    f.write('# Summary tables (generated by analysis/make_figures.py)\n\n## Model profile\n\n')
    cols = ['model', 'family', 'reasoning', 'rows_3k', 'acc_mode_3k', 'unanimous_3k', 'unparsed_3k', 'acc_mode_holdout',
            'acc_mode_shuffled', 'same_ending_share_shuffled', 'same_slot_share_shuffled']
    f.write(md(P[cols]) + '\n\n## Identification (top-1 accuracy; reference runs 0-4, probe runs 5-9)\n\n')
    f.write(md(ID[['set', 'k', 'all_answers', 'wrong_answers_only', 'chance']]) + '\n')
print(P[['model', 'rows_3k', 'acc_mode_3k', 'acc_mode_holdout', 'acc_mode_shuffled']].round(3).to_string(index=False))
print(ID.round(3).to_string(index=False))
