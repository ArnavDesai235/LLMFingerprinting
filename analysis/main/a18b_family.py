"""Per-family breakdown, fixed vs shuffled. Reuses a18's loading/pairs via exec. Usage same as a18."""
import sys
src = open(sys.argv[0].replace('a18b_family.py', 'a18_shuffle.py')).read()
exec(src.split("print('\\n== 1.")[0])
exec('def counts' + src.split('def counts')[1].split('rows = []')[0])
exec('def pairs' + src.split('def pairs')[1].split('PU, PS =')[0])
PU, PS = pairs(AU), pairs(AS)
out = []
for tag, P, col in [('fixed_excess', PU, 'excess'), ('fixed_raw', PU, 'share'), ('shuffled_raw', PS, 'share')]:
    Q = P[P[col].notna()].copy(); Q['same'] = [fam(a) == fam(b) for a, b in zip(Q.a, Q.b)]
    X = np.column_stack([np.ones(len(Q)), Q.accmin, Q.accdiff, Q.accmin**2]); cr = ~Q.same.values
    beta = la.lstsq(X[cr], Q[col].values[cr], rcond=None)[0]; Q['res'] = Q[col].values - X @ beta
    for f, g in Q[Q.same].groupby(Q[Q.same].a.map(fam)):
        out.append(dict(measure=tag, family=f, pairs=len(g), mean_resid=g.res.mean(), positive=f'{(g.res>0).sum()}/{len(g)}'))
    sd = Q.res[cr].std(); out.append(dict(measure=tag, family='ALL within', pairs=Q.same.sum(), mean_resid=Q.res[Q.same].mean(), positive=f'{(Q.res[Q.same]>0).sum()}/{Q.same.sum()}', ))
    print(f'{tag}: SD of cross-family residuals {sd:.3f}')
D = pd.DataFrame(out); print(D.pivot(index='family', columns='measure', values='mean_resid').round(3).to_string())
print(D.pivot(index='family', columns='measure', values='positive').to_string())
# top within-family pairs shuffled
Q = PS.copy(); Q['same'] = [fam(a) == fam(b) for a, b in zip(Q.a, Q.b)]
print(Q[Q.same].sort_values('share', ascending=False)[['a','b','share','nboth']].round(3).to_string())
