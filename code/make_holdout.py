"""Held-out HellaSwag set: 500 clean validation rows NOT in the 3k sample, stratified like the 3k (2031/969 WikiHow/ActivityNet), seed 1."""
import pandas as pd
full = pd.read_csv('data_clean/hellaswag_val_clean_reordered.csv')
k3 = pd.read_csv('data_clean/hellaswag_val_clean_3k.csv')
pool = full[~full['index'].isin(k3['index'])].copy()
pool['src'] = pool.source_id.str.split('~').str[0]
n_wiki = round(500 * 2031 / 3000)                      # 338
parts = [pool[pool.src == 'wikihow'].sample(n_wiki, random_state=1),
         pool[pool.src == 'activitynet'].sample(500 - n_wiki, random_state=1)]
out = pd.concat(parts).sample(frac=1, random_state=1).drop(columns='src')
assert not out['index'].isin(k3['index']).any() and out['index'].is_unique and len(out) == 500
out.to_csv('data_clean/hellaswag_val_clean_holdout500.csv', index=False)
print('pool', len(pool), '| holdout', len(out), out.source_id.str.split('~').str[0].value_counts().to_dict(),
      '| label dist', out.label.value_counts().sort_index().to_dict(), '| all intact', out.stripped_endings_intact.all())
