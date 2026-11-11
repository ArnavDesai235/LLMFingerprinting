import glob, re
import pandas as pd
import config

ds = pd.read_csv(config.DATASET_PATH)[["index"]]
groups = {}
for f in glob.glob("fingerprints/*_shard*of*.csv"):
    groups.setdefault(re.sub(r"_shard\d+of\d+\.csv$", "", f), []).append(f)

for base, files in sorted(groups.items()):
    df = pd.concat([pd.read_csv(f) for f in files])
    df = ds.merge(df, on="index", how="left")          # original row order
    col = base.split("/")[-1]
    done = df[f"{col}_mode"].notna().sum()
    acc = pd.to_numeric(df[f"{col}_mode_match"], errors="coerce").mean()
    df.to_csv(f"{base}.csv", index=False)
    print(f"{col}: {len(files)} shards, {done}/{len(df)} rows done, acc={acc:.4f} -> {base}.csv")
