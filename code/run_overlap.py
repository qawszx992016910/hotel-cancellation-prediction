"""補充分析（須在 run_experiments.py 之後執行）
1. 隨機切分：依測試列在訓練集中是否有完全相同的列分成兩群，比較 LightGBM（未加權）F1（5 種子）
2. 時間切分：訓練期與 2017 測試期的組成差異
輸出：results/e3_overlap.csv、results/e4_composition.csv
"""
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
import run_experiments as R

df, _ = R.load_clean()
cols = [c for c in df.columns if c not in R.DUP_COLS_EXCLUDE]
key = pd.util.hash_pandas_object(df[cols], index=False)
rows = []
for s in R.SEEDS:
    tr, va, te = R.random_split(df, s)
    p = np.load(R.OUT / "preds" / f"rand_LightGBM_bal0_s{s}_test.npy")
    inter = key.loc[te.index].isin(set(key.loc[tr.index])).values
    y = te[R.TARGET].values
    for name, m in [("訓練集有完全相同列", inter), ("訓練集無相同列", ~inter)]:
        rows.append(dict(seed=s, group=name, n=int(m.sum()), cancel_rate=y[m].mean(),
                         f1=f1_score(y[m], (p[m] >= 0.5).astype(int))))
o = pd.DataFrame(rows)
o.to_csv(R.OUT / "e3_overlap_per_seed.csv", index=False)
o.groupby("group")[["n", "cancel_rate", "f1"]].agg(["mean", "std"]).round(4).to_csv(R.OUT / "e3_overlap.csv")
print(o.groupby("group")[["n", "cancel_rate", "f1"]].agg(["mean", "std"]).round(4))

tr = df[df.arrival_date < "2016-10-01"]; te = df[df.arrival_date >= "2017-01-01"]
comp = []
for name, d in [("訓練期 2015-07～2016-09", tr), ("測試期 2017-01～2017-08", te)]:
    nd = d[d.deposit_type == "No Deposit"]
    comp.append(dict(period=name, n=len(d), cancel_rate=d[R.TARGET].mean(),
                     non_refund_share=(d.deposit_type == "Non Refund").mean(),
                     online_ta_share=(d.market_segment == "Online TA").mean(),
                     no_deposit_cancel_rate=nd[R.TARGET].mean()))
c = pd.DataFrame(comp).round(4); c.to_csv(R.OUT / "e4_composition.csv", index=False); print(c.to_string())
