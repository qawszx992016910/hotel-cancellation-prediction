"""補充分析（須在 run_experiments.py 之後執行；讀取 results/preds/ 的預測機率）
執行：python code/run_extra.py
1. RQ1 成對差異：同一切分下 LightGBM 與 RF、LR 之 F1／AUC 差
2. E2 門檻對照：未加權 LightGBM 在驗證集上找出與加權模型相同 Recall 的門檻，比較測試集結果
3. 驗證集選門檻：各模型以驗證集 F1 最大之門檻套用至測試集（隨機與時間切分）
4. E4 bootstrap：2017 測試集重抽 2,000 次，LightGBM − LR 之 F1、AUC 差的 95% 信賴區間
5. E4 固定輪數：LightGBM 以 E1 平均最佳輪數訓練、不提前停止（訓練期；訓練期＋驗證期）
6. E4 季節對照：隨機切分測試集中只取 1–8 月抵達訂單之 F1
7. 去重拆解：主實驗與去重實驗在 No Deposit 子群的 F1（5 種子）
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import run_experiments as R

OUT = R.OUT
P = OUT / "preds"
SEEDS = R.SEEDS
MODELS = R.MODELS
T = R.TARGET


def prf(y, p, thr):
    yhat = (p >= thr).astype(int)
    return dict(precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat),
                f1=f1_score(y, yhat))


def best_f1_thr(y, p):
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f = [f1_score(y, (p >= g).astype(int)) for g in grid]
    return float(grid[int(np.argmax(f))])


def thr_for_recall(y, p, target):
    """最大門檻使驗證集 Recall ≥ target"""
    grid = np.round(np.arange(0.01, 0.991, 0.005), 3)
    ok = [g for g in grid if recall_score(y, (p >= g).astype(int)) >= target]
    return float(max(ok))


def time_split(df):
    trva = df[df["arrival_date"] < "2017-01-01"]
    te = df[df["arrival_date"] >= "2017-01-01"]
    return trva[trva["arrival_date"] < "2016-10-01"], trva[trva["arrival_date"] >= "2016-10-01"], te, trva


def main():
    df, _ = R.load_clean()
    cat, num = R.features()
    splits = {s: R.random_split(df, s) for s in SEEDS}

    # ---- 1. 成對差異 ----
    e1 = pd.read_csv(OUT / "e1_e2_per_seed.csv")
    e1 = e1[e1.exp == "E1"].pivot(index="seed", columns="model", values=["f1", "auc_roc"])
    pair = []
    for base in ["RandomForest", "LogisticRegression"]:
        for met in ["f1", "auc_roc"]:
            d = e1[(met, "LightGBM")] - e1[(met, base)]
            pair.append(dict(comparison=f"LightGBM − {base}", metric=met, mean=d.mean(), min=d.min(),
                             max=d.max(), n_positive=int((d > 0).sum()), n=len(d)))
    pd.DataFrame(pair).round(4).to_csv(OUT / "rq1_paired_diff.csv", index=False)

    # ---- 2. E2 門檻對照 ＋ 3. 驗證集選門檻（隨機切分）----
    e2, vthr = [], []
    for s in SEEDS:
        tr, va, te = splits[s]
        yv, yt = va[T].values, te[T].values
        for m in MODELS:
            pv0 = np.load(P / f"rand_{m}_bal0_s{s}_val.npy"); pt0 = np.load(P / f"rand_{m}_bal0_s{s}_test.npy")
            pv1 = np.load(P / f"rand_{m}_bal1_s{s}_val.npy"); pt1 = np.load(P / f"rand_{m}_bal1_s{s}_test.npy")
            target = recall_score(yv, (pv1 >= 0.5).astype(int))
            thr = thr_for_recall(yv, pv0, target)
            e2.append(dict(seed=s, model=m, setting="加權，門檻 0.5", thr=0.5, **prf(yt, pt1, 0.5)))
            e2.append(dict(seed=s, model=m, setting="未加權，驗證集對齊 Recall 之門檻", thr=thr, **prf(yt, pt0, thr)))
            t_best = best_f1_thr(yv, pv0)
            vthr.append(dict(split="隨機", seed=s, model=m, thr=t_best, **prf(yt, pt0, t_best)))
    e2 = pd.DataFrame(e2)
    e2.to_csv(OUT / "e2_threshold_equiv_per_seed.csv", index=False)
    e2.groupby(["model", "setting"], sort=False)[["thr", "precision", "recall", "f1"]].agg(["mean", "std"]).round(4) \
        .to_csv(OUT / "e2_threshold_equiv.csv")

    # ---- 時間切分 ----
    tr, va, te, trva = time_split(df)
    yv, yt = va[T].values, te[T].values
    tp = {m: np.load(P / f"time_{m}_test.npy") for m in MODELS}
    for m in MODELS:
        pv = np.load(P / f"time_{m}_val.npy")
        t_best = best_f1_thr(yv, pv)
        vthr.append(dict(split="時間", seed=42, model=m, thr=t_best, **prf(yt, tp[m], t_best)))
    vthr = pd.DataFrame(vthr)
    vthr.to_csv(OUT / "val_threshold_per_seed.csv", index=False)
    vthr.groupby(["split", "model"], sort=False)[["thr", "precision", "recall", "f1"]].agg(["mean", "std"]).round(4) \
        .to_csv(OUT / "val_threshold.csv")
    thr_time = vthr[vthr.split == "時間"].set_index("model").thr

    # ---- 4. bootstrap ----
    # 抽樣單位：單列（列獨立假設），以及「完全重複列群組」（同組列一起抽，較保守）
    B, n = 2000, len(yt)
    dup_cols = [c for c in te.columns if c not in R.DUP_COLS_EXCLUDE]
    gid = te.groupby(dup_cols, dropna=False, sort=False).ngroup().values
    order = np.argsort(gid, kind="stable")
    bounds = np.flatnonzero(np.diff(gid[order])) + 1
    groups = np.split(order, bounds)
    print(f"E4 test rows {n}, duplicate groups {len(groups)}")
    rows = []
    for unit in ["單列", "重複列群組"]:
      rng = np.random.default_rng(42)
      for label, tl, tg in [("門檻 0.5", 0.5, 0.5), ("驗證集選門檻", thr_time["LightGBM"], thr_time["LogisticRegression"])]:
        dF, dA = [], []
        for _ in range(B):
            if unit == "單列":
                i = rng.integers(0, n, n)
            else:
                i = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
            y = yt[i]
            dF.append(f1_score(y, (tp["LightGBM"][i] >= tl).astype(int)) -
                      f1_score(y, (tp["LogisticRegression"][i] >= tg).astype(int)))
            dA.append(roc_auc_score(y, tp["LightGBM"][i]) - roc_auc_score(y, tp["LogisticRegression"][i]))
        for met, arr in [("F1", dF), ("AUC-ROC", dA)]:
            arr = np.array(arr)
            point = (f1_score(yt, (tp["LightGBM"] >= tl).astype(int)) - f1_score(yt, (tp["LogisticRegression"] >= tg).astype(int))
                     if met == "F1" else roc_auc_score(yt, tp["LightGBM"]) - roc_auc_score(yt, tp["LogisticRegression"]))
            rows.append(dict(unit=unit, threshold=label, metric=met, diff_LGBM_minus_LR=point,
                             ci_low=np.percentile(arr, 2.5), ci_high=np.percentile(arr, 97.5), B=B))
    pd.DataFrame(rows).round(4).to_csv(OUT / "e4_bootstrap.csv", index=False)

    # ---- 5. 固定輪數 ----
    e1all = pd.read_csv(OUT / "e1_e2_per_seed.csv")
    n_iter = int(round(e1all[(e1all.exp == "E1") & (e1all.model == "LightGBM")].best_iter.mean()))
    fx = []
    for label, train in [("訓練期 2015-07～2016-09", tr), ("訓練期＋驗證期 2015-07～2016-12", trva)]:
        cap = R.AdrCapper().fit(train)
        pre = R.make_pre(cat, num, scale=False)
        A_tr = pre.fit_transform(cap.transform(train)); A_te = pre.transform(cap.transform(te))
        m = lgb.LGBMClassifier(n_estimators=n_iter, learning_rate=0.05, num_leaves=63, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1).fit(A_tr, train[T])
        p = m.predict_proba(A_te)[:, 1]
        fx.append(dict(model=f"LightGBM 固定 {n_iter} 輪", train_data=label, n_train=len(train), **R.metrics(yt, p)))
    pd.DataFrame(fx).round(4).to_csv(OUT / "e4_extra.csv", index=False)

    # ---- 6. 季節對照 ----
    ss = []
    for s in SEEDS:
        _, _, tes = splits[s]
        mask = (tes["arrival_month"] <= 8).values
        for m in MODELS:
            pt = np.load(P / f"rand_{m}_bal0_s{s}_test.npy")
            ss.append(dict(seed=s, model=m, subset="隨機切分測試集中 1–8 月抵達", n=int(mask.sum()),
                           **prf(tes[T].values[mask], pt[mask], 0.5)))
    ss = pd.DataFrame(ss)
    ss.groupby(["model"])[["n", "precision", "recall", "f1"]].mean().round(4).to_csv(OUT / "e4_season_control.csv")

    # ---- 7. 去重拆解（5 種子，LightGBM 主實驗特徵）----
    dd = R.dedup(df)
    out = []
    for label, d in [("主實驗（保留重複列）", df), ("刪除完全重複列", dd)]:
        for s in SEEDS:
            tr_, va_, te_ = R.random_split(d, s)
            if label.startswith("主實驗"):
                p = np.load(P / f"rand_LightGBM_bal0_s{s}_test.npy")
            else:
                _, p, *_ = R.run_one("LightGBM", tr_, tr_[T], va_, va_[T], te_, te_[T], cat, num, s)
            t = te_.assign(pred=(p >= 0.5).astype(int))
            for dep, g in [("全部", t)] + list(t.groupby("deposit_type")):
                out.append(dict(variant=label, seed=s, deposit_type=dep, n=len(g), cancel_rate=g[T].mean(),
                                f1=f1_score(g[T], g.pred, zero_division=0)))
    res = pd.DataFrame(out)
    res.to_csv(OUT / "e3_dedup_by_deposit_per_seed.csv", index=False)
    res.groupby(["variant", "deposit_type"], sort=False)[["n", "cancel_rate", "f1"]].agg(["mean", "std"]).round(4) \
        .to_csv(OUT / "e3_dedup_by_deposit.csv")
    print("done")


if __name__ == "__main__":
    main()
