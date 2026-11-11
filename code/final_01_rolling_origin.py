"""期末 步驟 1：rolling-origin 時間驗證＋事先定義的模型選擇規則（報告 陸之二）
- 切點：2015-10-01、2016-01-01、2016-04-01、2016-07-01、2016-10-01
  每折以切點前全部資料訓練，切點後 3 個月驗證；2017 測試集只在最後評估一次
- 超參數、樹的數量、分類門檻只用驗證期選
- 選模規則：各驗證期合併預測，3 個種子，以完全重複列群組為單位 bootstrap 2,000 次；
  LightGBM − LogisticRegression 的 F1 差 95% CI 下界 > 0.01 才採用 LightGBM
執行：python code/final_01_rolling_origin.py （專案根目錄）
輸出：results/final/*
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import (TARGET, AdrCapper, DUP_COLS_EXCLUDE, features, load_clean,  # noqa: E402
                             make_pre, metrics)
from sklearn.linear_model import LogisticRegression  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
OUT.mkdir(parents=True, exist_ok=True)

CUTOFFS = ["2015-10-01", "2016-01-01", "2016-04-01", "2016-07-01", "2016-10-01"]
TEST_START = "2017-01-01"
SEEDS = [42, 43, 44]
LR_C = [0.1, 1.0, 10.0]
LGB_LEAVES = [15, 31, 63]
LGB_TREES = [50, 100, 200, 400, 800, 1600]
THRS = np.round(np.arange(0.20, 0.801, 0.02), 2)
N_BOOT = 2000
MIN_DIFF = 0.01


def folds(df):
    for c in CUTOFFS:
        c0 = pd.Timestamp(c)
        c1 = c0 + pd.DateOffset(months=3)
        tr = df[df.arrival_date < c0]
        va = df[(df.arrival_date >= c0) & (df.arrival_date < c1)]
        yield c, tr, va


def prep(tr, others, cat, num, scale):
    cap = AdrCapper().fit(tr)
    pre = make_pre(cat, num, scale)
    A_tr = pre.fit_transform(cap.transform(tr))
    return A_tr, [pre.transform(cap.transform(o)) for o in others]


def fit_predict(cfg, A_tr, y_tr, A_list, seed):
    """回傳 {config_key: [每個 A 的機率]}；LightGBM 一次訓練、以不同樹數預測"""
    out = {}
    if cfg[0] == "LogisticRegression":
        m = LogisticRegression(C=cfg[1], max_iter=3000).fit(A_tr, y_tr)
        out[cfg] = [m.predict_proba(A)[:, 1] for A in A_list]
    else:
        m = lgb.LGBMClassifier(n_estimators=max(LGB_TREES), learning_rate=0.05, num_leaves=cfg[1],
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               random_state=seed, verbose=-1).fit(A_tr, y_tr)
        for t in LGB_TREES:
            out[(cfg[0], cfg[1], t)] = [m.predict_proba(A, num_iteration=t)[:, 1] for A in A_list]
    return out


def best_thr(y, p):
    f = [f1_score(y, (p >= t).astype(int)) for t in THRS]
    return float(THRS[int(np.argmax(f))])


def group_counts(y, yhat, g):
    """每個重複列群組的 TP/FP/FN 數，供 bootstrap"""
    k = g.max() + 1
    tp = np.bincount(g, weights=((yhat == 1) & (y == 1)).astype(float), minlength=k)
    fp = np.bincount(g, weights=((yhat == 1) & (y == 0)).astype(float), minlength=k)
    fn = np.bincount(g, weights=((yhat == 0) & (y == 1)).astype(float), minlength=k)
    return tp, fp, fn


def main():
    t_start = time.perf_counter()
    df, _ = load_clean()
    df["dup_group"] = df.groupby([c for c in df.columns if c not in DUP_COLS_EXCLUDE + ["arrival_date"]],
                                 dropna=False, sort=False).ngroup()
    cat, num = features()
    cfgs = [("LogisticRegression", c) for c in LR_C] + [("LightGBM", l) for l in LGB_LEAVES]

    # ---------- 1. 各折驗證預測 ----------
    fold_info, val_rows = [], []
    preds = {}          # (seed, key) -> 合併驗證機率（依 val_index 順序）
    val_parts = []
    for c, tr, va in folds(df):
        fold_info.append(dict(cutoff=c, train=len(tr), val=len(va), train_cancel=tr[TARGET].mean(),
                              val_cancel=va[TARGET].mean()))
        val_parts.append(va.assign(fold=c))
        A = {}
        for scale in (True, False):
            A[scale] = prep(tr, [va], cat, num, scale)
        for seed in SEEDS:
            for cfg in cfgs:
                if cfg[0] == "LogisticRegression" and seed != SEEDS[0]:
                    continue    # LR 無隨機性，結果與種子無關
                A_tr, (A_va,) = A[cfg[0] == "LogisticRegression"]
                t0 = time.perf_counter()
                res = fit_predict(cfg, A_tr, tr[TARGET].values, [A_va], seed)
                for key, (p,) in res.items():
                    preds.setdefault((seed, key), []).append(p)
                print(f"{c} seed{seed} {cfg} {time.perf_counter() - t0:.1f}s", flush=True)
    for (s, key), lst in list(preds.items()):
        preds[(s, key)] = np.concatenate(lst)
    for key in [k for (s, k) in preds if s == SEEDS[0] and k[0] == "LogisticRegression"]:
        for s in SEEDS[1:]:
            preds[(s, key)] = preds[(SEEDS[0], key)]
    val = pd.concat(val_parts).reset_index(drop=True)
    y = val[TARGET].values
    pd.DataFrame(fold_info).to_csv(OUT / "rolling_folds.csv", index=False)

    # ---------- 2. 以驗證期選設定與門檻 ----------
    keys = sorted({k for (_, k) in preds}, key=str)
    for key in keys:
        for s in SEEDS:
            p = preds[(s, key)]
            thr = best_thr(y, p)
            per_fold = [(f1_score(y[m], (p[m] >= thr).astype(int)), roc_auc_score(y[m], p[m]))
                        for m in (val.fold == c for c in CUTOFFS)]
            val_rows.append(dict(model=key[0], param=key[1], trees=key[2] if len(key) > 2 else None, seed=s,
                                 thr=thr, f1_pooled=f1_score(y, (p >= thr).astype(int)),
                                 f1_fold_mean=np.mean([a for a, _ in per_fold]),
                                 auc_fold_mean=np.mean([b for _, b in per_fold]),
                                 **{f"f1_{c[:7]}": a for c, (a, _) in zip(CUTOFFS, per_fold)}))
    vs = pd.DataFrame(val_rows)
    vs.to_csv(OUT / "rolling_config_scores.csv", index=False)
    agg = (vs.groupby(["model", "param", "trees"], dropna=False)[["f1_fold_mean", "auc_fold_mean", "thr"]]
           .mean().reset_index())
    agg.to_csv(OUT / "rolling_config_summary.csv", index=False)
    best = {m: agg[agg.model == m].sort_values("f1_fold_mean", ascending=False).iloc[0] for m in
            ["LogisticRegression", "LightGBM"]}
    keyof = lambda r: (r.model, r.param) if r.model == "LogisticRegression" else (r.model, int(r.param), int(r.trees))

    # ---------- 3. 事先定義的選模規則：群組 bootstrap ----------
    rng = np.random.default_rng(2026)
    g = pd.factorize(val["dup_group"])[0]
    n_g = g.max() + 1
    cnt = {}
    for m in best:
        for s in SEEDS:
            p = preds[(s, keyof(best[m]))]
            thr = best_thr(y, p)
            cnt[(m, s)] = group_counts(y, (p >= thr).astype(int), g)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, n_g, n_g), minlength=n_g)
        d = []
        for s in SEEDS:
            f = {}
            for m in best:
                tp, fp, fn = (w @ x for x in cnt[(m, s)])
                f[m] = 2 * tp / (2 * tp + fp + fn)
            d.append(f["LightGBM"] - f["LogisticRegression"])
        diffs[b] = np.mean(d)
    point = float(np.mean([vs[(vs.model == "LightGBM") & (vs.param == best["LightGBM"].param) &
                             (vs.trees == best["LightGBM"].trees) & (vs.seed == s)].f1_pooled.iloc[0]
                          - vs[(vs.model == "LogisticRegression") & (vs.param == best["LogisticRegression"].param) &
                               (vs.seed == s)].f1_pooled.iloc[0] for s in SEEDS]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    chosen = "LightGBM" if lo > MIN_DIFF else "LogisticRegression"

    # ---------- 4. 2017 測試集：只評估一次 ----------
    trva = df[df.arrival_date < TEST_START]
    te = df[df.arrival_date >= TEST_START]
    test_rows = []
    for m in best:
        r = best[m]
        A_tr, (A_te,) = prep(trva, [te], cat, num, m == "LogisticRegression")
        for s in (SEEDS if m == "LightGBM" else SEEDS[:1]):
            cfg = (m, r.param) if m == "LogisticRegression" else (m, int(r.param))
            res = fit_predict(cfg, A_tr, trva[TARGET].values, [A_te], s)
            (p,) = res[keyof(r)]
            np.save(OUT / f"test_{m}_s{s}.npy", p)
            test_rows.append(dict(model=m, param=r.param, trees=r.trees, seed=s, thr=r.thr,
                                  **metrics(te[TARGET].values, p, r.thr),
                                  f1_at_0_5=f1_score(te[TARGET].values, (p >= 0.5).astype(int))))
    tdf = pd.DataFrame(test_rows)
    tdf.to_csv(OUT / "test_2017.csv", index=False)

    summary = dict(
        cutoffs=CUTOFFS, seeds=SEEDS, n_boot=N_BOOT, min_diff=MIN_DIFF,
        best_config={m: dict(param=float(best[m].param), trees=None if pd.isna(best[m].trees) else int(best[m].trees),
                             thr_mean=float(best[m].thr), val_f1_fold_mean=float(best[m].f1_fold_mean),
                             val_auc_fold_mean=float(best[m].auc_fold_mean)) for m in best},
        val_f1_diff_lgbm_minus_lr=dict(point=point, ci95=[float(lo), float(hi)]),
        decision=chosen,
        test_2017=tdf.groupby("model")[["f1", "auc_roc", "precision", "recall", "f1_at_0_5"]].mean().round(4).to_dict("index"),
        runtime_min=round((time.perf_counter() - t_start) / 60, 1),
    )
    (OUT / "model_selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
