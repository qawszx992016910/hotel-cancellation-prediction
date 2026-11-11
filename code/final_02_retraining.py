"""期末 步驟 2：重新訓練策略比較（報告 陸之二）
訓練視窗：expanding（切點前全部資料）、最近 3 個月、最近 6 個月
（12 個月視窗只在 2016-10 切點與 expanding 不同，不納入）
- 驗證：沿用步驟 1 的 5 個切點，每個策略各自以合併驗證預測選門檻
- 2017 測試：static（只在 2017-01 訓練一次）vs 每季重新訓練（2017-01／04／07），各策略各跑一次
- 模型設定沿用步驟 1 最佳設定（LR C=0.1；LightGBM 15 葉 100 棵）
執行：python code/final_02_retraining.py
輸出：results/final/retrain_*.csv、retrain_summary.json、fig_retrain.png
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import TARGET, DUP_COLS_EXCLUDE, features, load_clean, metrics  # noqa: E402
from final_01_rolling_origin import CUTOFFS, OUT, SEEDS, best_thr, fit_predict, group_counts, prep  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sel = json.loads((OUT / "model_selection.json").read_text(encoding="utf8"))["best_config"]
CFG = {"LogisticRegression": ("LogisticRegression", sel["LogisticRegression"]["param"]),
       "LightGBM": ("LightGBM", int(sel["LightGBM"]["param"]))}
KEY = {"LogisticRegression": CFG["LogisticRegression"],
       "LightGBM": ("LightGBM", int(sel["LightGBM"]["param"]), sel["LightGBM"]["trees"])}
WINDOWS = {"expanding": None, "sliding_6m": 6, "sliding_3m": 3}
TEST_CUTS = ["2017-01-01", "2017-04-01", "2017-07-01"]
TEST_END = "2017-09-01"
N_BOOT = 2000


def train_slice(df, cut, months):
    c0 = pd.Timestamp(cut)
    m = df.arrival_date < c0
    if months is not None:
        m &= df.arrival_date >= c0 - pd.DateOffset(months=months)
    return df[m]


def predict_block(df, cut, end, months, model, seed, cat, num):
    tr = train_slice(df, cut, months)
    blk = df[(df.arrival_date >= pd.Timestamp(cut)) & (df.arrival_date < pd.Timestamp(end))]
    A_tr, (A_b,) = prep(tr, [blk], cat, num, model == "LogisticRegression")
    (p,) = fit_predict(CFG[model], A_tr, tr[TARGET].values, [A_b], seed)[KEY[model]]
    return blk, p, len(tr)


def main():
    t0 = time.perf_counter()
    df, _ = load_clean()
    df["dup_group"] = df.groupby([c for c in df.columns if c not in DUP_COLS_EXCLUDE + ["arrival_date"]],
                                 dropna=False, sort=False).ngroup()
    cat, num = features()

    # ---------- 1. 驗證期（rolling-origin）----------
    val_pred, val_frame, sizes = {}, None, []
    for w, months in WINDOWS.items():
        for model in CFG:
            for s in (SEEDS if model == "LightGBM" else SEEDS[:1]):
                parts, ps = [], []
                for c in CUTOFFS:
                    end = pd.Timestamp(c) + pd.DateOffset(months=3)
                    blk, p, ntr = predict_block(df, c, end, months, model, s, cat, num)
                    parts.append(blk.assign(fold=c)); ps.append(p)
                    if model == "LogisticRegression":
                        sizes.append(dict(window=w, cutoff=c, train_rows=ntr))
                val_pred[(w, model, s)] = np.concatenate(ps)
                if val_frame is None:
                    val_frame = pd.concat(parts).reset_index(drop=True)
                print(f"val {w} {model} s{s} {time.perf_counter() - t0:.0f}s", flush=True)
    pd.DataFrame(sizes).pivot(index="cutoff", columns="window", values="train_rows").to_csv(OUT / "retrain_train_sizes.csv")
    y = val_frame[TARGET].values
    rows, thr = [], {}
    for (w, model, s), p in val_pred.items():
        t = best_thr(y, p); thr[(w, model, s)] = t
        r = dict(window=w, model=model, seed=s, thr=t, f1_pooled=f1_score(y, (p >= t).astype(int)))
        for c in CUTOFFS:
            m = (val_frame.fold == c).values
            r[f"f1_{c[:7]}"] = f1_score(y[m], (p[m] >= t).astype(int))
            r[f"auc_{c[:7]}"] = roc_auc_score(y[m], p[m])
        r["f1_fold_mean"] = np.mean([r[f"f1_{c[:7]}"] for c in CUTOFFS])
        r["auc_fold_mean"] = np.mean([r[f"auc_{c[:7]}"] for c in CUTOFFS])
        rows.append(r)
    vdf = pd.DataFrame(rows)
    vdf.to_csv(OUT / "retrain_val_per_seed.csv", index=False)
    vagg = vdf.groupby(["model", "window"], sort=False).mean(numeric_only=True).drop(columns="seed").round(4)
    vagg.to_csv(OUT / "retrain_val_summary.csv")

    # 群組 bootstrap：各視窗 − expanding 的驗證 F1 差
    rng = np.random.default_rng(2026)
    g = pd.factorize(val_frame["dup_group"])[0]; n_g = g.max() + 1
    W = [np.bincount(rng.integers(0, n_g, n_g), minlength=n_g) for _ in range(N_BOOT)]
    boot = {}
    for model in CFG:
        seeds = SEEDS if model == "LightGBM" else SEEDS[:1]
        cnt = {(w, s): group_counts(y, (val_pred[(w, model, s)] >= thr[(w, model, s)]).astype(int), g)
               for w in WINDOWS for s in seeds}
        f1 = lambda wt, k: (lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn))(*(wt @ x for x in cnt[k]))
        for w in WINDOWS:
            if w == "expanding":
                continue
            d = np.array([np.mean([f1(wt, (w, s)) - f1(wt, ("expanding", s)) for s in seeds]) for wt in W])
            boot[f"{model}:{w}-expanding"] = dict(
                point=float(vagg.loc[(model, w), "f1_pooled"] - vagg.loc[(model, "expanding"), "f1_pooled"]),
                ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])])

    # ---------- 2. 2017 測試：static vs 每季重新訓練 ----------
    te_rows, monthly = [], []
    for model in CFG:
        for s in (SEEDS if model == "LightGBM" else SEEDS[:1]):
            plans = {"static（2017-01 訓練一次）": [("2017-01-01", TEST_END, None)]}
            for w, months in WINDOWS.items():
                cuts = TEST_CUTS + [TEST_END]
                plans[f"每季重新訓練｜{w}"] = [(cuts[i], cuts[i + 1], months) for i in range(len(TEST_CUTS))]
            for plan, blocks in plans.items():
                w_for_thr = "expanding" if plan.startswith("static") else plan.split("｜")[1]
                t = thr[(w_for_thr, model, s)]
                parts, ps = [], []
                for c, e, months in blocks:
                    blk, p, _ = predict_block(df, c, e, months, model, s, cat, num)
                    parts.append(blk); ps.append(p)
                te = pd.concat(parts); p = np.concatenate(ps); yt = te[TARGET].values
                te_rows.append(dict(plan=plan, model=model, seed=s, thr=t, **metrics(yt, p, t)))
                mon = te.arrival_date.dt.to_period("M").astype(str).values
                for mo in sorted(set(mon)):
                    m = mon == mo
                    monthly.append(dict(plan=plan, model=model, seed=s, month=mo,
                                        f1=f1_score(yt[m], (p[m] >= t).astype(int)), auc=roc_auc_score(yt[m], p[m])))
                print(f"test {plan} {model} s{s} F1={te_rows[-1]['f1']:.3f}", flush=True)
    tdf = pd.DataFrame(te_rows); tdf.to_csv(OUT / "retrain_test_per_seed.csv", index=False)
    tagg = tdf.groupby(["model", "plan"], sort=False).mean(numeric_only=True).drop(columns="seed").round(4)
    tagg.to_csv(OUT / "retrain_test_summary.csv")
    mdf = pd.DataFrame(monthly).groupby(["model", "plan", "month"], sort=False)[["f1", "auc"]].mean().reset_index()
    mdf.to_csv(OUT / "retrain_test_monthly.csv", index=False)

    # 圖：2017 各月 F1
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, CFG):
        for plan, g_ in mdf[mdf.model == model].groupby("plan", sort=False):
            ax.plot(g_.month.str[5:], g_.f1, marker="o", label=plan)
        ax.set_title(f"{model}：2017 年各月 F1"); ax.set_xlabel("抵達月份（2017）"); ax.grid(alpha=.3)
    axes[0].set_ylabel("F1-score"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_retrain.png", dpi=200); plt.close(fig)

    summary = dict(configs={k: list(v) for k, v in KEY.items()},
                   val=vagg[["f1_pooled", "f1_fold_mean", "auc_fold_mean", "thr"]].reset_index().to_dict("records"),
                   val_diff_vs_expanding=boot,
                   test_2017=tagg[["f1", "auc_roc", "precision", "recall", "thr"]].reset_index().to_dict("records"),
                   runtime_min=round((time.perf_counter() - t0) / 60, 1))
    (OUT / "retrain_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
