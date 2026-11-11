"""期末 步驟 4：補充指標（報告 陸之二）
正式方案：LightGBM（步驟 1 規則選定）、base 特徵（步驟 3 決策）、每季重新訓練＋expanding（步驟 2 決策 A）
LR 並列對照。驗證期（5 切點）與 2017 測試期都報：
  - 整體 F1／Precision／Recall（門檻＝驗證期選定）
  - 排除不退款訂單後的 Recall／Precision／F1
  - 每個抵達日風險最高前 10% 訂單的 Precision（逐日平均與合併）與其 Recall、該日基準取消率
執行：python code/final_04_metrics.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import TARGET, features, load_clean  # noqa: E402
from final_01_rolling_origin import CUTOFFS, OUT, SEEDS, best_thr  # noqa: E402
from final_02_retraining import CFG, TEST_CUTS, TEST_END  # noqa: E402
from final_03_features import add_calendar, run_block  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TOP = 0.10


def top_k_by_day(frame, p):
    """每個抵達日取風險最高的 ceil(10%) 筆"""
    d = frame.assign(p=p).reset_index(drop=True)
    d["rank"] = d.groupby("arrival_date")["p"].rank(ascending=False, method="first")
    d["n_day"] = d.groupby("arrival_date")["p"].transform("size")
    d["sel"] = d["rank"] <= np.ceil(d["n_day"] * TOP)
    per_day = d[d.sel].groupby("arrival_date")[TARGET].mean()
    return dict(top10_precision_daily_mean=float(per_day.mean()),
                top10_precision_pooled=float(d.loc[d.sel, TARGET].mean()),
                top10_recall=float(d.loc[d.sel, TARGET].sum() / d[TARGET].sum()),
                base_cancel_rate=float(d[TARGET].mean()),
                top10_n=int(d.sel.sum()))


def evaluate(frame, p, t):
    y = frame[TARGET].values; yhat = (p >= t).astype(int)
    nr = (frame.deposit_type != "Non Refund").values
    out = dict(thr=t, f1=f1_score(y, yhat), precision=precision_score(y, yhat), recall=recall_score(y, yhat),
               excl_nonrefund_recall=recall_score(y[nr], yhat[nr]),
               excl_nonrefund_precision=precision_score(y[nr], yhat[nr]),
               excl_nonrefund_f1=f1_score(y[nr], yhat[nr]),
               excl_nonrefund_n=int(nr.sum()))
    out.update(top_k_by_day(frame, p))
    nrf = frame[nr]
    out.update({f"excl_nonrefund_{k}": v for k, v in top_k_by_day(nrf, p[nr]).items()
                if k.startswith("top10_precision")})
    return out


def main():
    t0 = time.perf_counter()
    df, _ = load_clean()
    df = add_calendar(df)
    cat, num = features()
    rows, curves = [], []
    for model in CFG:
        for s in (SEEDS if model == "LightGBM" else SEEDS[:1]):
            vparts, vps = [], []
            for c in CUTOFFS:
                blk, p = run_block(df, c, pd.Timestamp(c) + pd.DateOffset(months=3), model, s, [], cat, num)
                vparts.append(blk); vps.append(p)
            vf = pd.concat(vparts).reset_index(drop=True); vp = np.concatenate(vps)
            t = best_thr(vf[TARGET].values, vp)
            rows.append(dict(period="驗證期（5 切點合併）", model=model, seed=s, **evaluate(vf, vp, t)))
            cuts = TEST_CUTS + [TEST_END]
            tparts, tps = [], []
            for i in range(len(TEST_CUTS)):
                blk, p = run_block(df, cuts[i], cuts[i + 1], model, s, [], cat, num)
                tparts.append(blk); tps.append(p)
            tf = pd.concat(tparts).reset_index(drop=True); tp = np.concatenate(tps)
            rows.append(dict(period="2017 測試期", model=model, seed=s, **evaluate(tf, tp, t)))
            np.save(OUT / f"final_test_{model}_s{s}.npy", tp)
            if s == SEEDS[0]:
                nr = (tf.deposit_type != "Non Refund").values
                for thr in np.round(np.arange(0.1, 0.91, 0.05), 2):
                    yh = (tp >= thr).astype(int)
                    curves.append(dict(model=model, thr=thr, recall=recall_score(tf[TARGET], yh),
                                       precision=precision_score(tf[TARGET], yh, zero_division=0),
                                       recall_excl_nonrefund=recall_score(tf[TARGET].values[nr], yh[nr])))
            print(f"{model} s{s} done {time.perf_counter() - t0:.0f}s", flush=True)
    rdf = pd.DataFrame(rows); rdf.to_csv(OUT / "metrics_per_seed.csv", index=False)
    agg = rdf.groupby(["period", "model"], sort=False).mean(numeric_only=True).drop(columns="seed").round(4)
    agg.to_csv(OUT / "metrics_summary.csv")
    cdf = pd.DataFrame(curves); cdf.to_csv(OUT / "threshold_recall_curve_2017.csv", index=False)

    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for model, g in cdf.groupby("model", sort=False):
        ax.plot(g.thr, g.recall, marker="o", ms=3, label=f"{model} Recall（全部）")
        ax.plot(g.thr, g.recall_excl_nonrefund, "--", marker="s", ms=3, label=f"{model} Recall（排除不退款）")
        ax.plot(g.thr, g.precision, ":", label=f"{model} Precision")
    ax.axvline(agg.loc[("2017 測試期", "LightGBM"), "thr"], color="gray", lw=1)
    ax.set_xlabel("分類門檻"); ax.set_ylabel("指標值"); ax.set_title("2017 測試期：門檻—Recall／Precision 曲線")
    ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_threshold_recall.png", dpi=200); plt.close(fig)

    (OUT / "metrics_summary.json").write_text(json.dumps(
        dict(final_plan="LightGBM（15 葉、100 棵）＋base 特徵＋每季重新訓練（expanding）",
             summary=agg.reset_index().to_dict("records"), runtime_min=round((time.perf_counter() - t0) / 60, 1)),
        ensure_ascii=False, indent=2), encoding="utf8")
    print(agg.T.to_string())


if __name__ == "__main__":
    main()
