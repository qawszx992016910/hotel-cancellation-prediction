"""期末 階段 7：錯誤分析與可解釋性（新範本 肆之三）＋推論時間（伍之四）
正式方案：LightGBM（15 葉 100 棵）、base 特徵、每季重新訓練＋expanding、門檻 0.36（驗證期選定）
- 各類別效能（未取消／取消／macro 平均），3 個種子平均 ± 標準差（讀 final_04 存的 final_test_LightGBM_s*.npy）
- 混淆矩陣圖（seed 42）
- SHAP：LightGBM 內建 TreeSHAP（pred_contrib=True），one-hot 欄位加總回原始欄位；
  依 TP／FN／TN／FP 分組比較平均 SHAP 值（seed 42，重跑 3 個季度模型並核對機率與 final_04 一致）
- 推論時間：前處理＋模型，單筆逐筆 1,000 次與批次 40,000 筆
執行：python code/final_07_explain.py
輸出：results/final/explain_*.csv、fig_confusion_final.png、fig_shap_*.png、inference_time.csv
"""
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import TARGET, AdrCapper, features, load_clean, make_pre  # noqa: E402
from final_01_rolling_origin import OUT, SEEDS  # noqa: E402
from final_02_retraining import CFG, KEY, TEST_CUTS, TEST_END  # noqa: E402

import lightgbm as lgb  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
THR = 0.36
SEED = 42
N_LEAVES, N_TREES = KEY["LightGBM"][1], KEY["LightGBM"][2]


def test_frame(df):
    cuts = TEST_CUTS + [TEST_END]
    return [(cuts[i], cuts[i + 1]) for i in range(len(TEST_CUTS))]


def fit_block(df, cut, end, cat, num, seed):
    tr = df[df.arrival_date < pd.Timestamp(cut)]
    blk = df[(df.arrival_date >= pd.Timestamp(cut)) & (df.arrival_date < pd.Timestamp(end))]
    cap = AdrCapper().fit(tr)
    pre = make_pre(cat, num, False)
    A_tr = pre.fit_transform(cap.transform(tr))
    m = lgb.LGBMClassifier(n_estimators=1600, learning_rate=0.05, num_leaves=N_LEAVES,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=seed, verbose=-1).fit(A_tr, tr[TARGET].values)
    return tr, blk, cap, pre, m


def orig_col(name, cat, num):
    """ColumnTransformer 輸出名 → 原始欄位"""
    s = name.split("__", 1)[1]
    if name.startswith("num__"):
        return s
    for c in sorted(cat, key=len, reverse=True):
        if s.startswith(c + "_") or s == c:
            return c
    return s


def main():
    t0 = time.perf_counter()
    df, _ = load_clean()
    cat, num = features()
    blocks = test_frame(df)
    tf = pd.concat([df[(df.arrival_date >= pd.Timestamp(a)) & (df.arrival_date < pd.Timestamp(b))]
                    for a, b in blocks]).reset_index(drop=True)
    y = tf[TARGET].values

    # 1. 各類別效能（3 種子）
    rows = []
    for s in SEEDS:
        p = np.load(OUT / f"final_test_LightGBM_s{s}.npy")
        yh = (p >= THR).astype(int)
        pr, rc, f1, sup = precision_recall_fscore_support(y, yh, labels=[0, 1])
        for k, lab in enumerate(["未取消（0）", "取消（1）"]):
            rows.append(dict(seed=s, cls=lab, precision=pr[k], recall=rc[k], f1=f1[k], support=int(sup[k])))
        rows.append(dict(seed=s, cls="macro 平均", precision=pr.mean(), recall=rc.mean(), f1=f1.mean(),
                         support=int(sup.sum())))
    pc = pd.DataFrame(rows)
    pc.to_csv(OUT / "explain_per_class_per_seed.csv", index=False, encoding="utf-8-sig")
    summ = pc.groupby("cls", sort=False).agg(precision_mean=("precision", "mean"), precision_sd=("precision", "std"),
                                             recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
                                             f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
                                             support=("support", "first")).reset_index()
    summ.to_csv(OUT / "explain_per_class.csv", index=False, encoding="utf-8-sig")
    print(summ.round(4).to_string())

    # 2. 混淆矩陣（seed 42）
    p42 = np.load(OUT / f"final_test_LightGBM_s{SEED}.npy")
    yh42 = (p42 >= THR).astype(int)
    cm = confusion_matrix(y, yh42, labels=[0, 1])
    pd.DataFrame(cm, index=["真實 未取消", "真實 取消"], columns=["預測 未取消", "預測 取消"]).to_csv(
        OUT / "explain_confusion_s42.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(4.6, 3.9), dpi=200)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "#1a1a1a")
    ax.set_xticks([0, 1], ["未取消", "取消"]); ax.set_yticks([0, 1], ["未取消", "取消"])
    ax.set_xlabel("預測類別"); ax.set_ylabel("真實類別")
    fig.tight_layout(); fig.savefig(OUT / "fig_confusion_final.png"); plt.close(fig)

    # 3. SHAP（seed 42，重跑 3 個季度模型）
    contribs, probs, names_all = [], [], None
    for a, b in blocks:
        tr, blk, cap, pre, m = fit_block(df, a, b, cat, num, SEED)
        A_b = pre.transform(cap.transform(blk))
        names = list(pre.get_feature_names_out())
        c = m.booster_.predict(A_b, num_iteration=N_TREES, pred_contrib=True)
        probs.append(m.predict_proba(A_b, num_iteration=N_TREES)[:, 1])
        cdf = pd.DataFrame(c[:, :-1], columns=names)
        agg = cdf.T.groupby([orig_col(n, cat, num) for n in names]).sum().T
        agg["_base"] = c[:, -1]
        contribs.append(agg)
        last = (tr, cap, pre, m)
    sh = pd.concat(contribs).fillna(0).reset_index(drop=True)
    pchk = np.concatenate(probs)
    diff = float(np.abs(pchk - p42).max())
    print(f"重跑機率與 final_04 最大差 {diff:.2e}")
    assert diff < 1e-9, diff

    base = sh.pop("_base")
    grp = np.where(y == 1, np.where(yh42 == 1, "TP", "FN"), np.where(yh42 == 1, "FP", "TN"))
    imp = sh.abs().mean().sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(OUT / "explain_shap_importance.csv", encoding="utf-8-sig")
    top = imp.index[:10].tolist()
    by_grp = sh[top].groupby(grp).mean().T[["TP", "FN", "TN", "FP"]]
    by_grp.to_csv(OUT / "explain_shap_by_group.csv", encoding="utf-8-sig")
    cnt = pd.Series(grp).value_counts()
    cnt.to_csv(OUT / "explain_group_counts.csv", encoding="utf-8-sig")
    print(imp.head(12).round(4).to_string()); print(by_grp.round(3).to_string()); print(cnt.to_string())

    # 子群錯誤分布（seed 42）
    seg_rows = []
    for col in ("deposit_type", "market_segment", "hotel", "customer_type"):
        for v, g in tf.assign(yh=yh42, grp=grp).groupby(col):
            tp_, fn_, fp_ = (g.grp == "TP").sum(), (g.grp == "FN").sum(), (g.grp == "FP").sum()
            seg_rows.append(dict(column=col, value=v, n=len(g), cancel_rate=g[TARGET].mean(),
                                 recall=tp_ / max(tp_ + fn_, 1), precision=tp_ / max(tp_ + fp_, 1),
                                 fn=int(fn_), fn_share=fn_ / cnt["FN"], fp=int(fp_), fp_share=fp_ / cnt["FP"]))
    seg = pd.DataFrame(seg_rows)
    seg.to_csv(OUT / "explain_segments_s42.csv", index=False, encoding="utf-8-sig")
    print(seg[seg.n >= 500].round(3).to_string(index=False))
    prof = tf.assign(grp=grp).groupby("grp")[["lead_time", "total_of_special_requests", "adr"]].mean()
    prof.to_csv(OUT / "explain_group_profile_s42.csv", encoding="utf-8-sig")
    print(prof.round(2).to_string())

    # 圖 A：平均 |SHAP| 前 12
    t12 = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    ax.barh(t12.index, t12.values, color="#3b6ea8")
    for i, v in enumerate(t12.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("平均 |SHAP 值|（對數勝算）")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_shap_importance.png"); plt.close(fig)

    # 圖 B：TP 對 FN（同為真實取消，模型抓到 vs 漏掉）平均 SHAP
    t8 = top[:8][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    yy = np.arange(len(t8)); h = 0.38
    ax.barh(yy + h / 2, by_grp.loc[t8, "TP"], h, color="#3b6ea8", label=f"抓到的取消（TP，{cnt['TP']:,} 筆）")
    ax.barh(yy - h / 2, by_grp.loc[t8, "FN"], h, color="#d08a2e", label=f"漏掉的取消（FN，{cnt['FN']:,} 筆）")
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_yticks(yy, t8); ax.set_xlabel("平均 SHAP 值（正＝推向取消）")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_shap_tp_fn.png"); plt.close(fig)

    # 4. 推論時間（最後一季模型；含 adr 截尾、前處理、模型）
    tr, cap, pre, m = last
    sample = tf.sample(1000, random_state=0)
    t1 = time.perf_counter()
    for i in range(len(sample)):
        r = sample.iloc[[i]]
        m.predict_proba(pre.transform(cap.transform(r)), num_iteration=N_TREES)
    single_ms = (time.perf_counter() - t1) / len(sample) * 1000
    t1 = time.perf_counter()
    m.predict_proba(pre.transform(cap.transform(tf)), num_iteration=N_TREES)
    batch_s = time.perf_counter() - t1
    it = pd.DataFrame([dict(mode="逐筆（1,000 筆平均）", ms_per_row=single_ms, total_s=single_ms * 1000 / 1000),
                       dict(mode=f"批次（{len(tf):,} 筆）", ms_per_row=batch_s / len(tf) * 1000, total_s=batch_s)])
    it.to_csv(OUT / "inference_time.csv", index=False, encoding="utf-8-sig")
    print(it.to_string())
    print(f"done {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
