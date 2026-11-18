"""期末 階段 8：消融實驗（新範本 肆之二）與表 4-1 補充欄位
完整模型＝LightGBM 15 葉 100 棵、門檻 0.36（rolling-origin 驗證選定）、base 特徵、每季重新訓練 expanding
變體（每個變體的門檻都在同樣 5 個驗證期上重新選，測試期一律每季重新訓練）：
  - 門檻改回 0.5（不經驗證期選門檻）
  - 改用期中設定：63 葉、1,600 棵（期中隨機切分提前停止的平均輪數）
  - 加入類別權重 balanced
（「只訓練一次」「＋agent 歷史取消率」「＋日期特徵」已由 final_02、final_03 算出，報告直接引用）
另算：正式方案與 LR 在 2017 測試期的 Accuracy／AUC、單次訓練時間（2017-07 切點）
執行：python code/final_08_ablation.py
輸出：results/final/ablation_*.csv、final_table41.csv
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import TARGET, features, load_clean  # noqa: E402
from final_01_rolling_origin import CUTOFFS, OUT, SEEDS, best_thr, prep  # noqa: E402
from final_02_retraining import TEST_CUTS, TEST_END  # noqa: E402

VARIANTS = {
    "完整模型": dict(leaves=15, trees=100, cw=None),
    "期中設定（63 葉、1,600 棵）": dict(leaves=63, trees=1600, cw=None),
    "加入類別權重（balanced）": dict(leaves=15, trees=100, cw="balanced"),
}


def lgbm(v, seed):
    return lgb.LGBMClassifier(n_estimators=v["trees"], learning_rate=0.05, num_leaves=v["leaves"],
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              class_weight=v["cw"], random_state=seed, verbose=-1)


def blocks(df, cuts):
    for a, b in zip(cuts[:-1], cuts[1:]):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        yield df[df.arrival_date < a], df[(df.arrival_date >= a) & (df.arrival_date < b)]


def run(df, cuts, v, seed, cat, num):
    ys, ps = [], []
    for tr, blk in blocks(df, cuts):
        A_tr, (A_b,) = prep(tr, [blk], cat, num, False)
        m = lgbm(v, seed).fit(A_tr, tr[TARGET].values)
        ys.append(blk[TARGET].values); ps.append(m.predict_proba(A_b)[:, 1])
    return np.concatenate(ys), np.concatenate(ps)


def scores(y, p, t):
    yh = (p >= t).astype(int)
    return dict(thr=t, accuracy=accuracy_score(y, yh), precision=precision_score(y, yh),
                recall=recall_score(y, yh), f1=f1_score(y, yh), auc=roc_auc_score(y, p))


def main():
    t0 = time.perf_counter()
    df, _ = load_clean()
    cat, num = features()
    val_cuts = []  # 每個驗證期：(切點, 切點+3 個月)
    test_cuts = TEST_CUTS + [TEST_END]
    rows = []
    for name, v in VARIANTS.items():
        for s in SEEDS:
            vy, vp = [], []
            for c in CUTOFFS:
                y, p = run(df, [c, str((pd.Timestamp(c) + pd.DateOffset(months=3)).date())], v, s, cat, num)
                vy.append(y); vp.append(p)
            vy, vp = np.concatenate(vy), np.concatenate(vp)
            t = best_thr(vy, vp)
            ty, tp = run(df, test_cuts, v, s, cat, num)
            if name == "完整模型":
                ref = np.load(OUT / f"final_test_LightGBM_s{s}.npy")
                assert np.abs(ref - tp).max() < 1e-9, "完整模型與 final_04 不一致"
                rows.append(dict(variant="門檻改回 0.5", seed=s, val_f1=f1_score(vy, (vp >= 0.5).astype(int)),
                                 **scores(ty, tp, 0.5)))
            rows.append(dict(variant=name, seed=s, val_f1=f1_score(vy, (vp >= t).astype(int)), **scores(ty, tp, t)))
            print(f"{name} s{s} thr={t} test F1={rows[-1]['f1']:.4f} {time.perf_counter() - t0:.0f}s", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "ablation_per_seed.csv", index=False, encoding="utf-8-sig")
    summ = per.groupby("variant", sort=False).agg(["mean", "std"]).drop(columns="seed")
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(OUT / "ablation_summary.csv", encoding="utf-8-sig")
    print(summ.round(4).to_string())

    # 表 4-1：2017 測試期，每季重新訓練
    ty = pd.concat([b for _, b in blocks(df, test_cuts)])[TARGET].values
    t41 = []
    lr = np.load(OUT / "final_test_LogisticRegression_s42.npy")
    t41.append(dict(model="Logistic Regression（C＝0.1）", seed="42", **scores(ty, lr, 0.36)))
    for s in SEEDS:
        t41.append(dict(model="LightGBM（15 葉、100 棵）", seed=str(s),
                        **scores(ty, np.load(OUT / f"final_test_LightGBM_s{s}.npy"), 0.36)))
    # 訓練時間：2017-07 切點的訓練資料（含前處理擬合），各量 3 次取平均
    tr = df[df.arrival_date < pd.Timestamp(TEST_CUTS[-1])]
    tim = {}
    for model in ("LR", "LGBM"):
        ts = []
        for s in SEEDS:
            t1 = time.perf_counter()
            A_tr, _ = prep(tr, [], cat, num, model == "LR")
            if model == "LR":
                LogisticRegression(C=0.1, max_iter=3000).fit(A_tr, tr[TARGET].values)
            else:
                lgbm(VARIANTS["完整模型"], s).fit(A_tr, tr[TARGET].values)
            ts.append(time.perf_counter() - t1)
        tim[model] = (np.mean(ts), np.std(ts, ddof=1), len(tr))
    t41 = pd.DataFrame(t41)
    t41["train_sec_mean"] = np.where(t41.model.str.startswith("Logistic"), tim["LR"][0], tim["LGBM"][0])
    t41["train_sec_sd"] = np.where(t41.model.str.startswith("Logistic"), tim["LR"][1], tim["LGBM"][1])
    t41["train_rows"] = tim["LR"][2]
    t41.to_csv(OUT / "final_table41.csv", index=False, encoding="utf-8-sig")
    print(t41.round(4).to_string())
    print(f"done {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
