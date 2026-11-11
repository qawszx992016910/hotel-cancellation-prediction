"""期末 步驟 5：選模規則的敏感度分析（回應審查 F1）
1. 逐一剔除各驗證期（leave-one-fold-out）後，LightGBM − LR 的 F1 差與 95% CI 下界
2. bootstrap 隨機種子（5 組）與 B＝10,000 下的下界變動
3. 6 個月視窗 vs 全部歷史（LightGBM）的同樣檢查
4. 2017 年 7、8 月：排除不退款訂單後，訓練一次 vs 每季重新訓練的 F1（檢查回升是否只來自訂金組成）
執行：python code/final_05_sensitivity.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import TARGET, DUP_COLS_EXCLUDE, features, load_clean  # noqa: E402
from final_01_rolling_origin import CUTOFFS, OUT, SEEDS, best_thr, group_counts  # noqa: E402
from final_02_retraining import CFG, TEST_CUTS, TEST_END, predict_block  # noqa: E402


def val_preds(df, cat, num, months):
    out, frame = {}, None
    for model in CFG:
        for s in (SEEDS if model == "LightGBM" else SEEDS[:1]):
            parts, ps = [], []
            for c in CUTOFFS:
                blk, p, _ = predict_block(df, c, pd.Timestamp(c) + pd.DateOffset(months=3), months, model, s, cat, num)
                parts.append(blk.assign(fold=c)); ps.append(p)
            out[(model, s)] = np.concatenate(ps)
            if frame is None:
                frame = pd.concat(parts).reset_index(drop=True)
    return out, frame


def boot_diff(y, g, pa, pb, seeds_a, seeds_b, B, rng_seed, mask=None):
    """pa/pb: {seed: prob}；回傳 (點估計, 下界, 上界)；門檻在 mask 內重新選"""
    m = np.ones(len(y), bool) if mask is None else mask
    y, g = y[m], pd.factorize(g[m])[0]
    n_g = g.max() + 1
    ca = {s: group_counts(y, (pa[s][m] >= best_thr(y, pa[s][m])).astype(int), g) for s in seeds_a}
    cb = {s: group_counts(y, (pb[s][m] >= best_thr(y, pb[s][m])).astype(int), g) for s in seeds_b}
    f = lambda w, c: (lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn))(*(w @ x for x in c))
    one = np.ones(n_g)
    point = np.mean([f(one, ca[s]) for s in seeds_a]) - np.mean([f(one, cb[s]) for s in seeds_b])
    rng = np.random.default_rng(rng_seed)
    d = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, n_g, n_g), minlength=n_g)
        d[b] = np.mean([f(w, ca[s]) for s in seeds_a]) - np.mean([f(w, cb[s]) for s in seeds_b])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    df, _ = load_clean()
    df["dup_group"] = df.groupby([c for c in df.columns if c not in DUP_COLS_EXCLUDE + ["arrival_date"]],
                                 dropna=False, sort=False).ngroup()
    cat, num = features()
    exp, frame = val_preds(df, cat, num, None)
    six, _ = val_preds(df, cat, num, 6)
    y, g = frame[TARGET].values, frame["dup_group"].values
    lg = {s: exp[("LightGBM", s)] for s in SEEDS}
    lr = {SEEDS[0]: exp[("LogisticRegression", SEEDS[0])]}
    lg6 = {s: six[("LightGBM", s)] for s in SEEDS}
    rows = []
    for name, (pa, pb, sa, sb) in {"LightGBM − LR": (lg, lr, SEEDS, SEEDS[:1]),
                                   "LightGBM 6 個月 − 全部歷史": (lg6, lg, SEEDS, SEEDS)}.items():
        pt, lo, hi = boot_diff(y, g, pa, pb, sa, sb, 2000, 2026)
        rows.append(dict(comparison=name, setting="主分析（B=2000, seed 2026）", point=pt, lo=lo, hi=hi))
        for rs in [1, 2, 3, 4, 5]:
            pt, lo, hi = boot_diff(y, g, pa, pb, sa, sb, 2000, rs)
            rows.append(dict(comparison=name, setting=f"bootstrap seed {rs}", point=pt, lo=lo, hi=hi))
        pt, lo, hi = boot_diff(y, g, pa, pb, sa, sb, 10000, 2026)
        rows.append(dict(comparison=name, setting="B=10000", point=pt, lo=lo, hi=hi))
        for c in CUTOFFS:
            pt, lo, hi = boot_diff(y, g, pa, pb, sa, sb, 2000, 2026, mask=(frame.fold != c).values)
            rows.append(dict(comparison=name, setting=f"剔除 {c[:7]} 起的驗證期", point=pt, lo=lo, hi=hi))
        print(name, "done", flush=True)
    sens = pd.DataFrame(rows).round(4)
    sens.to_csv(OUT / "sensitivity_rule.csv", index=False)

    # 2017 年 7、8 月：排除不退款後，訓練一次 vs 每季重新訓練
    mon = []
    for model in CFG:
        for s in (SEEDS if model == "LightGBM" else SEEDS[:1]):
            thr = best_thr(y, exp[(model, s)])
            blk1, p1, _ = predict_block(df, "2017-01-01", TEST_END, None, model, s, cat, num)
            blk2, p2, _ = predict_block(df, TEST_CUTS[2], TEST_END, None, model, s, cat, num)
            m1 = (blk1.arrival_date >= TEST_CUTS[2]).values
            a, pa_ = blk1[m1], p1[m1]
            for lab, sub, pp in [("訓練一次", a, pa_), ("每季重新訓練", blk2, p2)]:
                for grp, mk in [("全部", np.ones(len(sub), bool)), ("排除不退款", (sub.deposit_type != "Non Refund").values)]:
                    yy = sub[TARGET].values[mk]
                    mon.append(dict(model=model, seed=s, plan=lab, subset=grp, n=int(mk.sum()),
                                    f1=f1_score(yy, (pp[mk] >= thr).astype(int))))
    mdf = pd.DataFrame(mon).groupby(["model", "plan", "subset"], sort=False)[["n", "f1"]].mean().round(4)
    mdf.to_csv(OUT / "jul_aug_nonrefund_check.csv")
    print(sens.to_string()); print(mdf.to_string())
    (OUT / "sensitivity_summary.json").write_text(json.dumps(dict(
        rule=sens.to_dict("records"), jul_aug=mdf.reset_index().to_dict("records")), ensure_ascii=False, indent=2),
        encoding="utf8")


if __name__ == "__main__":
    main()
