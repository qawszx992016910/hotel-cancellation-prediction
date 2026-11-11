"""期末 步驟 3：衍生特徵（報告 陸之二）
正式方案（使用者決策 A）：expanding 視窗、每季重新訓練；模型設定沿用步驟 1
特徵組：
  calendar  抵達日星期、是否週末抵達、抵達日是否葡萄牙國定假日、住宿期間是否含國定假日、距最近國定假日天數
  agent     同一 agent 的歷史取消率：只用「訓練集中、抵達日早於本筆訂房日（抵達日 − lead_time）」的訂單，
            以全體訓練取消率平滑（k=20），不含自身與未來標籤
比較：base / +calendar / +agent / +both，驗證（5 切點）＋群組 bootstrap，2017 測試（每季重新訓練）
另報：無訂金、Online TA、Direct 三個子群的 F1
執行：python code/final_03_features.py
"""
import json
import sys
import time
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import TARGET, DUP_COLS_EXCLUDE, features, load_clean, metrics  # noqa: E402
from final_01_rolling_origin import CUTOFFS, OUT, SEEDS, best_thr, fit_predict, group_counts, prep  # noqa: E402
from final_02_retraining import CFG, KEY, TEST_CUTS, TEST_END  # noqa: E402

K_SMOOTH = 20
N_BOOT = 2000
CAL = ["arrival_dow", "arrival_weekend", "arrival_holiday", "stay_has_holiday", "days_to_holiday"]
SETS = {"base": [], "+calendar": CAL, "+agent": ["agent_cancel_rate", "agent_hist_n"],
        "+both": CAL + ["agent_cancel_rate", "agent_hist_n"]}
SEGMENTS = {"無訂金": lambda d: d.deposit_type == "No Deposit",
            "Online TA": lambda d: d.market_segment == "Online TA",
            "Direct": lambda d: d.market_segment == "Direct"}


def add_calendar(df):
    """只依日期與固定假日曆，不用任何標籤，可在切分前計算"""
    hol = holidays.Portugal(years=range(2015, 2018))
    hd = pd.to_datetime(sorted(hol.keys()))
    a = df.arrival_date
    df["arrival_dow"] = a.dt.dayofweek
    df["arrival_weekend"] = (a.dt.dayofweek >= 4).astype(int)       # 週五、六、日抵達
    df["arrival_holiday"] = a.isin(hd).astype(int)
    dep = a + pd.to_timedelta(df.total_nights, unit="D")
    idx_a = np.searchsorted(hd.values, a.values, side="left")
    idx_d = np.searchsorted(hd.values, dep.values, side="left")
    df["stay_has_holiday"] = (idx_d > idx_a).astype(int)           # [抵達, 離開) 之間有假日
    nxt = hd.values[np.minimum(idx_a, len(hd) - 1)]
    prv = hd.values[np.maximum(idx_a - 1, 0)]
    df["days_to_holiday"] = np.minimum(np.abs((nxt - a.values) / np.timedelta64(1, "D")),
                                       np.abs((a.values - prv) / np.timedelta64(1, "D"))).astype(int)
    df["booking_date"] = a - pd.to_timedelta(df.lead_time, unit="D")
    return df


def agent_rate(hist, query, prior):
    """query 每列：hist 中同 agent、抵達日 < query 訂房日 的取消數／筆數（平滑）"""
    rate = np.full(len(query), prior); n_out = np.zeros(len(query))
    hist = hist.sort_values("arrival_date")
    hg = {a: (g.arrival_date.values, np.cumsum(g[TARGET].values)) for a, g in hist.groupby("agent")}
    q = query.reset_index(drop=True)
    for a, idx in q.groupby("agent").groups.items():
        if a not in hg:
            continue
        dates, cs = hg[a]
        n = np.searchsorted(dates, q.loc[idx, "booking_date"].values, side="left")
        c = np.where(n > 0, cs[np.maximum(n - 1, 0)], 0)
        rate[idx] = (c + K_SMOOTH * prior) / (n + K_SMOOTH)
        n_out[idx] = n
    return rate, n_out


def with_agent(tr, others):
    prior = tr[TARGET].mean()
    out = []
    for d in [tr] + others:
        r, n = agent_rate(tr, d, prior)
        out.append(d.assign(agent_cancel_rate=r, agent_hist_n=np.log1p(n)))
    return out[0], out[1:]


def run_block(df, cut, end, model, seed, extra, cat, num):
    c0, c1 = pd.Timestamp(cut), pd.Timestamp(end)
    tr = df[df.arrival_date < c0]
    blk = df[(df.arrival_date >= c0) & (df.arrival_date < c1)]
    if any(f.startswith("agent_") for f in extra):
        tr, (blk,) = with_agent(tr, [blk])
    A_tr, (A_b,) = prep(tr, [blk], cat, num + extra, model == "LogisticRegression")
    (p,) = fit_predict(CFG[model], A_tr, tr[TARGET].values, [A_b], seed)[KEY[model]]
    return blk, p


def seg_f1(frame, y, p, t):
    return {f"f1_{k}": f1_score(y[m], (p[m] >= t).astype(int)) for k, f in SEGMENTS.items()
            for m in [f(frame).values]}


def main():
    t0 = time.perf_counter()
    df, _ = load_clean()
    df["dup_group"] = df.groupby([c for c in df.columns if c not in DUP_COLS_EXCLUDE + ["arrival_date"]],
                                 dropna=False, sort=False).ngroup()
    df = add_calendar(df)
    cat, num = features()

    # sanity：抽 500 筆，以暴力法重算 agent 歷史率，確認只用「抵達日 < 訂房日」的訓練訂單
    tr_chk = df[df.arrival_date < "2016-10-01"]
    q_chk = df[df.arrival_date >= "2016-10-01"].sample(250, random_state=0)
    q_chk = pd.concat([q_chk, tr_chk.sample(250, random_state=0)])
    prior = tr_chk[TARGET].mean()
    r_fast, n_fast = agent_rate(tr_chk, q_chk, prior)
    brute_r, brute_n = [], []
    for _, row in q_chk.iterrows():
        h = tr_chk[(tr_chk.agent == row.agent) & (tr_chk.arrival_date < row.booking_date)]
        brute_n.append(len(h)); brute_r.append((h[TARGET].sum() + K_SMOOTH * prior) / (len(h) + K_SMOOTH))
    leak_chk = dict(n_checked=len(q_chk), max_abs_rate_diff=float(np.max(np.abs(r_fast - np.array(brute_r)))),
                    count_mismatch=int(np.sum(n_fast != np.array(brute_n))),
                    note="自身抵達日 ≥ 訂房日，條件「抵達日 < 訂房日」必然排除自身")
    assert leak_chk["max_abs_rate_diff"] < 1e-9 and leak_chk["count_mismatch"] == 0, leak_chk
    print("agent-rate check OK", leak_chk, flush=True)

    # ---------- 驗證 ----------
    vp, vframe = {}, None
    for fs, extra in SETS.items():
        for model in CFG:
            for s in (SEEDS if model == "LightGBM" else SEEDS[:1]):
                parts, ps = [], []
                for c in CUTOFFS:
                    blk, p = run_block(df, c, pd.Timestamp(c) + pd.DateOffset(months=3), model, s, extra, cat, num)
                    parts.append(blk.assign(fold=c)); ps.append(p)
                vp[(fs, model, s)] = np.concatenate(ps)
                if vframe is None:
                    vframe = pd.concat(parts).reset_index(drop=True)
                print(f"val {fs} {model} s{s} {time.perf_counter() - t0:.0f}s", flush=True)
    y = vframe[TARGET].values
    rows, thr = [], {}
    for (fs, model, s), p in vp.items():
        t = best_thr(y, p); thr[(fs, model, s)] = t
        r = dict(features=fs, model=model, seed=s, thr=t, f1_pooled=f1_score(y, (p >= t).astype(int)),
                 auc_pooled=roc_auc_score(y, p),
                 f1_fold_mean=np.mean([f1_score(y[m], (p[m] >= t).astype(int))
                                       for m in ((vframe.fold == c).values for c in CUTOFFS)]),
                 **seg_f1(vframe, y, p, t))
        rows.append(r)
    vdf = pd.DataFrame(rows); vdf.to_csv(OUT / "feat_val_per_seed.csv", index=False)
    vagg = vdf.groupby(["model", "features"], sort=False).mean(numeric_only=True).drop(columns="seed").round(4)
    vagg.to_csv(OUT / "feat_val_summary.csv")

    rng = np.random.default_rng(2026)
    g = pd.factorize(vframe["dup_group"])[0]; n_g = g.max() + 1
    W = [np.bincount(rng.integers(0, n_g, n_g), minlength=n_g) for _ in range(N_BOOT)]
    boot = {}
    for model in CFG:
        seeds = SEEDS if model == "LightGBM" else SEEDS[:1]
        cnt = {(fs, s): group_counts(y, (vp[(fs, model, s)] >= thr[(fs, model, s)]).astype(int), g)
               for fs in SETS for s in seeds}
        f1 = lambda wt, k: (lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn))(*(wt @ x for x in cnt[k]))
        for fs in SETS:
            if fs == "base":
                continue
            d = np.array([np.mean([f1(wt, (fs, s)) - f1(wt, ("base", s)) for s in seeds]) for wt in W])
            boot[f"{model}:{fs}-base"] = dict(
                point=float(vagg.loc[(model, fs), "f1_pooled"] - vagg.loc[(model, "base"), "f1_pooled"]),
                ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                passes_rule=bool(np.percentile(d, 2.5) > 0.01))

    # ---------- 2017 測試：每季重新訓練（expanding）----------
    cuts = TEST_CUTS + [TEST_END]
    trows = []
    for fs, extra in SETS.items():
        for model in CFG:
            for s in (SEEDS if model == "LightGBM" else SEEDS[:1]):
                parts, ps = [], []
                for i in range(len(TEST_CUTS)):
                    blk, p = run_block(df, cuts[i], cuts[i + 1], model, s, extra, cat, num)
                    parts.append(blk); ps.append(p)
                te = pd.concat(parts); p = np.concatenate(ps); yt = te[TARGET].values; t = thr[(fs, model, s)]
                trows.append(dict(features=fs, model=model, seed=s, thr=t, **metrics(yt, p, t), **seg_f1(te, yt, p, t)))
                print(f"test {fs} {model} s{s} F1={trows[-1]['f1']:.3f}", flush=True)
    tdf = pd.DataFrame(trows); tdf.to_csv(OUT / "feat_test_per_seed.csv", index=False)
    tagg = tdf.groupby(["model", "features"], sort=False).mean(numeric_only=True).drop(columns="seed").round(4)
    tagg.to_csv(OUT / "feat_test_summary.csv")

    summary = dict(k_smooth=K_SMOOTH, leak_check=leak_chk,
                   val=vagg.reset_index().to_dict("records"), val_diff_vs_base=boot,
                   test_2017=tagg.reset_index().to_dict("records"),
                   runtime_min=round((time.perf_counter() - t0) / 60, 1))
    (OUT / "feat_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(boot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
