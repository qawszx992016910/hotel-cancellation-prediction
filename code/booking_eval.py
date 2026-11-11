"""訂房取消預測評估工具（期末系統核心）

用途：輸入與 Hotel Booking Demand 相同欄位結構的 CSV，一律重新訓練，輸出評估報告：
  1. 洩漏檢查（兩層）
     第一層：對照欄位確定時點清單 → 標為「事後資訊」或「入住時才確定」
     第二層：統計規則（某類別取消率 ≤1% 或 ≥99%、或單一欄位 AUC ≥ 0.85）→ 只標「需人工確認」
     並對每個被標欄位報告「加回／刪除」的 F1 差
  2. 隨機切分、時間切分、去重後隨機切分三種評估的效能落差
  3. 門檻—Recall／Precision 曲線（時間切分測試集），含排除不退款訂單的 Recall
  4. 每個抵達日風險最高前 10% 訂單的 Precision
模型：Logistic Regression（C=0.1）與 LightGBM（15 葉、100 棵；依期末步驟 1 規則選定）；門檻一律在驗證集選

用法：
  python code/booking_eval.py --data data/hotel_bookings.csv --out results/eval_report
  Colab：!python booking_eval.py --data hotel_bookings.csv --out report
"""
import argparse
import base64
import io
import json
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET = "is_canceled"
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                      "August", "September", "October", "November", "December"], 1)}
# 第一層：欄位確定時點清單（依 António et al., 2017b, 2019b；本研究期中第參章）
TIMING = {
    "reservation_status": ("事後資訊", "訂單最後狀態，Canceled／No-Show 直接對應答案"),
    "reservation_status_date": ("事後資訊", "最後狀態的日期"),
    "country": ("入住時才確定", "訂房時多預設為葡萄牙，入住出示證件才更正"),
    "assigned_room_type": ("入住時才確定", "實際分配房型，可能因超賣或客人要求改變"),
    "required_car_parking_spaces": ("入住時才確定", "停車需求多在抵達時登記"),
    "booking_changes": ("可能含事後資訊", "統計到入住或取消為止；預設保留，請看加回／刪除差"),
    "arrival_date_year": ("不可外推", "年份只有少數值，無法外推到未來"),
}
EXCLUDE_DEFAULT = [k for k, (lvl, _) in TIMING.items() if lvl != "可能含事後資訊"]
CAT_HINT = ["hotel", "meal", "market_segment", "distribution_channel", "reserved_room_type",
            "deposit_type", "customer_type", "agent", "company", "country", "assigned_room_type",
            "reservation_status", "arrival_date_month"]
DERIVED = ["arrival_month", "total_nights", "total_guests"]
NEVER = [TARGET, "arrival_date", "reservation_status_date", "arrival_date_month"]
RATE_LO, RATE_HI, AUC_FLAG, MIN_N = 0.01, 0.99, 0.85, 50
LR_C, LGB_LEAVES, LGB_TREES = 0.1, 15, 100
THRS = np.round(np.arange(0.10, 0.901, 0.02), 2)


# ---------------- 資料 ----------------
def load(path):
    df = pd.read_csv(path)
    missing = [c for c in [TARGET, "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]
               if c not in df.columns]
    if missing:
        raise SystemExit(f"缺少必要欄位：{missing}")
    log = {"raw_rows": len(df)}
    kids = df["children"].fillna(0) if "children" in df else 0
    if {"adults", "babies"} <= set(df.columns):
        g = df["adults"] + kids + df["babies"]
        bad = g == 0
        if "adr" in df:
            bad |= df["adr"] < 0
        log["removed_invalid_rows"] = int(bad.sum())
        df = df[~bad].copy()
        df["total_guests"] = df["adults"] + df["children"].fillna(0) + df["babies"]
    if {"stays_in_weekend_nights", "stays_in_week_nights"} <= set(df.columns):
        df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["arrival_month"] = df["arrival_date_month"].map(MONTHS)
    df["arrival_date"] = pd.to_datetime(dict(year=df["arrival_date_year"], month=df["arrival_month"],
                                             day=df["arrival_date_day_of_month"]))
    for c in ["agent", "company"]:
        if c in df:
            df[c] = df[c].fillna(-1).astype(int).astype(str).replace("-1", "None")
    for c in df.columns:
        if c == "arrival_date":
            continue
        if c in CAT_HINT or not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna("Unknown").astype(str)
    log["rows"] = len(df)
    log["cancel_rate"] = float(df[TARGET].mean())
    log["period"] = f"{df.arrival_date.min().date()} ~ {df.arrival_date.max().date()}"
    return df.reset_index(drop=True), log


def col_types(df, cols):
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    return cat, [c for c in cols if c not in cat]


def base_features(df):
    return [c for c in df.columns if c not in NEVER + EXCLUDE_DEFAULT]


# ---------------- 切分 ----------------
def split_random(df, seed):
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp[TARGET], random_state=seed)
    return tr, va, te


def split_time(df):
    """依抵達日排序，前 70%／中 15%／後 15%（同一天不跨集合）"""
    days = df.arrival_date.sort_values().values
    d1, d2 = days[int(len(days) * 0.70)], days[int(len(days) * 0.85)]
    return df[df.arrival_date < d1], df[(df.arrival_date >= d1) & (df.arrival_date < d2)], df[df.arrival_date >= d2]


def dedup(df):
    return df.drop_duplicates(subset=[c for c in df.columns if c != "arrival_date"]).reset_index(drop=True)


# ---------------- 模型 ----------------
def fit_predict(model, tr, others, cols):
    cat, num = col_types(tr, cols)
    tr = tr.copy(); others = [o.copy() for o in others]
    if "adr" in num:
        cap = float(np.percentile(tr["adr"], 99.9))
        for d in [tr] + others:
            d["adr"] = d["adr"].clip(upper=cap)
    num_pipe = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if model == "LR" else [])
    pre = ColumnTransformer([("num", Pipeline(num_pipe), num),
                             ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001,
                                                   sparse_output=False), cat)])
    A = pre.fit_transform(tr[cols])
    if model == "LR":
        m = LogisticRegression(C=LR_C, max_iter=3000)
    else:
        m = lgb.LGBMClassifier(n_estimators=LGB_TREES, learning_rate=0.05, num_leaves=LGB_LEAVES, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
    m.fit(A, tr[TARGET].values)
    return [m.predict_proba(pre.transform(o[cols]))[:, 1] for o in others]


def best_thr(y, p):
    return float(THRS[int(np.argmax([f1_score(y, (p >= t).astype(int)) for t in THRS]))])


def scores(y, p, t):
    yh = (p >= t).astype(int)
    return dict(thr=t, f1=f1_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh), auc=roc_auc_score(y, p))


def evaluate(model, tr, va, te, cols):
    pv, pt = fit_predict(model, tr, [va, te], cols)
    t = best_thr(va[TARGET].values, pv)
    return scores(te[TARGET].values, pt, t), pt, t


# ---------------- 洩漏檢查 ----------------
def stat_flags(tr, va, cols):
    """第二層：只用訓練集統計，驗證集算單欄 AUC"""
    rows = []
    for c in cols:
        if c in (TARGET, "arrival_date"):
            continue
        s_tr, s_va = tr[c], va[c]
        if pd.api.types.is_numeric_dtype(s_tr) and s_tr.nunique() > 20:
            q = np.unique(np.nanquantile(s_tr, np.linspace(0, 1, 11)))
            b_tr = pd.cut(s_tr, q, include_lowest=True, duplicates="drop").astype(str)
            b_va = pd.cut(s_va.clip(q[0], q[-1]), q, include_lowest=True, duplicates="drop").astype(str)
            grp = pd.Series(np.where(s_tr > 0, ">0", "=0"), index=s_tr.index)
        else:
            b_tr, b_va = s_tr.astype(str), s_va.astype(str)
            grp = b_tr
        st = tr.groupby(grp)[TARGET].agg(["mean", "size"])
        st = st[st["size"] >= max(MIN_N, 0.005 * len(tr))]
        extreme = st[(st["mean"] <= RATE_LO) | (st["mean"] >= RATE_HI)]
        enc = tr.groupby(b_tr)[TARGET].mean()
        pv = b_va.map(enc).fillna(tr[TARGET].mean()).values
        auc = roc_auc_score(va[TARGET], pv) if len(np.unique(pv)) > 1 else 0.5
        auc = max(auc, 1 - auc)
        reasons = []
        if len(extreme):
            reasons.append("；".join(f"{k} 取消率 {v['mean']:.1%}（n={int(v['size']):,}）" for k, v in extreme.head(3).iterrows()))
        if auc >= AUC_FLAG:
            reasons.append(f"單欄 AUC {auc:.3f}")
        rows.append(dict(column=c, single_auc=round(auc, 4), extreme_levels=len(extreme), stat_flag=bool(reasons),
                         stat_reason="；".join(reasons)))
    return pd.DataFrame(rows)


def leak_check(df, seed):
    tr, va, te = split_random(df, seed)
    cand = [c for c in df.columns if c not in (TARGET, "arrival_date")]
    st = stat_flags(tr, va, cand)
    st["timing_level"] = st.column.map(lambda c: TIMING.get(c, ("", ""))[0])
    st["timing_note"] = st.column.map(lambda c: TIMING.get(c, ("", ""))[1])
    st["in_base"] = st.column.isin(base_features(df))
    flagged = st[(st.timing_level != "") | st.stat_flag].copy()
    base = base_features(df)
    ref, _, _ = evaluate("LGBM", tr, va, te, base)
    deltas = []
    for _, r in flagged.iterrows():
        c = r.column
        if c in ("reservation_status_date",):
            deltas.append((np.nan, "日期字串欄位，未測")); continue
        cols = [x for x in base if x != c] if c in base else base + [c]
        res, _, _ = evaluate("LGBM", tr, va, te, cols)
        d = res["f1"] - ref["f1"]
        deltas.append((d if c not in base else -d, "刪除後的 F1 差（取負）" if c in base else "加回後的 F1 差"))
    flagged["f1_gain_if_used"] = [round(d, 4) if d == d else d for d, _ in deltas]
    flagged["how"] = [h for _, h in deltas]
    flagged["verdict"] = np.select(
        [flagged.timing_level.isin(["事後資訊", "入住時才確定", "不可外推"]), flagged.stat_flag],
        ["排除（第一層）", "需人工確認（第二層）"], "保留但檢查（第一層）")
    return st, flagged.sort_values(["verdict", "f1_gain_if_used"], ascending=[True, False]), ref


# ---------------- 圖 ----------------
def fig_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description="訂房取消預測評估工具")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="eval_report")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    t0 = time.perf_counter()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    df, log = load(a.data)
    print(f"[1/4] 讀入 {log['rows']:,} 筆，期間 {log['period']}", flush=True)

    st, flagged, _ = leak_check(df, a.seed)
    st.to_csv(out / "leak_stats_all_columns.csv", index=False, encoding="utf-8-sig")
    flagged.to_csv(out / "leak_flagged.csv", index=False, encoding="utf-8-sig")
    print(f"[2/4] 洩漏檢查：標出 {len(flagged)} 欄", flush=True)

    base = base_features(df)
    dd = dedup(df)
    settings = {"隨機切分": split_random(df, a.seed), "去重後隨機切分": split_random(dd, a.seed), "時間切分": split_time(df)}
    rows, store = [], {}
    for sname, (tr, va, te) in settings.items():
        for m in ["LR", "LGBM"]:
            res, pt, t = evaluate(m, tr, va, te, base)
            rows.append(dict(setting=sname, model=m, train=len(tr), test=len(te), **res))
            store[(sname, m)] = (te, pt, t)
    ev = pd.DataFrame(rows)
    ev.to_csv(out / "evaluation_by_split.csv", index=False, encoding="utf-8-sig")
    gap = (ev.pivot(index="model", columns="setting", values="f1")
           .assign(**{"隨機−時間": lambda x: x["隨機切分"] - x["時間切分"],
                      "隨機−去重": lambda x: x["隨機切分"] - x["去重後隨機切分"]}))
    print(f"[3/4] 三種切分評估完成", flush=True)

    curve, top = [], []
    for m in ["LR", "LGBM"]:
        te, pt, t = store[("時間切分", m)]
        y = te[TARGET].values
        nr = (te["deposit_type"] != "Non Refund").values if "deposit_type" in te else np.ones(len(te), bool)
        for thr in THRS:
            yh = (pt >= thr).astype(int)
            curve.append(dict(model=m, thr=thr, recall=recall_score(y, yh),
                              precision=precision_score(y, yh, zero_division=0),
                              recall_excl_nonrefund=recall_score(y[nr], yh[nr])))
        d = te.assign(p=pt)
        d["rk"] = d.groupby("arrival_date").p.rank(ascending=False, method="first")
        sel = d.rk <= np.ceil(d.groupby("arrival_date").p.transform("size") * 0.10)
        yh = (pt >= t).astype(int)
        top.append(dict(model=m, thr=t, recall_excl_nonrefund=recall_score(y[nr], yh[nr]),
                        top10_precision_daily=float(d[sel].groupby("arrival_date")[TARGET].mean().mean()),
                        top10_recall=float(d.loc[sel, TARGET].sum() / y.sum()), base_rate=float(y.mean())))
    cdf, tdf = pd.DataFrame(curve), pd.DataFrame(top)
    cdf.to_csv(out / "threshold_curve_time_split.csv", index=False, encoding="utf-8-sig")
    tdf.to_csv(out / "practical_metrics_time_split.csv", index=False, encoding="utf-8-sig")

    figs = {}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    piv = ev.pivot(index="setting", columns="model", values="f1").loc[list(settings)]
    piv.plot.bar(ax=ax, rot=0, color=["#4C78A8", "#F58518"])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    ax.set_ylabel("測試集 F1"); ax.set_xlabel(""); ax.set_title("三種評估方式的 F1"); ax.set_ylim(0, 1)
    figs["fig_split_gap.png"] = fig_b64(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m, g in cdf.groupby("model", sort=False):
        ax.plot(g.thr, g.recall, label=f"{m} Recall")
        ax.plot(g.thr, g.recall_excl_nonrefund, "--", label=f"{m} Recall（排除不退款）")
        ax.plot(g.thr, g.precision, ":", label=f"{m} Precision")
    ax.set_xlabel("分類門檻"); ax.set_ylabel("指標值"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_title("時間切分測試集：門檻—Recall／Precision")
    figs["fig_threshold_curve.png"] = fig_b64(fig)
    for k, v in figs.items():
        (out / k).write_bytes(v)

    # ---------------- 報告 ----------------
    show = flagged[["column", "verdict", "timing_level", "timing_note", "stat_reason", "single_auc",
                    "f1_gain_if_used", "how"]].rename(columns={
        "column": "欄位", "verdict": "判定", "timing_level": "第一層", "timing_note": "第一層說明",
        "stat_reason": "第二層理由", "single_auc": "單欄 AUC", "f1_gain_if_used": "使用此欄的 F1 增量", "how": "算法"})
    evs = ev.round(3).rename(columns={"setting": "評估方式", "model": "模型", "train": "訓練筆數", "test": "測試筆數",
                                       "thr": "門檻（驗證集選）"})
    gaps = gap.round(3)
    tds = tdf.round(3).rename(columns={"model": "模型", "thr": "門檻", "recall_excl_nonrefund": "排除不退款 Recall",
                                       "top10_precision_daily": "每日前 10% Precision", "top10_recall": "前 10% 涵蓋的取消比例",
                                       "base_rate": "基準取消率"})
    notes = [
        "第二層統計規則只是提示，不代表欄位一定洩漏；本研究的 country 不會被統計規則標出，而保留的 deposit_type 會被標出。",
        "「使用此欄的 F1 增量」在隨機切分下計算，數值大不代表欄位可用，事後資訊會讓隨機切分的效能偏高。",
        "門檻一律在驗證集上選擇（最大化 F1），測試集只評估一次。",
        "時間切分依抵達日排序取前 70%／15%／15%；效能落差（隨機−時間）愈大，表示隨機切分愈高估模型對未來的表現。",
        "去重只刪除所有欄位完全相同的列；資料沒有訂單編號，無法分辨團體訂單與重複登錄。",
    ]
    md = [f"# 訂房取消預測評估報告", "",
          f"- 資料：`{a.data}`；{log['rows']:,} 筆（已刪除 {log.get('removed_invalid_rows', 0)} 筆人數為 0 或房價為負）；期間 {log['period']}；取消率 {log['cancel_rate']:.1%}",
          f"- 模型：Logistic Regression（C={LR_C}）、LightGBM（{LGB_LEAVES} 葉、{LGB_TREES} 棵）；種子 {a.seed}",
          f"- 使用特徵（{len(base)} 欄）：{', '.join(base)}", "",
          "## 一、洩漏檢查", "", show.to_markdown(index=False), "",
          "## 二、三種評估方式", "", evs.to_markdown(index=False), "", "F1 落差：", "", gaps.to_markdown(), "",
          "![](fig_split_gap.png)", "",
          "## 三、實務指標（時間切分測試集）", "", tds.to_markdown(index=False), "", "![](fig_threshold_curve.png)", "",
          "## 註", ""] + [f"- {n}" for n in notes] + ["", f"執行時間 {time.perf_counter() - t0:.0f} 秒"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")

    def img(k):
        return f'<img src="data:image/png;base64,{base64.b64encode(figs[k]).decode()}" style="max-width:100%">'
    css = ("body{font-family:'Microsoft JhengHei','Noto Sans TC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;"
           "color:#222;background:#fff}table{border-collapse:collapse;font-size:13px;margin:8px 0}"
           "td,th{border:1px solid #ccc;padding:4px 8px;vertical-align:top}th{background:#f2f2f2}h2{margin-top:32px}")
    html = (f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><title>取消預測評估報告</title>"
            f"<style>{css}</style></head><body><h1>訂房取消預測評估報告</h1><ul>"
            + "".join(f"<li>{x[2:]}</li>" for x in md[2:5]) + "</ul>"
            + "<h2>一、洩漏檢查</h2>" + show.to_html(index=False, na_rep="—")
            + "<h2>二、三種評估方式</h2>" + evs.to_html(index=False) + "<p>F1 落差：</p>" + gaps.to_html()
            + img("fig_split_gap.png")
            + "<h2>三、實務指標（時間切分測試集）</h2>" + tds.to_html(index=False) + img("fig_threshold_curve.png")
            + "<h2>註</h2><ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul></body></html>")
    (out / "report.html").write_text(html, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(dict(log=log, evaluation=ev.to_dict("records"),
                                                      practical=tdf.to_dict("records"),
                                                      flagged=flagged.column.tolist()),
                                                 ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[4/4] 報告：{out / 'report.html'}（{time.perf_counter() - t0:.0f} 秒）", flush=True)


if __name__ == "__main__":
    main()
