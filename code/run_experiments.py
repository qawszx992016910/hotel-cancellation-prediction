"""飯店訂房取消預測：E1–E4 實驗
資料：Kaggle jessemostipak/hotel-booking-demand（Antonio et al., 2019, CC BY 4.0）
執行：python code/run_experiments.py   （於專案根目錄）
輸出：results/*.csv、results/fig_*.png
"""
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hotel_bookings.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
(OUT / "preds").mkdir(exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
TARGET = "is_canceled"
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}

# ---------- 特徵集合 ----------
LEAK_ALWAYS = ["reservation_status", "reservation_status_date"]   # 目標的最終狀態，必刪
LEAK_RISK_CAT = ["assigned_room_type", "country"]                 # 入住時才確定（António et al., 2017b, 2019a; Silvestre et al., 2025）
LEAK_RISK_NUM = ["required_car_parking_spaces"]                   # >0 者取消率 0%，António et al. (2017b) 亦刪除
DROP_OTHER = ["arrival_date_year"]                                # 年份無法外推
CAT_BASE = ["hotel", "meal", "market_segment", "distribution_channel",
            "reserved_room_type", "deposit_type", "customer_type", "agent", "company"]
NUM_BASE = ["lead_time", "arrival_month", "arrival_date_week_number", "arrival_date_day_of_month",
            "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
            "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
            "booking_changes", "days_in_waiting_list", "adr",
            "total_of_special_requests", "total_nights", "total_guests"]


def load_clean():
    """讀檔＋以固定規則處理異常值（規則不依賴資料統計量，切分前執行不會洩漏）"""
    df = pd.read_csv(DATA)
    log = {"raw_rows": len(df)}
    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    bad = (guests == 0) | (df["adr"] < 0)
    log["removed_zero_guests"] = int((guests == 0).sum())
    log["removed_negative_adr"] = int((df["adr"] < 0).sum())
    df = df[~bad].copy()
    log["clean_rows"] = len(df)
    df["arrival_month"] = df["arrival_date_month"].map(MONTHS)
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"].fillna(0) + df["babies"]
    # agent／company 的 NULL 表示「不適用」（Antonio et al., 2019），轉成類別 None
    for c in ["agent", "company"]:
        df[c] = df[c].fillna(-1).astype(int).astype(str).replace("-1", "None")
    df["country"] = df["country"].fillna("Unknown")
    df["arrival_date"] = pd.to_datetime(dict(year=df["arrival_date_year"], month=df["arrival_month"],
                                             day=df["arrival_date_day_of_month"]))
    return df.reset_index(drop=True), log


class AdrCapper:
    """以訓練集 99.9 百分位數截尾 adr（只用訓練集擬合）"""
    def fit(self, X, y=None):
        self.cap_ = float(np.percentile(X["adr"], 99.9)); return self
    def transform(self, X):
        X = X.copy(); X["adr"] = X["adr"].clip(upper=self.cap_); return X
    def get_params(self, deep=True): return {}
    def set_params(self, **p): return self


def make_pre(cat, num, scale):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001,
                              sparse_output=False), cat),
    ])


def make_model(name, seed, balanced):
    cw = "balanced" if balanced else None
    if name == "LogisticRegression":
        return LogisticRegression(C=1.0, max_iter=3000, class_weight=cw), True
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                      random_state=seed, class_weight=cw), False
    if name == "LightGBM":
        return lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, num_leaves=63,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  random_state=seed, class_weight=cw, verbose=-1), False
    raise ValueError(name)


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yhat), precision=precision_score(y, yhat),
                recall=recall_score(y, yhat), f1=f1_score(y, yhat),
                auc_roc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p))


def run_one(name, Xtr, ytr, Xva, yva, Xte, yte, cat, num, seed, balanced=False):
    model, scale = make_model(name, seed, balanced)
    t0 = time.perf_counter()
    cap = AdrCapper().fit(Xtr)
    pre = make_pre(cat, num, scale)
    A_tr = pre.fit_transform(cap.transform(Xtr))
    A_va = pre.transform(cap.transform(Xva))
    A_te = pre.transform(cap.transform(Xte))
    if name == "LightGBM":
        model.fit(A_tr, ytr, eval_set=[(A_va, yva)], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(A_tr, ytr)
    train_sec = time.perf_counter() - t0
    p_te = model.predict_proba(A_te)[:, 1]
    p_va = model.predict_proba(A_va)[:, 1]
    res = metrics(yte, p_te)
    res.update(val_f1=f1_score(yva, (p_va >= 0.5).astype(int)), train_sec=train_sec,
               n_features=A_tr.shape[1])
    if name == "LightGBM":
        res["best_iter"] = model.best_iteration_
    res["_p_va"] = p_va
    return res, p_te, model, pre


def random_split(df, seed):
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp[TARGET], random_state=seed)
    return tr, va, te


def features(add_cat=(), add_num=(), drop=()):
    cat = [c for c in CAT_BASE if c not in drop] + list(add_cat)
    num = [c for c in NUM_BASE if c not in drop] + list(add_num)
    return cat, num


DUP_COLS_EXCLUDE = ["arrival_date"]   # 去重依據：全部原始欄位＋衍生欄位（arrival_date 由原始欄位組成）


def dedup(df):
    return df.drop_duplicates(subset=[c for c in df.columns if c not in DUP_COLS_EXCLUDE]).reset_index(drop=True)


def clean_res(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


MODELS = ["LogisticRegression", "RandomForest", "LightGBM"]


def main():
    df, log = load_clean()
    cat, num = features()
    rows, store = [], {}

    # ---------- E1 模型比較 ＋ E2 類別權重（隨機分層切分，5 種子）----------
    for seed in SEEDS:
        tr, va, te = random_split(df, seed)
        if seed == 42:
            log["split_sizes_seed42"] = dict(train=len(tr), val=len(va), test=len(te))
            log["cancel_rate_seed42"] = dict(train=tr[TARGET].mean(), val=va[TARGET].mean(),
                                             test=te[TARGET].mean())
        for bal in [False, True]:
            for m in MODELS:
                r, p, model, pre = run_one(m, tr, tr[TARGET], va, va[TARGET], te, te[TARGET],
                                           cat, num, seed, bal)
                rows.append(dict(exp="E1" if not bal else "E2", model=m, balanced=bal, seed=seed, **clean_res(r)))
                np.save(OUT / "preds" / f"rand_{m}_bal{int(bal)}_s{seed}_test.npy", p)
                np.save(OUT / "preds" / f"rand_{m}_bal{int(bal)}_s{seed}_val.npy", r["_p_va"])
                print(f"seed{seed} {m:18s} bal={bal} F1={r['f1']:.3f} AUC={r['auc_roc']:.3f} {r['train_sec']:.1f}s")
                if seed == 42:
                    store[(m, bal)] = (te, p, model, pre)
    pd.DataFrame(rows).to_csv(OUT / "e1_e2_per_seed.csv", index=False)

    # ---------- E3 洩漏風險特徵敏感度（LightGBM，5 種子）＋ 去重資料三模型 ----------
    e3 = []
    variants = {
        "主實驗": dict(),
        "只加回 country": dict(add_cat=["country"]),
        "只加回 assigned_room_type": dict(add_cat=["assigned_room_type"]),
        "只加回 required_car_parking_spaces": dict(add_num=["required_car_parking_spaces"]),
        "三欄全部加回": dict(add_cat=LEAK_RISK_CAT, add_num=LEAK_RISK_NUM),
        "再刪除 booking_changes": dict(drop=["booking_changes"]),
        "再刪除 deposit_type": dict(drop=["deposit_type"]),
        "誤用 reservation_status（示範洩漏）": dict(add_cat=["reservation_status"]),
    }
    for vname, v in variants.items():
        c, n = features(**v)
        for seed in SEEDS:
            tr, va, te = random_split(df, seed)
            r, *_ = run_one("LightGBM", tr, tr[TARGET], va, va[TARGET], te, te[TARGET], c, n, seed)
            e3.append(dict(exp="E3", variant=vname, model="LightGBM", rows=len(df), seed=seed, **clean_res(r)))
            print(f"E3 {vname} seed{seed} F1={r['f1']:.3f}", flush=True)
    dd = dedup(df)
    log["dedup"] = dict(rows_after=len(dd), removed=len(df) - len(dd), cancel_rate_after=dd[TARGET].mean(),
                        removed_cancel_rate=float(df.loc[df.duplicated(subset=[c for c in df.columns if c not in DUP_COLS_EXCLUDE]), TARGET].mean()))
    for seed in SEEDS:
        tr, va, te = random_split(dd, seed)
        for m in MODELS:
            r, *_ = run_one(m, tr, tr[TARGET], va, va[TARGET], te, te[TARGET], cat, num, seed)
            e3.append(dict(exp="E3", variant="刪除完全重複列", model=m, rows=len(dd), seed=seed, **clean_res(r)))
            print(f"E3 dedup {m} seed{seed} F1={r['f1']:.3f}", flush=True)
    pd.DataFrame(e3).to_csv(OUT / "e3_per_seed.csv", index=False)

    # ---------- E4 時間切分 ----------
    e4 = []
    trva = df[df["arrival_date"] < "2017-01-01"]
    te = df[df["arrival_date"] >= "2017-01-01"]
    tr = trva[trva["arrival_date"] < "2016-10-01"]
    va = trva[trva["arrival_date"] >= "2016-10-01"]
    log["temporal_split"] = dict(train=len(tr), val=len(va), test=len(te),
                                 train_period="2015-07-01~2016-09-30", val_period="2016-10-01~2016-12-31",
                                 test_period="2017-01-01~2017-08-31",
                                 cancel_rate=dict(train=tr[TARGET].mean(), val=va[TARGET].mean(),
                                                  test=te[TARGET].mean()))
    for m in MODELS:
        r, p, *_ = run_one(m, tr, tr[TARGET], va, va[TARGET], te, te[TARGET], cat, num, 42)
        e4.append(dict(exp="E4", split="時間切分", model=m, **clean_res(r)))
        np.save(OUT / "preds" / f"time_{m}_test.npy", p)
        np.save(OUT / "preds" / f"time_{m}_val.npy", r["_p_va"])
        print(f"E4 {m} F1={r['f1']:.3f} AUC={r['auc_roc']:.3f}")
        if m == "LightGBM":
            store["E4_lgbm"] = (te, p)
    pd.DataFrame(e4).to_csv(OUT / "e4_temporal.csv", index=False)

    # ---------- 彙整表 ----------
    per = pd.DataFrame(rows)
    agg = (per.groupby(["exp", "model", "balanced"])
           [["accuracy", "precision", "recall", "f1", "auc_roc", "pr_auc", "train_sec"]]
           .agg(["mean", "std"]).round(4))
    agg.to_csv(OUT / "summary_e1_e2.csv")
    agg3 = (pd.DataFrame(e3).groupby(["variant", "model"], sort=False)
            [["accuracy", "precision", "recall", "f1", "auc_roc", "pr_auc"]].agg(["mean", "std"]).round(4))
    agg3.to_csv(OUT / "summary_e3.csv")

    # ---------- 錯誤分析與圖（seed 42、LightGBM 無權重）----------
    te42, p42, lgbm, pre = store[("LightGBM", False)]
    y42 = te42[TARGET].values
    yhat = (p42 >= 0.5).astype(int)
    cm = confusion_matrix(y42, yhat)
    pd.DataFrame(cm, index=["實際未取消", "實際取消"], columns=["預測未取消", "預測取消"]).to_csv(OUT / "confusion_lgbm_seed42.csv")
    err = te42.assign(pred=yhat, prob=p42)
    err["type"] = np.select([(err.pred == 1) & (err[TARGET] == 1), (err.pred == 0) & (err[TARGET] == 0),
                             (err.pred == 1) & (err[TARGET] == 0)], ["TP", "TN", "FP"], "FN")
    seg = []
    for col in ["hotel", "deposit_type", "market_segment", "customer_type"]:
        g = err.groupby(col).apply(lambda x: pd.Series(dict(
            n=len(x), cancel_rate=x[TARGET].mean(),
            recall=recall_score(x[TARGET], x.pred, zero_division=0),
            precision=precision_score(x[TARGET], x.pred, zero_division=0),
            f1=f1_score(x[TARGET], x.pred, zero_division=0))), include_groups=False)
        g.insert(0, "feature", col); seg.append(g.reset_index().rename(columns={col: "level"}))
    pd.concat(seg).round(4).to_csv(OUT / "error_by_segment_lgbm_seed42.csv", index=False)
    prof = err.groupby("type")[["lead_time", "adr", "total_of_special_requests", "required_car_parking_spaces",
                               "previous_cancellations", "booking_changes", "total_nights"]].mean().round(3)
    prof["n"] = err["type"].value_counts()
    prof["non_refund_share"] = err.groupby("type")["deposit_type"].apply(lambda s: (s == "Non Refund").mean()).round(4)
    prof.to_csv(OUT / "error_profile_lgbm_seed42.csv")

    # 特徵重要度（gain，依原始欄位彙總 one-hot）
    names = pre.get_feature_names_out()
    gain = lgbm.booster_.feature_importance("gain")
    imp = pd.DataFrame({"feature": names, "gain": gain})
    imp["source"] = imp.feature.str.replace(r"^(num|cat)__", "", regex=True)
    for c in cat:
        imp.loc[imp.source.str.startswith(c + "_"), "source"] = c
    imp_src = imp.groupby("source").gain.sum().sort_values(ascending=False)
    (imp_src / imp_src.sum()).round(4).rename("gain_share").to_csv(OUT / "feature_importance_lgbm_seed42.csv")

    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 圖：混淆矩陣
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["未取消", "取消"]); ax.set_yticks([0, 1], ["未取消", "取消"])
    ax.set_xlabel("預測類別"); ax.set_ylabel("實際類別"); ax.set_title("LightGBM 混淆矩陣（測試集，seed 42）")
    fig.tight_layout(); fig.savefig(OUT / "fig_confusion_lgbm.png", dpi=200); plt.close(fig)
    # 圖：ROC
    fig, ax = plt.subplots(figsize=(5, 4.4))
    for m in MODELS:
        t, p, *_ = store[(m, False)]
        fpr, tpr, _ = roc_curve(t[TARGET], p)
        ax.plot(fpr, tpr, label=f"{m}（AUC = {roc_auc_score(t[TARGET], p):.3f}）")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("偽陽性率（False Positive Rate）"); ax.set_ylabel("真陽性率（True Positive Rate）")
    ax.set_title("三種模型 ROC 曲線（測試集，seed 42）"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_roc.png", dpi=200); plt.close(fig)
    # 圖：類別分布（依飯店）
    ct = pd.crosstab(df["hotel"], df[TARGET])
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    x = np.arange(len(ct))
    ax.bar(x - 0.2, ct[0], 0.4, label="未取消", color="#4C78A8")
    ax.bar(x + 0.2, ct[1], 0.4, label="取消", color="#E45756")
    for i, h in enumerate(ct.index):
        ax.text(i - 0.2, ct.loc[h, 0], f"{ct.loc[h, 0]:,}", ha="center", va="bottom", fontsize=9)
        ax.text(i + 0.2, ct.loc[h, 1], f"{ct.loc[h, 1]:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, ["City Hotel（城市）", "Resort Hotel（度假）"]); ax.set_ylabel("訂單數")
    ax.set_title("各飯店訂單之取消分布（清理後）"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_class_distribution.png", dpi=200); plt.close(fig)
    # 圖：特徵重要度 top 12
    top = (imp_src / imp_src.sum()).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.barh(top.index, top.values, color="#4C78A8")
    ax.set_xlabel("Gain 佔比"); ax.set_title("LightGBM 特徵重要度前 12 名")
    fig.tight_layout(); fig.savefig(OUT / "fig_feature_importance.png", dpi=200); plt.close(fig)

    log["environment"] = dict(python=platform.python_version(), sklearn=sklearn.__version__,
                              pandas=pd.__version__, lightgbm=lgb.__version__, numpy=np.__version__,
                              os=platform.platform(), seeds=SEEDS)
    (OUT / "run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2, default=float), encoding="utf8")
    print(json.dumps(log, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
