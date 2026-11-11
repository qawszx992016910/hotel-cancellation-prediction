"""期末 網頁展示：產生單檔 demo/index.html（不需伺服器，瀏覽器直接開）
內容：
  1. 預先計算的成果摘要（results/final、results/eval_report）
  2. 每日聯繫名單示範：2017 測試期每季重新訓練的 LightGBM 預測，一般訂單與不退款／團體訂單分開排序
  3. 單筆訂單預測：示範模型（LightGBM 15 葉 100 棵，以 2017-07-01 前資料訓練）匯出成 JSON，於瀏覽器內推論
     ＋固定模板的文字說明（同類訂單歷史取消率，屬描述統計，非因果）
並以 node 驗證 JS 推論與 Python 結果一致。
執行：python code/build_demo.py （須先跑 final_01～final_04 與 booking_eval）
"""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import TARGET, AdrCapper, features, load_clean, make_pre  # noqa: E402
from final_02_retraining import TEST_CUTS, TEST_END  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIN = ROOT / "results" / "final"
EVR = ROOT / "results" / "eval_report"
OUT = ROOT / "demo"
OUT.mkdir(exist_ok=True)
DEMO_CUT = "2017-07-01"
LEAD_BINS = [-1, 7, 30, 90, 180, 10_000]
LEAD_LBL = ["0–7 天", "8–30 天", "31–90 天", "91–180 天", "180 天以上"]


def export_model(df, cat, num, thr):
    tr = df[df.arrival_date < DEMO_CUT]
    cap = AdrCapper().fit(tr)
    pre = make_pre(cat, num, scale=False)
    A = pre.fit_transform(cap.transform(tr))
    m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=42, verbose=-1).fit(A, tr[TARGET].values)
    imp = pre.named_transformers_["num"].named_steps["imp"]
    ohe = pre.named_transformers_["cat"]
    cats = []
    for c, cs, inf in zip(cat, ohe.categories_, ohe.infrequent_categories_):
        inf = [] if inf is None else [str(x) for x in inf]
        cats.append(dict(name=c, frequent=[str(x) for x in cs if str(x) not in inf], infrequent=inf))
    trees = []
    for t in m.booster_.dump_model()["tree_info"]:
        nodes = []
        def walk(n):
            i = len(nodes); nodes.append(None)
            if "leaf_value" in n:
                nodes[i] = [n["leaf_value"]]
            else:
                assert n["decision_type"] == "<=", n["decision_type"]
                l = walk(n["left_child"]); r = walk(n["right_child"])
                nodes[i] = [n["split_feature"], n["threshold"], l, r]
            return i
        walk(t["tree_structure"]); trees.append(nodes)
    model = dict(num=num, medians=[float(x) for x in imp.statistics_], adr_cap=cap.cap_, cats=cats, trees=trees,
                 thr=thr, n_features=A.shape[1], train_rows=len(tr), train_period=f"2015-07-01 ~ 2017-06-30")
    # 模板說明用的描述統計（只用示範模型的訓練資料）
    lb = pd.cut(tr.lead_time, LEAD_BINS, labels=LEAD_LBL).astype(str)
    stats = dict(base=float(tr[TARGET].mean()),
                 deposit_type=tr.groupby("deposit_type")[TARGET].agg(["mean", "size"]).to_dict("index"),
                 market_segment=tr.groupby("market_segment")[TARGET].agg(["mean", "size"]).to_dict("index"),
                 lead=tr.groupby(lb)[TARGET].agg(["mean", "size"]).to_dict("index"),
                 prev_cancel={k: v for k, v in tr.groupby(tr.previous_cancellations > 0)[TARGET]
                              .agg(["mean", "size"]).rename(index={True: "有", False: "無"}).to_dict("index").items()},
                 special={k: v for k, v in tr.groupby(tr.total_of_special_requests > 0)[TARGET]
                          .agg(["mean", "size"]).rename(index={True: "有", False: "無"}).to_dict("index").items()})
    # 驗證樣本：取 2017-07 之後 300 筆給 node 比對
    te = df[df.arrival_date >= DEMO_CUT].sample(300, random_state=1)
    p = m.predict_proba(pre.transform(cap.transform(te)))[:, 1]
    sample = [dict(row={c: (None if pd.isna(r[c]) else (r[c] if isinstance(r[c], str) else float(r[c]))) for c in cat + num},
                   p=float(pp)) for (_, r), pp in zip(te.iterrows(), p)]
    return model, stats, sample


def daily_lists(df):
    blocks = TEST_CUTS + [TEST_END]
    parts = [df[(df.arrival_date >= blocks[i]) & (df.arrival_date < blocks[i + 1])] for i in range(len(TEST_CUTS))]
    te = pd.concat(parts)
    p = np.load(FIN / "final_test_LightGBM_s42.npy")
    assert len(p) == len(te)
    te = te.assign(p=p)
    cols = ["arrival_date", "hotel", "lead_time", "market_segment", "deposit_type", "customer_type", "adr",
            "total_nights", "total_guests", "p", TARGET]
    t = te[cols].copy()
    t["arrival_date"] = t.arrival_date.dt.strftime("%Y-%m-%d")
    t["p"] = t.p.round(4); t["adr"] = t.adr.round(1)
    return {c: t[c].tolist() for c in cols}


def b64(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def main():
    df, _ = load_clean()
    cat, num = features()
    sel = json.loads((FIN / "model_selection.json").read_text(encoding="utf8"))
    thr = sel["best_config"]["LightGBM"]["thr_mean"]
    model, stats, sample = export_model(df, cat, num, round(thr, 2))
    week = (df.assign(md=df.arrival_date.dt.strftime("%m-%d")).groupby("md").arrival_date_week_number
            .agg(lambda s: int(s.mode().iloc[0])).to_dict())
    lists = daily_lists(df)
    choices = {c: sorted(df[c].astype(str).unique().tolist()) for c in
               ["hotel", "meal", "market_segment", "distribution_channel", "reserved_room_type", "deposit_type",
                "customer_type"]}
    choices["agent"] = df.agent.value_counts().head(40).index.tolist()
    choices["company"] = ["None"] + [c for c in df.company.value_counts().head(20).index if c != "None"]

    retrain = json.loads((FIN / "retrain_summary.json").read_text(encoding="utf8"))
    metrics = pd.read_csv(FIN / "metrics_summary.csv")
    ev = pd.read_csv(EVR / "evaluation_by_split.csv")
    flagged = pd.read_csv(EVR / "leak_flagged.csv")
    data = dict(model=model, stats=stats, week=week, lists=lists, choices=choices,
                summary=dict(selection=sel, retrain_test=retrain["test_2017"],
                             metrics=metrics.to_dict("records"), eval=ev.round(4).to_dict("records"),
                             flagged=flagged[["column", "verdict", "f1_gain_if_used", "stat_reason"]]
                             .fillna("").to_dict("records")),
                figs=dict(split=b64(EVR / "fig_split_gap.png"), retrain=b64(FIN / "fig_retrain.png"),
                          thr=b64(FIN / "fig_threshold_recall.png")))
    js_data = json.dumps(data, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tpl = (Path(__file__).parent / "demo_template.html").read_text(encoding="utf-8")
    model_js = (Path(__file__).parent / "demo_model.js").read_text(encoding="utf-8")
    html = tpl.replace("/*__MODELJS__*/", model_js).replace("/*__DATA__*/null", js_data.replace("</", "<\\/"))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("saved", OUT / "index.html", f"{(OUT / 'index.html').stat().st_size / 1e6:.1f} MB")

    # node 驗證：JS 推論 vs Python
    (OUT / "_sample.json").write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")
    js = (Path(__file__).parent / "demo_model.js").read_text(encoding="utf-8")
    check = js + """
const fs=require('fs');const M=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const S=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));let mx=0;
for(const s of S){const p=predict(M,s.row);mx=Math.max(mx,Math.abs(p-s.p));}
console.log(JSON.stringify({n:S.length,max_abs_diff:mx}));"""
    (OUT / "_check.js").write_text(check, encoding="utf-8")
    (OUT / "_model.json").write_text(json.dumps(model), encoding="utf-8")
    if shutil.which("node"):
        r = subprocess.run(["node", str(OUT / "_check.js"), str(OUT / "_model.json"), str(OUT / "_sample.json")],
                           capture_output=True, text=True)
        print("node check:", r.stdout.strip(), r.stderr.strip()[:500])
        res = json.loads(r.stdout)
        assert res["max_abs_diff"] < 1e-6, res
    else:
        print("找不到 node，略過 JS 推論驗證")
    for f in ["_sample.json", "_check.js", "_model.json"]:
        (OUT / f).unlink()


if __name__ == "__main__":
    main()
