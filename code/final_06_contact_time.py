"""期末 階段 4 補充：只計「聯繫當天仍有效」的訂單時，每日前 10% 的 Precision 與涵蓋率
仍有效＝抵達前 h 天已訂房（lead_time >= h）且當時尚未取消；reservation_status_date 只用於篩選，不是模型特徵。
另輸出網頁名單定義（排除不退款與團體，seed 42）的同樣指標。需先執行 final_04_metrics.py。
執行：python code/final_06_contact_time.py → results/final/contact_time_check.csv、contact_time_demo_def.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import load_clean  # noqa: E402
from final_02_retraining import TEST_CUTS, TEST_END  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "final"
HORIZONS = [0, 7, 14, 30]


def top10(d, p):
    d = d.assign(p=p)
    d["rk"] = d.groupby("arrival_date").p.rank(ascending=False, method="first")
    sel = d.rk <= np.ceil(d.groupby("arrival_date").p.transform("size") * 0.1)
    return (d[sel].groupby("arrival_date").is_canceled.mean().mean(),
            d.loc[sel, "is_canceled"].sum() / max(d.is_canceled.sum(), 1), d.is_canceled.mean())


def main():
    df, _ = load_clean()
    c = TEST_CUTS + [TEST_END]
    te = pd.concat([df[(df.arrival_date >= c[i]) & (df.arrival_date < c[i + 1])] for i in range(3)]).reset_index(drop=True)
    canc = te.is_canceled == 1
    cd = (te.arrival_date - pd.to_datetime(te.reservation_status_date)).dt.days
    print("已取消訂單：取消日距抵達日中位數", cd[canc].median(), "；≥7 天", round((cd[canc] >= 7).mean(), 3),
          "；≥30 天", round((cd[canc] >= 30).mean(), 3))
    live = lambda h: np.ones(len(te), bool) if h == 0 else ((te.lead_time >= h) & (~canc | (cd < h))).values
    rows = []
    for h in HORIZONS:
        for grp, gm in [("全部", np.ones(len(te), bool)), ("排除不退款", (te.deposit_type != "Non Refund").values)]:
            m = live(h) & gm
            for mod, seeds in [("LightGBM", [42, 43, 44]), ("LogisticRegression", [42])]:
                r = [top10(te[m], np.load(OUT / f"final_test_{mod}_s{s}.npy")[m]) for s in seeds]
                rows.append(dict(contact_days_before=h, subset=grp, model=mod, n=int(m.sum()), base=round(r[0][2], 4),
                                 top10_prec=round(np.mean([x[0] for x in r]), 4),
                                 top10_cover=round(np.mean([x[1] for x in r]), 4)))
    out = pd.DataFrame(rows); out.to_csv(OUT / "contact_time_check.csv", index=False); print(out.to_string())
    gen = ((te.deposit_type != "Non Refund") & (te.market_segment != "Groups")).values
    p = np.load(OUT / "final_test_LightGBM_s42.npy")
    demo = []
    for h in [0, 30, 7]:
        m = live(h) & gen
        d = te[m].assign(p=p[m])
        d["rk"] = d.groupby("arrival_date").p.rank(ascending=False, method="first")
        dp = d[d.rk <= np.ceil(d.groupby("arrival_date").p.transform("size") * 0.1)].groupby("arrival_date").is_canceled.mean()
        demo.append(dict(contact_days_before=h, n=int(m.sum()), base=round(d.is_canceled.mean(), 4),
                         top10_prec=round(dp.mean(), 4), q25=round(dp.quantile(.25), 2), q75=round(dp.quantile(.75), 2)))
    demo = pd.DataFrame(demo); demo.to_csv(OUT / "contact_time_demo_def.csv", index=False); print(demo.to_string())


if __name__ == "__main__":
    main()
