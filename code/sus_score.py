"""使用者測試計分：讀 data/sus_responses.csv → results/final/sus_summary.csv、sus_per_user.csv
格式見 drafts_final2/使用者測試_SUS.md
執行：python code/sus_score.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "sus_responses.csv"
OUT = ROOT / "results" / "final"


def main():
    d = pd.read_csv(SRC, encoding="utf-8")
    q = [f"q{i}" for i in range(1, 11)]
    assert d[q].isin([1, 2, 3, 4, 5]).all().all(), "SUS 題目須為 1–5"
    odd = sum(d[f"q{i}"] - 1 for i in (1, 3, 5, 7, 9))
    even = sum(5 - d[f"q{i}"] for i in (2, 4, 6, 8, 10))
    d["sus"] = (odd + even) * 2.5
    d[["user", "role", "sus"] + [c for c in d.columns if c.startswith("T")]].to_csv(
        OUT / "sus_per_user.csv", index=False, encoding="utf-8-sig")
    rows = [dict(item="受測人數", value=len(d)),
            dict(item="SUS 平均", value=round(d.sus.mean(), 1)),
            dict(item="SUS 標準差", value=round(d.sus.std(), 1)),
            dict(item="SUS 最低／最高", value=f"{d.sus.min():.1f}／{d.sus.max():.1f}")]
    if "hotel_exp" in d:
        rows.append(dict(item="有飯店／訂房工作經驗", value=f"{d.hotel_exp.sum()}/{len(d)}"))
    for t in ("T1", "T2", "T3"):
        rows.append(dict(item=f"{t} 完成率", value=f"{d[t + '_done'].sum()}/{len(d)}"))
        if t + "_sec" in d:
            rows.append(dict(item=f"{t} 完成者平均秒數", value=round(d.loc[d[t + '_done'] == 1, t + "_sec"].mean(), 1)))
    if "total_min" in d:
        rows.append(dict(item="三任務自報總時間平均（分鐘）", value=round(d.total_min.mean(), 1)))
        rows.append(dict(item="三任務自報總時間最短／最長（分鐘）", value=f"{d.total_min.min():g}／{d.total_min.max():g}"))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "sus_summary.csv", index=False, encoding="utf-8-sig")
    print(s.to_string(index=False))
    print("\n開放回饋：")
    for _, r in d.iterrows():
        print(f"{r.user}（{r.role}）困惑：{r.confusing}｜希望：{r.wish}")


if __name__ == "__main__":
    main()
