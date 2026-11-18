"""Google 表單回覆（xlsx）→ data/sus_responses.csv（code/sus_score.py 的輸入）
欄位依表單題號（drafts_final2/使用者測試_操作教學.md）：0 時間、1 背景、2 飯店經驗、3–5 任務一至三、
6 任務三說明、7 總分鐘、8–17 SUS 1–10、18 困惑、19 希望功能。
判分：任務答案與網頁顯示值相差 0.5 個百分點內為完成（任務一 0.680、任務二 75%、任務三 27.3%）。
執行：python code/sus_convert.py "訂房取消預測網頁 使用者測試 (回覆).xlsx"
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANS = {"T1": 0.680, "T2": 0.75, "T3": 0.273}


def num(v):
    """'75%'、'0.75'、75 → 0.75；無數字回傳 None。"""
    m = re.search(r"\d+(?:\.\d+)?", str(v))
    if not m:
        return None
    x = float(m.group())
    return x / 100 if ("%" in str(v) or x > 1) else x


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "訂房取消預測網頁 使用者測試 (回覆).xlsx"
    x = pd.read_excel(src)
    c = x.columns
    out = pd.DataFrame({"user": [f"U{i + 1}" for i in range(len(x))],
                        "role": x[c[1]].astype(str).str.strip(),
                        "hotel_exp": (x[c[2]].astype(str).str.strip() == "有").astype(int)})
    for t, col in zip(ANS, c[3:6]):
        out[t + "_ans"] = x[col]
        out[t + "_done"] = [int(num(v) is not None and abs(num(v) - ANS[t]) <= 0.005) for v in x[col]]
    out["T3_explain"] = x[c[6]].astype(str).str.strip()
    out["total_min"] = [float(m.group()) if (m := re.search(r"\d+(?:\.\d+)?", str(v))) else None for v in x[c[7]]]
    for i in range(10):
        out[f"q{i + 1}"] = x[c[8 + i]].astype(int)
    out["confusing"] = x[c[18]].astype(str).str.strip()
    out["wish"] = x[c[19]].astype(str).str.strip()
    dst = ROOT / "data" / "sus_responses.csv"
    out.to_csv(dst, index=False, encoding="utf-8")
    print(out.drop(columns=["confusing", "wish", "T3_explain"]).to_string(index=False))
    print("→", dst)


if __name__ == "__main__":
    main()
