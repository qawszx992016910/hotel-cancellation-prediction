"""表 4-1 補充：期中 E4（時間切分、單一驗證期、驗證集選定門檻）三模型的 Accuracy
讀 results/preds/time_*_test.npy 與 results/val_threshold.csv 的門檻（LR 0.29、RF 0.20、LightGBM 0.20）
執行：python code/final_09_e4_accuracy.py → results/final/e4_validated_metrics.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import TARGET, load_clean  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
THR = {"LogisticRegression": 0.29, "RandomForest": 0.20, "LightGBM": 0.20}

df, _ = load_clean()
y = df[df.arrival_date >= "2017-01-01"][TARGET].values
rows = []
for m, t in THR.items():
    p = np.load(ROOT / "results" / "preds" / f"time_{m}_test.npy")
    yh = (p >= t).astype(int)
    rows.append(dict(model=m, thr=t, accuracy=accuracy_score(y, yh), precision=precision_score(y, yh),
                     recall=recall_score(y, yh), f1=f1_score(y, yh), auc=roc_auc_score(y, p)))
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "final" / "e4_validated_metrics.csv", index=False, encoding="utf-8-sig")
print(out.round(4).to_string(index=False))
