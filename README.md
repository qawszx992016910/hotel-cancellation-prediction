# 飯店訂房取消預測：排除資料洩漏與時間外推之評估

Hotel Booking Cancellation Prediction: A Leakage-Aware Evaluation with Temporal Validation

人工智慧課程期中／期末專題（115 學年度第 1 學期）之程式與結果。報告中所有數字皆由本儲存庫的程式產生。

## 在 Colab 執行

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qawszx992016910/hotel-cancellation-prediction/blob/main/hotel_cancellation_colab.ipynb)

## 在本機執行

```bash
pip install -r requirements.txt
python code/run_experiments.py   # E1–E4 主實驗，輸出 results/*.csv、results/fig_*.png
python code/run_extra.py         # 門檻、bootstrap 與對照分析（須在上一步之後執行）
python code/run_overlap.py       # 重複列分群與時間組成分析（須在第一步之後執行）
```

原始結果於 Windows 11、Python 3.13.3、AMD Ryzen 5 7535HS（CPU）產生；套件版本見 `requirements.txt`。隨機種子固定為 42–46。在不同作業系統或 CPU 上，LightGBM 與 Random Forest 的結果可能在小數點後第三位有些微差異。

## 期末：時間驗證、評估工具與網頁展示

```bash
python code/final_01_rolling_origin.py   # rolling-origin 時間驗證＋事先定義的選模規則
python code/final_02_retraining.py       # 重新訓練策略：expanding／6 個月／3 個月視窗
python code/final_03_features.py         # 衍生特徵：週末、葡萄牙國定假日、agent 歷史取消率
python code/final_04_metrics.py          # 排除不退款 Recall、每日前 10% Precision、門檻—Recall 曲線
python code/booking_eval.py --data data/hotel_bookings.csv --out results/eval_report   # 評估工具
python code/build_demo.py                # 產生 demo/index.html（單檔網頁展示）
```

步驟 1–4 須依序執行（後面的程式讀前面的結果），合計約 20 分鐘；結果在 `results/final/`。

**正式方案**：LightGBM（15 葉、100 棵，門檻 0.36）＋基本特徵＋每季重新訓練（使用全部歷史資料）。選擇只依 rolling-origin 驗證期與事先定義的規則（F1 差 95% 信賴區間下界 > 0.01），2017 年測試集只評估一次。

**評估工具** `booking_eval.py`：輸入相同欄位結構的 CSV，一律重新訓練，輸出 `report.html`（圖內嵌的單一檔案）、`report.md` 與各項 csv：

1. 洩漏檢查：第一層對照欄位確定時點清單；第二層統計規則（類別取消率 ≤1% 或 ≥99%、單欄 AUC ≥ 0.85）只標「需人工確認」；每個被標欄位附加回／刪除的 F1 差
2. 隨機切分、去重後隨機切分、時間切分三種評估的效能落差
3. 時間切分測試集的門檻—Recall／Precision 曲線（含排除不退款訂單）與每日前 10% Precision

**網頁展示** `demo/index.html`：不需伺服器，瀏覽器直接開啟。包含成果摘要、每日聯繫名單（一般訂單與不退款／團體訂單分開排序）、單筆訂單預測（LightGBM 匯出為 JSON 於瀏覽器內推論，與 Python 結果差異 < 1e-15）與固定模板的文字說明。示範模型只適用於本資料的兩家飯店。

## 檔案

| 路徑 | 內容 |
|---|---|
| `code/run_experiments.py` | 資料清理、前處理、三種模型（Logistic Regression、Random Forest、LightGBM）、E1–E4 實驗、錯誤分析與圖表 |
| `code/run_extra.py` | RQ1 成對差異、E2 門檻對照、驗證集選門檻、E4 bootstrap、固定輪數、季節對照、去重後依訂金類型分析 |
| `code/run_overlap.py` | 測試列是否在訓練集有相同列之分群 F1、訓練期與測試期組成比較 |
| `code/make_framework_fig.py` | 報告圖 3-1 研究架構圖 |
| `code/final_01_rolling_origin.py`～`final_04_metrics.py` | 期末步驟 1–4 |
| `code/booking_eval.py` | 評估工具（單檔，可在 Colab 或命令列執行） |
| `code/build_demo.py`、`demo_template.html`、`demo_model.js` | 網頁展示產生器與瀏覽器端推論 |
| `demo/index.html` | 網頁展示 |
| `data/hotel_bookings.csv` | 原始資料（119,390 筆 × 32 欄） |
| `results/` | 期中執行結果（csv、png、run_log.json）；`results/final/` 期末結果；`results/eval_report/` 評估工具輸出 |

## 資料來源與授權

資料取自 Kaggle「Hotel booking demand」（jessemostipak/hotel-booking-demand，版本 1），原始資料出自：

António, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief, 22*, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

資料以 CC BY 4.0 授權釋出，本儲存庫未修改原始檔案。
