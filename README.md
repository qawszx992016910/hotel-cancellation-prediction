# 飯店訂房取消預測：排除資料洩漏與時間外推之評估

Hotel Booking Cancellation Prediction: A Leakage-Aware Evaluation with Temporal Validation

人工智慧課程期中專題（115 學年度第 1 學期）之程式與結果。報告中所有數字皆由本儲存庫的程式產生。

## 在 Colab 執行

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qawszx992016910/hotel-cancellation-prediction/blob/main/hotel_cancellation_colab.ipynb)

## 在本機執行

```bash
pip install -r requirements.txt
python code/run_experiments.py   # E1–E4 主實驗，輸出 results/*.csv、results/fig_*.png
python code/run_extra.py         # 門檻、bootstrap 與對照分析（須在上一步之後執行）
```

原始結果於 Windows 11、Python 3.13.3、AMD Ryzen 5 7535HS（CPU）產生；套件版本見 `requirements.txt`。隨機種子固定為 42–46。在不同作業系統或 CPU 上，LightGBM 與 Random Forest 的結果可能在小數點後第三位有些微差異。

## 檔案

| 路徑 | 內容 |
|---|---|
| `code/run_experiments.py` | 資料清理、前處理、三種模型（Logistic Regression、Random Forest、LightGBM）、E1–E4 實驗、錯誤分析與圖表 |
| `code/run_extra.py` | RQ1 成對差異、E2 門檻對照、驗證集選門檻、E4 bootstrap、固定輪數、季節對照、去重後依訂金類型分析 |
| `code/make_framework_fig.py` | 報告圖 3-1 研究架構圖 |
| `data/hotel_bookings.csv` | 原始資料（119,390 筆 × 32 欄） |
| `results/` | 執行結果（csv、png、run_log.json） |

## 資料來源與授權

資料取自 Kaggle「Hotel booking demand」（jessemostipak/hotel-booking-demand，版本 1），原始資料出自：

António, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief, 22*, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

資料以 CC BY 4.0 授權釋出，本儲存庫未修改原始檔案。
