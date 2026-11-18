"""圖 5-1 系統架構圖：離線 Python（評估工具、訓練與預先計算）→ 單檔網頁（瀏覽器內推論，無後端）
執行：python code/make_system_fig.py → results/final/fig_system.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
NAVY, LIGHT, ORANGE, OL = "#1F3864", "#EEF2F8", "#C55A11", "#FBE5D6"

fig, ax = plt.subplots(figsize=(11.3, 4.9), dpi=200)
ax.set_xlim(0, 11.3); ax.set_ylim(0, 4.9); ax.axis("off")


def box(x, y, w, h, title, lines, fc=LIGHT, ec=NAVY, tc=NAVY, fs=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="center", fontsize=10.5, weight="bold", color=tc)
    for j, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.55 - j * 0.29, s, ha="center", va="center", fontsize=fs)


def band(x, w, label, color):
    ax.add_patch(FancyBboxPatch((x, 0.1), w, 4.65, boxstyle="round,pad=0.02", fc="white", ec=color, lw=1.4, ls="--"))
    ax.text(x + w / 2, 4.55, label, ha="center", va="center", fontsize=11, weight="bold", color=color)


def arrow(x0, y0, x1, y1, text=None, color=NAVY):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.14, text, ha="center", va="bottom", fontsize=8, color="#444")


band(0.1, 2.0, "資料", NAVY)
band(2.35, 4.35, "離線處理（Python，命令列或 Colab）", NAVY)
band(6.95, 4.25, "網頁展示（單檔 HTML，瀏覽器內執行）", ORANGE)

box(0.25, 2.0, 1.7, 2.1, "訂房資料 CSV", ["Kaggle Hotel", "Booking Demand", "或飯店自有資料", "（同欄位格式）"])

box(2.5, 2.55, 1.95, 1.75, "評估工具", ["booking_eval.py", "兩層洩漏檢查", "隨機／去重／時間", "三種評估比較"])
box(4.6, 2.55, 1.95, 1.75, "訓練與預測", ["final_01～06", "rolling-origin 選模", "每季重訓 expanding", "LightGBM 機率"])
box(2.5, 0.3, 1.95, 1.85, "評估報告", ["report.html", "report.md、csv", "（圖表內嵌）"], fc="white")
box(4.6, 0.3, 1.95, 1.85, "網頁產生器", ["build_demo.py", "模型樹結構→JSON", "預先計算聯繫名單", "node 核對推論"], fc="white")

box(7.6, 3.15, 3.4, 1.15, "成果摘要", ["選模、重訓與指標結果"], fc=OL, ec=ORANGE, tc=ORANGE)
box(7.6, 1.75, 3.4, 1.2, "每日聯繫名單", ["選抵達日、飯店、聯繫時點；一般與不退款／團體分列"], fc=OL, ec=ORANGE, tc=ORANGE)
box(7.6, 0.3, 3.4, 1.25, "單筆預測", ["輸入訂單欄位 → JS 走訪 100 棵樹", "取消機率＋模板文字說明"], fc=OL, ec=ORANGE, tc=ORANGE)

arrow(1.95, 3.4, 2.5, 3.4)
ax.plot([1.95, 4.53, 4.53], [2.35, 2.35, 3.0], color=NAVY, lw=1.5)
arrow(4.53, 3.0, 4.6, 3.0)
arrow(3.47, 2.55, 3.47, 2.15)
arrow(5.57, 2.55, 5.57, 2.15)
ax.plot([7.3, 7.3], [0.92, 3.72], color=ORANGE, lw=1.5)
for yy in (3.72, 2.35, 0.92):
    arrow(7.3, yy, 7.6, yy, color=ORANGE)
arrow(6.55, 1.25, 7.3, 1.25, color=ORANGE)
ax.text(6.92, 1.33, "index.html", ha="center", va="bottom", fontsize=8, color="#444")

plt.tight_layout()
plt.savefig("results/final/fig_system.png", bbox_inches="tight")
print("ok")
