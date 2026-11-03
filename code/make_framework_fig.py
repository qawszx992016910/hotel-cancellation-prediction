"""圖 3-1 研究架構圖"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"]
boxes = [
    ("資料蒐集", ["Kaggle Hotel Booking", "Demand（CC BY 4.0）", "119,390 筆、32 欄"]),
    ("資料前處理", ["刪除洩漏欄位", "異常值與缺失值", "衍生特徵、編碼", "（只以訓練集擬合）"]),
    ("模型建構", ["Baseline：LR、RF", "提出方法：LightGBM", "5 個隨機種子"]),
    ("模型評估", ["E1 模型比較", "E2 類別權重", "E3 洩漏與重複列", "E4 時間切分"]),
    ("錯誤分析與\n應用展示", ["錯誤案例特徵", "特徵重要度", "期末評估工具"]),
]
fig, ax = plt.subplots(figsize=(11.3, 3.6), dpi=200)
ax.set_xlim(0, 11.3); ax.set_ylim(0, 3.6); ax.axis("off")
w, gap = 1.9, 0.35
for i, (t, items) in enumerate(boxes):
    x = 0.15 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 2.55), w, 0.8, boxstyle="round,pad=0.02", fc="#1F3864", ec="#1F3864"))
    ax.text(x + w / 2, 2.95, t, ha="center", va="center", color="white", fontsize=12, weight="bold")
    ax.add_patch(FancyBboxPatch((x, 0.2), w, 2.2, boxstyle="round,pad=0.02", fc="#EEF2F8", ec="#1F3864", lw=1))
    for j, s in enumerate(items):
        ax.text(x + w / 2, 2.05 - j * 0.45, s, ha="center", va="center", fontsize=9.5)
    if i < len(boxes) - 1:
        ax.annotate("", xy=(x + w + gap - 0.03, 2.95), xytext=(x + w + 0.03, 2.95),
                    arrowprops=dict(arrowstyle="-|>", color="#1F3864", lw=1.6))
plt.tight_layout()
plt.savefig("results/fig_framework.png", bbox_inches="tight")
