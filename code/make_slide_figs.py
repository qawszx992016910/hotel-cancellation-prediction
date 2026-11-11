"""簡報用示意圖：梯度提升決策樹（LightGBM）原理"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"]
NAVY, LIGHT, GOLD = "#1F2D3D", "#EEF1F4", "#B8860B"

def tree(ax, x, y, s=0.32):
    pts = [(x, y), (x - s, y - 0.45), (x + s, y - 0.45), (x - s * 1.5, y - 0.9), (x - s * 0.5, y - 0.9)]
    for a, b in [(0, 1), (0, 2), (1, 3), (1, 4)]:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=NAVY, lw=1.2, zorder=1)
    for i, (px, py) in enumerate(pts):
        ax.scatter(px, py, s=90, color=NAVY if i < 3 else GOLD, zorder=2)

fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=200)
plt.rcParams["font.size"] = 12
ax.set_xlim(0, 8.6); ax.set_ylim(0, 5.6); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.2, 4.35), 8.2, 0.9, boxstyle="round,pad=0.02", fc=LIGHT, ec=NAVY, lw=1))
ax.text(4.3, 4.8, "輸入：訂單特徵（One-hot 後約 145 維）", ha="center", va="center", fontsize=15)
xs = [1.1, 3.0, 4.9, 7.4]
labels = ["樹 1", "樹 2", "樹 3", "樹 M"]
for x, lab in zip(xs, labels):
    tree(ax, x, 3.7)
    ax.text(x, 2.5, lab, ha="center", fontsize=16, weight="bold")
ax.text(6.15, 3.25, "…", ha="center", va="center", fontsize=22)
for x0, x1 in [(1.5, 2.6), (3.4, 4.5)]:
    ax.annotate("", xy=(x1, 3.25), xytext=(x0, 3.25), arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.6))
ax.text(2.05, 2.9, "擬合殘差\n（負梯度）", ha="center", fontsize=12, color=GOLD)
ax.text(3.95, 2.9, "擬合殘差\n（負梯度）", ha="center", fontsize=12, color=GOLD)
ax.add_patch(FancyBboxPatch((0.2, 0.95), 8.2, 1.15, boxstyle="round,pad=0.02", fc=NAVY, ec=NAVY))
ax.text(4.3, 1.75, "取消機率 = sigmoid（樹 1 ＋ 樹 2 ＋ … ＋ 樹 M 的加權輸出）", ha="center", va="center", color="white", fontsize=15.5, weight="bold")
ax.text(4.3, 1.28, "M 由驗證集 log-loss 決定：連續 100 輪未改善即停止", ha="center", va="center", color="white", fontsize=14)
ax.text(4.3, 0.45, "learning_rate 0.05｜num_leaves 63｜subsample 0.8｜colsample_bytree 0.8｜輪數上限 5,000", ha="center", fontsize=13, color="#333333")
plt.tight_layout(); plt.savefig("results/fig_gbdt_schematic.png", bbox_inches="tight"); plt.close(fig)


# ---- 三類方法比較示意（簡報理論基礎頁，直式） ----
fig, ax = plt.subplots(figsize=(6.9, 4.7), dpi=220)
ax.set_xlim(0, 6.9); ax.set_ylim(0, 4.7); ax.axis("off")
rows = [("Logistic Regression", "特徵線性加權後取 sigmoid\n係數可解讀；交互作用需人工設計", "lr"),
        ("Random Forest（平行）", "多棵樹以自助抽樣資料訓練\n機率取平均（Breiman, 2001）", "rf"),
        ("梯度提升 GBDT（依序）", "新樹擬合前一輪的負梯度，輸出加總\nXGBoost、LightGBM（Ke et al., 2017）", "gb")]
for k, (title, desc, kind) in enumerate(rows):
    y0 = 4.7 - (k + 1) * 1.55 + 0.1
    ax.add_patch(FancyBboxPatch((0.05, y0), 6.8, 1.35, boxstyle="round,pad=0.02", fc=LIGHT, ec="#C9CED3"))
    if kind == "lr":
        for i, yy in enumerate([y0 + 1.05, y0 + 0.68, y0 + 0.31]):
            ax.text(0.35, yy, f"x{i+1}", ha="center", va="center", fontsize=12)
            ax.annotate("", xy=(1.3, y0 + 0.68), xytext=(0.55, yy), arrowprops=dict(arrowstyle="-|>", color=GOLD))
        ax.add_patch(FancyBboxPatch((1.3, y0 + 0.45), 0.9, 0.46, boxstyle="round,pad=0.02", fc=NAVY, ec=NAVY))
        ax.text(1.75, y0 + 0.68, "Σwx", ha="center", va="center", color="white", fontsize=12)
    else:
        for x in [0.5, 1.2, 1.9]: tree(ax, x, y0 + 1.15, s=0.17)
        if kind == "gb":
            for x0 in [0.72, 1.42]:
                ax.annotate("", xy=(x0 + 0.3, y0 + 0.78), xytext=(x0, y0 + 0.78), arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.4))
    ax.text(2.55, y0 + 1.02, title, ha="left", va="center", fontsize=14, weight="bold", color=NAVY)
    ax.text(2.55, y0 + 0.45, desc, ha="left", va="center", fontsize=11.5, linespacing=1.5)
plt.savefig("results/fig_methods.png", bbox_inches="tight"); plt.close(fig)

# ---- 文獻中的取消率（簡報研究背景頁） ----
rows = [("António et al. (2017a)\nAlgarve 4 家度假飯店，2015", 11.8, 26.4),
        ("António et al. (2019a)\n葡萄牙 8 家飯店，2016–2017", 12.2, 40.0),
        ("本研究資料（António et al., 2019b）\n度假飯店 H1、城市飯店 H2", 27.8, 41.7)]
fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
for i, (lab, lo, hi) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([lo, hi], [y, y], color=NAVY, lw=7, solid_capstyle="round")
    ax.text(lo - 1.2, y, f"{lo}%", ha="right", va="center", fontsize=13)
    ax.text(hi + 1.2, y, f"{hi}%", ha="left", va="center", fontsize=13)
ax.scatter([37.0], [0], color=GOLD, s=90, zorder=3)
ax.text(37.0, -0.32, "整體 37.0%", ha="center", va="top", fontsize=13, color=GOLD)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=12.5)
ax.set_xlim(0, 50); ax.set_ylim(-0.8, 2.5); ax.set_xlabel("取消率（%）", fontsize=13)
ax.set_title("各研究中飯店的取消率範圍", fontsize=15, weight="bold", color=NAVY)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.grid(axis="x", color="#DDDDDD", lw=0.8); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig("results/fig_cancel_rates.png", bbox_inches="tight"); plt.close(fig)
