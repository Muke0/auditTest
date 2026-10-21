"""P5：生成 3 张报告图（保存到 figures/）。配色采用 dataviz 参考调色板。"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体 + 减号
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 调色板（dataviz 参考实例）
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

df = pd.read_csv("audit_records_processed.csv")
df["audit_time"] = pd.to_datetime(df["audit_time"])
df["voucher_synced_at"] = pd.to_datetime(df["voucher_synced_at"])
df["lag"] = (df["audit_time"] - df["voucher_synced_at"]).dt.total_seconds() / 60
df["is_fail"] = (df["audit_result"] == 0).astype(int)
df["is_empty"] = df["audit_result"].isna().astype(int)

import os
os.makedirs("figures", exist_ok=True)


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


# ---- 图1：失败率 vs 同步滞后 ----
bins = [-1e9, 0, 15, 30, 45, 60, 1e9]
labels = ["≤0", "0–15", "15–30", "30–45", "45–60", ">60"]
df["bkt"] = pd.cut(df["lag"], bins=bins, labels=labels)
g = df.groupby("bkt", observed=True).agg(条数=("audit_id", "count"), 失败=("is_fail", "sum"))
g["失败率%"] = g["失败"] / g["条数"] * 100
baseline = df[df["lag"].between(0, 15)]["is_fail"].mean() * 100

fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
x = range(len(g))
ax.plot(x, g["失败率%"], color=BLUE, marker="o", markersize=6, linewidth=2)
ax.axhline(baseline, color=ORANGE, linestyle="--", linewidth=1.2)
ax.text(0.02, baseline + 0.8, f"新鲜基准 {baseline:.1f}%", color=ORANGE, fontsize=8)
for xi, (n, r) in enumerate(zip(g["条数"], g["失败率%"])):
    ax.text(xi, r + 1.0, f"{r:.1f}%\n(n={n})", ha="center", fontsize=8, color=INK)
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_xlabel("凭证数据同步滞后（audit_time − voucher_synced_at，分钟）")
ax.set_ylabel("稽核失败率（%）")
ax.set_title("失败率随同步滞后单调上升")
ax.set_ylim(0, 38)
style_ax(ax)
plt.tight_layout()
plt.savefig("figures/fail_rate_by_lag.png", bbox_inches="tight")
plt.close()

# ---- 图2：净节省 vs 触发阈值 ----
rows = []
for T in [15, 30, 45, 60]:
    m = df["lag"] > T
    n = int(m.sum())
    cur_fail = int(df.loc[m, "is_fail"].sum())
    expect = n * (baseline / 100)
    avoid = cur_fail - expect
    net = avoid * 75 - n * 10
    rows.append((f">{T}", net, avoid))
rdf = pd.DataFrame(rows, columns=["阈值", "净节省min", "可避免失败"])
colors = [ORANGE if t == ">30" else BLUE for t in rdf["阈值"]]

fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
bars = ax.bar(rdf["阈值"], rdf["净节省min"], color=colors, width=0.55)
for b, (net, avoid) in zip(bars, zip(rdf["净节省min"], rdf["可避免失败"])):
    ax.text(b.get_x() + b.get_width() / 2, net + 40, f"{net:.0f}\n(避免{avoid:.0f}条)", ha="center", fontsize=8, color=INK)
ax.axhline(0, color=MUTED, linewidth=0.8)
ax.set_xlabel("触发同步的滞后阈值（分钟）")
ax.set_ylabel("净节省（分钟）")
ax.set_title("「先同步再稽核」净节省：阈值 30min 最优")
style_ax(ax)
plt.tight_layout()
plt.savefig("figures/net_saving_by_threshold.png", bbox_inches="tight")
plt.close()

# ---- 图3：各机构 新鲜 vs 陈旧 失败率 ----
df["stale"] = df["lag"] > 30
piv = df.pivot_table(index="branch_name", columns="stale", values="is_fail", aggfunc="mean") * 100
piv.columns = ["新鲜(≤30min)", "陈旧(>30min)"]
piv = piv.sort_values("陈旧(>30min)", ascending=False)

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
y = range(len(piv))
h = 0.38
ax.barh([i + h / 2 for i in y], piv["新鲜(≤30min)"], height=h, color=BLUE, label="新鲜(≤30min)")
ax.barh([i - h / 2 for i in y], piv["陈旧(>30min)"], height=h, color=ORANGE, label="陈旧(>30min)")
for i, name in enumerate(piv.index):
    fresh = piv.loc[name, "新鲜(≤30min)"]
    stale = piv.loc[name, "陈旧(>30min)"]
    ax.text(fresh + 0.7, i + h / 2, f"{fresh:.1f}", va="center", fontsize=8, color=INK)
    ax.text(stale + 0.7, i - h / 2, f"{stale:.1f}", va="center", fontsize=8, color=INK)
ax.set_yticks(list(y))
ax.set_yticklabels(piv.index)
ax.set_xlabel("稽核失败率（%）")
ax.set_title("各机构失败率：区域中心随滞后暴涨，总部直管几乎不变")
ax.legend(frameon=False, fontsize=8)
style_ax(ax)
plt.tight_layout()
plt.savefig("figures/fail_rate_by_branch.png", bbox_inches="tight")
plt.close()

print("saved 3 figures -> figures/")
print("baseline(新鲜)失败率 = %.2f%%" % baseline)
