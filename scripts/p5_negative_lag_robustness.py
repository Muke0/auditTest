"""P5：负滞后(数据竞争)排查 + 主结论稳健性(敏感性)分析。"""
import pandas as pd
import numpy as np

df = pd.read_csv("audit_records_processed.csv")
df["audit_time"] = pd.to_datetime(df["audit_time"])
df["voucher_synced_at"] = pd.to_datetime(df["voucher_synced_at"])
df["lag"] = (df["audit_time"] - df["voucher_synced_at"]).dt.total_seconds() / 60
df["is_fail"] = (df["audit_result"] == 0).astype(int)

print("=" * 70)
print("A. 负滞后(lag<=0) 排查")
print("=" * 70)
neg = df[df["lag"] <= 0]
print(f"lag<=0 记录 {len(neg)} 条，失败 {neg['is_fail'].sum()} 条（失败率 {neg['is_fail'].mean()*100:.1f}%）")
print("\n负滞后的分布（分钟，越负=凭证同步时间越晚于稽核发起）:")
print(neg["lag"].describe().round(2).to_string())
print("\n按负滞后大小分桶:")
nb = pd.cut(-neg["lag"], bins=[-1e-9, 0, 1, 5, 10, 30, 1e9], labels=["0", "(0,1]", "(1,5]", "(5,10]", "(10,30]", ">30"])
t = neg.groupby(nb, observed=True).agg(条数=("audit_id", "count"), 失败=("is_fail", "sum"))
print(t.to_string())
print("\n负滞后 按 branch_name:")
print(neg["branch_name"].value_counts().to_string())
print("\n负滞后 按 business_category:")
print(neg["business_category"].value_counts().to_string())

print()
print("=" * 70)
print("B. 主结论稳健性：lag 各桶的失败率与超额失败率")
print("=" * 70)
bins = [-1e9, 0, 15, 30, 45, 60, 1e9]
labels = ["<=0", "0-15", "15-30", "30-45", "45-60", ">60"]
df["bkt"] = pd.cut(df["lag"], bins=bins, labels=labels)
g = df.groupby("bkt", observed=True).agg(条数=("audit_id", "count"), 失败=("is_fail", "sum"))
g["失败率%"] = (g["失败"] / g["条数"] * 100).round(1)
baseline = df[df["lag"].between(0, 15)]["is_fail"].mean()
print(f"基准(新鲜0-15min)失败率 = {baseline*100:.2f}%")
g["超额失败率(pp)"] = ((g["失败"] / g["条数"] - baseline) * 100).round(1)
print(g.to_string())

print()
print("=" * 70)
print("C. 断点分析：lag>30 规则的盈亏平衡同步成本")
print("=" * 70)
m = df["lag"] > 30
fail_rate = df.loc[m, "is_fail"].mean()
excess = fail_rate - baseline
print(f"lag>30 失败率 = {fail_rate*100:.1f}%，超额 = {excess*100:.1f}pp")
print(f"每条同步可省的期望人工 = 超额 × 75 = {excess*75:.1f} min")
print(f"=> 盈亏平衡同步成本 = {excess*75:.1f} min（同步成本超过此值则规则转亏）")

print()
print("=" * 70)
print("D. 敏感性：同步成本变化时的最优阈值与净节省（人工成本固定 75min）")
print("=" * 70)
print(f"{'同步成本':>8}{'最优阈值':>10}{'净节省(min)':>14}")
for c_sync in [5, 10, 15, 20, 25, 30]:
    best = (-1e9, None)
    for T in [15, 30, 45, 60]:
        mT = df["lag"] > T
        nT = int(mT.sum())
        if nT == 0:
            continue
        excessT = df.loc[mT, "is_fail"].mean() - baseline
        net = nT * (excessT * 75 - c_sync)
        if net > best[0]:
            best = (net, T)
    print(f"{c_sync:>8}{best[1]:>10}{best[0]:>14.0f}")

print()
print("=" * 70)
print("E. 敏感性：人工成本变化时的最优阈值与净节省（同步成本固定 10min）")
print("=" * 70)
print(f"{'人工成本':>8}{'最优阈值':>10}{'净节省(min)':>14}")
for c_fail in [40, 60, 75, 100, 120]:
    best = (-1e9, None)
    for T in [15, 30, 45, 60]:
        mT = df["lag"] > T
        nT = int(mT.sum())
        if nT == 0:
            continue
        excessT = df.loc[mT, "is_fail"].mean() - baseline
        net = nT * (excessT * c_fail - 10)
        if net > best[0]:
            best = (net, T)
    print(f"{c_fail:>8}{best[1]:>10}{best[0]:>14.0f}")
