"""验证 credit_balance/invoice_tax_amount 是否解释剩余失败，并量化「先同步再稽核」的 ROI。"""
import pandas as pd
import numpy as np

df = pd.read_csv("audit_records_processed.csv")
df["audit_time"] = pd.to_datetime(df["audit_time"])
df["voucher_synced_at"] = pd.to_datetime(df["voucher_synced_at"])
df["lag"] = (df["audit_time"] - df["voucher_synced_at"]).dt.total_seconds() / 60
df["gap"] = df["credit_balance"] - df["invoice_tax_amount"]
df["is_fail"] = (df["audit_result"] == 0).astype(int)
df["is_empty"] = df["audit_result"].isna().astype(int)
df["result_label"] = df["audit_result"].apply(
    lambda r: "未回传" if pd.isna(r) else ("一致" if r == 1 else "失败")
)

print("=" * 80)
print("A. credit_balance 分桶 → 失败率 / 未回传率")
print("=" * 80)
cb = pd.cut(df["credit_balance"], bins=[-1, 5, 10, 15, 20, 25, 31], labels=["0-5", "6-10", "11-15", "16-20", "21-25", "26-30"])
a = df.groupby(cb, observed=True).agg(总数=("audit_id", "count"), 失败率=("is_fail", "mean"), 未回传率=("is_empty", "mean"), 平均滞后=("lag", "mean"))
a["失败率"] = (a["失败率"] * 100).round(1)
a["未回传率"] = (a["未回传率"] * 100).round(1)
a["平均滞后"] = a["平均滞后"].round(1)
print(a.to_string())

print()
print("=" * 80)
print("B. invoice_tax_amount 分桶 → 失败率 / 未回传率")
print("=" * 80)
tb = pd.cut(df["invoice_tax_amount"], bins=[-1, 30, 45, 60, 80, 107], labels=["0-30", "31-45", "46-60", "61-80", "81-106"])
b = df.groupby(tb, observed=True).agg(总数=("audit_id", "count"), 失败率=("is_fail", "mean"), 未回传率=("is_empty", "mean"), 平均滞后=("lag", "mean"))
b["失败率"] = (b["失败率"] * 100).round(1)
b["未回传率"] = (b["未回传率"] * 100).round(1)
b["平均滞后"] = b["平均滞后"].round(1)
print(b.to_string())

print()
print("=" * 80)
print("C. 未回传(空) 记录画像 vs 全量")
print("=" * 80)
empty = df[df["is_empty"] == 1]
print(f"未回传条数: {len(empty)}，其平均滞后 %.1f min（全量 %.1f min）" % (empty["lag"].mean(), df["lag"].mean()))
print("未回传 按 branch_name 分布:")
print(empty["branch_name"].value_counts().to_string())
print("未回传 按 business_category 分布:")
print(empty["business_category"].value_counts().to_string())
print("未回传 的 audit_time 是否集中在某些时段:")
print("未回传记录 audit_time 最早=%s 最晚=%s" % (df["audit_time"].min(), df["audit_time"].max()))
print(empty["audit_time"].dt.hour.value_counts().sort_index().to_string())

print()
print("=" * 80)
print("D. 失败是否被「滞后」之外的因素解释：lag 与 credit 的联合")
print("=" * 80)
df["stale"] = df["lag"] > 30
joint = df.pivot_table(index="stale", columns="credit_balance", values="is_fail", aggfunc="mean") * 100
print("失败率(%), 行=是否滞后>30min, 列=credit_balance:")
print(joint.round(1).to_string())

print()
print("=" * 80)
print("E. ROI：按「滞后>阈值即先同步(10min)」测算净节省")
print("=" * 80)
BASELINE_FAIL_RATE = df[df["lag"].between(0, 15)]["is_fail"].mean()  # 新鲜数据基准失败率
print(f"新鲜数据(0-15min)基准失败率 = {BASELINE_FAIL_RATE*100:.1f}%")
print(f"当前失败总数 = {int(df['is_fail'].sum())}，人工成本 = {int(df['is_fail'].sum()) * 75} 分钟")
print()
print(f"{'阈值':<8}{'触发同步条数':>12}{'当前失败数':>12}{'预期残留失败':>14}{'可避免失败':>12}{'省人工(min)':>12}{'同步成本(min)':>14}{'净节省(min)':>12}")
for thr in [15, 30, 45, 60]:
    mask = df["lag"] > thr
    n_sync = int(mask.sum())
    cur_fail = int(df.loc[mask, "is_fail"].sum())
    expect_fail = n_sync * BASELINE_FAIL_RATE
    avoid = cur_fail - expect_fail
    save = avoid * 75
    cost = n_sync * 10
    net = save - cost
    print(f"{thr:<8}{n_sync:>12}{cur_fail:>12}{expect_fail:>14.1f}{avoid:>12.1f}{save:>12.0f}{cost:>14}{net:>12.0f}")

print()
print("=" * 80)
print("F. 若对所有滞后>30min 的请求统一先同步，明细")
print("=" * 80)
m = df["lag"] > 30
print(f"滞后>30min 记录: {int(m.sum())} 条，其中当前失败 {int(df.loc[m,'is_fail'].sum())} 条")
print(f"同步后按基准失败率仅剩 {m.sum()*BASELINE_FAIL_RATE:.1f} 条失败，可避免 {int(df.loc[m,'is_fail'].sum()) - m.sum()*BASELINE_FAIL_RATE:.1f} 条")
