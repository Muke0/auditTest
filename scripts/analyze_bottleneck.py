"""分析稽核结果与 branch_name / business_category / 同步滞后 的关系。

- 同步滞后 lag_minutes = audit_time - voucher_synced_at（分钟，正=凭证数据较旧）。
- 只看 audit_result 为 0（异常需人工）或空（未回传）的记录，并按机构/大类统计。
- 对比不同 audit_result 组的同步滞后分布，判断是否相关。
"""
import pandas as pd

df = pd.read_csv("audit_records_processed.csv")

# 时区已统一为 +08:00，直接相减
df["audit_time"] = pd.to_datetime(df["audit_time"])
df["voucher_synced_at"] = pd.to_datetime(df["voucher_synced_at"])
df["lag_minutes"] = (df["audit_time"] - df["voucher_synced_at"]).dt.total_seconds() / 60

# 结果标签
def label(r):
    if pd.isna(r):
        return "空(未回传)"
    return "1(一致)" if r == 1 else "0(异常)"

df["result_label"] = df["audit_result"].apply(label)

print("=" * 70)
print("一、audit_result 总体分布")
print("=" * 70)
print(df["result_label"].value_counts())
print(f"失败+未回传合计: {(df['result_label'] != '1(一致)').sum()} / {len(df)}")

print()
print("=" * 70)
print("二、失败(0)与未回传(空) —— 按 branch_name 统计")
print("=" * 70)
bad = df[df["result_label"] != "1(一致)"]
by_branch = (
    bad.groupby("branch_name")
    .agg(
        失败=("audit_result", lambda s: (s == 0).sum()),
        未回传=("audit_result", lambda s: s.isna().sum()),
        合计=("audit_id", "count"),
        平均滞后分钟=("lag_minutes", "mean"),
    )
    .sort_values("合计", ascending=False)
)
print(by_branch.to_string(float_format=lambda x: f"{x:.1f}"))

print()
print("=" * 70)
print("三、失败(0)与未回传(空) —— 按 business_category 统计")
print("=" * 70)
by_cat = (
    bad.groupby("business_category")
    .agg(
        失败=("audit_result", lambda s: (s == 0).sum()),
        未回传=("audit_result", lambda s: s.isna().sum()),
        合计=("audit_id", "count"),
        平均滞后分钟=("lag_minutes", "mean"),
    )
    .sort_values("合计", ascending=False)
)
print(by_cat.to_string(float_format=lambda x: f"{x:.1f}"))

print()
print("=" * 70)
print("四、同步滞后 lag_minutes 按结果分组对比")
print("=" * 70)
lag_stat = (
    df.groupby("result_label")["lag_minutes"]
    .agg(条数="count", 均值="mean", 中位数="median", 最小="min", 最大="max", 正滞后占比=lambda s: (s > 0).mean())
)
print(lag_stat.to_string(float_format=lambda x: f"{x:.2f}"))

print()
print("=" * 70)
print("五、滞后分桶 × 结果 交叉表（判断失败是否随滞后增大而增多）")
print("=" * 70)
bins = [-1e9, 0, 15, 30, 60, 120, 240, 1e9]
labels = ["<=0", "0-15", "15-30", "30-60", "60-120", "120-240", ">240"]
df["lag_bucket"] = pd.cut(df["lag_minutes"], bins=bins, labels=labels)
ct = pd.crosstab(df["lag_bucket"], df["result_label"], normalize="index") * 100
print(ct.to_string(float_format=lambda x: f"{x:.1f}%"))
