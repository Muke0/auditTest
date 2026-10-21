"""补充分析：按机构/大类的失败率（含分母），以及滞后与失败率的单调关系。"""
import pandas as pd

df = pd.read_csv("audit_records_processed.csv")
df["audit_time"] = pd.to_datetime(df["audit_time"])
df["voucher_synced_at"] = pd.to_datetime(df["voucher_synced_at"])
df["lag_minutes"] = (df["audit_time"] - df["voucher_synced_at"]).dt.total_seconds() / 60

is_fail = df["audit_result"] == 0
is_empty = df["audit_result"].isna()
is_bad = is_fail | is_empty

def full_stats(g, col):
    return pd.DataFrame(
        {
            "总数": g.size(),
            "一致": (~is_bad[g.index]).sum() if False else g.apply(lambda i: 0),  # placeholder
        }
    )

# 更直接：按列分组后统计
def summarize(col):
    rows = []
    for name, grp in df.groupby(col):
        n = len(grp)
        fail = (grp["audit_result"] == 0).sum()
        empty = grp["audit_result"].isna().sum()
        rows.append(
            {
                col: name,
                "总数": n,
                "失败数": int(fail),
                "未回传数": int(empty),
                "失败率%": round(fail / n * 100, 1),
                "异常合计率%": round((fail + empty) / n * 100, 1),
                "平均滞后分钟": round(grp["lag_minutes"].mean(), 1),
                "中位滞后分钟": round(grp["lag_minutes"].median(), 1),
            }
        )
    return pd.DataFrame(rows).set_index(col)

print("=" * 80)
print("A. 按 branch_name（按 失败率 降序）")
print("=" * 80)
b = summarize("branch_name").sort_values("失败率%", ascending=False)
print(b.to_string())

print()
print("=" * 80)
print("B. 按 business_category（按 失败率 降序）")
print("=" * 80)
c = summarize("business_category").sort_values("失败率%", ascending=False)
print(c.to_string())

print()
print("=" * 80)
print("C. 滞后分桶 → 失败率（单调性检验）")
print("=" * 80)
bins = [-1e9, 0, 15, 30, 60, 1e9]
labels = ["<=0", "0-15", "15-30", "30-60", ">60"]
df["bucket"] = pd.cut(df["lag_minutes"], bins=bins, labels=labels)
t = (
    df.groupby("bucket", observed=True)
    .agg(
        总数=("audit_id", "count"),
        失败数=("audit_result", lambda s: (s == 0).sum()),
        未回传数=("audit_result", lambda s: s.isna().sum()),
        平均滞后=("lag_minutes", "mean"),
    )
)
t["失败率%"] = (t["失败数"] / t["总数"] * 100).round(1)
t["异常合计率%"] = ((t["失败数"] + t["未回传数"]) / t["总数"] * 100).round(1)
print(t.to_string(float_format=lambda x: f"{x:.1f}"))

print()
print("=" * 80)
print("D. 滞后整体分布（分钟）")
print("=" * 80)
print(df["lag_minutes"].describe().round(2).to_string())
print(
    "分位数: 25%%=%.1f 50%%=%.1f 75%%=%.1f 90%%=%.1f"
    % (
        df["lag_minutes"].quantile(0.25),
        df["lag_minutes"].quantile(0.50),
        df["lag_minutes"].quantile(0.75),
        df["lag_minutes"].quantile(0.90),
    )
)
