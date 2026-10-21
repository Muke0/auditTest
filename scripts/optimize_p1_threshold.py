"""P1：对「真实不匹配」分流规则做最优阈值优化。

范围：新鲜记录(lag<=30)，因为陈旧失败由同步解决、不属于「真实不匹配」。
规则：credit_balance <= c 或 invoice_tax_amount >= t  → 判「疑似真实失败」。
扫描所有 (c, t)，按 F1 与净节省选最优。

成本模型（假设，口径未给预检成本）：
  失败人工 = 75 min/条；分流预检 = 30 min/条（假设）。
  净节省 = TP*(75-30) - FP*30 = TP*45 - FP*30。
"""
import pandas as pd

df = pd.read_csv("audit_records_processed.csv")
df["audit_time"] = pd.to_datetime(df["audit_time"])
df["voucher_synced_at"] = pd.to_datetime(df["voucher_synced_at"])
df["lag"] = (df["audit_time"] - df["voucher_synced_at"]).dt.total_seconds() / 60
df["is_fail"] = (df["audit_result"] == 0).astype(int)

fresh = df[df["lag"] <= 30].copy()
n = len(fresh)
n_fail = int(fresh["is_fail"].sum())
print(f"新鲜记录 n={n}，真实失败 n={n_fail}，基准失败率 {n_fail/n*100:.1f}%")

cred = fresh["credit_balance"]
tax = fresh["invoice_tax_amount"]
y = fresh["is_fail"]

def metrics(mask):
    tp = int((mask & (y == 1)).sum())
    fp = int((mask & (y == 0)).sum())
    fn = int((~mask & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    net = tp * 45 - fp * 30
    return tp, fp, fn, prec, rec, f1, net

print("\n=== A. 单变量 credit_balance 阈值扫描 (规则 credit <= c) 前5 ===\n")
rows = []
for c in range(0, 31):
    tp, fp, fn, prec, rec, f1, net = metrics(cred <= c)
    rows.append((f1, c, tp, fp, prec, rec, net))
for f1, c, tp, fp, prec, rec, net in sorted(rows, reverse=True)[:5]:
    print(f"c={c:2d}: 命中{int((cred<=c).sum()):4d} TP={tp:2d} FP={fp:3d} 精确率={prec*100:4.1f}% 召回率={rec*100:4.1f}% F1={f1:.3f} 净省={net:5d}min")

print("\n=== B. 单变量 invoice_tax_amount 阈值扫描 (规则 tax >= t) 前5 ===\n")
rows = []
for t in range(0, 107):
    tp, fp, fn, prec, rec, f1, net = metrics(tax >= t)
    rows.append((f1, t, tp, fp, prec, rec, net))
for f1, t, tp, fp, prec, rec, net in sorted(rows, reverse=True)[:5]:
    print(f"t={t:3d}: 命中{int((tax>=t).sum()):4d} TP={tp:2d} FP={fp:3d} 精确率={prec*100:4.1f}% 召回率={rec*100:4.1f}% F1={f1:.3f} 净省={net:5d}min")

print("\n=== C. 联合规则 (credit<=c 或 tax>=t) 全扫描，按 F1 与净节省各取前5 ===\n")
by_f1 = []
by_net = []
for c in range(0, 31):
    for t in range(0, 107):
        tp, fp, fn, prec, rec, f1, net = metrics((cred <= c) | (tax >= t))
        by_f1.append((f1, c, t, tp, fp, prec, rec, net))
        by_net.append((net, c, t, tp, fp, prec, rec, f1))

print("--- 按 F1 前5 ---")
for f1, c, t, tp, fp, prec, rec, net in sorted(by_f1, reverse=True)[:5]:
    print(f"c={c:2d},t={t:3d}: TP={tp:2d} FP={fp:3d} 精确率={prec*100:4.1f}% 召回率={rec*100:4.1f}% F1={f1:.3f} 净省={net:5d}min")

print("\n--- 按净节省 前5 ---")
for net, c, t, tp, fp, prec, rec, f1 in sorted(by_net, reverse=True)[:5]:
    print(f"c={c:2d},t={t:3d}: TP={tp:2d} FP={fp:3d} 精确率={prec*100:4.1f}% 召回率={rec*100:4.1f}% F1={f1:.3f} 净省={net:5d}min")

print("\n=== D. 推荐规则 vs 原人工规则(credit<=5 或 tax>80) ===\n")
for label, mask in [
    ("原规则 credit<=5 或 tax>80", (cred <= 5) | (tax > 80)),
    ("原规则 credit<=5 或 tax>=80", (cred <= 5) | (tax >= 80)),
]:
    tp, fp, fn, prec, rec, f1, net = metrics(mask)
    print(f"{label}: 命中{int(mask.sum()):4d} TP={tp:2d} FP={fp:3d} 精确率={prec*100:4.1f}% 召回率={rec*100:4.1f}% 净省={net:5d}min")
