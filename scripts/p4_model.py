"""P4：精准拦截模型 vs 固定 lag>30 规则。

用 numpy 手写逻辑回归（无 sklearn 依赖），时间切分 train/test，
比较「lag 单特征」与「lag+credit+tax+类别+机构」全模型的拦截效果。

目标：预测 audit_result==0（失败）。未回传(空) 记录不参与本模型。
指标：AUC + 召回率@同步数 曲线 + 与固定规则的头对头比较。
"""
import pandas as pd
import numpy as np

df = pd.read_csv("audit_records_processed.csv")
df["audit_time"] = pd.to_datetime(df["audit_time"])
df["voucher_synced_at"] = pd.to_datetime(df["voucher_synced_at"])
df["lag"] = (df["audit_time"] - df["voucher_synced_at"]).dt.total_seconds() / 60
df["is_fail"] = (df["audit_result"] == 0).astype(int)

# 只保留 0/1（剔除未回传）
m = df["audit_result"].notna()
d = df[m].copy()
d = d.sort_values("audit_time").reset_index(drop=True)

# 时间切分：前 80% 训练，后 20% 测试
cut = int(len(d) * 0.8)
tr, te = d.iloc[:cut], d.iloc[cut:]
print(f"训练 {len(tr)} (失败 {tr['is_fail'].sum()}) / 测试 {len(te)} (失败 {te['is_fail'].sum()})")

# one-hot 类别/机构
cat_cols = ["branch_name", "business_category"]
dummies = pd.get_dummies(d[cat_cols], drop_first=True).astype(float)


def build_features(Xd, cont_cols, cont_mean=None, cont_std=None):
    """连续特征标准化 + one-hot。返回矩阵与标准化参数。"""
    Xc = Xd[cont_cols].values.astype(float)
    if cont_mean is None:
        cont_mean = Xc.mean(axis=0)
        cont_std = Xc.std(axis=0) + 1e-9
    Xc = (Xc - cont_mean) / cont_std
    Xd_dum = dummies.loc[Xd.index].values
    return np.hstack([Xc, Xd_dum]), cont_mean, cont_std


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_fit(X, y, lr=0.5, iters=2000, l2=1e-2):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = sigmoid(z)
        g = (p - y) / n
        w -= lr * (X.T @ g + l2 * w)
        b -= lr * g.sum()
    return w, b


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def auc(y, s):
    order = np.argsort(s)
    ys = y[order]
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = np.where(ys == 1)[0] + 1
    return (ranks.sum() - npos * (npos + 1) / 2) / (npos * nneg)


CONT = ["lag", "credit_balance", "invoice_tax_amount"]

# 基线：lag 单特征
X_tr1, mu, sd = build_features(tr, ["lag"])
X_te1, _, _ = build_features(te, ["lag"], mu, sd)
w1, b1 = logistic_fit(X_tr1, tr["is_fail"].values)
s1 = predict_proba(X_te1, w1, b1)

# 全模型
X_tr, _, _ = build_features(tr, CONT)
X_te, _, _ = build_features(te, CONT, mu, sd)
w, b = logistic_fit(X_tr, tr["is_fail"].values)
s = predict_proba(X_te, w, b)

y_te = te["is_fail"].values
print("\n=== 测试集 AUC（越高越好，0.5=随机）===")
print(f"lag 单特征:      AUC={auc(y_te, s1):.3f}")
print(f"全模型(全部特征): AUC={auc(y_te, s):.3f}")

# 召回率@同步数 曲线
print("\n=== 召回率 vs 同步数（测试集）===")
print(f"{'同步数':>6}{'lag规则抓到失败':>14}{'全模型抓到失败':>14}")
F_te = int(y_te.sum())
order1 = np.argsort(-s1)  # lag 降序
order = np.argsort(-s)    # 全模型降序
for K in [50, 100, 150, 200, 286, 400]:
    if K > len(y_te):
        K = len(y_te)
    c1 = int(y_te[order1[:K]].sum())
    c = int(y_te[order[:K]].sum())
    print(f"{K:>6}{c1:>14}{c:>14}")

# 头对头：固定 lag>30 规则 vs 模型同预算
print("\n=== 头对头：测试集上，相同同步预算下的净节省 ===")
lag_rule = te["lag"] > 30
K_rule = int(lag_rule.sum())
c_rule = int(y_te[lag_rule].sum())
c_model = int(y_te[order[:K_rule]].sum())

def net(caught, synced):
    return caught * 75 - synced * 10  # 相对“什么都不做”的净节省

print(f"固定规则 lag>30:  同步 {K_rule} 条, 抓到失败 {c_rule}/{F_te}, 净节省 {net(c_rule, K_rule)} min")
print(f"全模型同预算:      同步 {K_rule} 条, 抓到失败 {c_model}/{F_te}, 净节省 {net(c_model, K_rule)} min")

# 全模型最优工作点：扫阈值，最大化净节省
print("\n=== 全模型最优工作点（按测试集风险分降序，扫同步数）===")
best = (-1e9, None)
for K in range(0, len(y_te) + 1, 5):
    c = int(y_te[order[:K]].sum())
    v = net(c, K)
    if v > best[0]:
        best = (v, K, c)
print(f"最优：同步 {best[1]} 条, 抓到失败 {best[2]}/{F_te}, 净节省 {best[0]:.0f} min")
print(f"对照：固定 lag>30 规则净节省 {net(c_rule, K_rule)} min")
