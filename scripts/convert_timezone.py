"""将 audit_records.csv 复制一份，并把 voucher_synced_at 从 UTC(+00:00) 转为 +08:00。

- 原文件 audit_records.csv 保持不变（源数据，不动）。
- 输出 audit_records_processed.csv：除 voucher_synced_at 外，其余字段与原文逐字符一致
  （含 audit_result 的空值「未回传」原样保留）。
- 时区语义：UTC 时刻换算到东八区，保留同一时刻（22:54:00Z -> 次日 06:54:00+08:00）。
"""
import csv
from datetime import datetime, timezone, timedelta

SRC = "audit_records.csv"
DST = "audit_records_processed.csv"
TZ_CN = timezone(timedelta(hours=8))


def utc_to_cn(s: str) -> str:
    """'2026-08-31T22:54:00Z' -> '2026-09-01T06:54:00+08:00'"""
    # Z 等价于 +00:00，统一解析后再转东八区
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt.astimezone(TZ_CN).strftime("%Y-%m-%dT%H:%M:%S+08:00")


def main() -> None:
    with open(SRC, newline="", encoding="utf-8") as fin, open(
        DST, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        n = 0
        for row in reader:
            row["voucher_synced_at"] = utc_to_cn(row["voucher_synced_at"])
            writer.writerow(row)
            n += 1
    print(f"wrote {n} rows -> {DST}")


if __name__ == "__main__":
    main()
