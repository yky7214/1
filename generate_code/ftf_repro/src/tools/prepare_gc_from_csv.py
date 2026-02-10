# tools/prepare_gc_from_csv.py
import os
import pandas as pd

CSV_PATH = r"C:\Users\Yukiya\Desktop\GOLDstrategy\DeepCode\deepcode_lab\papers\1\GC_in_daily_new.csv"
OUT_PATH = r"ftf_repro\data\processed\gc_continuous.parquet"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

df = pd.read_csv(CSV_PATH)
# あなたのCSVは datetime / open / high / low / close / volume になってる
df["datetime"] = pd.to_datetime(df["datetime"])
df["date"] = df["datetime"].dt.normalize()

df = df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")

# 必須列（この3つは絶対）
keep = ["close", "high", "low"]
# あれば残す（任意）
for c in ["open", "volume"]:
    if c in df.columns:
        keep.append(c)

df = df[keep].astype(float)

# 念のため：NaN行を落とす（最低 close/high/low が揃う行だけ）
df = df.dropna(subset=["close", "high", "low"])

df.to_parquet(OUT_PATH)
print("saved:", OUT_PATH)
print("range:", df.index.min().date(), "->", df.index.max().date())
print("rows:", len(df))
