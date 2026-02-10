import pandas as pd

RUN_DIR = "reports/base_fast_full"
OOS_PATH = f"{RUN_DIR}/reports/oos_daily.parquet"

oos = pd.read_parquet(OOS_PATH).sort_index()

# Strategy equity (net of costs already)
oos["strategy_equity"] = (1.0 + oos["net_ret"].fillna(0.0)).cumprod()

# Buy & Hold equity on the same dates
# (use close returns; assumes close is in the stitched table)
bh_ret = oos["close"].pct_change().fillna(0.0)
oos["buyhold_equity"] = (1.0 + bh_ret).cumprod()

out = oos[["close", "net_ret", "gross_ret", "strategy_equity", "buyhold_equity"]]

OUT_CSV = f"{RUN_DIR}/reports/equity_compare.csv"
out.to_csv(OUT_CSV)

print("loaded:", OOS_PATH)
print("saved :", OUT_CSV)
print("range :", out.index.min().date(), "->", out.index.max().date())
print("final strategy:", float(out["strategy_equity"].iloc[-1]))
print("final buyhold :", float(out["buyhold_equity"].iloc[-1]))
