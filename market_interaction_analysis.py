import pandas as pd

df = pd.read_csv("outputs/events_with_market_context.csv")

# -------------------------------
# 1. Drop strength buckets
# -------------------------------
df["drop_bucket_simple"] = pd.cut(
    df["drop_pct"],
    bins=[0, 0.05, 0.07, 0.10, 1],
    labels=["3-5%", "5-7%", "7-10%", "10%+"]
)

# -------------------------------
# 2. Interaction: drop × regime
# -------------------------------
interaction = (
    df.groupby(["drop_bucket_simple", "risk_regime"])
    .agg(
        total_events=("ticker", "count"),
        success_rate=("success", "mean"),
        avg_return=("final_return", "mean"),
        avg_drawdown=("max_drawdown", "mean"),
    )
    .reset_index()
)

print("\n=== DROP BUCKET × MARKET REGIME ===")
print(interaction.to_string(index=False))

# -------------------------------
# 3. Interaction: drop × BAD tags
# -------------------------------
bad_tags = ["rates_up", "valuation_reset", "growth_fear"]

df["has_bad_tag"] = df["primary_tags_list"].apply(
    lambda x: any(tag in x for tag in bad_tags)
)

bad_interaction = (
    df.groupby(["drop_bucket_simple", "has_bad_tag"])
    .agg(
        total_events=("ticker", "count"),
        success_rate=("success", "mean"),
        avg_return=("final_return", "mean"),
        avg_drawdown=("max_drawdown", "mean"),
    )
    .reset_index()
)

print("\n=== DROP BUCKET × BAD TAG ===")
print(bad_interaction.to_string(index=False))

# -------------------------------
# 4. Extreme filter test
# -------------------------------
extreme_bad = df[df["has_bad_tag"]]

print("\n=== BAD TAG ONLY ===")
print({
    "count": len(extreme_bad),
    "success_rate": extreme_bad["success"].mean(),
    "avg_return": extreme_bad["final_return"].mean(),
})

# -------------------------------
# 5. GOOD tag interaction
# -------------------------------
good_tags = ["oversold_rebound", "liquidity_stress", "ai_leadership"]

df["has_good_tag"] = df["primary_tags_list"].apply(
    lambda x: any(tag in x for tag in good_tags)
)

good_interaction = (
    df.groupby(["drop_bucket_simple", "has_good_tag"])
    .agg(
        total_events=("ticker", "count"),
        success_rate=("success", "mean"),
        avg_return=("final_return", "mean"),
    )
    .reset_index()
)

print("\n=== DROP BUCKET × GOOD TAG ===")
print(good_interaction.to_string(index=False))