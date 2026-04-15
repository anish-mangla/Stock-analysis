import pandas as pd
import json

with open("market_weekly_context.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.to_csv("outputs/market_weekly_context.csv", index=False)