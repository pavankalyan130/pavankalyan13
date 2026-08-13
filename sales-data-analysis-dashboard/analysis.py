"""Sales analysis using a small reproducible sample dataset."""
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt

CSV = """date,region,category,revenue
2025-01-05,Hyderabad,Electronics,24500
2025-01-12,Bengaluru,Home,12800
2025-02-03,Hyderabad,Home,15400
2025-02-11,Chennai,Electronics,27300
2025-03-08,Bengaluru,Electronics,22100
2025-03-19,Chennai,Home,18700
2025-04-02,Hyderabad,Electronics,31600
2025-04-17,Bengaluru,Home,14200
"""

sales = pd.read_csv(StringIO(CSV), parse_dates=["date"])
sales["month"] = sales["date"].dt.strftime("%b")

total_revenue = sales["revenue"].sum()
top_region = sales.groupby("region")["revenue"].sum().idxmax()
top_category = sales.groupby("category")["revenue"].sum().idxmax()

print(f"Total revenue: ₹{total_revenue:,.0f}")
print(f"Top region: {top_region}")
print(f"Top category: {top_category}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sales.groupby("month", sort=False)["revenue"].sum().plot(
    kind="bar", ax=axes[0], title="Monthly Revenue", color="#0A66C2"
)
sales.groupby("region")["revenue"].sum().sort_values().plot(
    kind="barh", ax=axes[1], title="Revenue by Region", color="#17A589"
)
for ax in axes:
    ax.set_ylabel("Revenue (₹)")
plt.tight_layout()
plt.savefig("sales_dashboard.png", dpi=160)
print("Saved chart: sales_dashboard.png")
