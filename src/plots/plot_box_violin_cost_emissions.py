import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # you can pip install seaborn if not already

# Load per-origin summary
per_origin = pd.read_csv("data/processed/ev_outputs/per_origin_summary.csv")

# Focus on deltas (cost and emissions differences between cheapest and cleanest)
# You may want to filter to a specific price_basis, e.g. retailA_c_per_kwh
basis = "retailA_c_per_kwh"
df = per_origin[per_origin["price_basis"] == basis]

# --- Boxplot of cost delta ---
plt.figure(figsize=(6,4), dpi=150)
sns.boxplot(y="cost_delta_AUD", data=df, color="lightblue")
plt.axhline(0, color="black", linestyle="--")
plt.ylabel("Cost difference (AUD, cheapest – cleanest)")
plt.title("Distribution of cost deltas across origins")
plt.tight_layout()
plt.savefig("fig_box_cost_delta.png")
plt.close()

# --- Boxplot of emissions delta ---
plt.figure(figsize=(6,4), dpi=150)
sns.boxplot(y="emissions_delta_kg", data=df, color="lightgreen")
plt.axhline(0, color="black", linestyle="--")
plt.ylabel("Emissions difference (kg CO₂, cheapest – cleanest)")
plt.title("Distribution of emissions savings across origins")
plt.tight_layout()
plt.savefig("fig_box_emissions_delta.png")
plt.close()

# --- Violin plots (if you want to see density) ---
plt.figure(figsize=(6,4), dpi=150)
sns.violinplot(y="cost_delta_AUD", data=df, inner="quartile", color="lightblue")
plt.axhline(0, color="black", linestyle="--")
plt.ylabel("Cost difference (AUD, cheapest – cleanest)")
plt.title("Distribution of cost deltas across origins (violin)")
plt.tight_layout()
plt.savefig("fig_violin_cost_delta.png")
plt.close()

plt.figure(figsize=(6,4), dpi=150)
sns.violinplot(y="emissions_delta_kg", data=df, inner="quartile", color="lightgreen")
plt.axhline(0, color="black", linestyle="--")
plt.ylabel("Emissions difference (kg CO₂, cheapest – cleanest)")
plt.title("Distribution of emissions savings across origins (violin)")
plt.tight_layout()
plt.savefig("fig_violin_emissions_delta.png")
plt.close()
