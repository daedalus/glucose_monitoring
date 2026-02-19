# Takes in sibionics gs1 sensor data
# Outputs an image with a circadian glucose profile including variability bands
# + TIR / TAR / TBR / ROC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

LOW = 70
HIGH = 180

# -------------------------
# 1) Read XLS file
# -------------------------
df = pd.read_excel(sys.argv[1])

# -------------------------
# 2) Clean & parse time
# -------------------------
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.dropna(subset=["Time", "Sensor Reading(mg/dL)"])
df = df.sort_values("Time")

# -------------------------
# 3) Compute ROC (mg/dL per minute)
# -------------------------
df["delta_glucose"] = df["Sensor Reading(mg/dL)"].diff()
df["delta_minutes"] = df["Time"].diff().dt.total_seconds() / 60.0
df["ROC"] = df["delta_glucose"] / df["delta_minutes"]

# -------------------------
# 4) TIR / TAR / TBR
# -------------------------
total = len(df)
tir = ((df["Sensor Reading(mg/dL)"] >= LOW) &
       (df["Sensor Reading(mg/dL)"] <= HIGH)).sum() / total * 100

tar = (df["Sensor Reading(mg/dL)"] > HIGH).sum() / total * 100
tbr = (df["Sensor Reading(mg/dL)"] < LOW).sum() / total * 100

# -------------------------
# 5) Seconds from midnight
# -------------------------
df["seconds"] = (
    df["Time"].dt.hour * 3600 +
    df["Time"].dt.minute * 60 +
    df["Time"].dt.second
)

# -------------------------
# 6) 5-minute bins
# -------------------------
bin_size = 300
df["bin"] = (df["seconds"] / bin_size).round().astype(int)

# -------------------------
# 7) Aggregate glucose stats
# -------------------------
result = df.groupby("bin")["Sensor Reading(mg/dL)"].agg(
    mean="mean",
    std="std",
    median="median",
    min="min",
    max="max",
    count="count"
).reset_index()

# Aggregate ROC
roc_profile = df.groupby("bin")["ROC"].mean().reset_index()
roc_profile = roc_profile.rename(columns={"ROC": "roc_mean"})

# Merge
result = result.merge(roc_profile, on="bin", how="left")

# Optional: remove bins with very few samples
min_samples = 5
result.loc[result["count"] < min_samples,
           ["mean", "std", "median", "min", "max", "roc_mean"]] = np.nan

result = result.sort_values("bin")
result["minutes"] = result["bin"] * 5

# -------------------------
# 8) Plot
# -------------------------
fig, ax1 = plt.subplots(figsize=(14, 7))

x = result["minutes"]

# Glucose axis
ax1.plot(x, result["mean"], linewidth=2, label="Mean")
ax1.plot(x, result["median"], linestyle="--", linewidth=1.5, label="Median")

ax1.fill_between(
    x,
    result["mean"] - result["std"],
    result["mean"] + result["std"],
    alpha=0.25,
    label="Mean ± Std"
)

ax1.fill_between(
    x,
    result["min"],
    result["max"],
    alpha=0.12,
    label="Min–Max"
)

ax1.axhline(LOW, linestyle=":", linewidth=1)
ax1.axhline(HIGH, linestyle=":", linewidth=1)

ax1.set_xlabel("Time of Day")
ax1.set_ylabel("Glucose (mg/dL)")
ax1.grid(True)

# ROC axis
ax2 = ax1.twinx()
ax2.plot(x, result["roc_mean"], linestyle=":", linewidth=2, label="ROC (mg/dL/min)")
ax2.set_ylabel("Rate of Change (mg/dL/min)")

# Format X axis
xticks = np.arange(0, 1441, 120)
xtick_labels = [f"{int(t//60):02d}:00" for t in xticks]
ax1.set_xticks(xticks)
ax1.set_xticklabels(xtick_labels)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Add TIR/TAR/TBR text box
textstr = (
    f"TIR ({LOW}-{HIGH}): {tir:.1f}%\n"
    f"TAR (> {HIGH}): {tar:.1f}%\n"
    f"TBR (< {LOW}): {tbr:.1f}%"
)

plt.gcf().text(0.80, 0.75, textstr, fontsize=10,
               bbox=dict(boxstyle="round", alpha=0.2))


plt.title("Circadian Glucose Profile with Variability + ROC")
plt.tight_layout()
plt.savefig("average_daily_profile.png", dpi=300)
plt.show()
plt.close()

