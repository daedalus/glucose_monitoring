# Takes in sibionics gs1 sensor data
# Outputs an image with a circadian glucose profile.

import pandas as pd
import sys

# 1) Read XLS file
df = pd.read_excel(sys.argv[1])

# 2) Convert Time to datetime
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

# Drop invalid columns
df = df.dropna(subset=["Time", "Sensor Reading(mg/dL)"])

# 3) Calculate seconds from midnight
df["seconds"] = (
    df["Time"].dt.hour * 3600 +
    df["Time"].dt.minute * 60 +
    df["Time"].dt.second
)

# 4) make bins of 5 minutes (300 secs)
bin_size = 300
df["bin"] = (df["seconds"] / bin_size).round().astype(int)

"""
# 5) average by bin
result = df.groupby("bin")["Sensor Reading(mg/dL)"].mean().reset_index()

# 6) Convert bin to Time
result["time_of_day"] = pd.to_timedelta(result["bin"] * bin_size, unit="s")

# Ordenar
result = result.sort_values("bin")
"""

result = df.groupby("bin")["Sensor Reading(mg/dL)"].agg(
    mean="mean",
    std="std",
    median="median",
    min="min",
    max="max"
).reset_index()

result = result.sort_values("bin")
result["minutes"] = result["bin"] * 5


# 7) Optional: convert to HH:MM
result["time_of_day"] = result["time_of_day"].dt.components.apply(
    lambda x: f"{int(x.hours):02d}:{int(x.minutes):02d}", axis=1
)

print(result[["time_of_day", "Sensor Reading(mg/dL)"]])


import matplotlib.pyplot as plt
import numpy as np

# Convert time_of_day to minutes
result["minutes"] = result["bin"] * 5

"""
# Create graphic
plt.figure(figsize=(12, 6))
plt.plot(result["minutes"], result["Sensor Reading(mg/dL)"])

# Format X axis as HH:MM
xticks = np.arange(0, 1441, 120)  # Each 2 hours
xtick_labels = [f"{int(x//60):02d}:00" for x in xticks]

plt.xticks(xticks, xtick_labels)

plt.xlabel("Time of Day")
plt.ylabel("Average Glucose (mg/dL)")
plt.title("Average Daily Glucose Profile (5-minute bins)")
plt.grid(True)

plt.tight_layout()
plt.show()

output_file = "average_daily_profile.png"
plt.savefig(output_file, dpi=300)

plt.close()
"""

plt.figure(figsize=(12, 6))

x = result["minutes"]

mean = result["mean"]
std = result["std"]
median = result["median"]
minv = result["min"]
maxv = result["max"]

# Mean line
plt.plot(x, mean, linewidth=2, label="Mean")

# Median line
plt.plot(x, median, linestyle="--", linewidth=1.5, label="Median")

# Std band (mean ± std)
plt.fill_between(
    x,
    mean - std,
    mean + std,
    alpha=0.2,
    label="Mean ± Std"
)

# Min–Max envelope
plt.fill_between(
    x,
    minv,
    maxv,
    alpha=0.1,
    label="Min–Max"
)

# Format X axis
xticks = np.arange(0, 1441, 120)
xtick_labels = [f"{int(t//60):02d}:00" for t in xticks]
plt.xticks(xticks, xtick_labels)

plt.xlabel("Time of Day")
plt.ylabel("Glucose (mg/dL)")
plt.title("Circadian Glucose Profile with Variability")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("average_daily_profile.png", dpi=300)
plt.show()
plt.close()

