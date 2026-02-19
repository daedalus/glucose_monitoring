# Unified Robust AGP + GMI + AUC + Full Clinical Metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import timedelta

LOW = 70
HIGH = 180
BIN_MINUTES = 5
MIN_SAMPLES_PER_BIN = 5
ROC_CLIP = 10  # mg/dL/min physiological guardrail

# --------------------------------------------------
# 1) Read Data
# --------------------------------------------------
df = pd.read_excel(sys.argv[1])

required = ["Time", "Sensor Reading(mg/dL)"]
if not all(col in df.columns for col in required):
    raise ValueError("Missing required columns.")

df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.dropna(subset=required)
df = df.sort_values("Time").drop_duplicates(subset=["Time"])

if len(df) == 0:
    raise ValueError("No valid glucose data found.")

# Filter physiologically impossible values
glucose = df["Sensor Reading(mg/dL)"].astype(float)
glucose = glucose.clip(20, 600)  # Physiological limits
df["Sensor Reading(mg/dL)"] = glucose

# --------------------------------------------------
# 2) Rate of Change (ROC)
# --------------------------------------------------
df["delta_glucose"] = glucose.diff()
df["delta_minutes"] = df["Time"].diff().dt.total_seconds() / 60.0
df["ROC"] = df["delta_glucose"] / df["delta_minutes"]
df.loc[df["delta_minutes"] <= 0, "ROC"] = np.nan
df["ROC"] = df["ROC"].clip(-ROC_CLIP, ROC_CLIP)

# --------------------------------------------------
# 3) Core Metrics
# --------------------------------------------------
mean_glucose = glucose.mean()
std_glucose = glucose.std(ddof=0)
cv_percent = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else np.nan

# GMI
gmi = 3.31 + (0.02392 * mean_glucose)

# J-Index (combines mean and variability)
j_index = 0.001 * (mean_glucose + std_glucose) ** 2

# --------------------------------------------------
# 4) MAGE (smoothed)
# --------------------------------------------------
def compute_mage(series):
    s = series.rolling(3, center=True).mean().dropna().values
    if len(s) < 3:
        return np.nan

    sd = np.std(s, ddof=0)
    diffs = np.diff(s)
    signs = np.sign(diffs)

    turning = []
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1] and signs[i] != 0:
            turning.append(i)

    excursions = []
    for i in range(1, len(turning)):
        delta = abs(s[turning[i]] - s[turning[i - 1]])
        if delta > sd:
            excursions.append(delta)

    return np.mean(excursions) if excursions else np.nan

mage = compute_mage(glucose)

# --------------------------------------------------
# 5) MODD (Mean of Daily Differences)
# --------------------------------------------------
def compute_modd(df):
    """Calculate Mean of Daily Differences - day-to-day glucose variability"""
    df_daily = df.copy()
    df_daily['date'] = df_daily['Time'].dt.date
    df_daily['time'] = df_daily['Time'].dt.time
    
    # Create time index for matching across days
    df_daily['time_seconds'] = (
        df_daily['Time'].dt.hour * 3600 +
        df_daily['Time'].dt.minute * 60 +
        df_daily['Time'].dt.second
    )
    
    # Pivot to get glucose by time across days
    pivot = df_daily.pivot_table(
        values='Sensor Reading(mg/dL)',
        index='time_seconds',
        columns='date',
        aggfunc='mean'
    )
    
    # Calculate differences between consecutive days
    diffs = []
    dates = sorted(pivot.columns)
    for i in range(len(dates) - 1):
        day1 = pivot[dates[i]].dropna()
        day2 = pivot[dates[i + 1]].dropna()
        
        # Find common time points
        common_times = day1.index.intersection(day2.index)
        if len(common_times) > 0:
            day_diff = np.abs(day1.loc[common_times] - day2.loc[common_times])
            diffs.extend(day_diff.dropna().values)
    
    return np.mean(diffs) if diffs else np.nan

modd = compute_modd(df)

# --------------------------------------------------
# 6) ADRR (Average Daily Risk Range)
# --------------------------------------------------
def compute_adrr(values, dates):
    """Calculate Average Daily Risk Range"""
    def risk_function(g):
        # Convert glucose to risk score
        g = np.clip(g, 20, 600)
        return 10 * (1.509 * ((np.log(g)) ** 1.084 - 5.381)) ** 2
    
    # Group by date
    df_risk = pd.DataFrame({'glucose': values, 'date': dates})
    daily_risk_range = []
    
    for date, group in df_risk.groupby('date'):
        if len(group) >= 12:  # At least 12 readings per day (every 2 hours)
            risks = risk_function(group['glucose'].values)
            daily_range = np.max(risks) - np.min(risks)
            daily_risk_range.append(daily_range)
    
    return np.mean(daily_risk_range) if daily_risk_range else np.nan

adrr = compute_adrr(glucose, df['Time'].dt.date)

# --------------------------------------------------
# 7) CONGA
# --------------------------------------------------
def compute_conga(df, lag_minutes=60, tolerance=5):
    df_temp = df.set_index("Time")
    target_index = df_temp.index - pd.Timedelta(minutes=lag_minutes)

    lagged = df_temp["Sensor Reading(mg/dL)"].reindex(
        target_index,
        method="nearest",
        tolerance=pd.Timedelta(minutes=tolerance)
    )

    diff = df_temp["Sensor Reading(mg/dL)"].values - lagged.values
    return np.nanstd(diff, ddof=0)

conga = compute_conga(df)

# --------------------------------------------------
# 8) Risk Indices
# --------------------------------------------------
def compute_risk_indices(values):
    g = np.clip(values.values, 1, None)
    f = 1.509 * ((np.log(g) ** 1.084) - 5.381)
    risk = 10 * (f ** 2)

    lbgi = np.mean(risk[f < 0]) if np.any(f < 0) else 0
    hbgi = np.mean(risk[f > 0]) if np.any(f > 0) else 0
    return lbgi, hbgi

lbgi, hbgi = compute_risk_indices(glucose)

# --------------------------------------------------
# 9) TIR / TAR / TBR with levels
# --------------------------------------------------
total = len(glucose)

# Time in Range
tir = ((glucose >= LOW) & (glucose <= HIGH)).sum() / total * 100

# Time Above Range (with levels)
tar_level1 = ((glucose > HIGH) & (glucose <= 250)).sum() / total * 100  # Level 1: 181-250
tar_level2 = (glucose > 250).sum() / total * 100  # Level 2: >250
tar = tar_level1 + tar_level2

# Time Below Range (with levels)
tbr_level2 = (glucose < 54).sum() / total * 100  # Level 2: <54 (clinically serious)
tbr_level1 = ((glucose >= 54) & (glucose < LOW)).sum() / total * 100  # Level 1: 54-69
tbr = tbr_level1 + tbr_level2

# --------------------------------------------------
# 10) AUC Metrics
# --------------------------------------------------
df_auc = df.sort_values("Time").copy()
df_auc["time_minutes"] = (
    (df_auc["Time"] - df_auc["Time"].iloc[0]).dt.total_seconds() / 60.0
)

times = df_auc["time_minutes"].values
values = df_auc["Sensor Reading(mg/dL)"].values

auc_total = np.trapezoid(values, times)
auc_high = np.trapezoid(np.maximum(values - HIGH, 0), times)
auc_low = np.trapezoid(np.maximum(LOW - values, 0), times)

# --------------------------------------------------
# 11) Data Quality Metrics
# --------------------------------------------------
days_of_data = (df['Time'].max() - df['Time'].min()).days
hours_of_data = (df['Time'].max() - df['Time'].min()).total_seconds() / 3600
readings_per_day = len(df) / days_of_data if days_of_data > 0 else len(df)

# --------------------------------------------------
# 12) Circadian Binning
# --------------------------------------------------
df["seconds"] = (
    df["Time"].dt.hour * 3600 +
    df["Time"].dt.minute * 60 +
    df["Time"].dt.second
)

df["bin"] = (df["seconds"] // (BIN_MINUTES * 60)).astype(int)

grouped = df.groupby("bin")["Sensor Reading(mg/dL)"]

result = grouped.agg(
    count="count",
    p5=lambda x: np.percentile(x, 5),
    p10=lambda x: np.percentile(x, 10),
    p25=lambda x: np.percentile(x, 25),
    median="median",
    mean="mean",
    p75=lambda x: np.percentile(x, 75),
    p90=lambda x: np.percentile(x, 90),
    p95=lambda x: np.percentile(x, 95),
).reset_index()

roc_profile = df.groupby("bin")["ROC"].mean().reset_index()
roc_profile.rename(columns={"ROC": "roc_mean"}, inplace=True)

result = result.merge(roc_profile, on="bin", how="left")

result.loc[result["count"] < MIN_SAMPLES_PER_BIN,
           result.columns.difference(["bin", "count"])] = np.nan

result = result.sort_values("bin")
result["minutes"] = result["bin"] * BIN_MINUTES

# --------------------------------------------------
# 13) Plot AGP with Internal Metrics Box
# --------------------------------------------------
fig, ax1 = plt.subplots(figsize=(16, 9))  # Slightly larger figure for better fit
x = result["minutes"]

# Add target zone highlighting (70-180)
ax1.axhspan(70, 180, alpha=0.1, color='green', label='Target Range (70-180)')

ax1.fill_between(x, result["p5"], result["p95"], alpha=0.15, label="5–95%")
ax1.fill_between(x, result["p25"], result["p75"], alpha=0.35, label="IQR")
ax1.plot(x, result["median"], linewidth=2.5, label="Median")
ax1.plot(x, result["mean"], linestyle="--", linewidth=1.5, label="Mean")

ax1.axhline(LOW, linestyle=":", linewidth=1, color='red', alpha=0.5)
ax1.axhline(HIGH, linestyle=":", linewidth=1, color='red', alpha=0.5)

# Shade nighttime hours (10pm - 6am)
ax1.axvspan(22*60, 24*60, alpha=0.05, color='gray', label='Night Hours')
ax1.axvspan(0, 6*60, alpha=0.05, color='gray')

ax1.set_xlabel("Time of Day", fontsize=12)
ax1.set_ylabel("Glucose (mg/dL)", fontsize=12)
ax1.grid(True, alpha=0.3)

# ROC axis
ax2 = ax1.twinx()
ax2.plot(x, result["roc_mean"], linestyle=":", linewidth=2,
         label="ROC (mg/dL/min)", color='orange')
ax2.set_ylabel("Rate of Change (mg/dL/min)", fontsize=12)

xticks = np.arange(0, 1441, 120)
ax1.set_xticks(xticks)
ax1.set_xticklabels([f"{int(t//60):02d}:00" for t in xticks])

# Set y-axis limits to leave space at top for metrics box
current_ylim = ax1.get_ylim()
ax1.set_ylim(current_ylim[0], current_ylim[1] * 1.15)  # Add 15% headroom

# --------------------------------------------------
# Internal Metrics Box (Positioned inside plot)
# --------------------------------------------------
textstr = (
    f"TIME IN RANGE\n"
    f"TIR (70-180): {tir:.1f}%\n"
    f"TAR >180: {tar:.1f}% (>{HIGH}-250: {tar_level1:.1f}%, >250: {tar_level2:.1f}%)\n"
    f"TBR <70: {tbr:.1f}% (54-69: {tbr_level1:.1f}%, <54: {tbr_level2:.1f}%)\n\n"
    f"GLUCOSE STATS\n"
    f"Mean: {mean_glucose:.1f} mg/dL\n"
    f"GMI: {gmi:.2f}%\n"
    f"CV: {cv_percent:.1f}%\n"
    f"J-Index: {j_index:.1f}\n\n"
    f"VARIABILITY\n"
    f"MAGE: {mage:.1f}\n"
    f"MODD: {modd:.1f}\n"
    f"CONGA(1h): {conga:.1f}\n\n"
    f"RISK\n"
    f"LBGI: {lbgi:.2f}\n"
    f"HBGI: {hbgi:.2f}\n"
    f"ADRR: {adrr:.1f}\n\n"
    f"AUC\n"
    f"Total: {auc_total:.0f}\n"
    f">{HIGH}: {auc_high:.0f}\n"
    f"<{LOW}: {auc_low:.0f}\n\n"
    f"DATA QUALITY\n"
    f"Days: {days_of_data:.1f}\n"
    f"Readings/day: {readings_per_day:.0f}"
)

# Position the text box in the upper right corner of the plot
# Use axes coordinates (0 to 1) for reliable positioning
plt.gcf().text(0.78, 0.88, textstr, fontsize=9,
               bbox=dict(boxstyle="round", facecolor='white', alpha=0.55,
                        edgecolor='gray', linewidth=1),
               verticalalignment='top',
               horizontalalignment='left',
               transform=ax1.transAxes)  # Use axes coordinates

# Adjust legend position to avoid overlapping with metrics box
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

plt.title("Ambulatory Glucose Profile (Full Clinical Version)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("agp_profile_full.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Print warning if insufficient data
if days_of_data < 5:
    print(f"Warning: Only {days_of_data:.1f} days of data. AGP typically requires ≥5 days for reliability.")
if readings_per_day < 24:
    print(f"Warning: Low reading frequency ({readings_per_day:.0f} readings/day). Continuous glucose monitor expected.")
