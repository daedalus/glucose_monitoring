import sys
import pandas as pd
import numpy as np


def load_and_preprocess(input_file, cfg, verbose=False):
    """Load Excel file, validate columns, parse datetimes, deduplicate, compute ROC.

    Returns a cleaned DataFrame with an added 'ROC' column.
    """
    if verbose:
        print(f"Loading data from: {input_file}")
        print(f"Glucose thresholds: Low={cfg['LOW']}, High={cfg['HIGH']}, "
              f"Tight={cfg['TIGHT_LOW']}-{cfg['TIGHT_HIGH']}")

    try:
        df = pd.read_excel(input_file)
        if verbose:
            print(f"Successfully loaded {len(df)} rows from {input_file}")
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        sys.exit(1)

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

    # Rate of Change (ROC)
    df["delta_glucose"] = glucose.diff()
    df["delta_minutes"] = df["Time"].diff().dt.total_seconds() / 60.0
    df["ROC"] = df["delta_glucose"] / df["delta_minutes"]
    df.loc[df["delta_minutes"] <= 0, "ROC"] = np.nan
    df["ROC"] = df["ROC"].clip(-cfg['ROC_CLIP'], cfg['ROC_CLIP'])

    return df
