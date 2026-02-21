import os
import sys
import pandas as pd
import numpy as np

_READERS = {
    ".xlsx": lambda p: pd.read_excel(p, engine="openpyxl"),
    ".xls":  lambda p: pd.read_excel(p, engine="xlrd"),
    ".ods":  lambda p: pd.read_excel(p, engine="odf"),
    ".csv":  lambda p: pd.read_csv(p),
}


def _read_input(input_file: str) -> pd.DataFrame:
    """Dispatch to the correct pandas reader based on file extension."""
    ext = os.path.splitext(input_file)[1].lower()
    reader = _READERS.get(ext)
    if reader is None:
        supported = ", ".join(sorted(_READERS))
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: {supported}"
        )
    return reader(input_file)


def load_and_preprocess(input_file, cfg, verbose=False):
    """Load data file (xlsx/xls/csv/ods), validate columns, parse datetimes, deduplicate, compute ROC.

    Returns a cleaned DataFrame with an added 'ROC' column.
    """
    if verbose:
        print(f"Loading data from: {input_file}")
        print(f"Glucose thresholds: Low={cfg['LOW']}, High={cfg['HIGH']}, "
              f"Tight={cfg['TIGHT_LOW']}-{cfg['TIGHT_HIGH']}")

    try:
        df = _read_input(input_file)
        if verbose:
            print(f"Successfully loaded {len(df)} rows from {input_file}")
    except ValueError:
        raise
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
