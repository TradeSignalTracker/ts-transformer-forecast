import os
import pathlib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values`.*")
from transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerForPrediction

os.environ["TRANSFORMERS_NO_TF"] = "1"

# --- Config ---
MODEL_PATH = pathlib.Path(__file__).parent / "model"
CSV_PATH = pathlib.Path(__file__).parent / "EURUSD15.csv"
COLUMN = "close"
MAX_BARS_AROUND_PREDICTION = 500  # Number of bars before and after the prediction point for plotting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_start_idx_by_date_or_steps(df_index, context_length, prediction_length, start_date=None, steps_back=None):
    """
    Determine the start index for prediction based on either a given date or
    a number of steps (bars) back from the dataset's end.
    Ensures index is within valid bounds.
    """
    max_start = len(df_index) - (context_length + prediction_length)
    if start_date is not None:
        dt = pd.to_datetime(start_date)
        possible_idx = df_index.searchsorted(dt, side='left')
        if possible_idx == len(df_index) or df_index[possible_idx] > dt:
            possible_idx -= 1
        start_idx = possible_idx
    elif steps_back is not None:
        start_idx = max_start - steps_back
    else:
        start_idx = max_start

    # Clamp start_idx to valid range
    start_idx = max(0, min(start_idx, max_start))
    return start_idx

def interactive_input(df_index, context_length, prediction_length):
    """
    Interactive user input to select the prediction start:
    - Either by date (YYYY-MM-DD)
    - Or by number of bars back from dataset end
    """
    print("Enter the prediction start date (YYYY-MM-DD), or press Enter to skip:")
    start_date = input().strip()
    if start_date == "":
        start_date = None

    if start_date is None:
        print("Enter how many bars back from the dataset end (integer), or press Enter for latest:")
        steps_back = input().strip()
        if steps_back == "":
            steps_back = 0
        else:
            try:
                steps_back = int(steps_back)
            except ValueError:
                print("Invalid number, defaulting to 0")
                steps_back = 0
    else:
        steps_back = None

    start_idx = get_start_idx_by_date_or_steps(df_index, context_length, prediction_length, start_date=start_date, steps_back=steps_back)
    return start_idx

# --- Load Data ---
df = pd.read_csv(CSV_PATH, header=None, sep="\t")
df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

values = df[COLUMN].astype("float32").values

# --- Load Model ---
model = TimeSeriesTransformerForPrediction.from_pretrained(
    str(MODEL_PATH.resolve()),
    local_files_only=True
).to(device)
model.eval()

# --- Model Parameters ---
max_lag = max(model.config.lags_sequence)
subsequences_length = model.config.context_length
prediction_length = model.config.prediction_length
context_length = max_lag + subsequences_length

if len(values) < context_length + prediction_length:
    raise ValueError(f"Not enough data: have {len(values)}, need {context_length + prediction_length}")

# --- Get start_idx via interactive input ---
start_idx = interactive_input(df.index, context_length, prediction_length)
print(f"Selected start_idx: {start_idx}, date: {df.index[start_idx]}")

# --- Select context and future data ---
context = values[start_idx : start_idx + context_length]
future_context = values[start_idx : start_idx + context_length + prediction_length]

# Scale only by context min and max for normalization
min_val, max_val = context.min(), context.max()
scaled_context = (context - min_val) / (max_val - min_val)

past_values = torch.tensor(scaled_context, dtype=torch.float32).unsqueeze(0).to(device)

# --- Prepare past time features ---
timestamps = df.index[start_idx : start_idx + context_length]
dow = timestamps.dayofweek.values
dow_1h = np.eye(7)[dow]
doy = timestamps.dayofyear.values
doy_sin = np.sin(2 * np.pi * doy / 365)
doy_cos = np.cos(2 * np.pi * doy / 365)
past_time_features_np = np.concatenate([dow_1h, doy_sin[:, None], doy_cos[:, None]], axis=1)
past_time_features = torch.tensor(past_time_features_np, dtype=torch.float32).unsqueeze(0).to(device)

past_observed_mask = torch.ones_like(past_values)

# --- Prepare future time features ---
step = timestamps[-1] - timestamps[-2]
future_timestamps = pd.date_range(start=timestamps[-1] + step, periods=prediction_length, freq=step)
dow_f = future_timestamps.dayofweek.values
dow_1h_f = np.eye(7)[dow_f]
doy_f = future_timestamps.dayofyear.values
doy_sin_f = np.sin(2 * np.pi * doy_f / 365)
doy_cos_f = np.cos(2 * np.pi * doy_f / 365)
future_time_features_np = np.concatenate([dow_1h_f, doy_sin_f[:, None], doy_cos_f[:, None]], axis=1)
future_time_features = torch.tensor(future_time_features_np, dtype=torch.float32).unsqueeze(0).to(device)

# --- Generate forecast ---
with torch.no_grad():
    outputs = model.generate(
        past_values=past_values,
        past_time_features=past_time_features,
        past_observed_mask=past_observed_mask,
        future_time_features=future_time_features,
    )

forecast_scaled = outputs.sequences.squeeze().cpu().numpy()
if forecast_scaled.ndim > 1:
    forecast_scaled = forecast_scaled[-1]

print("Forecast scaled range:", forecast_scaled.min(), forecast_scaled.max())

# Unscale forecast back to original price scale
forecast = forecast_scaled * (max_val - min_val) + min_val

# --- Plotting with limited bars around prediction ---
plot_start = max(start_idx + context_length - MAX_BARS_AROUND_PREDICTION, 0)
plot_end = min(start_idx + context_length + prediction_length + MAX_BARS_AROUND_PREDICTION, len(values))

x_full = np.arange(plot_start, plot_end)
y_full = values[plot_start:plot_end]

forecast_x = np.arange(start_idx + context_length, start_idx + context_length + prediction_length)
forecast_y = forecast

real_future_y = values[start_idx + context_length : start_idx + context_length + prediction_length]

plt.figure(figsize=(14, 7))
plt.plot(x_full, y_full, label="History (price)", color="blue")
plt.plot(forecast_x, forecast_y, label="Forecast", color="orange")
plt.plot(forecast_x, real_future_y, label="Real price (future)", color="green", linestyle="dashed")
plt.axvline(start_idx + context_length - 1, color="gray", linestyle="--", label="Forecast start")
plt.title("Comparison of History, Forecast, and Real Future Price")
plt.xlabel("Index in dataset")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
