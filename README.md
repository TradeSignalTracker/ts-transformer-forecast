# Time Series Transformer Forecasting

This repository contains a script to forecast time series data (e.g., EURUSD prices) using a pre-trained Time Series Transformer model.

The model is loaded directly from Hugging Face’s model hub:

**Model:** [huggingface/time-series-transformer-tourism-monthly](https://huggingface.co/huggingface/time-series-transformer-tourism-monthly)

---

## Features

* Interactive input for start date or steps back for prediction window
* Scales input context dynamically for better prediction accuracy
* Visualizes historical prices, forecast, and actual future prices together
* Limits plotting to a configurable number of bars around prediction point for clarity

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install torch pandas numpy matplotlib transformers
```

---

## Usage

Run the forecasting script:

```bash
python predict.py
```

You will be prompted to enter:

* Start date for the forecast (YYYY-MM-DD), or press Enter to skip
* Number of records backward from dataset end if no date is entered

The script will then load the model, run prediction, and show the plot.

---
🔍 Disclaimer on Model Source

    ⚠️ Note: The base model used here is time-series-transformer-tourism-monthly, originally trained on monthly tourism datasets.
    While it can sometimes capture trends even in high-frequency financial data like EURUSD M15, the domain mismatch is significant. The model was never trained on financial time series, so its outputs should be treated as experimental and non-reliable for direct trading decisions.

    That said, the script remains a powerful sandbox — and with proper fine-tuning or model replacement, it can become a strong foundation for forecasting pipelines in finance.

## Notes

* The model will be downloaded and cached automatically by the `transformers` library.
* Requires GPU for best performance, but CPU also works.
