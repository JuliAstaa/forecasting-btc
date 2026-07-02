# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Streamlit web app forecasting Bitcoin (BTC-USD) daily close, comparing two time-series models: **Prophet** and **LSTM**. UI text and docs are Indonesian; keep that language for user-facing strings. Python 3.12.

## Commands

```bash
pip install -r requirements.txt   # dependencies (pinned versions — Prophet + TF are strict)
python train_models.py            # run ONCE — trains models, writes artifact files
streamlit run app.py              # serve app at localhost:8501
```

No test suite, no linter configured.

## Architecture

Three-stage pipeline; artifacts are the contract between stages.

1. **`btc_3y.csv`** — yfinance export, 2023–2026 daily BTC-USD. Non-standard header: real column names in row 1, rows 2–3 are junk (`Ticker`, blank `Date`), so it is loaded with `skiprows=[1, 2]` and `Price` column renamed to `Date`. Any code reading this CSV must replicate that.

2. **`train_models.py`** — trains offline, dumps artifacts (must be committed to git so Streamlit Cloud can load them):
   - `prophet_model.pkl` — Prophet fit on raw prices.
   - `lstm_model.keras` + `lstm_scaler.pkl` — LSTM predicts **log returns** (not prices), StandardScaler'd, direct multi-step (one shot outputs full 30-day `HORIZON` vector, no recursion).
   - `meta.pkl` — bundles `last_price`, `window`(60), `horizon`(30), `ret_scaled_tail` (last 60 scaled returns to seed inference), `last_date`, and full `history` price series for plotting.
   - Last `TEST_SIZE`=30 days held out as test set (train/test split only; eval metrics live in README, not computed here).

3. **`app.py`** — loads artifacts via `@st.cache_resource`, never retrains. Sidebar picks model + horizon (7–30 days). Renders Plotly chart + tables.

### LSTM inference reconstruction (key detail)

LSTM outputs scaled **log returns**, not prices. `predict_lstm()` reverses the training transform:
`scaler.inverse_transform` → `last_price * np.exp(np.cumsum(pred_logret))`.
Break this chain and prices go wrong silently. Prophet by contrast predicts prices directly (`make_future_dataframe` → `yhat`).

## Constraints when editing

- `WINDOW=60` / `HORIZON=30` must match between `train_models.py` and the shapes in `meta.pkl`; the app trusts `meta["window"]`. Changing either requires retraining.
- After any change to `train_models.py`, artifacts are stale — rerun it, then commit the regenerated `.pkl`/`.keras` files (Streamlit Cloud deploy loads them as-is, does not train).
- Dependency versions are pinned tight because Prophet/TensorFlow break across minor versions.
