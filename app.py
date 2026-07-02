"""
app.py
Aplikasi demo peramalan harga Bitcoin (BTC-USD).
Model: Prophet dan LSTM. Menampilkan prediksi 30 hari ke depan.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Prediksi Harga Bitcoin", page_icon="chart", layout="wide")

# ------------------- Load model (sekali, di-cache) -------------------
@st.cache_resource
def load_all():
    with open("prophet_model.pkl", "rb") as f:
        prophet = pickle.load(f)
    lstm = load_model("lstm_model.keras")
    with open("lstm_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return prophet, lstm, scaler, meta

prophet, lstm, scaler, meta = load_all()
history = pd.DataFrame(meta["history"])
history["Date"] = pd.to_datetime(history["Date"])
last_price = meta["last_price"]
window = meta["window"]
horizon = meta["horizon"]

# ------------------- Sidebar -------------------
st.sidebar.title("Pengaturan")
model_choice = st.sidebar.radio("Pilih model peramalan:", ["Prophet", "LSTM", "Bandingkan Keduanya"])
n_days = st.sidebar.slider("Jumlah hari peramalan:", 7, 30, 30)
st.sidebar.markdown("---")
st.sidebar.caption("Data historis: BTC-USD harian 2023 sampai 2026.")

# ------------------- Fungsi prediksi -------------------
def predict_prophet(days):
    future = prophet.make_future_dataframe(periods=days)
    fc = prophet.predict(future)
    tail = fc.tail(days)
    return tail["ds"].values, tail["yhat"].values

def predict_lstm(days):
    seq = np.array(meta["ret_scaled_tail"]).reshape(1, window, 1)
    pred_scaled = lstm.predict(seq, verbose=0)[0]
    pred_logret = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    prices = last_price * np.exp(np.cumsum(pred_logret))
    prices = prices[:days]
    start = pd.to_datetime(meta["last_date"]) + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=days)
    return dates.values, prices

def hitung_error(dates, preds):
    """Bandingkan prediksi dengan data aktual pada tanggal yang sama (periode uji)."""
    actual = history.set_index("Date")["Close"]
    pred = pd.Series(preds, index=pd.to_datetime(dates))
    common = pred.index.intersection(actual.index)
    if len(common) == 0:
        return None
    a, p = actual.loc[common], pred.loc[common]
    mape = float(np.mean(np.abs((a - p) / a)) * 100)
    return {"mape": mape, "n": len(common)}

# ------------------- Header -------------------
st.title("Prediksi Harga Penutupan Bitcoin (BTC-USD)")
st.markdown("Aplikasi ini meramalkan harga Bitcoin menggunakan model **Prophet** dan **LSTM**.")

col1, col2 = st.columns(2)
col1.metric("Harga terakhir (data latih)", f"${last_price:,.2f}")
col2.metric("Horizon peramalan", f"{n_days} hari")

# ------------------- Plot -------------------
fig = go.Figure()
hist_tail = history.tail(120)
fig.add_trace(go.Scatter(x=hist_tail["Date"], y=hist_tail["Close"],
                         name="Historis", line=dict(color="gray")))

results = {}
if model_choice in ["Prophet", "Bandingkan Keduanya"]:
    d, y = predict_prophet(n_days)
    results["Prophet"] = (d, y)
    fig.add_trace(go.Scatter(x=d, y=y, name="Prophet",
                             line=dict(color="blue", dash="dash")))
if model_choice in ["LSTM", "Bandingkan Keduanya"]:
    d, y = predict_lstm(n_days)
    results["LSTM"] = (d, y)
    fig.add_trace(go.Scatter(x=d, y=y, name="LSTM",
                             line=dict(color="green", dash="dash")))

fig.update_layout(title="Peramalan Harga Bitcoin",
                  xaxis_title="Tanggal", yaxis_title="Harga (USD)",
                  hovermode="x unified", height=500)
st.plotly_chart(fig, use_container_width=True)

# ------------------- Kartu akurasi -------------------
st.subheader("Seberapa Meleset Prediksi dari Data Aktual?")
err_cols = st.columns(len(results))
for col, (name, (d, y)) in zip(err_cols, results.items()):
    err = hitung_error(d, y)
    if err is None:
        col.metric(f"MAPE {name}", "N/A")
        col.caption("Tidak ada data aktual pada periode prediksi.")
    else:
        col.metric(f"MAPE {name}", f"±{err['mape']:.2f}%")
        col.caption(
            f"Rata-rata prediksi meleset {err['mape']:.2f}% dari harga aktual "
            f"(bisa di atas maupun di bawah), dihitung dari {err['n']} hari data uji."
        )

# ------------------- Tabel hasil -------------------
st.subheader("Detail Hasil Peramalan")
actual_by_date = history.set_index("Date")["Close"]
for name, (d, y) in results.items():
    st.markdown(f"**Model {name}**")
    dates = pd.to_datetime(d)
    aktual = actual_by_date.reindex(dates)
    selisih_pct = (pd.Series(y, index=dates) - aktual) / aktual * 100
    tabel = pd.DataFrame({
        "Tanggal": dates.strftime("%d %b %Y"),
        "Prediksi Harga (USD)": [f"${v:,.2f}" for v in y],
        "Harga Aktual (USD)": [
            f"${v:,.2f}" if pd.notna(v) else "-" for v in aktual
        ],
        "Meleset (%)": [
            f"{v:+.2f}%" if pd.notna(v) else "-" for v in selisih_pct
        ],
    })
    st.dataframe(tabel, use_container_width=True, height=250)
st.caption(
    "Meleset (%): positif berarti prediksi di atas harga aktual, "
    "negatif berarti di bawah. Tanda '-' berarti data aktual belum tersedia."
)

st.markdown("---")
st.caption("Catatan: Prophet menjadi model terbaik pada evaluasi (MAPE 6,37 persen). "
           "Peramalan harga kripto bersifat tidak pasti dan tidak boleh dijadikan saran investasi.")
