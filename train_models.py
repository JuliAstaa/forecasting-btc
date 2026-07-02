"""
train_models.py
Jalankan sekali untuk melatih dan menyimpan model Prophet dan LSTM.
Output: prophet_model.pkl, lstm_model.keras, lstm_scaler.pkl, meta.pkl
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from prophet import Prophet

TEST_SIZE = 30
WINDOW, HORIZON = 60, 30

# 1. Load data
df = pd.read_csv("btc_3y.csv", skiprows=[1, 2], parse_dates=["Price"])
df = df.rename(columns={"Price": "Date"}).set_index("Date").sort_index()

train = df["Close"][:-TEST_SIZE]
test = df["Close"][-TEST_SIZE:]

# 2. Train Prophet
print("Melatih Prophet...")
tp = train.reset_index()
tp.columns = ["ds", "y"]
m_prophet = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.05,
)
m_prophet.fit(tp)
with open("prophet_model.pkl", "wb") as f:
    pickle.dump(m_prophet, f)

# 3. Train LSTM (direct multi-step + log return)
print("Melatih LSTM...")
train_logret = np.log(train).diff().dropna()
scaler = StandardScaler()
ret_scaled = scaler.fit_transform(train_logret.values.reshape(-1, 1))
last_price = float(train.iloc[-1])

def create_sequences_multi(data, window, horizon):
    X, y = [], []
    for i in range(window, len(data) - horizon + 1):
        X.append(data[i - window:i, 0])
        y.append(data[i:i + horizon, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences_multi(ret_scaled, WINDOW, HORIZON)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

model_lstm = Sequential([
    LSTM(64, return_sequences=False, input_shape=(WINDOW, 1)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(HORIZON),
])
model_lstm.compile(optimizer="adam", loss="mae")
es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
model_lstm.fit(X_train, y_train, epochs=150, batch_size=32,
               validation_split=0.1, callbacks=[es], verbose=1)
model_lstm.save("lstm_model.keras")

with open("lstm_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 4. Simpan metadata dan data historis untuk aplikasi
meta = {
    "last_price": last_price,
    "window": WINDOW,
    "horizon": HORIZON,
    "ret_scaled_tail": ret_scaled[-WINDOW:].tolist(),
    "last_date": str(train.index[-1].date()),
    "history": df["Close"].reset_index().assign(
        Date=lambda d: d["Date"].astype(str)).to_dict("list"),
}
with open("meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("Selesai. Model dan metadata tersimpan.")
