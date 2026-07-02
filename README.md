# Prediksi Harga Bitcoin (BTC-USD) - LSTM vs Prophet

Aplikasi web peramalan harga penutupan Bitcoin menggunakan dua model deret waktu: Prophet dan LSTM. Bagian dari tugas forecasting.

## Alur Kerja Sistem

Sistem terdiri dari tiga komponen utama:

1. **Data (`btc_3y.csv`)**
   Data historis harga harian BTC-USD periode 2023 sampai 2026, diambil dari yfinance.

2. **Pelatihan Model (`train_models.py`)**
   Melatih kedua model dan menyimpannya ke file:
   - Prophet dilatih pada harga asli, disimpan ke `prophet_model.pkl`.
   - LSTM dilatih pada log return (direct multi-step 30 hari), disimpan ke `lstm_model.keras` beserta `lstm_scaler.pkl`.
   - Metadata dan data historis disimpan ke `meta.pkl` untuk dipakai aplikasi.
   Skrip ini dijalankan sekali saja sebelum deployment.

3. **Aplikasi Web (`app.py`)**
   Antarmuka Streamlit. Memuat model yang sudah dilatih (tidak melatih ulang), menerima input pilihan model dan jumlah hari, lalu menampilkan hasil peramalan dalam bentuk grafik interaktif dan tabel.

Peran tiap komponen: data menyediakan input mentah, skrip pelatihan menghasilkan model siap pakai, aplikasi menyajikan prediksi ke pengguna.

## Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
python train_models.py     # jalankan sekali, menghasilkan file model
streamlit run app.py       # membuka aplikasi di localhost:8501
```

## Deploy ke Streamlit Cloud (Live Online)

1. Push seluruh file ke repository GitHub (termasuk file model hasil `train_models.py`).
2. Buka https://share.streamlit.io, login dengan GitHub.
3. Klik "New app", pilih repository, branch, dan file `app.py`.
4. Klik Deploy. Aplikasi akan online dengan URL publik.

Catatan: file model (`.pkl`, `.keras`) harus ikut di-push ke GitHub agar aplikasi tidak perlu melatih ulang saat online.

## Hasil Evaluasi

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| LSTM | 9.451,62 | 11.671,22 | 14,52% |
| Prophet | 4.243,34 | 5.035,60 | 6,37% |

Prophet menjadi model terbaik dengan MAPE 6,37 persen, memenuhi target di bawah 10 persen.
