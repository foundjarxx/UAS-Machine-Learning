# Aplikasi Klasifikasi Data Teks — Analisis Sentimen Bahasa Indonesia

Proyek UAS Machine Learning dengan fokus pada klasifikasi sentimen teks ulasan produk.

## Struktur folder
- `data/dataset_ulasan.csv` — dataset ulasan berlabel (`teks`, `label`)
- `preprocessing.py` — fungsi pembersihan dan normalisasi teks
- `train.py` — pelatihan dan perbandingan beberapa model klasifikasi
- `app.py` — antarmuka web Streamlit untuk demo prediksi
- `model/` — model terlatih dan file metrik evaluasi

## Alur kerja
1. Dataset dibaca dari `data/dataset_ulasan.csv`
2. Teks dibersihkan menggunakan `clean_text()`
3. Teks diubah menjadi vektor TF-IDF
4. Model klasifikasi dilatih dan dibandingkan
5. Model terbaik disimpan ke `model/model.joblib`
6. Hasil prediksi ditampilkan melalui Streamlit

## Cara menjalankan
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Aplikasi akan terbuka di http://localhost:8501

## Model yang digunakan
- TF-IDF dengan n-gram unigram + bigram
- Model yang dibandingkan: Multinomial Naive Bayes, Linear SVM, Logistic Regression
- Model terbaik dipilih berdasarkan skor macro-F1

