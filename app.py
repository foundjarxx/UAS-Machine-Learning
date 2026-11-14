"""Streamlit UI untuk klasifikasi sentimen teks.

Cara menjalankan:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import clean_text

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "model" / "model.joblib"
METRICS_PATH = ROOT_DIR / "model" / "metrics.json"

EMOJI = {
    "positive": "😊",
    "negative": "😠",
    "neutral": "😐",
}


@st.cache_resource
def load_model():
    """Muat model yang sudah dilatih."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    """Muat metrik evaluasi dari file JSON."""
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def softmax_from_logits(logits):
    """Konversi nilai keputusan model menjadi probabilitas."""
    logits = np.asarray(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def predict(model, texts):
    """Preprocessing teks, lalu menghasilkan label dan probabilitas."""
    cleaned_texts = [clean_text(text) for text in texts]
    classifier = model.named_steps["clf"]
    vectorizer = model.named_steps["tfidf"]
    features = vectorizer.transform(cleaned_texts)

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)
    else:
        scores = classifier.decision_function(features)
        probabilities = softmax_from_logits(scores)

    return classifier.classes_, probabilities


st.set_page_config(page_title="Klasifikasi Sentimen Teks", page_icon="💬", layout="centered")

model = load_model()
metrics = load_metrics()

st.title("💬 Klasifikasi Sentimen Teks")
st.caption(f"Proyek UAS Machine Learning · TF-IDF + {metrics['best_model']}")

tab_kategori, tab_csv, tab_eval, tab_about = st.tabs(
    ["Klasifikasi Teks", "Klasifikasi File CSV", "Evaluasi Model", "Tentang"]
)

with tab_kategori:
    contoh_teks = {
        "(ketik sendiri)": "",
        "Positif": "makanan di restoran ini enak banget, pelayanan nya ramah dan tempat nya nyaman. pasti balik lagi.",
        "Negatif": "gue kecewa banget sama pelayanan nya. tempat nya kotor dan makanan nya tidak enak. tidak mau lagi ke sana.",
        "Netral": "kemarin gue pergi ke mall baru di dekat rumah. tempat nya cukup luas dan banyak toko.",
    }

    pilihan = st.selectbox("Contoh kalimat", list(contoh_teks.keys()))
    teks = st.text_area(
        "Masukkan teks:",
        value=contoh_teks[pilihan],
        height=130,
        placeholder="Tulis teks di sini...",
    )

    if st.button("Klasifikasikan", type="primary"):
        if not teks.strip():
            st.warning("Teks tidak boleh kosong.")
        else:
            classes, probabilities = predict(model, [teks])
            index_terbaik = int(probabilities[0].argmax())
            label_terbaik = classes[index_terbaik]
            confidence = probabilities[0][index_terbaik] * 100

            st.success(
                f"Hasil prediksi: **{label_terbaik.upper()}** {EMOJI.get(label_terbaik, '')} "
                f"(keyakinan {confidence:.1f}%)"
            )
            st.bar_chart(pd.DataFrame({"Probabilitas": probabilities[0]}, index=classes))

            with st.expander("Teks setelah preprocessing"):
                st.code(clean_text(teks) or "(kosong)")

with tab_csv:
    st.write("Unggah file CSV dengan kolom **teks**.")
    uploaded_file = st.file_uploader("File CSV", type="csv")

    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        if "teks" not in dataframe.columns:
            st.error("Kolom 'teks' tidak ditemukan.")
        else:
            classes, probabilities = predict(model, dataframe["teks"].fillna("").tolist())
            dataframe["prediksi"] = classes[probabilities.argmax(axis=1)]
            dataframe["keyakinan_%"] = (probabilities.max(axis=1) * 100).round(1)

            st.dataframe(dataframe, use_container_width=True)
            st.bar_chart(dataframe["prediksi"].value_counts())

            st.download_button(
                "Unduh hasil (CSV)",
                dataframe.to_csv(index=False).encode("utf-8"),
                "hasil_prediksi.csv",
                "text/csv",
            )

with tab_eval:
    st.subheader("Perbandingan Model")
    metrics_df = pd.DataFrame(metrics["results"]).set_index("model")
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader(f"Confusion Matrix · {metrics['best_model']}")
    confusion_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"aktual {label}" for label in metrics["labels"]],
        columns=[f"prediksi {label}" for label in metrics["labels"]],
    )
    st.dataframe(confusion_df, use_container_width=True)
    st.caption(f"Data latih: {metrics['n_train']} · Data uji: {metrics['n_test']}")

with tab_about:
    st.markdown(
        """
        **Alur sistem:** teks → preprocessing (lowercase, hapus simbol/URL, normalisasi slang, stopword) → TF-IDF (unigram + bigram) → klasifikator → label sentimen.

        **Kelas:** positive · negative · neutral

        **Pelatihan ulang:** ganti `data/dataset_ulasan.csv` (kolom `teks`, `label`) lalu jalankan `python train.py`.
        """
    )
