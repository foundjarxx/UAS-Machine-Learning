"""Fungsi utilitas untuk membersihkan teks ulasan."""

import re

STOPWORDS = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "untuk",
    "pada",
    "dengan",
    "ini",
    "itu",
    "adalah",
    "atau",
    "juga",
    "saja",
    "sih",
    "lah",
    "dong",
    "nya",
    "ya",
    "sudah",
    "akan",
    "sebagai",
    "oleh",
    "dalam",
    "karena",
    "sebuah",
    "para",
    "tersebut",
    "agar",
    "saya",
    "aku",
    "kami",
    "kita",
    "anda",
    "dia",
    "mereka",
}

# Kata negasi dan penentu sentimen sengaja tidak dibuang.
SLANG = {
    "bgt": "banget",
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "yg": "yang",
    "sgt": "sangat",
    "bagu": "bagus",
    "kecwa": "kecewa",
    "aja": "saja",
    "udah": "sudah",
}


def clean_text(text: str) -> str:
    """Normalisasi teks dari ulasan menjadi bentuk yang siap diproses model."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = [SLANG.get(token, token) for token in text.split()]
    tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    return " ".join(tokens)
