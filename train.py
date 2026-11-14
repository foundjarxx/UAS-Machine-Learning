"""Latih dan bandingkan beberapa model klasifikasi teks."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import clean_text

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "dataset_ulasan.csv"
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

MODEL_REGISTRY = {
    "Multinomial Naive Bayes": MultinomialNB,
    "Linear SVM": lambda: LinearSVC(class_weight="balanced"),
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Baca dataset, validasi kolom yang dibutuhkan, dan bersihkan teks."""
    dataframe = pd.read_csv(path)
    required_columns = {"teks", "label"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset kurang kolom: {missing_text}")

    dataframe = dataframe.dropna(subset=["teks", "label"]).copy()
    dataframe["teks"] = dataframe["teks"].astype(str)
    dataframe["bersih"] = dataframe["teks"].apply(clean_text)
    return dataframe


def build_pipeline(estimator):
    """Membuat pipeline TF-IDF + model klasifikasi."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", estimator),
        ]
    )


def evaluate_model(model_name: str, estimator_factory, X_train, y_train, X_test, y_test):
    """Latih model, evaluasi CV dan data uji, lalu kembalikan hasilnya."""
    estimator = estimator_factory()
    pipeline = build_pipeline(estimator)
    cv_accuracy = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(y_test, predictions)

    result = {
        "model": model_name,
        "cv_acc": round(cv_accuracy, 4),
        "test_acc": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1_score, 4),
        "predictions": predictions,
        "pipeline": pipeline,
    }

    print(f"\n=== {model_name} ===")
    print(f"CV acc: {cv_accuracy:.4f} | Test acc: {accuracy:.4f}")
    print(classification_report(y_test, predictions, zero_division=0))
    return result


def save_model_and_metrics(best_result, labels, X_train, X_test, y_test, results):
    """Simpan model terbaik dan metrik evaluasi ke folder model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_pipeline = best_result["pipeline"]
    best_predictions = best_result["predictions"]
    confusion = confusion_matrix(y_test, best_predictions, labels=labels)

    joblib.dump(best_pipeline, MODEL_PATH)

    metrics = {
        "best_model": best_result["model"],
        "labels": labels,
        "results": [
            {
                "model": result["model"],
                "cv_acc": result["cv_acc"],
                "test_acc": result["test_acc"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            }
            for result in results
        ],
        "confusion_matrix": confusion.tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)


def main():
    """Proses utama pelatihan dan evaluasi model."""
    dataset = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["bersih"],
        dataset["label"],
        test_size=0.2,
        stratify=dataset["label"],
        random_state=42,
    )

    results = []
    best_result = None

    for model_name, estimator_factory in MODEL_REGISTRY.items():
        result = evaluate_model(model_name, estimator_factory, X_train, y_train, X_test, y_test)
        results.append(result)

        if best_result is None or result["f1"] > best_result["f1"]:
            best_result = result

    labels = sorted(dataset["label"].unique())
    save_model_and_metrics(best_result, labels, X_train, X_test, y_test, results)

    print(f"\nModel terbaik: {best_result['model']} (macro-F1={best_result['f1']:.4f}) -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
