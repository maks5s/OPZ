import io
import json
import os
import random
import tempfile
import time
import zipfile
from datetime import datetime
from typing import Callable, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_curve, precision_recall_curve

MODEL_FORMAT_VERSION = 1
MODEL_DECISION_THRESHOLD = 0.5
MODEL_FILENAME = "model.keras"
METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics.json"


def configure_seed(seed: int) -> None:
    seed = int(seed)
    keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_model(
        feature_count: int,
        hidden_layers: int = 1,
        hidden_units: int = 32,
        learning_rate: float = 0.001,
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-3,
        label_smoothing: float = 0.05,
) -> keras.Model:
    if hidden_layers not in (0, 1):
        raise ValueError("hidden_layers має бути 0 або 1")
    if not 1 <= int(hidden_units) <= 256:
        raise ValueError("hidden_units має бути в межах 1–256")

    regularizer = keras.regularizers.l2(float(l2_reg))

    inputs = keras.Input(shape=(int(feature_count),), name="features")

    x = keras.layers.Normalization(axis=-1, name="normalization")(inputs)

    if hidden_layers == 1:
        x = keras.layers.Dense(
            int(hidden_units),
            activation="relu",
            kernel_regularizer=regularizer,
            name="hidden",
        )(x)
        x = keras.layers.Dropout(float(dropout_rate), name="dropout")(x)

    outputs = keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=regularizer,
        name="output",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="defect_classifier")

    loss_fn = keras.losses.BinaryCrossentropy(
        label_smoothing=float(label_smoothing),
        name="binary_crossentropy"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss=loss_fn,
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


class _EpochProgressCallback(keras.callbacks.Callback):

    def __init__(self, callback: Optional[Callable[[int, float, float], None]]):
        super().__init__()
        self._callback = callback

    def on_epoch_end(self, epoch, logs=None):
        if self._callback is None:
            return

        logs = logs or {}
        train_accuracy = float(logs.get("accuracy", 0.0)) * 100.0
        test_accuracy = float(logs.get("val_accuracy", 0.0)) * 100.0
        self._callback(int(epoch) + 1, train_accuracy, test_accuracy)


def _round(value, digits: int = 4):
    if value is None:
        return None
    return round(float(value), digits)


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = (np.asarray(probabilities).reshape(-1) >= threshold).astype(int)

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    try:
        roc_auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        roc_auc = None

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)) * 100.0, 2),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)) * 100.0, 2),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)) * 100.0, 2),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)) * 100.0, 2),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)) * 100.0, 2),
        "roc_auc": _round(roc_auc, 4),
        "confusion_matrix": cm.astype(int).tolist(),
        "per_class": {
            "class0": {
                "precision": round(float(per_precision[0]) * 100.0, 2),
                "recall": round(float(per_recall[0]) * 100.0, 2),
                "f1": round(float(per_f1[0]) * 100.0, 2),
                "support": int(per_support[0]),
            },
            "class1": {
                "precision": round(float(per_precision[1]) * 100.0, 2),
                "recall": round(float(per_recall[1]) * 100.0, 2),
                "f1": round(float(per_f1[1]) * 100.0, 2),
                "support": int(per_support[1]),
            },
        },
    }


def train_defect_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 200,
    learning_rate: float = 0.001,
    hidden_layers: int = 1,
    hidden_units: int = 16,
    test_size: float = 0.10,
    random_state: int = 42,
    progress_callback: Optional[Callable[[int, float, float], None]] = None,
) -> Tuple[keras.Model, dict]:
    started_at = time.perf_counter()
    configure_seed(random_state)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int).reshape(-1)
    feature_count = int(X.shape[1])
    epochs = int(epochs)
    hidden_layers = int(hidden_layers)
    hidden_units = int(hidden_units)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=int(random_state),
        stratify=y,
    )

    model = build_model(
        feature_count=feature_count,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )

    normalization = model.get_layer("normalization")
    normalization.adapt(X_train)

    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train,
    )
    class_weights = {
        0: float(class_weight_values[0]),
        1: float(class_weight_values[1]),
    }

    batch_size = min(32, max(4, len(X_train) // 10))
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        verbose=0,
        shuffle=True,
        callbacks=[_EpochProgressCallback(progress_callback)],
    )

    train_probabilities = model.predict(X_train, verbose=0).reshape(-1)
    test_probabilities = model.predict(X_test, verbose=0).reshape(-1)

    optimal_threshold = find_optimal_threshold(y_test, test_probabilities, method="youden")
    optimal_threshold = round(float(optimal_threshold), 4)

    train_metrics = _classification_metrics(
        y_train, train_probabilities, threshold=optimal_threshold
    )
    test_metrics = _classification_metrics(
        y_test, test_probabilities, threshold=optimal_threshold
    )

    train_loss = float(model.evaluate(X_train, y_train, verbose=0, batch_size=batch_size)[0])
    test_loss = float(model.evaluate(X_test, y_test, verbose=0, batch_size=batch_size)[0])

    class_distribution = {
        "class0": int(np.sum(y == 0)),
        "class1": int(np.sum(y == 1)),
    }

    metrics = {
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "train_loss": round(train_loss, 4),
        "test_loss": round(test_loss, 4),
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "per_class": test_metrics["per_class"],
        "class_distribution": class_distribution,
        "class_weights": {
            "class0": round(class_weights[0], 4),
            "class1": round(class_weights[1], 4),
        },
        "epochs": epochs,
        "learning_rate": float(learning_rate),
        "hidden_layers": hidden_layers,
        "hidden_units": hidden_units,
        "feature_count": feature_count,
        "threshold": optimal_threshold,
        "trained_at": datetime.now().astimezone().isoformat(),
        "duration_seconds": round(time.perf_counter() - started_at, 3),
    }

    return model, metrics


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray, method: str = "youden") -> float:
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)

        j_scores = tpr - fpr

        optimal_idx = int(np.argmax(j_scores))
        optimal_threshold = float(thresholds[optimal_idx])

    elif method == "f1":
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        f1_scores = (2 * precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
        optimal_idx = int(np.argmax(f1_scores))
        optimal_threshold = float(thresholds[optimal_idx])

    return float(np.clip(optimal_threshold, 0.05, 0.95))


def predict_sample(model: keras.Model, features: np.ndarray) -> float:
    x = np.asarray(features, dtype=np.float32).reshape(1, -1)
    probability = model.predict(x, verbose=0)[0][0]
    return float(probability)


def export_model_bundle(model: keras.Model, metadata: dict, metrics: dict) -> bytes:
    buffer = io.BytesIO()

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, MODEL_FILENAME)
        model.save(model_path)

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(model_path, MODEL_FILENAME)
            archive.writestr(
                METADATA_FILENAME,
                json.dumps(metadata, ensure_ascii=False, indent=2),
            )
            archive.writestr(
                METRICS_FILENAME,
                json.dumps(metrics, ensure_ascii=False, indent=2),
            )

    return buffer.getvalue()


def load_model_bundle(contents: bytes) -> Tuple[keras.Model, dict, dict]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(contents))
    except zipfile.BadZipFile as error:
        raise ValueError("Файл не є коректним ZIP-архівом") from error

    with archive:
        names = set(archive.namelist())
        missing = [
            name
            for name in (MODEL_FILENAME, METADATA_FILENAME, METRICS_FILENAME)
            if name not in names
        ]
        if missing:
            raise ValueError(
                "У ZIP-архіві відсутні обов'язкові файли: " + ", ".join(missing)
            )

        try:
            metadata = json.loads(archive.read(METADATA_FILENAME).decode("utf-8"))
            metrics = json.loads(archive.read(METRICS_FILENAME).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("metadata.json або metrics.json має некоректний JSON") from error

        if not isinstance(metadata, dict):
            raise ValueError("metadata.json має містити JSON-об'єкт")
        if not isinstance(metrics, dict):
            raise ValueError("metrics.json має містити JSON-об'єкт")

        model_bytes = archive.read(MODEL_FILENAME)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, MODEL_FILENAME)
        with open(model_path, "wb") as model_file:
            model_file.write(model_bytes)

        try:
            model = keras.models.load_model(model_path, safe_mode=True)
        except Exception as error:
            raise ValueError(f"Не вдалося завантажити Keras-модель: {error}") from error

    return model, metadata, metrics
