"""Train a 1D CNN using the same 26x3 input shape as the embedded model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from wisdm_common import LABELS, encode_labels, split_by_user


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(26, 3)),
            tf.keras.layers.Conv1D(16, 3, activation="relu"),
            tf.keras.layers.Conv1D(8, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Prepared .npz file")
    parser.add_argument("--model", type=Path, default=Path("ml/models/cnn.keras"))
    parser.add_argument("--report", type=Path, default=Path("ml/results/cnn_report.txt"))
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    data = np.load(args.dataset)
    x_train, x_test, y_train_text, y_test_text = split_by_user(
        data["x"], data["y"], data["users"]
    )
    y_train = encode_labels(y_train_text)
    y_test = encode_labels(y_test_text)
    model = build_model()
    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=64,
        verbose=2,
    )
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    report = classification_report(y_test, predictions, target_names=LABELS, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model)
    args.report.write_text(
        f"Classification report\n\n{report}\nConfusion matrix\n{matrix}\n",
        encoding="utf-8",
    )
    print(report)
    print(f"Saved model to {args.model}")


if __name__ == "__main__":
    main()
