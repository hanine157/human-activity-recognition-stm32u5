"""Train and evaluate a Random Forest baseline on WISDM windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from wisdm_common import LABELS, encode_labels, split_by_user


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Prepared .npz file")
    parser.add_argument("--model", type=Path, default=Path("ml/models/random_forest.joblib"))
    parser.add_argument("--report", type=Path, default=Path("ml/results/random_forest_report.txt"))
    args = parser.parse_args()

    data = np.load(args.dataset)
    x_train, x_test, y_train_text, y_test_text = split_by_user(
        data["x"], data["y"], data["users"]
    )
    y_train = encode_labels(y_train_text)
    y_test = encode_labels(y_test_text)
    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    classifier.fit(x_train.reshape(len(x_train), -1), y_train)
    predictions = classifier.predict(x_test.reshape(len(x_test), -1))
    report = classification_report(y_test, predictions, target_names=LABELS, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, args.model)
    args.report.write_text(
        f"Classification report\n\n{report}\nConfusion matrix\n{matrix}\n",
        encoding="utf-8",
    )
    print(report)
    print(f"Saved model to {args.model}")


if __name__ == "__main__":
    main()
