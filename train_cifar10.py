from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import Subset
from torchvision import datasets

from model import FeatureEncoder


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def extract_features(
    encoder: FeatureEncoder,
    dataset,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for start in range(0, len(dataset), batch_size):
        batch_labels = []
        batch_images = []
        end = min(start + batch_size, len(dataset))
        for index in range(start, end):
            image, label = dataset[index]
            batch_images.append(image)
            batch_labels.append(label)
        embeddings.append(encoder.embed_batch(batch_images))
        labels.append(np.asarray(batch_labels, dtype=np.int64))
        print(f"  encoded {min(start + batch_size, len(dataset))}/{len(dataset)}", end="\r")

    print()
    return np.vstack(embeddings), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_dir = Path("data/cifar10")
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading CIFAR-10...")
    train_set = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
    )

    if args.limit:
        train_set = Subset(train_set, range(min(args.limit, len(train_set))))
        test_set = Subset(test_set, range(min(args.limit // 5, len(test_set))))

    print("Extracting MobileNetV3 features...")
    encoder = FeatureEncoder()
    train_embeddings, train_labels = extract_features(
        encoder,
        train_set,
        args.batch_size,
    )
    test_embeddings, test_labels = extract_features(
        encoder,
        test_set,
        args.batch_size,
    )

    print("Training 10-class classifier...")
    classifier = LogisticRegression(
        C=1.0,
        max_iter=300,
        random_state=42,
    )
    classifier.fit(train_embeddings, train_labels)
    predictions = classifier.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, predictions)

    model_path = model_dir / "cifar10_classifier.pkl"
    model_path.write_bytes(pickle.dumps({
        "model": classifier,
        "classes": CIFAR10_CLASSES,
    }))

    metrics = {
        "dataset": "CIFAR-10",
        "classes": CIFAR10_CLASSES,
        "train_samples": len(train_embeddings),
        "test_samples": len(test_embeddings),
        "test_accuracy": float(accuracy),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_path": str(model_path),
    }
    (model_dir / "cifar10_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"Test accuracy: {accuracy:.1%}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
