from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class FeatureEncoder:
    def __init__(self) -> None:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        model.eval()
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.inference_mode()
    def embed(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        embedding = self.model(tensor).cpu().numpy()[0]
        norm = np.linalg.norm(embedding)
        if norm:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    @torch.inference_mode()
    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        tensors = [
            self.transform(image.convert("RGB"))
            for image in images
        ]
        embeddings = self.model(torch.stack(tensors)).cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        np.divide(
            embeddings,
            norms,
            out=embeddings,
            where=norms != 0,
        )
        return embeddings.astype(np.float32)


class ExampleMemory:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.directory / "embeddings.npy"
        self.metadata_path = self.directory / "metadata.json"
        self.embeddings = np.empty((0, 576), dtype=np.float32)
        self.metadata: list[dict] = []
        self.load()

    def load(self) -> None:
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path).astype(np.float32)
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def save(self) -> None:
        np.save(self.embeddings_path, self.embeddings)
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2),
            encoding="utf-8",
        )

    def add(self, embedding: np.ndarray, label: str) -> None:
        label = label.strip()
        if not label:
            raise ValueError("Label cannot be empty")
        embedding = embedding.reshape(1, -1).astype(np.float32)
        if self.embeddings.shape[0] == 0:
            self.embeddings = embedding
        else:
            if embedding.shape[1] != self.embeddings.shape[1]:
                raise ValueError("Encoder embedding dimension changed")
            self.embeddings = np.vstack([self.embeddings, embedding])
        self.metadata.append({
            "label": label,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self.save()

    def remove(self, indexes: Iterable[int]) -> None:
        keep = np.ones(len(self.metadata), dtype=bool)
        for index in indexes:
            if 0 <= index < len(self.metadata):
                keep[index] = False
        self.embeddings = self.embeddings[keep]
        self.metadata = [item for item, retained in zip(self.metadata, keep) if retained]
        self.save()

    def clear(self) -> None:
        self.embeddings = np.empty((0, 576), dtype=np.float32)
        self.metadata = []
        self.save()

    def labels(self) -> list[str]:
        return [item["label"] for item in self.metadata]


class CosineMemoryClassifier:
    def __init__(self, memory: ExampleMemory) -> None:
        self.memory = memory

    def predict(
        self,
        embedding: np.ndarray,
        k: int = 5,
    ) -> dict:
        if not self.memory.metadata:
            return {
                "prediction": None,
                "top_score": 0.0,
                "margin": 0.0,
                "scores": {},
                "nearest": [],
            }

        norm = np.linalg.norm(embedding)
        query = embedding / norm if norm else embedding
        similarities = self.memory.embeddings @ query
        k = min(k, len(similarities))
        nearest_indexes = np.argsort(similarities)[-k:][::-1]

        class_scores: dict[str, list[float]] = {}
        for index in nearest_indexes:
            label = self.memory.metadata[int(index)]["label"]
            score = float(similarities[int(index)])
            class_scores.setdefault(label, []).append(score)

        aggregated = {
            label: float(np.mean(scores))
            for label, scores in class_scores.items()
        }
        ranked = sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
        top_score = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        nearest = [
            {
                "label": self.memory.metadata[int(index)]["label"],
                "score": float(similarities[int(index)]),
            }
            for index in nearest_indexes
        ]
        return {
            "prediction": ranked[0][0],
            "top_score": top_score,
            "margin": top_score - second_score,
            "scores": dict(ranked),
            "nearest": nearest,
        }


class DatasetClassifier:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.model = None
        self.classes: list[str] = []
        self.load()

    def load(self) -> None:
        if not self.model_path.exists():
            return
        payload = pickle.loads(self.model_path.read_bytes())
        self.model = payload["model"]
        self.classes = payload["classes"]

    def predict(self, embedding: np.ndarray) -> dict:
        if self.model is None:
            return {"prediction": None, "scores": {}, "top_score": 0.0}
        probabilities = self.model.predict_proba(embedding.reshape(1, -1))[0]
        ranked_indexes = np.argsort(probabilities)[::-1]
        ranked = [
            (self.classes[int(index)], float(probabilities[int(index)]))
            for index in ranked_indexes
        ]
        return {
            "prediction": ranked[0][0],
            "top_score": ranked[0][1],
            "scores": dict(ranked),
        }


def should_ask(result: dict, similarity_threshold: float, margin_threshold: float) -> bool:
    if result["prediction"] is None:
        return True
    return (
        result["top_score"] < similarity_threshold
        or result["margin"] < margin_threshold
    )
