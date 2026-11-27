import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from model import (
    CosineMemoryClassifier,
    DatasetClassifier,
    ExampleMemory,
    FeatureEncoder,
    should_ask,
)


DATA_DIR = Path("data")
LOG_PATH = DATA_DIR / "trials.csv"
MEMORY_DIR = DATA_DIR / "memory"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def load_encoder() -> FeatureEncoder:
    return FeatureEncoder()


def log_trial(record: dict) -> None:
    fields = [
        "timestamp", "condition", "ground_truth", "prediction", "top_score",
        "margin", "queried", "correct", "latency_ms",
    ]
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in fields})


st.set_page_config(page_title="Human-Guided Webcam Object Learning", layout="wide")
st.title("Human-Guided Webcam Object Learning")

memory = ExampleMemory(MEMORY_DIR)
encoder = load_encoder()
classifier = CosineMemoryClassifier(memory)
dataset_classifier = DatasetClassifier(DATA_DIR / "models" / "cifar10_classifier.pkl")

dataset_metrics_path = DATA_DIR / "models" / "cifar10_metrics.json"

with st.sidebar:
    st.header("Experiment")
    condition = st.selectbox(
        "Condition",
        ["selective_feedback", "no_feedback", "always_ask"],
    )
    similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.75, 0.01)
    margin_threshold = st.slider("Margin threshold", 0.0, 1.0, 0.10, 0.01)
    k_neighbors = st.slider("k nearest neighbors", 1, 15, 5)

    st.header("CIFAR-10 model")
    if dataset_metrics_path.exists():
        dataset_metrics = json.loads(dataset_metrics_path.read_text(encoding="utf-8"))
        st.metric("Test accuracy", f"{dataset_metrics['test_accuracy']:.1%}")
        st.caption(f"{dataset_metrics['train_samples']} training samples")
    else:
        st.caption("No trained CIFAR-10 model yet.")
    st.code("python train_cifar10.py", language="powershell")

    st.header("Virtual sorting")
    bins = {
        "bin_1": st.text_input("Bin 1 label", "Recycle"),
        "bin_2": st.text_input("Bin 2 label", "Storage"),
        "bin_3": st.text_input("Bin 3 label", "Discard"),
    }
    st.caption("Assign confirmed class labels in the feedback panel.")

st.markdown(
    "Show one tabletop object in a simple background. "
    "The camera frame is compared with local labeled embeddings."
)

left, right = st.columns([1.25, 1])
with left:
    photo = st.camera_input("Capture an object")

with right:
    if photo is None:
        st.info("Capture a frame to begin.")
    else:
        start = time.perf_counter()
        image = Image.open(photo)
        embedding = encoder.embed(image)
        result = classifier.predict(embedding, k=k_neighbors)
        dataset_result = dataset_classifier.predict(embedding)
        latency_ms = (time.perf_counter() - start) * 1000

        uncertain = should_ask(result, similarity_threshold, margin_threshold)
        queried = uncertain or condition == "always_ask"

        st.subheader("Decision")
        if queried:
            st.warning("The system is uncertain and is asking for help.")
        else:
            st.success(f"Prediction: **{result['prediction']}**")

        c1, c2 = st.columns(2)
        c1.metric("Top similarity", f"{result['top_score']:.3f}")
        c2.metric("Top-two margin", f"{result['margin']:.3f}")
        st.caption(f"Inference: {latency_ms:.1f} ms")

        if dataset_result["prediction"]:
            st.subheader("CIFAR-10 prediction")
            st.success(
                f"Prediction: **{dataset_result['prediction']}** "
                f"({dataset_result['top_score']:.1%})"
            )

        ground_truth = st.text_input("Ground-truth label (for this trial)")
        if queried:
            label = st.text_input("Correct label")
            confirmed = st.button("Confirm and save example")
            if confirmed:
                if not label.strip():
                    st.error("Enter a label before saving.")
                elif condition != "no_feedback":
                    memory.add(embedding, label)
                    st.success(f"Saved labeled example: {label}")
                    st.rerun()
                else:
                    st.info("No-feedback condition does not add labels.")
        else:
            label = result["prediction"]

        if st.button("Log trial"):
            prediction = result["prediction"] if not queried else ""
            correct = (
                ground_truth.strip().lower() == prediction.lower()
                if prediction and ground_truth else None
            )
            log_trial({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "condition": condition,
                "ground_truth": ground_truth.strip(),
                "prediction": prediction or "",
                "top_score": result["top_score"],
                "margin": result["margin"],
                "queried": queried,
                "correct": correct if correct is not None else "",
                "latency_ms": round(latency_ms, 2),
            })
            st.success("Trial logged.")

        if result["nearest"]:
            st.subheader("Nearest examples")
            st.dataframe(
                pd.DataFrame(result["nearest"]),
                use_container_width=True,
                hide_index=True,
            )

        if not queried and result["prediction"]:
            st.subheader("Virtual sorting bin")
            label_key = result["prediction"].strip().lower()
            assigned_bin = "No bin assignment"
            for number in range(1, 4):
                class_labels = st.session_state.get(f"bin_{number}_classes", "")
                if label_key in {item.strip().lower() for item in class_labels.split(",") if item.strip()}:
                    assigned_bin = bins[f"bin_{number}"]
                    break
            st.markdown(f"**Destination: {assigned_bin}**")
            st.caption("Configure class membership in Memory below.")

st.divider()
tab_memory, tab_results = st.tabs(["Memory", "Results"])

with tab_memory:
    st.subheader("Labeled memory")
    if memory.metadata:
        table = pd.DataFrame(memory.metadata)
        table.insert(0, "index", range(len(table)))
        st.dataframe(table, use_container_width=True, hide_index=True)

        remove_text = st.text_input("Indexes to remove (comma-separated)")
        col1, col2 = st.columns(2)
        if col1.button("Remove selected"):
            try:
                indexes = [int(item.strip()) for item in remove_text.split(",") if item.strip()]
                memory.remove(indexes)
                st.success("Removed selected examples.")
                st.rerun()
            except ValueError:
                st.error("Indexes must be integers.")
        if col2.button("Delete all memory"):
            memory.clear()
            st.success("Memory deleted.")
            st.rerun()
    else:
        st.info("No examples in memory.")

    st.subheader("Bin class membership")
    for number in range(1, 4):
        st.text_input(
            f"Classes for {bins[f'bin_{number}']} (comma-separated)",
            key=f"bin_{number}_classes",
        )

with tab_results:
    st.subheader("Experiment log")
    if LOG_PATH.exists():
        trials = pd.read_csv(LOG_PATH)
        if trials.empty:
            st.info("No logged trials.")
        else:
            st.dataframe(trials, use_container_width=True, hide_index=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Trials", len(trials))
            col2.metric("Query rate", f"{trials['queried'].fillna(False).mean():.1%}")
            answered = trials[trials["prediction"].fillna("").astype(bool)]
            accuracy = (
                answered["correct"].dropna().mean()
                if "correct" in answered and answered["correct"].notna().any()
                else None
            )
            col3.metric("Accuracy", f"{accuracy:.1%}" if accuracy is not None else "N/A")

            if "correct" in trials:
                confusions = trials.dropna(subset=["ground_truth", "prediction"])
                if not confusions.empty:
                    labels = sorted(set(confusions["ground_truth"]) | set(confusions["prediction"]))
                    matrix = pd.crosstab(confusions["ground_truth"], confusions["prediction"])
                    fig, ax = plt.subplots()
                    ax.imshow(matrix.reindex(index=labels, columns=labels, fill_value=0).values)
                    ax.set_xticks(range(len(labels)))
                    ax.set_yticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha="right")
                    ax.set_yticklabels(labels)
                    ax.set_xlabel("Prediction")
                    ax.set_ylabel("Ground truth")
                    st.pyplot(fig)
    else:
        st.info("No trials logged yet.")

