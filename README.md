# Human-Guided Webcam Object Learning

Local webcam prototype implementing the PRD:

- frozen pretrained MobileNetV3-Small encoder
- cosine nearest-neighbor labeled memory
- top-score and top-two-margin uncertainty gate
- selective feedback and memory updates
- baseline and always-ask experiment conditions
- trial logging, confusion matrix, and virtual sorting

This prototype performs incremental learning only by appending confirmed embeddings.
It does not retrain or fine-tune the image encoder.

## Setup

```powershell
py -3.9 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

Browser camera permission is required.

## Data

- `data/memory/embeddings.npy`: stored embeddings
- `data/memory/metadata.json`: labels and timestamps
- `data/trials.csv`: experiment records
- `data/cifar10`: downloaded CIFAR-10 images
- `data/models/cifar10_classifier.pkl`: trained 10-class logistic classifier
- `data/models/cifar10_metrics.json`: CIFAR-10 training metrics

## Train the CIFAR-10 model

```powershell
python train_cifar10.py
```

The script downloads CIFAR-10, extracts frozen MobileNetV3-Small features,
trains a 10-class classifier, and saves the model under `data/models`.
Use `python train_cifar10.py --limit 2000 --batch-size 16` for a quick test run.
