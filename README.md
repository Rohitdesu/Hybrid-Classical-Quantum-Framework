# Breast Cancer Prediction Web App

This web app loads the saved checkpoint:

- [best_model.pt](C:\Users\rohit\Documents\New project\best_model.pt)

and serves a polished upload interface plus a JSON prediction API.

## Run locally

```powershell
python -m pip install -r requirements_web.txt
python app.py
```

Open:

- [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Features

- Upload histopathology images from the browser
- Uses the saved `best_model.pt` checkpoint for inference
- Shows predicted label, confidence, and per-class probabilities
- Displays image metadata and preview
- Includes `POST /api/predict` for integrations

## API example

```powershell
curl.exe -X POST -F "image=@C:\path\to\sample.jpg" http://127.0.0.1:5000/api/predict
```

# Breast Cancer Colab Project

This project reproduces the paper setup as closely as possible on a Google Colab T4 GPU:

- `ResNet50` backbone with ImageNet weights
- `Quantum-inspired processing layer`
- `Self-attention`
- `224x224` inputs
- stratified `70/15/15` train/val/test split
- weighted sampling for imbalance
- mixed precision for faster T4 training

Main script:

- [breast_cancer_colab_project.py](C:\Users\rohit\Documents\New project\breast_cancer_colab_project.py)

Default dataset path inside Colab:

- `/content/drive/MyDrive/dataset_cancer_v1`

Default output directory:

- `/content/drive/MyDrive/breast_cancer_project_outputs`

Run in Colab:

```python
!python breast_cancer_colab_project.py \
  --dataset-root /content/drive/MyDrive/dataset_cancer_v1 \
  --output-dir /content/drive/MyDrive/breast_cancer_project_outputs \
  --epochs 12 \
  --batch-size 32 \
  --num-workers 2 \
  --max-train-minutes 80
```

Important:

- The paper target is `92.34%` accuracy.
- The script reports the real achieved accuracy in `results_summary.json`.
- It also saves `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`, `classification_report.txt`, and `best_model.pt`.
