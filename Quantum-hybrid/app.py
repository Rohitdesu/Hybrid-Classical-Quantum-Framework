from pathlib import Path
from io import BytesIO

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from model_utils import (
    build_inference_transform,
    image_to_data_url,
    load_checkpoint_model,
    predict_pil_image,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pt"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

model, device, checkpoint = load_checkpoint_model(MODEL_PATH)
transform = build_inference_transform()


def build_result_payload(uploaded_file) -> dict:
    filename = uploaded_file.filename or "uploaded_image"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload a medical image in PNG, JPG, JPEG, BMP, TIFF, or WEBP format.")

    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    prediction = predict_pil_image(image, model, device, transform)
    severity_tone = "alert" if prediction["predicted_label"] == "Malignant" else "calm"

    return {
        "filename": filename,
        "filesize_kb": round(len(file_bytes) / 1024, 2),
        "dimensions": f"{image.width} x {image.height}",
        "mode": image.mode,
        "image_data_url": image_to_data_url(image),
        "result": prediction,
        "severity_tone": severity_tone,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "result_payload": None,
        "error_message": None,
        "model_info": {
            "checkpoint_name": MODEL_PATH.name,
            "best_val_accuracy": checkpoint.get("best_val_accuracy"),
            "best_epoch": checkpoint.get("epoch"),
            "device": str(device).upper(),
        },
    }

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if not uploaded_file or not uploaded_file.filename:
            context["error_message"] = "Choose an image first so the model has something to analyze."
            return render_template("index.html", **context)
        try:
            context["result_payload"] = build_result_payload(uploaded_file)
        except ValueError as exc:
            context["error_message"] = str(exc)

    return render_template("index.html", **context)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    uploaded_file = request.files.get("image")
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"error": "No image file was uploaded."}), 400

    try:
        result_payload = build_result_payload(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result_payload)


@app.errorhandler(413)
def file_too_large(_error):
    return (
        render_template(
            "index.html",
            result_payload=None,
            error_message="That file is too large. Please keep uploads under 12 MB.",
            model_info={
                "checkpoint_name": MODEL_PATH.name,
                "best_val_accuracy": checkpoint.get("best_val_accuracy"),
                "best_epoch": checkpoint.get("epoch"),
                "device": str(device).upper(),
            },
        ),
        413,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
