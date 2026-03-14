import os
import argparse
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient

# -----------------------------
# Ground-truth label mapping
# -----------------------------
LABEL_MAP = {
    "Adult Female": "adult_female",
    "Adult/Subadult Male Seal": "adult_male",
    "Pup": "young_seal",
    "Weaner": "young_seal"
}

# -----------------------------
# Predicted label mapping
# -----------------------------
PRED_LABEL_MAP = {
    "Adult Male": "adult_male",
    "Adult Female": "adult_female",
    "Adult/Subadult Male Seal": "adult_male",
    "Pup": "young_seal",
    "Weaner": "young_seal",
    "Young Seal": "young_seal"
}

VALID_CLASSES = ["adult_male", "adult_female", "young_seal"]


# -----------------------------
# Normalize filenames so they match
# -----------------------------
def normalize_filename(name):
    if pd.isna(name):
        return None

    name = str(name).strip()

    # Remove "Copy of " prefix if present
    if name.startswith("Copy of "):
        name = name[len("Copy of "):]

    base, ext = os.path.splitext(name)
    ext = ext.lower()

    # Normalize extensions
    if ext in [".tif", ".tiff", ".jpeg"]:
        ext = ".jpg"

    return base + ext


# -----------------------------
# Load ground truth from Excel
# -----------------------------
def load_ground_truth(xlsx_path):
    df = pd.read_excel(xlsx_path)

    df = df.rename(columns={
        "individual ": "filename",
        "Seal Type": "label"
    })

    # Map labels into your 3 classes
    df["label"] = df["label"].map(LABEL_MAP)

    # Drop rows that became NaN (e.g. Other or blanks)
    df = df.dropna(subset=["label"])

    # Normalize filenames
    df["filename"] = df["filename"].apply(normalize_filename)

    return df[["filename", "label"]]


# -----------------------------
# Classify all images in folder
# -----------------------------
def classify_folder(folder, api_key, model_id="individual-classification/1"):
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )

    predictions = []

    for file in os.listdir(folder):
        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue

        path = os.path.join(folder, file)

        result = client.infer(path, model_id=model_id)

        pred_class = result["top"]
        pred_class = PRED_LABEL_MAP.get(pred_class, pred_class)

        confidence = result["confidence"]

        predictions.append({
            "filename": normalize_filename(file),
            "predicted": pred_class,
            "confidence": confidence
        })

        print(f"{file} -> {pred_class} ({confidence:.3f})")

    return pd.DataFrame(predictions)


# -----------------------------
# Main evaluation
# -----------------------------
def evaluate(folder, xlsx_path, api_key):
    gt = load_ground_truth(xlsx_path)
    preds = classify_folder(folder, api_key)

    merged = pd.merge(preds, gt, on="filename", how="inner")

    print(f"\nPredictions made: {len(preds)}")
    print(f"Ground-truth rows kept: {len(gt)}")
    print(f"Matched rows: {len(merged)}")

    if merged.empty:
        print("\nNo matching filenames found between predictions and labels.")
        print("\nSample predicted filenames:")
        print(preds["filename"].head(10).tolist())
        print("\nSample ground-truth filenames:")
        print(gt["filename"].head(10).tolist())
        return

    y_true = merged["label"]
    y_pred = merged["predicted"]

    cm = confusion_matrix(y_true, y_pred, labels=VALID_CLASSES)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=VALID_CLASSES))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=VALID_CLASSES
    )

    disp.plot()
    plt.title("Elephant Seal Classification Confusion Matrix")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.show()

    merged.to_csv("evaluation_results.csv", index=False)
    pd.DataFrame(cm, index=VALID_CLASSES, columns=VALID_CLASSES).to_csv(
        "confusion_matrix.csv"
    )

    print("\nSaved:")
    print("evaluation_results.csv")
    print("confusion_matrix.csv")
    print("confusion_matrix.png")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images",
        required=True,
        help="Path to folder with cropped seal images"
    )

    parser.add_argument(
        "--labels",
        required=True,
        help="Path to indivs_count.xlsx"
    )

    parser.add_argument(
        "--api-key",
        required=True,
        help="Roboflow API key"
    )

    args = parser.parse_args()

    evaluate(args.images, args.labels, args.api_key)