from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

MODEL_PATH = r"" # <--- Path of trained model
DATA_ROOT = Path("data_yolo_cls")


# LOAD DATASET
def load_split(split):
    base = DATA_ROOT / split

    class_names = sorted([d.name for d in base.iterdir() if d.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples = []
    for class_name in class_names:
        for img in (base / class_name).glob("*.*"):
            if img.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue
            samples.append((img, class_to_idx[class_name]))

    return samples, class_names


# PREDICT
def predict(model, img_path):
    result = model(img_path, verbose=False)[0]
    return int(result.probs.top1)


# EVALUATE
def evaluate(model, split):
    samples, class_names = load_split(split)

    y_true, y_pred = [], []

    for img, label in samples:
        pred = predict(model, str(img))
        y_true.append(label)
        y_pred.append(pred)

    print(f"\n=== {split.upper()} RESULTS ===")

    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='binary'):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='binary'):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred, average='binary'):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nFull Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


# MAIN
def main():
    model = YOLO(MODEL_PATH)

    evaluate(model, "val")
    evaluate(model, "test")


if __name__ == "__main__":
    main()