from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data="data_yolo_cls",
        epochs=20,
        imgsz=288,
        batch=32,
        device=0
    )
    
if __name__ == "__main__":
    main()