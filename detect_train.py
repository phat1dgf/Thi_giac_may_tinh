from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    MODEL_PATH = "weights/yolo11s.yaml"

    model = YOLO(MODEL_PATH)

    model.train(
        data="./dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        name='yolo-hand-detection',
        workers=4
    )
