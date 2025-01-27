from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    MODEL_PATH = "weights/yolo11l.pt"

    model = YOLO(MODEL_PATH)

    model.train(
        data="./dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=4,
        name='yolo-hand-detection',
        workers=4
    )
