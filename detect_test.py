from ultralytics import YOLO

model = YOLO("runs/detect/yolo-hand-detection3/weights/best.pt")

results = model.predict(source="hand.mp4", show=True)