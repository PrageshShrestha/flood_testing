from ultralytics import YOLO

# ==============================
# TRAIN MODEL
# ==============================
model = YOLO("yolo26l.pt")

results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=7,
    device=0,
    workers=8
)

print("Training completed!")