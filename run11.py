from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2

image = "test2.jpg"

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="best2.pt",
    confidence_threshold=0.1,   # lower = detect small people
    device="cuda"
)

result = get_sliced_prediction(
    image,
    model,
    slice_height=256,   # VERY IMPORTANT (small slices)
    slice_width=256,
    overlap_height_ratio=0.4,
    overlap_width_ratio=0.4
)

result.export_visuals(export_dir="output/")