from ultralytics import RTDETR
model = RTDETR("rtdetr-l.pt")
# Export to OpenVINO with INT8 quantization
model.export(format="openvino", int8=True, imgsz=480)