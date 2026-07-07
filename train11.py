from ultralytics import YOLO
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
# 1. Load the pre-trained yolo11x model
model = YOLO('best.pt')

# 2. Train the model
results = model.train(
    data='dataset.yaml',  # Path to your data configuration file
    epochs=40,                # Number of epochs
    imgsz=640,                 # Image size
    batch=8,                   # Batch size (decrease if running out of memory)
    device=0,                  # Device (e.g., device=0 or device='0,1' or device='cpu')
    project='yolo26x_custom',  # Project name for saving results
    name='training_run',        # Run name

)
