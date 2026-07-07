import torch
import torch.nn.utils.prune as prune
from ultralytics import RTDETR
import shutil
from pathlib import Path

# 1. Load your trained RT-DETR-L model
print("Loading model...")
model = RTDETR('rtdetr-l.pt')

def apply_selective_pruning(model, amount=0.30):
    """
    Prunes Conv2d layers but skips early backbone layers
    """
    print(f"Starting pruning (Target: {amount*100}% reduction)...")
    
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Skip early layers critical for small objects
            if any(layer_key in name for layer_key in ['stem', 'stage1', 'downsample']):
                print(f"Skipping {name} to preserve small object sensitivity.")
                continue
            
            # Apply L1 Unstructured pruning
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    
    return model

# 2. Create TWO copies of the pruned model
print("\n" + "="*60)
print("📦 Creating two model variants...")
print("="*60)

# Apply pruning to the original model
pruned_model = apply_selective_pruning(model, amount=0.25)

# Create two separate copies
model_for_finetuning = pruned_model  # This will be fine-tuned
model_for_inference = pruned_model   # This will stay pruned (no fine-tuning)

# Save both models with different names
print("\n💾 Saving model variants...")
model_for_finetuning.save('rtdetr_l_pruned_version_1.pt')  # Will be fine-tuned
model_for_inference.save('rtdetr_l_pruned_version_2.pt')   # Will stay as-is (inference only)

print("✅ Model version 1 (will be fine-tuned): rtdetr_l_pruned_version_1.pt")
print("✅ Model version 2 (inference-only): rtdetr_l_pruned_version_2.pt")

# 3. Fine-tune ONLY version 1
print("\n" + "="*60)
print("🎯 Starting FINE-TUNING (Version 1 only)...")
print("="*60)

fine_tune_results = model_for_finetuning.train(
    data='dataset.yaml', 
    epochs=15, 
    imgsz=1080, 
    lr0=0.0005, 
    batch=6
)

# Save the fine-tuned version 1
model_for_finetuning.save('rtdetr_l_pruned_version_1_finetuned.pt')
print("✅ Fine-tuned version 1 saved: rtdetr_l_pruned_version_1_finetuned.pt")

# 4. Now you have three models for comparison:
print("\n" + "="*60)
print("📊 Available Models Ready for Inference:")
print("="*60)
print("1. Original model: rtdetr-l.pt")
print("2. Pruned ONLY (version 2): rtdetr_l_pruned_version_2.pt (no fine-tuning)")
print("3. Pruned + Fine-tuned (version 1): rtdetr_l_pruned_version_1_finetuned.pt")
print("="*60)