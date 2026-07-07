from ultralytics import RTDETR
from pathlib import Path
import argparse

def finetune_pruned_model(pruned_model_path, data_yaml, epochs=50, lr=0.0001):
    """
    Fine-tune the pruned model to recover accuracy
    Based on research showing fine-tuning restores pruning-induced accuracy loss [citation:2][citation:8]
    """
    
    print("="*70)
    print("PHASE 3: FINE-TUNING PRUNED RT-DETR")
    print("="*70)
    
    print(f"\n🔧 Loading pruned model: {pruned_model_path}")
    model = RTDETR(pruned_model_path)
    
    print(f"🚀 Starting fine-tuning (epochs={epochs}, lr={lr})")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        lr0=lr,  # Lower learning rate for fine-tuning
        batch=6,
        imgsz=1088,
        device='cuda',
        workers=8,
        project='pruned_finetuning',
        name='finetune_run',
        pretrained=False  # Important: don't reload original weights
    )
    
    # Save fine-tuned model
    output_path = pruned_model_path.replace('.pt', '_finetuned.pt')
    model.save(output_path)
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"   Model saved: {output_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, default='dataset.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    
    args = parser.parse_args()
    
    finetune_pruned_model(args.model, args.data, args.epochs, args.lr)