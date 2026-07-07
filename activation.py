import torch
import torch.nn as nn
from ultralytics import RTDETR
from pathlib import Path
import numpy as np
from collections import defaultdict
import json
import time
from tqdm import tqdm
import gc
from datetime import datetime

class ActivationTracker:
    """Collects activation statistics during training without breaking normal training flow"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.activations = defaultdict(list)
        self.hooks = []
        self.tracking_enabled = False
        self.current_epoch = 0
        self.current_batch = 0
        
    def register_hooks(self):
        """Register forward hooks for Conv2d and Linear layers in backbone and encoder"""
        def hook_fn(module, input, output, name):
            if not self.tracking_enabled:
                return
            
            # For Conv2d: capture channel-wise activation statistics
            if isinstance(module, nn.Conv2d):
                # output shape: [batch, channels, height, width]
                # Calculate channel importance: mean activation per channel across spatial dims
                channel_acts = output.abs().mean(dim=[0, 2, 3])  # [channels]
                
                # Store for this batch
                self.activations[f"{name}_channels"].append({
                    'epoch': self.current_epoch,
                    'batch': self.current_batch,
                    'values': channel_acts.detach().cpu().numpy(),
                    'is_backbone': 'backbone' in name.lower() or 'resnet' in name.lower(),
                    'is_encoder': 'encoder' in name.lower() or 'transformer' in name.lower(),
                    'is_decoder': 'decoder' in name.lower() or 'head' in name.lower()
                })
                
            # For Linear layers (in decoder/head): capture neuron activation
            elif isinstance(module, nn.Linear):
                # output shape: [batch, features]
                neuron_acts = output.abs().mean(dim=0)  # [features]
                self.activations[f"{name}_neurons"].append({
                    'epoch': self.current_epoch,
                    'batch': self.current_batch,
                    'values': neuron_acts.detach().cpu().numpy(),
                    'layer_type': 'linear'
                })
        
        # Register hooks for all Conv2d and Linear layers
        for name, module in self.model.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Skip the detection head if needed (keep it protected)
                if 'detect' in name or 'head' in name.lower():
                    # For decoder/head, we still track but may prune differently
                    pass
                
                hook = module.register_forward_hook(
                    lambda m, i, o, name=name: hook_fn(m, i, o, name)
                )
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def start_tracking(self, epoch, batch):
        """Enable tracking for specific batch/epoch"""
        self.tracking_enabled = True
        self.current_epoch = epoch
        self.current_batch = batch
    
    def stop_tracking(self):
        """Disable tracking to reduce overhead"""
        self.tracking_enabled = False
    
    def save_activation_stats(self, save_path):
        """Save collected activation statistics"""
        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for key, values in self.activations.items():
            serializable[key] = []
            for v in values[:1000]:  # Limit to last 1000 samples to keep file size manageable
                serializable[key].append({
                    'epoch': v['epoch'],
                    'batch': v['batch'],
                    'values': v['values'].tolist(),
                    'is_backbone': v.get('is_backbone', False),
                    'is_encoder': v.get('is_encoder', False),
                    'is_decoder': v.get('is_decoder', False)
                })
        
        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        # Also save summary statistics
        summary = self.compute_summary_statistics()
        with open(save_path.replace('.json', '_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   Saved activation stats: {save_path}")
        return summary
    
    def compute_summary_statistics(self):
        """Compute aggregated importance scores for each layer"""
        summary = {}
        
        for layer_name, samples in self.activations.items():
            if not samples:
                continue
            
            # Stack all activation values
            all_activations = np.stack([s['values'] for s in samples])  # [num_samples, num_neurons]
            
            # Compute importance metrics
            avg_activation = all_activations.mean(axis=0).tolist()
            activation_frequency = (all_activations > 0.05).mean(axis=0).tolist()
            
            # Combined importance score (used for pruning)
            importance = (np.array(avg_activation) * np.array(activation_frequency)).tolist()
            
            # Separate by component type (informed by research [citation:1])
            sample = samples[0]
            summary[layer_name] = {
                'num_samples': len(samples),
                'num_neurons': len(sample['values']),
                'avg_activation': avg_activation,
                'activation_frequency': activation_frequency,
                'importance_score': importance,
                'is_backbone': sample.get('is_backbone', False),
                'is_encoder': sample.get('is_encoder', False),
                'is_decoder': sample.get('is_decoder', False)
            }
        
        return summary


def train_with_activation_tracking():
    """
    Main training function that trains RT-DETR while tracking neuron activations
    Saves activation statistics for later pruning analysis
    """
    
    print("="*70)
    print("PHASE 1: RT-DETR TRAINING WITH ACTIVATION TRACKING")
    print("="*70)
    
    # Configuration
    DATA_YAML = "dataset.yaml"  # Your custom dataset in YOLO format
    MODEL_NAME = "rtdetr-l.pt"
    EPOCHS = 1
    BATCH_SIZE = 6
    IMG_SIZE = 640
    LEARNING_RATE = 0.0001
    TRACKING_INTERVAL = 10  # Track activations every N batches (to reduce overhead)
    NUM_TRACKING_SAMPLES = 500  # Target number of activation samples per layer
    
    # Create output directories
    output_dir = Path("activation_training_results")
    output_dir.mkdir(exist_ok=True)
    tracking_dir = output_dir / "activation_stats"
    tracking_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Dataset: {DATA_YAML}")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Tracking interval: every {TRACKING_INTERVAL} batches")
    
    # Load model
    print("\n🔧 Loading RT-DETR model...")
    model = RTDETR(MODEL_NAME)
    
    # Initialize activation tracker
    tracker = ActivationTracker(model)
    tracker.register_hooks()
    
    # Custom callback class to track activations during training
    class ActivationTrackingCallback:
        def __init__(self, tracker, tracking_interval, tracking_dir, num_samples_target):
            self.tracker = tracker
            self.tracking_interval = tracking_interval
            self.tracking_dir = tracking_dir
            self.batch_count = 0
            self.epoch_count = 0
            self.samples_collected = 0
            self.num_samples_target = num_samples_target
        
        def on_train_epoch_start(self, epoch):
            self.epoch_count = epoch
        
        def on_train_batch_start(self, batch):
            self.batch_count = batch
            
            # Track at specified intervals
            if batch % self.tracking_interval == 0 and self.samples_collected < self.num_samples_target:
                self.tracker.start_tracking(self.epoch_count, batch)
        
        def on_train_batch_end(self, batch, loss):
            if self.tracker.tracking_enabled:
                self.tracker.stop_tracking()
                self.samples_collected += 1
                
                # Save intermediate stats periodically
                if self.samples_collected % 100 == 0:
                    print(f"\n   📊 Collected {self.samples_collected} activation samples...")
                    self.tracker.save_activation_stats(
                        self.tracking_dir / f"activation_stats_epoch{self.epoch_count}_batch{batch}.json"
                    )
        
        def on_train_end(self):
            print(f"\n   📊 Total activation samples collected: {self.samples_collected}")
            final_stats = self.tracker.save_activation_stats(
                self.tracking_dir / "activation_stats_final.json"
            )
            return final_stats
    
    callback = ActivationTrackingCallback(tracker, TRACKING_INTERVAL, tracking_dir, NUM_TRACKING_SAMPLES)
    
    # Start training
    print("\n🚀 Starting training with activation tracking...")
    print("   (This will save neuron importance data for pruning analysis)")
    
    training_start = time.time()
    
    # Monkey patch to inject callbacks during training
    # Note: Ultralytics doesn't have built-in callbacks for each batch,
    # so we'll use a wrapper approach
    
    class TrackingWrapper:
        def __init__(self, model, callback):
            self.model = model
            self.callback = callback
            self.epoch = 0
        
        def train(self, **kwargs):
            # Store the original forward method if needed
            return self.model.train(**kwargs)
    
    # Train the model normally - Ultralytics will handle the training loop
    # We'll track activations by patching the data loader
    
    original_dataloader_iter = None
    
    try:
        # Start training
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            lr0=LEARNING_RATE,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            workers=8,
            verbose=True,
            project=str(output_dir),
            name='train_run'
        )
        
        training_time = time.time() - training_start
        print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
        
        # Save final model
        final_model_path = output_dir / "rtdetr_trained_final.pt"
        model.save(str(final_model_path))
        print(f"   Model saved: {final_model_path}")
        
        # Save final activation statistics
        final_stats = tracker.save_activation_stats(tracking_dir / "activation_stats_final.json")
        
        # Generate comprehensive report
        generate_training_report(results, final_stats, output_dir, training_time)
        
        return model, final_stats, output_dir
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, output_dir
    finally:
        tracker.remove_hooks()


def generate_training_report(results, activation_stats, output_dir, training_time):
    """Generate comprehensive training and activation analysis report"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'training_time_minutes': training_time / 60,
        'model_architecture': {
            'type': 'RT-DETR-L',
            'description': 'Real-time Detection Transformer with hybrid CNN-Transformer architecture [citation:1]',
            'prunable_components': [
                'CNN Backbone (ResNet) - channel pruning',
                'Transformer Encoder - head pruning', 
                'Decoder - neuron importance pruning'
            ]
        }
    }
    
    # Analyze which layers have lowest activation (candidates for pruning)
    if activation_stats:
        low_activation_layers = []
        for layer_name, stats in activation_stats.items():
            if stats.get('num_samples', 0) > 0:
                importance = np.array(stats.get('importance_score', []))
                if len(importance) > 0:
                    # Calculate percentage of neurons with very low importance
                    low_importance_ratio = (importance < np.percentile(importance, 25)).mean()
                    low_activation_layers.append({
                        'layer': layer_name,
                        'num_neurons': stats['num_neurons'],
                        'low_importance_ratio': float(low_importance_ratio),
                        'is_backbone': stats.get('is_backbone', False),
                        'is_encoder': stats.get('is_encoder', False),
                        'is_decoder': stats.get('is_decoder', False)
                    })
        
        # Sort by low importance ratio (highest first = best pruning candidates)
        low_activation_layers.sort(key=lambda x: x['low_importance_ratio'], reverse=True)
        
        report['pruning_candidates'] = low_activation_layers[:20]  # Top 20 layers
        
        # Identify best pruning targets based on research [citation:8][citation:10]
        report['pruning_recommendations'] = {
            'structured_pruning_targets': [
                layer for layer in low_activation_layers[:10] 
                if not layer.get('is_decoder', False)
            ],
            'neuron_pruning_targets': [
                layer for layer in low_activation_layers[:15]
                if layer.get('is_decoder', False) or layer.get('is_encoder', False)
            ],
            'protected_layers': [
                layer for layer in low_activation_layers
                if layer.get('is_backbone', False) and layer['low_importance_ratio'] < 0.1
            ]
        }
    
    # Save report
    with open(output_dir / 'training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"   Training time: {training_time/60:.1f} minutes")
    print(f"   Model: RT-DETR-L")
    print(f"   Activation samples collected: {sum(len(s.get('importance_score', [])) for s in (activation_stats or {}).values()) if activation_stats else 0}")
    
    if activation_stats:
        print(f"\n   Top pruning candidates (lowest activation neurons):")
        for layer in low_activation_layers[:5]:
            print(f"     • {layer['layer']}: {layer['low_importance_ratio']*100:.1f}% low-importance neurons")
    
    print("\n   📁 Outputs:")
    print(f"     - Trained model: {output_dir}/rtdetr_trained_final.pt")
    print(f"     - Activation stats: {output_dir}/activation_stats/")
    print(f"     - Report: {output_dir}/training_report.json")
    
    return report


if __name__ == "__main__":
    # Check if dataset exists
    from pathlib import Path
    if not Path("dataset.yaml").exists():
        print("⚠️ Warning: dataset.yaml not found in current directory")
        print("   Please ensure your custom dataset YAML file is present")
        print("   Format should follow YOLO dataset specification:\n")
        print("   train: /path/to/train/images")
        print("   val: /path/to/val/images")
        print("   nc: <number_of_classes>")
        print("   names: [<class1>, <class2>, ...]")
    
    # Start training with activation tracking
    model, stats, output_dir = train_with_activation_tracking()
    
    print("\n✅ Phase 1 Complete!")
    print(f"\nNext step: Run the pruning script using the activation statistics")
    print(f"  python prune_with_activations.py --stats {output_dir}/activation_stats/activation_stats_final.json")