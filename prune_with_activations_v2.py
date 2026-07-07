import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import RTDETR
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
import argparse
import time
import gc
import shutil
from datetime import datetime
import psutil
import cv2

class ActivationBasedPruner:
    """
    Prunes RT-DETR based on activation statistics and saves THREE model versions:
    1. rtdetr-l.pt (original, untouched - already exists)
    2. rtdetr-l-pruned.pt (pruned only, no fine-tuning)  
    3. rtdetr-l-pruned-finetuned.pt (pruned + fine-tuned)
    """
    
    def __init__(self, model_path, activation_stats_path, output_dir=".", device='cuda'):
        print(f"\n{'='*70}")
        print("ACTIVATION-BASED PRUNING - THREE MODEL VERSIONS")
        print(f"{'='*70}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"📊 Loading activation statistics: {activation_stats_path}")
        
        # Load original model (this will be saved as rtdetr-l.pt if not exists)
        print(f"\n🔧 Loading model: {model_path}")
        self.model = RTDETR(model_path)
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Create a deep copy for pruning (Version 2)
        self.pruned_model = self._copy_model(self.model)
        
        # Load activation stats
        with open(activation_stats_path, 'r') as f:
            self.activation_stats = json.load(f)
        
        # Build importance mapping
        self.layer_importance = {}
        self._build_importance_mapping()
        
        # Tracking metrics
        self.pruning_metrics = {
            'timestamp': datetime.now().isoformat(),
            'original_model': model_path,
            'pruned_model': None,
            'finetuned_model': None,
            'pruning_stats': {},
            'model_sizes_mb': {},
            'parameter_counts': {}
        }
    
    def _copy_model(self, model):
        """Create a deep copy of the model for pruning"""
        import copy
        return copy.deepcopy(model)
    
    def _build_importance_mapping(self):
        """Build mapping of layer names to importance scores from activation stats"""
        for layer_name, stats in self.activation_stats.items():
            importance = stats.get('importance_score', [])
            if importance:
                self.layer_importance[layer_name] = {
                    'importance': np.array(importance),
                    'num_neurons': len(importance),
                    'is_backbone': stats.get('is_backbone', False),
                    'is_encoder': stats.get('is_encoder', False),
                    'is_decoder': stats.get('is_decoder', False)
                }
        
        print(f"   Mapped {len(self.layer_importance)} layers with importance scores")
    
    def apply_activation_pruning(self, prune_percent=0.25, protect_backbone=True):
        """
        Apply pruning based on activation importance scores to create Version 2
        """
        print(f"\n✂️ Creating Version 2: rtdetr-l-pruned.pt")
        print(f"   Prune percent: {prune_percent*100:.0f}%")
        print(f"   Protect backbone: {protect_backbone}")
        
        total_params_before = 0
        total_params_after = 0
        total_neurons_before = 0
        total_neurons_removed = 0
        
        pruned_layers_info = []
        
        # Protected layer patterns (early backbone - critical for small objects)
        protected_patterns = ['stem', 'conv1', 'layer1', 'backbone.0', 'backbone.1']
        
        for name, module in self.pruned_model.model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            
            # Skip protected layers
            should_skip = False
            for pattern in protected_patterns:
                if pattern in name.lower():
                    should_skip = True
                    break
            
            if protect_backbone and ('backbone' in name.lower() or 'resnet' in name.lower()):
                if any(p in name.lower() for p in ['layer1', 'layer2']):
                    should_skip = True
            
            if should_skip:
                print(f"   ⚠️ Skipping (protected): {name}")
                continue
            
            # Check if we have importance scores for this layer
            if name not in self.layer_importance:
                print(f"   ⚠️ No activation data for: {name}")
                continue
            
            importance = self.layer_importance[name]['importance']
            num_neurons = len(importance)
            total_neurons_before += num_neurons
            
            # Determine number of neurons to prune
            num_to_prune = int(num_neurons * prune_percent)
            
            # Get indices of least important neurons
            least_important_indices = np.argsort(importance)[:num_to_prune]
            
            # Count parameters before
            params_before = module.weight.numel()
            total_params_before += params_before
            
            # Apply pruning based on layer type
            if isinstance(module, nn.Conv2d):
                self._prune_conv_channels(module, least_important_indices)
            elif isinstance(module, nn.Linear):
                self._prune_linear_neurons(module, least_important_indices)
            
            # Count parameters after (non-zero)
            params_after = (module.weight != 0).sum().item()
            total_params_after += params_after
            
            neurons_removed = num_to_prune
            total_neurons_removed += neurons_removed
            
            pruned_layers_info.append({
                'layer_name': name,
                'layer_type': 'Conv2d' if isinstance(module, nn.Conv2d) else 'Linear',
                'total_neurons': num_neurons,
                'neurons_pruned': neurons_removed,
                'prune_percent': (neurons_removed / num_neurons) * 100,
                'params_before': params_before,
                'params_kept': params_after,
                'compression_ratio': params_after / params_before if params_before > 0 else 1.0
            })
            
            print(f"   ✓ {name}: {neurons_removed}/{num_neurons} neurons pruned "
                  f"({neurons_removed/num_neurons*100:.1f}%)")
        
        # Make pruning permanent (remove masks) for Version 2
        for name, module in self.pruned_model.model.named_modules():
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
        
        # Store pruning stats
        self.pruning_metrics['pruning_stats'] = {
            'prune_percent': prune_percent,
            'protect_backbone': protect_backbone,
            'total_neurons_before': total_neurons_before,
            'total_neurons_removed': total_neurons_removed,
            'compression_ratio': total_params_after / total_params_before if total_params_before > 0 else 1.0,
            'pruned_layers_count': len(pruned_layers_info),
            'pruned_layers': pruned_layers_info
        }
        
        print(f"\n📊 Version 2 Pruning Summary:")
        print(f"   Total neurons before: {total_neurons_before:,}")
        print(f"   Total neurons removed: {total_neurons_removed:,}")
        print(f"   Compression ratio: {self.pruning_metrics['pruning_stats']['compression_ratio']:.2f}x")
        print(f"   Parameter reduction: {(1 - self.pruning_metrics['pruning_stats']['compression_ratio'])*100:.1f}%")
        
        return self.pruned_model
    
    def _prune_conv_channels(self, conv_layer, channel_indices):
        """Prune entire output channels from Conv2d layer"""
        mask = torch.ones(conv_layer.out_channels, device=self.device)
        mask[channel_indices] = 0
        weight_mask = mask.view(-1, 1, 1, 1).expand_as(conv_layer.weight)
        conv_layer.weight.data *= weight_mask
        
        if conv_layer.bias is not None:
            conv_layer.bias.data *= mask
    
    def _prune_linear_neurons(self, linear_layer, neuron_indices):
        """Prune output neurons from Linear layer"""
        mask = torch.ones(linear_layer.out_features, device=self.device)
        mask[neuron_indices] = 0
        weight_mask = mask.view(-1, 1).expand_as(linear_layer.weight)
        linear_layer.weight.data *= weight_mask
        
        if linear_layer.bias is not None:
            linear_layer.bias.data *= mask
    
    def save_models(self):
        """
        Save all three model versions:
        1. rtdetr-l.pt (original - copy if needed)
        2. rtdetr-l-pruned.pt (pruned only)
        """
        
        # Version 1: Original model (rtdetr-l.pt)
        v1_path = self.output_dir / "rtdetr-l.pt"
        if not v1_path.exists():
            print(f"\n💾 Saving Version 1 (original): {v1_path}")
            self.model.save(str(v1_path))
        else:
            print(f"\n✓ Version 1 already exists: {v1_path}")
        
        # Get original model size
        v1_size = v1_path.stat().st_size / (1024**2) if v1_path.exists() else 0
        
        # Version 2: Pruned only (rtdetr-l-pruned.pt)
        v2_path = self.output_dir / "rtdetr-l-pruned.pt"
        print(f"\n💾 Saving Version 2 (pruned only): {v2_path}")
        self.pruned_model.save(str(v2_path))
        v2_size = v2_path.stat().st_size / (1024**2)
        
        # Calculate parameter counts
        def count_params(model):
            return sum(p.numel() for p in model.parameters())
        
        v1_params = count_params(self.model)
        v2_params = count_params(self.pruned_model)
        
        # Update metrics
        self.pruning_metrics['model_sizes_mb'] = {
            'rtdetr-l.pt': v1_size,
            'rtdetr-l-pruned.pt': v2_size,
            'size_reduction_percent': (1 - v2_size/v1_size) * 100 if v1_size > 0 else 0
        }
        
        self.pruning_metrics['parameter_counts'] = {
            'rtdetr-l.pt': v1_params,
            'rtdetr-l-pruned.pt': v2_params,
            'parameter_reduction_percent': (1 - v2_params/v1_params) * 100 if v1_params > 0 else 0
        }
        
        print(f"\n📊 Model Size Comparison:")
        print(f"   Version 1 (rtdetr-l.pt):           {v1_size:.2f} MB ({v1_params:,} params)")
        print(f"   Version 2 (rtdetr-l-pruned.pt):    {v2_size:.2f} MB ({v2_params:,} params)")
        print(f"   Size reduction: {self.pruning_metrics['model_sizes_mb']['size_reduction_percent']:.1f}%")
        print(f"   Parameter reduction: {self.pruning_metrics['parameter_counts']['parameter_reduction_percent']:.1f}%")
        
        # Save pruning report
        self._save_pruning_report(v1_path, v2_path)
        
        return v1_path, v2_path
    
    def _save_pruning_report(self, v1_path, v2_path):
        """Save comprehensive pruning report"""
        
        report = {
            'pruning_info': self.pruning_metrics,
            'file_locations': {
                'original_model': str(v1_path),
                'pruned_model': str(v2_path),
                'finetuned_model': 'rtdetr-l-pruned-finetuned.pt (to be created after fine-tuning)'
            },
            'usage_instructions': {
                'inference_pruned': f"model = RTDETR('{v2_path}')",
                'inference_original': f"model = RTDETR('{v1_path}')",
                'fine_tune_pruned': f"model = RTDETR('{v2_path}')\nmodel.train(data='dataset.yaml', epochs=50, lr0=0.0001)"
            }
        }
        
        report_path = self.output_dir / "pruning_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Pruning report saved: {report_path}")
    
    def validate_model(self, test_image_path="test2.png"):
        """Quick validation to ensure pruned model still works"""
        print("\n🔍 Validating pruned model (Version 2)...")
        
        img = cv2.imread(test_image_path)
        if img is None:
            print("   ⚠️ No test image found, skipping validation")
            return None
        
        with torch.no_grad():
            results = self.pruned_model(img, verbose=False, conf=0.25)
        
        if len(results) > 0 and results[0].boxes is not None:
            num_dets = len(results[0].boxes)
            print(f"   ✓ Pruned model produces {num_dets} detections on test image")
        else:
            print("   ⚠️ No detections produced (may need fine-tuning)")
        
        return results


class FineTuner:
    """Handle fine-tuning of pruned model to create Version 3"""
    
    def __init__(self, pruned_model_path, data_yaml, output_dir="."):
        self.pruned_model_path = Path(pruned_model_path)
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.finetuned_path = self.output_dir / "rtdetr-l-pruned-finetuned.pt"
    
    def finetune(self, epochs=50, lr=0.0001, batch=6, imgsz=1088):
        """
        Fine-tune the pruned model to recover accuracy
        Creates Version 3: rtdetr-l-pruned-finetuned.pt
        """
        
        print(f"\n{'='*70}")
        print("CREATING VERSION 3: rtdetr-l-pruned-finetuned.pt")
        print(f"{'='*70}")
        
        if not self.pruned_model_path.exists():
            print(f"❌ Pruned model not found: {self.pruned_model_path}")
            return None
        
        print(f"\n🔧 Loading pruned model: {self.pruned_model_path}")
        model = RTDETR(str(self.pruned_model_path))
        
        print(f"🚀 Starting fine-tuning...")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {lr}")
        print(f"   Batch size: {batch}")
        print(f"   Image size: {imgsz}")
        
        start_time = time.time()
        
        # Train the model
        results = model.train(
            data=self.data_yaml,
            epochs=epochs,
            lr0=lr,
            batch=batch,
            imgsz=imgsz,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            workers=8,
            project=str(self.output_dir),
            name='finetune_run',
            pretrained=False,
            verbose=True
        )
        
        finetune_time = time.time() - start_time
        
        # Save Version 3
        model.save(str(self.finetuned_path))
        v3_size = self.finetuned_path.stat().st_size / (1024**2)
        
        print(f"\n✅ Version 3 created: {self.finetuned_path}")
        print(f"   File size: {v3_size:.2f} MB")
        print(f"   Fine-tuning time: {finetune_time/60:.1f} minutes")
        
        # Save fine-tuning report
        self._save_finetune_report(finetune_time, v3_size)
        
        return model
    
    def _save_finetune_report(self, finetune_time, v3_size):
        """Save fine-tuning report"""
        
        # Get sizes of other versions
        v1_path = self.output_dir / "rtdetr-l.pt"
        v2_path = self.pruned_model_path
        
        v1_size = v1_path.stat().st_size / (1024**2) if v1_path.exists() else 0
        v2_size = v2_path.stat().st_size / (1024**2) if v2_path.exists() else 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'finetuning_config': {
                'epochs': 50,
                'learning_rate': 0.0001,
                'batch_size': 6,
                'image_size': 1088
            },
            'finetuning_time_minutes': finetune_time / 60,
            'model_sizes_mb': {
                'rtdetr-l.pt (original)': v1_size,
                'rtdetr-l-pruned.pt (pruned)': v2_size,
                'rtdetr-l-pruned-finetuned.pt (finetuned)': v3_size
            },
            'size_reductions': {
                'pruned_vs_original_percent': (1 - v2_size/v1_size) * 100 if v1_size > 0 else 0,
                'finetuned_vs_original_percent': (1 - v3_size/v1_size) * 100 if v1_size > 0 else 0,
                'finetuned_vs_pruned_percent': (1 - v3_size/v2_size) * 100 if v2_size > 0 else 0
            }
        }
        
        report_path = self.output_dir / "finetuning_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Fine-tuning report saved: {report_path}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("FINAL MODEL SUMMARY")
        print(f"{'='*70}")
        print(f"   Version 1 (original):           {v1_size:.2f} MB")
        print(f"   Version 2 (pruned only):        {v2_size:.2f} MB")
        print(f"   Version 3 (pruned+finetuned):   {v3_size:.2f} MB")
        print(f"\n   Total reduction (v3 vs v1):     {(1 - v3_size/v1_size)*100:.1f}%")
        print(f"{'='*70}")


def analyze_activation_stats(activation_stats_path):
    """Generate detailed analysis of activation statistics"""
    
    with open(activation_stats_path, 'r') as f:
        stats = json.load(f)
    
    print("\n" + "="*70)
    print("ACTIVATION STATISTICS ANALYSIS")
    print("="*70)
    
    all_importances = []
    layer_summaries = []
    
    for layer_name, layer_stats in stats.items():
        importance = layer_stats.get('importance_score', [])
        if importance:
            imp_array = np.array(importance)
            all_importances.extend(imp_array)
            layer_summaries.append({
                'layer': layer_name,
                'mean_importance': imp_array.mean(),
                'num_neurons': len(imp_array),
                'is_backbone': layer_stats.get('is_backbone', False)
            })
    
    if all_importances:
        print(f"\n📊 Overall Importance Statistics:")
        print(f"   Mean importance: {np.mean(all_importances):.4f}")
        print(f"   Std importance: {np.std(all_importances):.4f}")
        print(f"   25th percentile: {np.percentile(all_importances, 25):.4f}")
        print(f"   Median: {np.percentile(all_importances, 50):.4f}")
    
    # Identify best pruning candidates
    layer_summaries.sort(key=lambda x: x['mean_importance'])
    
    print(f"\n🎯 Top 10 Pruning Candidates (lowest mean activation):")
    for i, layer in enumerate(layer_summaries[:10], 1):
        print(f"   {i}. {layer['layer']}: mean={layer['mean_importance']:.4f}, "
              f"neurons={layer['num_neurons']}")
    
    return layer_summaries


def main():
    parser = argparse.ArgumentParser(description='Prune RT-DETR using activation statistics - Creates 3 model versions')
    parser.add_argument('--model', type=str, default='rtdetr-l.pt',
                        help='Path to trained model')
    parser.add_argument('--stats', type=str, required=True,
                        help='Path to activation statistics JSON file')
    parser.add_argument('--prune-percent', type=float, default=0.25,
                        help='Percentage of least active neurons to prune')
    parser.add_argument('--protect-backbone', action='store_true', default=True,
                        help='Protect early backbone layers from pruning')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for models')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze activation stats without pruning')
    parser.add_argument('--finetune', action='store_true',
                        help='Also fine-tune after pruning to create Version 3')
    parser.add_argument('--data', type=str, default='dataset.yaml',
                        help='Dataset YAML for fine-tuning')
    parser.add_argument('--finetune-epochs', type=int, default=50,
                        help='Epochs for fine-tuning')
    
    args = parser.parse_args()
    
    print("="*70)
    print("RT-DETR ACTIVATION-BASED PRUNING - THREE VERSIONS")
    print("="*70)
    print("\nModel versions to be created:")
    print("   1. rtdetr-l.pt (original - already exists)")
    print("   2. rtdetr-l-pruned.pt (pruned only - created now)")
    print("   3. rtdetr-l-pruned-finetuned.pt (pruned+finetuned - optional)")
    
    # Analyze activation statistics first
    layer_summaries = analyze_activation_stats(args.stats)
    
    if args.analyze_only:
        print("\n✅ Analysis complete. Run without --analyze-only to perform pruning.")
        return
    
    # Create pruner and apply activation-based pruning
    pruner = ActivationBasedPruner(
        args.model, 
        args.stats, 
        output_dir=args.output_dir
    )
    
    # Apply pruning to create Version 2
    pruned_model = pruner.apply_activation_pruning(
        prune_percent=args.prune_percent,
        protect_backbone=args.protect_backbone
    )
    
    # Save Version 1 and Version 2
    v1_path, v2_path = pruner.save_models()
    
    # Validate Version 2
    pruner.validate_model()
    
    # Optionally fine-tune to create Version 3
    if args.finetune:
        finetuner = FineTuner(
            pruned_model_path=v2_path,
            data_yaml=args.data,
            output_dir=args.output_dir
        )
        
        v3_model = finetuner.finetune(
            epochs=args.finetune_epochs,
            lr=0.0001,
            batch=6,
            imgsz=1088
        )
    
    print("\n" + "="*70)
    print("✅ PRUNING COMPLETE")
    print("="*70)
    print("\n📁 Model files created:")
    print(f"   • rtdetr-l.pt                    (Version 1 - original)")
    print(f"   • rtdetr-l-pruned.pt             (Version 2 - pruned only)")
    if args.finetune:
        print(f"   • rtdetr-l-pruned-finetuned.pt  (Version 3 - pruned + fine-tuned)")
    
    print("\n📄 Reports saved:")
    print(f"   • pruning_report.json")
    if args.finetune:
        print(f"   • finetuning_report.json")
    
    print("\n🚀 Usage examples:")
    print(f"   # Load original model:")
    print(f"   model = RTDETR('rtdetr-l.pt')")
    print(f"\n   # Load pruned model (faster inference):")
    print(f"   model = RTDETR('rtdetr-l-pruned.pt')")
    if args.finetune:
        print(f"\n   # Load fine-tuned pruned model (best balance):")
        print(f"   model = RTDETR('rtdetr-l-pruned-finetuned.pt')")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()