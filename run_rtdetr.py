import cv2
import numpy as np
import time
import psutil
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import RTDETR
from typing import Dict, Tuple, List
import json

# -----------------------------
# CONFIGURATION
# -----------------------------
IMG_PATH = "test2.png"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25

# Model paths
MODELS = {
    'original': 'rtdetr-l.pt',
    'pruned_only': 'rtdetr_l_pruned_version_2.pt',
    'pruned_finetuned': 'rtdetr_l_pruned_version_1_finetuned.pt'
}

# Visualization colors
COLORS = {
    'original': (0, 255, 0),      # Green
    'pruned_only': (0, 165, 255), # Orange
    'pruned_finetuned': (255, 0, 0) # Blue
}

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def get_memory_usage() -> float:
    """Get current process memory usage in MB"""
    return psutil.Process().memory_info().rss / 1024**2

def calculate_model_size(model_path: str) -> float:
    """Calculate model file size in MB"""
    return Path(model_path).stat().st_size / 1024**2 if Path(model_path).exists() else 0.0

def compute_detection_metrics(predictions) -> Dict:
    """Compute detailed detection metrics from model predictions"""
    metrics = {
        'num_detections': 0,
        'avg_confidence': 0.0,
        'class_distribution': {}
    }
    
    if len(predictions) > 0 and predictions[0].boxes is not None:
        boxes = predictions[0].boxes
        metrics['num_detections'] = len(boxes)
        
        if len(boxes) > 0:
            confidences = boxes.conf.cpu().numpy()
            metrics['avg_confidence'] = float(np.mean(confidences))
            
            classes = boxes.cls.cpu().numpy().astype(int)
            for cls in classes:
                metrics['class_distribution'][int(cls)] = metrics['class_distribution'].get(int(cls), 0) + 1
    
    return metrics

def run_inference(model, image: np.ndarray, num_warmup: int = 3) -> Tuple[np.ndarray, Dict]:
    """Run inference with warmup and comprehensive timing"""
    
    # Warmup runs
    for _ in range(num_warmup):
        _ = model(image, verbose=False)
    
    # Clear cache
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Measure inference
    mem_before = get_memory_usage()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    start_time = time.perf_counter()
    results = model(image, verbose=False)
    inference_time = time.perf_counter() - start_time
    
    mem_after = get_memory_usage()
    gpu_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    # Compute metrics
    metrics = compute_detection_metrics(results)
    metrics.update({
        'inference_time_ms': inference_time * 1000,
        'memory_delta_mb': mem_after - mem_before,
        'peak_gpu_memory_mb': gpu_memory,
        'fps': 1.0 / inference_time
    })
    
    return results[0].plot(), metrics

# -----------------------------
# MAIN BENCHMARK PIPELINE
# -----------------------------
def benchmark_models() -> Dict:
    """Benchmark all three model variants with research-grade metrics"""
    
    # Load image
    image = cv2.imread(IMG_PATH)
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")
    
    original_height, original_width = image.shape[:2]
    
    results = {}
    visualizations = []
    
    print("\n" + "="*80)
    print("RT-DETR MODEL BENCHMARK REPORT")
    print("="*80)
    print(f"Image: {IMG_PATH} ({original_width}x{original_height})")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*80 + "\n")
    
    for model_name, model_path in MODELS.items():
        print(f"🔬 Benchmarking: {model_name.upper()}")
        print(f"   Model path: {model_path}")
        
        # Check if model exists
        if not Path(model_path).exists():
            print(f"   ⚠️ Model not found: {model_path}")
            continue
        
        # Load model
        load_start = time.perf_counter()
        model = RTDETR(model_path)
        load_time = time.perf_counter() - load_start
        
        # Run inference
        vis_image, metrics = run_inference(model, image)
        
        # Calculate model size
        model_size = calculate_model_size(model_path)
        
        # Store results
        results[model_name] = {
            'model_path': model_path,
            'model_size_mb': model_size,
            'load_time_seconds': load_time,
            'inference_time_ms': metrics['inference_time_ms'],
            'fps': metrics['fps'],
            'memory_delta_mb': metrics['memory_delta_mb'],
            'peak_gpu_memory_mb': metrics['peak_gpu_memory_mb'],
            'num_detections': metrics['num_detections'],
            'avg_confidence': metrics['avg_confidence'],
            'class_distribution': metrics['class_distribution']
        }
        
        # Add label to visualization
        label = f"{model_name.upper()} | {metrics['inference_time_ms']:.1f}ms | {metrics['fps']:.1f}FPS"
        cv2.putText(vis_image, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[model_name], 2)
        
        visualizations.append(cv2.resize(vis_image, (640, 640)))
        
        print(f"   ✓ Load time: {load_time*1000:.1f}ms")
        print(f"   ✓ Inference: {metrics['inference_time_ms']:.1f}ms ({metrics['fps']:.1f} FPS)")
        print(f"   ✓ Detections: {metrics['num_detections']}")
        print(f"   ✓ Memory: +{metrics['memory_delta_mb']:.1f}MB")
        print(f"   ✓ Model size: {model_size:.2f}MB\n")
    
    return results, visualizations

# -----------------------------
# ANALYSIS AND REPORTING
# -----------------------------
def generate_research_report(results: Dict) -> None:
    """Generate comprehensive research-grade report"""
    
    if not results:
        print("No results to report")
        return
    
    # Calculate baseline for comparison
    baseline_model = list(results.keys())[0]
    baseline_time = results[baseline_model]['inference_time_ms']
    baseline_memory = results[baseline_model]['memory_delta_mb']
    baseline_size = results[baseline_model]['model_size_mb']
    
    # Create DataFrame
    df_data = []
    for model_name, metrics in results.items():
        df_data.append({
            'Model': model_name.upper(),
            'Inference (ms)': f"{metrics['inference_time_ms']:.2f}",
            'Speedup (x)': f"{(baseline_time / metrics['inference_time_ms']):.2f}",
            'FPS': f"{metrics['fps']:.1f}",
            'Memory Δ (MB)': f"{metrics['memory_delta_mb']:.1f}",
            'Memory vs Baseline': f"{(metrics['memory_delta_mb'] / baseline_memory):.2f}x",
            'Model Size (MB)': f"{metrics['model_size_mb']:.2f}",
            'Size Reduction': f"{(1 - metrics['model_size_mb'] / baseline_size)*100:.1f}%",
            'Detections': metrics['num_detections'],
            'Avg Confidence': f"{metrics['avg_confidence']:.3f}"
        })
    
    df = pd.DataFrame(df_data)
    
    # Save to CSV
    csv_path = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False)
    
    # Generate JSON report
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'image_path': IMG_PATH,
            'image_size': cv2.imread(IMG_PATH).shape[:2],
            'device': 'CUDA' if torch.cuda.is_available() else 'CPU',
            'conf_threshold': CONF_THRESHOLD
        },
        'models': results,
        'comparison': {
            'speedup': {
                model: baseline_time / metrics['inference_time_ms']
                for model, metrics in results.items()
            },
            'memory_efficiency': {
                model: baseline_memory / metrics['memory_delta_mb']
                for model, metrics in results.items()
            },
            'size_reduction': {
                model: (1 - metrics['model_size_mb'] / baseline_size) * 100
                for model, metrics in results.items()
            }
        }
    }
    
    json_path = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print report
    print("\n" + "="*80)
    print("RESEARCH-GRADE BENCHMARK REPORT")
    print("="*80)
    print(f"\nBaseline Model: {baseline_model.upper()}")
    print(f"Baseline Inference: {baseline_time:.2f}ms")
    print(f"Baseline Memory: {baseline_memory:.1f}MB")
    print(f"Baseline Model Size: {baseline_size:.2f}MB\n")
    
    print(df.to_string(index=False))
    
    print("\n" + "-"*80)
    print("KEY FINDINGS:")
    print("-"*80)
    
    for model_name, metrics in results.items():
        if model_name == baseline_model:
            continue
            
        speedup = baseline_time / metrics['inference_time_ms']
        memory_ratio = metrics['memory_delta_mb'] / baseline_memory
        size_reduction = (1 - metrics['model_size_mb'] / baseline_size) * 100
        
        print(f"\n{model_name.upper()}:")
        print(f"  • {speedup:.2f}x faster inference")
        print(f"  • {memory_ratio:.2f}x memory usage")
        print(f"  • {size_reduction:.1f}% smaller model size")
        print(f"  • {metrics['num_detections']} detections (baseline: {results[baseline_model]['num_detections']})")
        
        if metrics['avg_confidence'] > 0:
            conf_change = (metrics['avg_confidence'] / results[baseline_model]['avg_confidence'] - 1) * 100
            print(f"  • {conf_change:+.1f}% confidence change")
    
    print("\n" + "="*80)
    print(f"✓ CSV Report: {csv_path}")
    print(f"✓ JSON Report: {json_path}")
    print("="*80)

# -----------------------------
# VISUALIZATION
# -----------------------------
def create_comparison_panel(visualizations: List[np.ndarray], results: Dict) -> np.ndarray:
    """Create comprehensive comparison panel"""
    
    if len(visualizations) != 3:
        print(f"Warning: Expected 3 visualizations, got {len(visualizations)}")
        return visualizations[0] if visualizations else np.zeros((640, 640, 3), dtype=np.uint8)
    
    panel = np.hstack(visualizations)
    
    # Add header
    header_height = 60
    header = np.zeros((header_height, panel.shape[1], 3), dtype=np.uint8)
    
    # Add titles
    titles = ['ORIGINAL MODEL', 'PRUNED ONLY', 'PRUNED + FINETUNED']
    colors = [COLORS['original'], COLORS['pruned_only'], COLORS['pruned_finetuned']]
    
    for i, (title, color) in enumerate(zip(titles, colors)):
        x_pos = i * 640 + 20
        cv2.putText(header, title, (x_pos, 30), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        
        # Add metrics
        model_key = ['original', 'pruned_only', 'pruned_finetuned'][i]
        if model_key in results:
            metrics = results[model_key]
            cv2.putText(header, f"{metrics['inference_time_ms']:.1f}ms | {metrics['fps']:.1f}FPS", 
                       (x_pos, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    panel_with_header = np.vstack([header, panel])
    
    return panel_with_header

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    try:
        # Run benchmark
        results, visualizations = benchmark_models()
        
        # Generate research report
        generate_research_report(results)
        
        # Create and save comparison panel
        if len(visualizations) == 3:
            panel = create_comparison_panel(visualizations, results)
            
            output_path = f"rtdetr_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(output_path, panel)
            
            # Display
            cv2.imshow("RT-DETR Model Comparison", panel)
            print(f"\n✓ Visualization saved: {output_path}")
            print("Press any key to close visualization...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("\n⚠️ Incomplete visualizations - check model files exist")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Ensure all model files exist:")
        for name, path in MODELS.items():
            exists = Path(path).exists()
            print(f"  • {name}: {path} ({'✓' if exists else '✗'})")