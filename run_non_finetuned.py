import cv2
import numpy as np
import time
import psutil
import torch
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import RTDETR
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "rtdetr-l.pt"
IMAGE_PATH = "test2.jpg"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
NUM_WARMUP_RUNS = 3

# Get screen resolution for window sizing
try:
    import tkinter as tk
    root = tk.Tk()
    SCREEN_WIDTH = root.winfo_screenwidth()
    SCREEN_HEIGHT = root.winfo_screenheight()
    root.destroy()
except:
    # Fallback to common resolutions if tkinter fails
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

# Calculate optimal window size (90% of screen, maintain aspect ratio)
MAX_WINDOW_WIDTH = int(SCREEN_WIDTH * 0.9)
MAX_WINDOW_HEIGHT = int(SCREEN_HEIGHT * 0.85)

# -----------------------------
# SYSTEM INFORMATION
# -----------------------------
def get_system_info() -> Dict:
    """Collect system hardware and software information"""
    return {
        'cpu_count': psutil.cpu_count(logical=True),
        'cpu_freq_mhz': psutil.cpu_freq().max if psutil.cpu_freq() else None,
        'ram_total_gb': psutil.virtual_memory().total / (1024**3),
        'ram_available_gb': psutil.virtual_memory().available / (1024**3),
        'screen_resolution': f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}",
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else None,
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'python_version': __import__('sys').version
    }

def get_memory_usage() -> Dict:
    """Get current memory usage statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024**2,
        'vms_mb': memory_info.vms / 1024**2,
        'percent': process.memory_percent(),
        'system_ram_percent': psutil.virtual_memory().percent
    }

# -----------------------------
# MODEL ANALYSIS
# -----------------------------
def analyze_model_file(model_path: str) -> Dict:
    """Analyze model file properties"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        return {'exists': False, 'error': 'Model file not found'}
    
    stats = model_path.stat()
    
    return {
        'exists': True,
        'size_mb': stats.st_size / 1024**2,
        'size_bytes': stats.st_size,
        'modified_time': datetime.fromtimestamp(stats.st_mtime).isoformat(),
        'path': str(model_path.absolute())
    }

def load_model_with_benchmark(model_path: str) -> Tuple[RTDETR, Dict]:
    """Load model with comprehensive timing and resource tracking"""
    
    mem_before = get_memory_usage()
    load_start = time.perf_counter()
    
    model = RTDETR(model_path)
    
    load_end = time.perf_counter()
    mem_after = get_memory_usage()
    
    metrics = {
        'load_time_ms': (load_end - load_start) * 1000,
        'memory_delta_mb': mem_after['rss_mb'] - mem_before['rss_mb'],
        'memory_rss_mb': mem_after['rss_mb']
    }
    
    return model, metrics

# -----------------------------
# INFERENCE ANALYSIS
# -----------------------------
def analyze_detections(results) -> Dict:
    """Comprehensive analysis of detection results"""
    
    if len(results) == 0 or results[0].boxes is None:
        return {
            'num_detections': 0,
            'objects_detected': [],
            'avg_confidence': 0.0,
            'confidence_std': 0.0,
            'confidences': [],
            'class_distribution': {},
            'bbox_areas': [],
            'avg_bbox_area': 0.0
        }
    
    boxes = results[0].boxes
    num_detections = len(boxes)
    
    if num_detections == 0:
        return {
            'num_detections': 0,
            'objects_detected': [],
            'avg_confidence': 0.0,
            'confidence_std': 0.0,
            'confidences': [],
            'class_distribution': {},
            'bbox_areas': [],
            'avg_bbox_area': 0.0
        }
    
    # Extract detection data
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    
    # Calculate bbox areas
    bbox_areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in xyxy]
    
    # Class distribution
    class_dist = Counter(classes)
    
    # COCO class names mapping
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic_light',
        'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard',
        'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear',
        'hair_drier', 'toothbrush'
    ]
    
    # Map classes to names
    objects_detected = [coco_classes[cls] if cls < len(coco_classes) else f'class_{cls}' for cls in classes]
    
    return {
        'num_detections': num_detections,
        'objects_detected': objects_detected,
        'avg_confidence': float(np.mean(confidences)),
        'confidence_std': float(np.std(confidences)),
        'confidences': confidences.tolist(),
        'class_distribution': {int(k): int(v) for k, v in class_dist.items()},
        'class_names': {int(k): coco_classes[k] if k < len(coco_classes) else f'class_{k}' for k in class_dist.keys()},
        'bbox_areas': [float(area) for area in bbox_areas],
        'avg_bbox_area': float(np.mean(bbox_areas)) if bbox_areas else 0.0,
        'min_bbox_area': float(np.min(bbox_areas)) if bbox_areas else 0.0,
        'max_bbox_area': float(np.max(bbox_areas)) if bbox_areas else 0.0
    }

def run_inference_benchmark(model: RTDETR, image: np.ndarray, num_runs: int = 10) -> Dict:
    """Run inference with statistical benchmarking"""
    
    # Warmup runs
    for _ in range(NUM_WARMUP_RUNS):
        _ = model(image, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark runs
    inference_times = []
    memory_samples = []
    
    for _ in range(num_runs):
        mem_before = get_memory_usage()
        
        start_time = time.perf_counter()
        results = model(image, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
        end_time = time.perf_counter()
        
        mem_after = get_memory_usage()
        
        inference_times.append((end_time - start_time) * 1000)
        memory_samples.append(mem_after['rss_mb'] - mem_before['rss_mb'])
    
    # Calculate statistics
    times_array = np.array(inference_times)
    
    # Detection analysis
    detection_metrics = analyze_detections(results)
    
    return {
        'inference_time_ms': {
            'mean': float(np.mean(times_array)),
            'std': float(np.std(times_array)),
            'min': float(np.min(times_array)),
            'max': float(np.max(times_array)),
            'median': float(np.median(times_array)),
            'p95': float(np.percentile(times_array, 95)),
            'p99': float(np.percentile(times_array, 99))
        },
        'fps': {
            'mean': 1000.0 / np.mean(times_array),
            'max': 1000.0 / np.min(times_array)
        },
        'memory_delta_mb': {
            'mean': float(np.mean(memory_samples)),
            'std': float(np.std(memory_samples))
        },
        'detection_metrics': detection_metrics,
        'raw_inference_times_ms': inference_times
    }

# -----------------------------
# RESIZE IMAGE FOR SCREEN
# -----------------------------
def resize_for_screen(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Resize image to fit screen while maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

# -----------------------------
# VISUALIZATION
# -----------------------------
def create_detection_visualization(image: np.ndarray, results, metrics: Dict) -> np.ndarray:
    """Create professional detection visualization with annotations"""
    
    vis_image = results[0].plot()
    
    # Calculate overlay size based on image dimensions
    overlay_width = min(400, int(vis_image.shape[1] * 0.3))
    overlay_height = min(180, int(vis_image.shape[0] * 0.25))
    
    # Add semi-transparent overlay for metrics
    overlay = vis_image[0:overlay_height, 0:overlay_width].copy()
    cv2.rectangle(overlay, (0, 0), (overlay_width, overlay_height), (0, 0, 0), -1)
    vis_image[0:overlay_height, 0:overlay_width] = cv2.addWeighted(overlay, 0.3, vis_image[0:overlay_height, 0:overlay_width], 0.7, 0)
    
    # Add metrics text with dynamic font scale
    font_scale = 0.5 if overlay_width < 300 else 0.6
    y_offset = 25
    
    metrics_text = [
        f"Model: Pruned Only (v2)",
        f"Detections: {metrics['detection_metrics']['num_detections']}",
        f"Avg Conf: {metrics['detection_metrics']['avg_confidence']:.3f}",
        f"Inference: {metrics['inference_time_ms']['mean']:.1f}ms",
        f"FPS: {metrics['fps']['mean']:.1f}",
        f"Memory: +{metrics['memory_delta_mb']['mean']:.1f}MB"
    ]
    
    for i, text in enumerate(metrics_text):
        y_pos = y_offset + i * 22
        if y_pos < overlay_height - 10:
            cv2.putText(vis_image, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    
    return vis_image

def generate_performance_plot(metrics: Dict, output_path: str):
    """Generate performance analysis plot"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Inference time distribution
    times = metrics['raw_inference_times_ms']
    axes[0, 0].hist(times, bins=20, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 0].axvline(metrics['inference_time_ms']['mean'], color='red', linestyle='--', 
                       label=f"Mean: {metrics['inference_time_ms']['mean']:.1f}ms")
    axes[0, 0].set_xlabel('Inference Time (ms)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Inference Time Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Confidence score distribution
    confidences = metrics['detection_metrics']['confidences']
    if confidences:
        axes[0, 1].hist(confidences, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].axvline(metrics['detection_metrics']['avg_confidence'], color='red', linestyle='--',
                          label=f"Mean: {metrics['detection_metrics']['avg_confidence']:.3f}")
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Detection Confidence Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No detections', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Detection Confidence Distribution')
    
    # 3. Bounding box area distribution
    bbox_areas = metrics['detection_metrics']['bbox_areas']
    if bbox_areas:
        axes[1, 0].hist(bbox_areas, bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Bounding Box Area (pixels²)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Object Size Distribution')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No detections', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Object Size Distribution')
    
    # 4. Class distribution (top 10)
    class_dist = metrics['detection_metrics']['class_distribution']
    class_names = metrics['detection_metrics']['class_names']
    
    if class_dist:
        top_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[:10]
        labels = [class_names.get(c, f'Class_{c}') for c, _ in top_classes]
        counts = [count for _, count in top_classes]
        
        axes[1, 1].barh(labels, counts, color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Count')
        axes[1, 1].set_title('Top 10 Detected Classes')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No detections', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Top 10 Detected Classes')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------
# REPORT GENERATION
# -----------------------------
def generate_json_report(system_info: Dict, model_info: Dict, load_metrics: Dict, 
                        inference_metrics: Dict, output_path: str):
    """Generate comprehensive JSON report"""
    
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_path': MODEL_PATH,
            'image_path': IMAGE_PATH,
            'confidence_threshold': CONF_THRESHOLD,
            'iou_threshold': IOU_THRESHOLD,
            'warmup_runs': NUM_WARMUP_RUNS,
            'screen_resolution': system_info['screen_resolution']
        },
        'system_info': system_info,
        'model_analysis': model_info,
        'load_performance': load_metrics,
        'inference_performance': inference_metrics,
        'summary': {
            'total_inference_time_ms': inference_metrics['inference_time_ms']['mean'],
            'fps': inference_metrics['fps']['mean'],
            'total_detections': inference_metrics['detection_metrics']['num_detections'],
            'average_confidence': inference_metrics['detection_metrics']['avg_confidence'],
            'memory_footprint_mb': load_metrics['memory_rss_mb']
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

def print_detailed_report(system_info: Dict, model_info: Dict, load_metrics: Dict, 
                         inference_metrics: Dict):
    """Print formatted research-grade report"""
    
    print("\n" + "="*80)
    print("RT-DETR PRUNED MODEL (v2) INFERENCE ANALYSIS")
    print("="*80)
    
    print("\n📊 SYSTEM SPECIFICATIONS")
    print("-"*40)
    print(f"  CPU Cores: {system_info['cpu_count']}")
    print(f"  RAM Total: {system_info['ram_total_gb']:.1f} GB")
    print(f"  RAM Available: {system_info['ram_available_gb']:.1f} GB")
    print(f"  Screen Resolution: {system_info['screen_resolution']}")
    if system_info['gpu_available']:
        print(f"  GPU: {system_info['gpu_name']}")
        print(f"  GPU Memory: {system_info['gpu_memory_gb']:.1f} GB")
    print(f"  PyTorch: {system_info['pytorch_version']}")
    print(f"  CUDA: {system_info['cuda_version'] if system_info['cuda_version'] else 'Not available'}")
    
    print("\n📁 MODEL INFORMATION")
    print("-"*40)
    print(f"  Path: {model_info['path']}")
    print(f"  Size: {model_info['size_mb']:.2f} MB ({model_info['size_bytes']:,} bytes)")
    print(f"  Modified: {model_info['modified_time']}")
    
    print("\n⚡ LOAD PERFORMANCE")
    print("-"*40)
    print(f"  Load Time: {load_metrics['load_time_ms']:.1f} ms")
    print(f"  Memory Delta: +{load_metrics['memory_delta_mb']:.1f} MB")
    print(f"  Peak RSS: {load_metrics['memory_rss_mb']:.1f} MB")
    
    print("\n🎯 INFERENCE PERFORMANCE")
    print("-"*40)
    print(f"  Inference Time (mean ± std): {inference_metrics['inference_time_ms']['mean']:.1f} ± {inference_metrics['inference_time_ms']['std']:.1f} ms")
    print(f"  Inference Time (min/max): {inference_metrics['inference_time_ms']['min']:.1f} / {inference_metrics['inference_time_ms']['max']:.1f} ms")
    print(f"  Inference Time (p95/p99): {inference_metrics['inference_time_ms']['p95']:.1f} / {inference_metrics['inference_time_ms']['p99']:.1f} ms")
    print(f"  FPS (mean): {inference_metrics['fps']['mean']:.1f}")
    print(f"  Memory Delta: +{inference_metrics['memory_delta_mb']['mean']:.1f} ± {inference_metrics['memory_delta_mb']['std']:.1f} MB")
    
    print("\n🔍 DETECTION RESULTS")
    print("-"*40)
    det_metrics = inference_metrics['detection_metrics']
    print(f"  Total Detections: {det_metrics['num_detections']}")
    
    if det_metrics['num_detections'] > 0:
        print(f"  Average Confidence: {det_metrics['avg_confidence']:.3f} ± {det_metrics['confidence_std']:.3f}")
        print(f"  Confidence Range: [{np.min(det_metrics['confidences']):.3f}, {np.max(det_metrics['confidences']):.3f}]")
        print(f"  Average BBox Area: {det_metrics['avg_bbox_area']:.0f} pixels²")
        print(f"  BBox Area Range: [{det_metrics['min_bbox_area']:.0f}, {det_metrics['max_bbox_area']:.0f}] pixels²")
        
        print("\n  Class Distribution:")
        for class_id, count in sorted(det_metrics['class_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
            class_name = det_metrics['class_names'].get(class_id, f'Class_{class_id}')
            print(f"    • {class_name}: {count}")
    else:
        print("  No objects detected")
    
    print("\n" + "="*80)

# -----------------------------
# MAIN EXECUTION
# -----------------------------
def main():
    """Main execution pipeline"""
    
    print(f"\n🚀 Starting inference benchmark on pruned model (v2)")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Image: {IMAGE_PATH}")
    print(f"   Confidence threshold: {CONF_THRESHOLD}")
    print(f"   Screen: {SCREEN_WIDTH}x{SCREEN_HEIGHT} (max window: {MAX_WINDOW_WIDTH}x{MAX_WINDOW_HEIGHT})")
    
    # Load image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    
    original_h, original_w = image.shape[:2]
    print(f"   Image size: {original_w}x{original_h}")
    
    # Get system info
    system_info = get_system_info()
    
    # Analyze model file
    model_info = analyze_model_file(MODEL_PATH)
    if not model_info['exists']:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    # Load model with benchmarking
    model, load_metrics = load_model_with_benchmark(MODEL_PATH)
    
    # Run inference benchmark
    inference_metrics = run_inference_benchmark(model, image, num_runs=10)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create visualization
    vis_image = create_detection_visualization(image, model(image, verbose=False, conf=CONF_THRESHOLD), inference_metrics)
    vis_path = f"inference_result_{timestamp}.png"
    cv2.imwrite(vis_path, vis_image)
    
    # Resize for screen display
    display_image = resize_for_screen(vis_image, MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT)
    
    # Generate performance plot
    plot_path = f"performance_analysis_{timestamp}.png"
    generate_performance_plot(inference_metrics, plot_path)
    
    # Generate JSON report
    report_path = f"inference_report_{timestamp}.json"
    generate_json_report(system_info, model_info, load_metrics, inference_metrics, report_path)
    
    # Print detailed report
    print_detailed_report(system_info, model_info, load_metrics, inference_metrics)
    
    # Save summary CSV
    summary_data = {
        'Metric': ['Inference Time (ms)', 'FPS', 'Memory Delta (MB)', 'Detections', 'Avg Confidence'],
        'Value': [
            f"{inference_metrics['inference_time_ms']['mean']:.2f} ± {inference_metrics['inference_time_ms']['std']:.2f}",
            f"{inference_metrics['fps']['mean']:.2f}",
            f"{inference_metrics['memory_delta_mb']['mean']:.2f} ± {inference_metrics['memory_delta_mb']['std']:.2f}",
            inference_metrics['detection_metrics']['num_detections'],
            f"{inference_metrics['detection_metrics']['avg_confidence']:.3f}"
        ]
    }
    df = pd.DataFrame(summary_data)
    csv_path = f"inference_summary_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  • Visualization: {vis_path}")
    print(f"  • Performance Plot: {plot_path}")
    print(f"  • JSON Report: {report_path}")
    print(f"  • CSV Summary: {csv_path}")
    
    # Create named window and resize to fit screen
    window_name = "RT-DETR Pruned Model (v2) - Inference Results"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_image.shape[1], display_image.shape[0])
    
    # Display the resized image
    cv2.imshow(window_name, display_image)
    
    print(f"\n📺 Display window size: {display_image.shape[1]}x{display_image.shape[0]}")
    print("Press any key to close visualization...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()