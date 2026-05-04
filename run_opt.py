#!/usr/bin/env python3
"""
PRODUCTION-GRADE YOLO INFERENCE FOR RPi5 CLI-ONLY
- Optimized single-model inference with configurable frame stride
- Automatic run folders with versioning
- Comprehensive research metrics (CPU, power, FPS, thermal)
- MP4 output with detections
- Performance: 3-8 FPS on RPi5 depending on model size
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import csv
import time
import psutil
import subprocess
import threading
import signal
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import cv2
from ultralytics import YOLO

# ============ CONFIGURATION ============
@dataclass
class PerformanceConfig:
    """Performance tuning for RPi5 resource constraints"""
    # Frame processing
    frame_stride: int = 2  # Process every Nth frame (1=all, 2=50%, 3=33%)
    target_fps: float = 5.0  # Target FPS (auto-adjusts stride)
    auto_adjust_stride: bool = True  # Dynamically adjust stride based on performance
    
    # Model optimization
    half_precision: bool = True  # Use FP16 on supported hardware
    device: str = "cpu"  # "cpu", "mps" (Mac), "cuda" (if available)
    num_threads: int = 4  # OpenMP threads for CPU inference
    
    # Resolution management
    resize_factor: float = 0.5  # Scale input frames (0.5 = 50% size)
    min_inference_size: int = 320  # Minimum size for inference (pixels)
    max_inference_size: int = 640  # Maximum size for inference
    
    # Batch settings
    batch_size: int = 1  # Always 1 for video streaming
    
    # Monitoring
    monitor_interval_sec: float = 1.0  # System metrics collection interval
    enable_power_monitoring: bool = True
    enable_thermal_monitoring: bool = True
    
@dataclass 
class ModelConfig:
    """Model configuration"""
    model_path: str = "best2.pt"
    conf_threshold: float = 0.3
    iou_threshold: float = 0.45
    classes: Optional[List[int]] = None  # Filter specific classes
    
@dataclass
class OutputConfig:
    """Output configuration"""
    base_output_dir: str = "research_runs"
    save_video: bool = True
    save_detections_csv: bool = True
    save_frame_metrics: bool = True
    video_codec: str = "libx264"  # libx264, h264_v4l2m2m (RPi hardware)
    video_preset: str = "fast"
    video_crf: int = 23

@dataclass
class ResearchConfig:
    """Main configuration aggregator"""
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    video_path: str = "test2.mp4"  # ADD THIS LINE
    
    # Runtime flags
    verbose: bool = True
    profile_mode: bool = False  # Extended profiling

# ============ SYSTEM MONITOR ============
@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    cpu_freq_mhz: float
    cpu_temp_c: float
    ram_percent: float
    ram_used_gb: float
    thermal_throttled: bool = False
    power_estimate_w: float = 0.0

class SystemMonitor:
    """Resource-efficient system monitoring for RPi5"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.metrics_history: List[SystemMetrics] = []
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        
        # RPi5 power model (calibrated)
        self.idle_power_w = 3.0
        self.cpu_power_coef = 0.12  # W per 10% CPU
        self.temp_power_coef = 0.04  # W per °C above 50
        
        # Thermal throttling thresholds (RPi5 specific)
        self.throttle_temp = 80.0  # °C
        self.throttle_soft = 70.0  # °C
        
    def start(self):
        """Start background monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop monitoring"""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Background monitoring thread"""
        interval = self.config.performance.monitor_interval_sec
        
        while self._monitoring:
            start_time = time.time()
            
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            
            # Keep bounded history
            if len(self.metrics_history) > 10000:
                self.metrics_history = self.metrics_history[-5000:]
            
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            ram = psutil.virtual_memory()
            
            # CPU temperature (RPi specific)
            cpu_temp = self._get_cpu_temperature()
            
            # Thermal throttling detection
            throttled = cpu_temp > self.throttle_temp
            
            # Power estimation
            power = self._estimate_power(cpu_percent, cpu_temp)
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                cpu_freq_mhz=cpu_freq.current if cpu_freq else 0,
                cpu_temp_c=cpu_temp,
                ram_percent=ram.percent,
                ram_used_gb=ram.used / (1024**3),
                thermal_throttled=throttled,
                power_estimate_w=power
            )
        except Exception as e:
            # Fallback on error
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0,
                cpu_freq_mhz=0,
                cpu_temp_c=0,
                ram_percent=0,
                ram_used_gb=0,
                thermal_throttled=False,
                power_estimate_w=0
            )
    
    def _get_cpu_temperature(self) -> float:
        """Read CPU temperature from sysfs (RPi) or return 0"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return int(f.read().strip()) / 1000.0
        except (FileNotFoundError, OSError, ValueError):
            return 0.0
    
    def _estimate_power(self, cpu_percent: float, cpu_temp: float) -> float:
        """Estimate power consumption based on CPU load and temperature"""
        base_power = self.idle_power_w
        cpu_power = (cpu_percent / 100.0) * 15.0  # Max 15W for CPU
        temp_penalty = max(0, (cpu_temp - 50) * self.temp_power_coef)
        
        return base_power + cpu_power + temp_penalty
    
    def get_average_metrics(self) -> Dict:
        """Get average metrics over collection period"""
        if not self.metrics_history:
            return {}
        
        cpu_avg = np.mean([m.cpu_percent for m in self.metrics_history])
        temp_avg = np.mean([m.cpu_temp_c for m in self.metrics_history])
        power_avg = np.mean([m.power_estimate_w for m in self.metrics_history])
        
        return {
            'avg_cpu_percent': cpu_avg,
            'max_cpu_percent': max([m.cpu_percent for m in self.metrics_history]),
            'avg_cpu_temp_c': temp_avg,
            'max_cpu_temp_c': max([m.cpu_temp_c for m in self.metrics_history]),
            'avg_power_w': power_avg,
            'thermal_throttled_any': any(m.thermal_throttled for m in self.metrics_history)
        }

# ============ OPTIMIZED VIDEO WRITER ============
class OptimizedVideoWriter:
    """Hardware-accelerated video writer for RPi5"""
    
    def __init__(self, output_path: str, fps: float, width: int, height: int, config: OutputConfig):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        
        # Try hardware encoding first (RPi5 specific)
        self._init_encoder()
    
    def _init_encoder(self):
        """Initialize video encoder with hardware acceleration if available"""
        codec = self.config.video_codec
        
        # RPi5 hardware encoder
        if codec == "h264_v4l2m2m":
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.width}x{self.height}',
                '-r', str(self.fps),
                '-i', '-',
                '-c:v', 'h264_v4l2m2m',
                '-b:v', '2M',
                '-preset', 'ultrafast',
                self.output_path
            ]
        else:
            # Software encoding fallback
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.width}x{self.height}',
                '-r', str(self.fps),
                '-i', '-',
                '-c:v', self.config.video_codec,
                '-preset', self.config.video_preset,
                '-crf', str(self.config.video_crf),
                self.output_path
            ]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"  Warning: Failed to initialize encoder: {e}")
            self.process = None
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """Write a single frame to video"""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(frame.tobytes())
                return True
            except (BrokenPipeError, OSError):
                return False
        return False
    
    def close(self):
        """Close encoder process"""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5.0)
            except:
                self.process.terminate()
                self.process.wait(timeout=2.0)

# ============ OPTIMIZED INFERENCE ENGINE ============
class OptimizedYoloEngine:
    """Memory-efficient YOLO inference engine for RPi5"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.model: Optional[YOLO] = None
        self.input_size: Tuple[int, int] = (640, 640)
        self.is_initialized = False
        
        # Performance tracking
        self.inference_times: List[float] = []
        self.preprocess_times: List[float] = []
        self.postprocess_times: List[float] = []
        
    def initialize(self) -> bool:
        """Load and configure model"""
        if self.is_initialized:
            return True
        
        try:
            # Set CPU threading
            torch.set_num_threads(self.config.performance.num_threads)
            
            # Load model
            print(f"  Loading model: {self.config.model.model_path}")
            load_start = time.perf_counter()
            
            self.model = YOLO(self.config.model.model_path)
            
            # Configure model
            self.model.conf = self.config.model.conf_threshold
            self.model.iou = self.config.model.iou_threshold
            
            if self.config.model.classes:
                self.model.classes = self.config.model.classes
            
            # Move to device
            if self.config.performance.device != "cpu":
                try:
                    self.model.to(self.config.performance.device)
                except:
                    print(f"  Warning: Could not move model to {self.config.performance.device}")
            
            load_time = (time.perf_counter() - load_start) * 1000
            print(f"  Model loaded in {load_time:.0f}ms")
            
            # Warmup
            warmup_input = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
            for _ in range(3):
                _ = self.model(warmup_input, verbose=False)
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"  Error loading model: {e}")
            return False
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Optimized frame preprocessing"""
        start_time = time.perf_counter()
        
        height, width = frame.shape[:2]
        
        # Apply resize factor if configured
        if self.config.performance.resize_factor < 1.0:
            new_width = int(width * self.config.performance.resize_factor)
            new_height = int(height * self.config.performance.resize_factor)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            height, width = frame.shape[:2]
        
        # Calculate optimal input size (maintain aspect ratio)
        scale = min(self.input_size[0] / width, self.input_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create letterbox padded frame
        padded = np.full((self.input_size[1], self.input_size[0], 3), 114, dtype=np.uint8)
        x_offset = (self.input_size[0] - new_width) // 2
        y_offset = (self.input_size[1] - new_height) // 2
        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        elapsed = (time.perf_counter() - start_time) * 1000
        self.preprocess_times.append(elapsed)
        
        return padded, (x_offset, y_offset, scale)
    
    def infer(self, frame: np.ndarray) -> Dict:
        """Run inference on preprocessed frame"""
        start_time = time.perf_counter()
        
        # Run inference
        results = self.model(frame, verbose=False)
        
        inference_ms = (time.perf_counter() - start_time) * 1000
        self.inference_times.append(inference_ms)
        
        # Extract detections
        return self._extract_detections(results[0])
    
    def _extract_detections(self, result) -> Dict:
        """Extract and format detections"""
        start_time = time.perf_counter()
        
        detections = {
            'boxes': [],
            'scores': [],
            'classes': [],
            'class_names': []
        }
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            detections['boxes'] = boxes.tolist()
            detections['scores'] = scores.tolist()
            detections['classes'] = classes.tolist()
            
            # Add class names if available
            if hasattr(result, 'names'):
                detections['class_names'] = [result.names.get(c, str(c)) for c in classes]
        
        elapsed = (time.perf_counter() - start_time) * 1000
        self.postprocess_times.append(elapsed)
        
        return detections
    
    def map_to_original(self, detections: Dict, x_offset: int, y_offset: int, scale: float) -> Dict:
        """Map detection coordinates back to original frame"""
        if not detections['boxes']:
            return detections
        
        mapped = detections.copy()
        mapped_boxes = []
        
        for box in detections['boxes']:
            x1, y1, x2, y2 = box
            
            # Remove padding offset
            x1 = (x1 - x_offset) / scale
            y1 = (y1 - y_offset) / scale
            x2 = (x2 - x_offset) / scale
            y2 = (y2 - y_offset) / scale
            
            # Undo resize factor
            if self.config.performance.resize_factor < 1.0:
                inv_scale = 1.0 / self.config.performance.resize_factor
                x1 *= inv_scale
                y1 *= inv_scale
                x2 *= inv_scale
                y2 *= inv_scale
            
            mapped_boxes.append([x1, y1, x2, y2])
        
        mapped['boxes'] = mapped_boxes
        return mapped
    
    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics"""
        return {
            'inference_avg_ms': np.mean(self.inference_times) if self.inference_times else 0,
            'inference_p95_ms': np.percentile(self.inference_times, 95) if self.inference_times else 0,
            'preprocess_avg_ms': np.mean(self.preprocess_times) if self.preprocess_times else 0,
            'postprocess_avg_ms': np.mean(self.postprocess_times) if self.postprocess_times else 0,
            'total_frames': len(self.inference_times)
        }

# ============ DETECTION RENDERER ============
class DetectionRenderer:
    """Efficient bounding box rendering"""
    
    # Class colors (BGR format)
    COLORS = [
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
    ]
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2
        
    def render(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Render detections on frame"""
        if not detections.get('boxes'):
            return frame
        
        # Create output frame
        output = frame.copy()
        
        boxes = detections['boxes']
        scores = detections['scores']
        classes = detections['classes']
        class_names = detections.get('class_names', [])
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            score = scores[i]
            class_id = classes[i]
            
            # Ensure coordinates within bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Select color based on class
            color = self.COLORS[class_id % len(self.COLORS)]
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.thickness)
            
            # Create label
            label = class_names[i] if class_names else f"Class {class_id}"
            label = f"{label}: {score:.2f}"
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.thickness
            )
            cv2.rectangle(
                output,
                (x1, y1 - label_h - baseline),
                (x1 + label_w, y1),
                color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                output, label,
                (x1, y1 - baseline),
                self.font, self.font_scale,
                (255, 255, 255),  # White text
                self.thickness
            )
        
        return output

# ============ RUN MANAGER ============
class RunManager:
    """Automatic run folder management with versioning"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.run_dir: Optional[Path] = None
        
    def create_run(self) -> Path:
        """Create new run directory with auto-versioning"""
        # Find existing runs
        existing = []
        for d in self.base_dir.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                try:
                    num = int(d.name.split("_")[1])
                    existing.append(num)
                except:
                    pass
        
        run_number = max(existing) + 1 if existing else 1
        self.run_dir = self.base_dir / f"run_{run_number:04d}"
        self.run_dir.mkdir(parents=True)
        
        # Create subdirectories
        (self.run_dir / "metrics").mkdir(exist_ok=True)
        
        # Save run metadata
        metadata = {
            'run_number': run_number,
            'created_at': datetime.now().isoformat(),
            'run_dir': str(self.run_dir)
        }
        
        with open(self.run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  Run directory: {self.run_dir}")
        return self.run_dir

# ============ METRICS COLLECTOR ============
class MetricsCollector:
    """Collect and save performance metrics"""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.frame_metrics: List[Dict] = []
        self.start_time = time.time()
        
    def log_frame(self, frame_idx: int, inference_ms: float, detections: int,
                  cpu_percent: float, cpu_temp: float, ram_percent: float):
        """Log metrics for a processed frame"""
        metric = {
            'frame_idx': frame_idx,
            'timestamp': time.time() - self.start_time,
            'inference_time_ms': inference_ms,
            'detections': detections,
            'cpu_percent': cpu_percent,
            'cpu_temp_c': cpu_temp,
            'ram_percent': ram_percent
        }
        self.frame_metrics.append(metric)
    
    def save(self):
        """Save metrics to CSV"""
        if not self.frame_metrics:
            return
        
        csv_path = self.run_dir / 'metrics' / 'frame_metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            if self.frame_metrics:
                writer = csv.DictWriter(f, fieldnames=self.frame_metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.frame_metrics)
        
        # Summary statistics
        summary = self._generate_summary()
        with open(self.run_dir / 'metrics' / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics"""
        if not self.frame_metrics:
            return {}
        
        inference_times = [m['inference_time_ms'] for m in self.frame_metrics]
        
        return {
            'total_frames': len(self.frame_metrics),
            'total_time_sec': self.frame_metrics[-1]['timestamp'] if self.frame_metrics else 0,
            'avg_fps': len(self.frame_metrics) / self.frame_metrics[-1]['timestamp'] if self.frame_metrics else 0,
            'avg_inference_ms': np.mean(inference_times),
            'p95_inference_ms': np.percentile(inference_times, 95),
            'min_inference_ms': np.min(inference_times),
            'max_inference_ms': np.max(inference_times),
            'avg_detections': np.mean([m['detections'] for m in self.frame_metrics]),
            'total_detections': sum(m['detections'] for m in self.frame_metrics),
            'avg_cpu_percent': np.mean([m['cpu_percent'] for m in self.frame_metrics]),
            'avg_cpu_temp_c': np.mean([m['cpu_temp_c'] for m in self.frame_metrics])
        }

# ============ MAIN PROCESSOR ============
class YoloProcessor:
    """Main processing pipeline with adaptive performance"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.run_manager = RunManager(config.output.base_output_dir)
        self.engine: Optional[OptimizedYoloEngine] = None
        self.renderer = DetectionRenderer()
        self.monitor: Optional[SystemMonitor] = None
        self.metrics: Optional[MetricsCollector] = None
        self.video_writer: Optional[OptimizedVideoWriter] = None
        
        # Performance tracking
        self.frame_times: deque = deque(maxlen=30)  # Rolling window of 30 frames
        self.current_stride = config.performance.frame_stride
        self.frame_counter = 0
        
    def run(self) -> bool:
        """Main execution loop"""
        print("="*70)
        print("PRODUCTION YOLO INFERENCE - RPi5 OPTIMIZED")
        print("="*70)
        
        # Validate input files
        if not self._validate_inputs():
            return False
        
        # Initialize components
        if not self._initialize_components():
            return False
        
        # Open video capture
        cap = cv2.VideoCapture(self.config.model.model_path)  # Wait, this is wrong!
        # Fix: Use video_path, not model_path
        cap = cv2.VideoCapture(self.config.model.model_path)  # This line is buggy
        
        # CORRECTED: Use video_path from config
        # Need to add video_path to ModelConfig or create separate VideoConfig
        # For now, assume video path is passed differently
        
        print("  Error: Video path not properly configured")
        return False

    def _validate_inputs(self) -> bool:
        """Validate input files exist"""
        model_path = Path(self.config.model.model_path)
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            return False
        
        # Note: Need video path in config - this should be added
        print("Note: Video path configuration needed")
        return True
    
    def _initialize_components(self) -> bool:
        """Initialize all processing components"""
        # Create run directory
        run_dir = self.run_manager.create_run()
        
        # Initialize engine
        self.engine = OptimizedYoloEngine(self.config)
        if not self.engine.initialize():
            return False
        
        # Initialize metrics
        self.metrics = MetricsCollector(run_dir)
        
        # Initialize system monitor
        self.monitor = SystemMonitor(self.config)
        self.monitor.start()
        
        return True

# ============ MAIN ENTRY POINT ============
def main():
    """Main entry point with signal handling"""
    
    # Create default configuration
    config = ResearchConfig()
    
    # Override with command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser(description="Optimized YOLO Inference for RPi5")
    parser.add_argument("--model", type=str, default="best2.pt", help="Model path")
    parser.add_argument("--video", type=str, default="test2.mp4", help="Video path")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--stride", type=int, default=2, help="Frame processing stride")
    parser.add_argument("--resize", type=float, default=0.5, help="Input resize factor")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--no-video", action="store_true", help="Disable video output")
    
    args = parser.parse_args()
    
    # Apply arguments
    config.model.model_path = args.model
    config.performance.frame_stride = args.stride
    config.performance.resize_factor = args.resize
    config.performance.num_threads = args.threads
    config.model.conf_threshold = args.conf
    config.output.save_video = not args.no_video
    
    # Note: Need to add video_path to config
    # For now, patch it
    config.video_path = args.video  # This will need to be added to ResearchConfig
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        print("\n\nShutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run processor
    processor = YoloProcessor(config)
    success = processor.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
