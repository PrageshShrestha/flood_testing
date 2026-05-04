#!/usr/bin/env python3
"""
OPTIMIZED YOLO INFERENCE FOR RPi5 - 3-REGION TILING FIXED
- Fixed: Proper region scaling (no more 11s per frame)
- All original functionality preserved
- CLI-only, research-grade metrics
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
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from ultralytics import YOLO

# ============ OPTIMIZED 3-REGION TILING ============
@dataclass
class TilingConfig:
    """Fixed 3-region tiling configuration"""
    enabled: bool = True
    # FIXED: Use percentage-based regions (0.0 to 1.0 instead of 0-100)
    regions: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=lambda: [
        ((0.0, 0.0), (0.5, 0.5)),      # Top-left: 50% x 50%
        ((0.5, 0.0), (1.0, 0.5)),      # Top-right: 50% x 50%
        ((0.0, 0.5), (1.0, 1.0))       # Bottom: 100% x 50%
    ])
    region_names: List[str] = field(default_factory=lambda: [
        "top-left", "top-right", "bottom"
    ])
    num_workers: int = 3
    merge_iou_threshold: float = 0.5
    # NEW: Performance optimizations
    resize_regions: bool = True
    region_scale: float = 0.5  # Scale regions by 50% for faster inference

@dataclass
class ResearchConfig:
    """Research-grade configuration"""
    model_path: str = "best2.pt"
    video_path: str = "test2.mp4"
    base_output_dir: str = "research_runs"
    
    tiling: TilingConfig = field(default_factory=TilingConfig)
    
    # Model settings
    imgsz: int = 640
    conf_threshold: float = 0.3
    iou_threshold: float = 0.45
    
    # NEW: Performance settings
    frame_stride: int = 1  # Process every frame (can increase for speed)
    resize_input: float = 1.0  # Scale entire frame (0.5 = half size)
    
    # Output settings
    save_video: bool = True
    save_detections_csv: bool = True
    save_metrics_json: bool = True
    save_frame_metrics: bool = True
    
    # Monitoring
    monitor_interval_ms: int = 1000
    enable_power_monitoring: bool = True
    enable_thermal_monitoring: bool = True

# ============ FIXED TILE MANAGER ============
class ThreeRegionTileManager:
    """Fixed region manager - uses percentage-based scaling"""
    
    def __init__(self, config: TilingConfig):
        self.config = config
        self.regions = config.regions
        self.region_names = config.region_names
        
    def get_regions_for_frame(self, frame: np.ndarray) -> List[Dict]:
        """Extract regions using proper percentage scaling"""
        height, width = frame.shape[:2]
        
        regions = []
        
        for region_id, (region_coords, region_name) in enumerate(zip(self.regions, self.region_names)):
            # FIXED: Use percentage coordinates (0.0-1.0) instead of 0-100 grid
            (x1_norm, y1_norm), (x2_norm, y2_norm) = region_coords
            
            # Scale to actual pixels
            x1 = int(x1_norm * width)
            y1 = int(y1_norm * height)
            x2 = int(x2_norm * width)
            y2 = int(y2_norm * height)
            
            # Ensure bounds
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))
            
            # Extract region
            region = frame[y1:y2, x1:x2]
            
            # OPTIMIZATION: Resize region for faster inference
            if self.config.resize_regions and self.config.region_scale < 1.0:
                new_w = int(region.shape[1] * self.config.region_scale)
                new_h = int(region.shape[0] * self.config.region_scale)
                region = cv2.resize(region, (new_w, new_h))
            
            if region.size == 0:
                continue
            
            regions.append({
                'region_id': region_id,
                'region': region,
                'x': x1,
                'y': y1,
                'width': x2 - x1,
                'height': y2 - y1,
                'scale_factor': self.config.region_scale if self.config.resize_regions else 1.0,
                'region_name': region_name,
                'original_width': width,
                'original_height': height
            })
        
        return regions
    
    def merge_detections(self, all_detections: List[Dict], original_shape: Tuple) -> Dict:
        """Merge detections with proper coordinate mapping"""
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for region_detections in all_detections:
            if not region_detections or len(region_detections.get('boxes', [])) == 0:
                continue
            
            boxes = region_detections['boxes']
            scores = region_detections['scores']
            classes = region_detections['classes']
            region_x = region_detections['region_x']
            region_y = region_detections['region_y']
            scale_factor = region_detections.get('scale_factor', 1.0)
            
            # Map back to original coordinates
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                
                # Reverse region scaling if applied
                if scale_factor != 1.0:
                    inv_scale = 1.0 / scale_factor
                    x1 *= inv_scale
                    y1 *= inv_scale
                    x2 *= inv_scale
                    y2 *= inv_scale
                
                # Add region offset
                orig_x1 = x1 + region_x
                orig_y1 = y1 + region_y
                orig_x2 = x2 + region_x
                orig_y2 = y2 + region_y
                
                # Clip to frame
                orig_x1 = max(0, min(orig_x1, original_shape[1]))
                orig_y1 = max(0, min(orig_y1, original_shape[0]))
                orig_x2 = max(0, min(orig_x2, original_shape[1]))
                orig_y2 = max(0, min(orig_y2, original_shape[0]))
                
                all_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2])
                all_scores.append(float(scores[i]))
                all_classes.append(int(classes[i]))
        
        # Apply NMS
        if all_boxes:
            return self._apply_nms(np.array(all_boxes), np.array(all_scores), 
                                   np.array(all_classes), self.config.merge_iou_threshold)
        return {'boxes': [], 'scores': [], 'classes': []}
    
    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray, 
                   classes: np.ndarray, iou_threshold: float) -> Dict:
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return {'boxes': [], 'scores': [], 'classes': []}
        
        # Simple efficient NMS
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        
        return {
            'boxes': boxes[keep].tolist(),
            'scores': scores[keep].tolist(),
            'classes': classes[keep].tolist()
        }

# ============ OPTIMIZED FRAME PROCESSOR ============
class ThreeRegionFrameProcessor:
    """Processes frames with fixed 3-region tiling"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.region_manager = ThreeRegionTileManager(config.tiling)
        self.model = None
        self.telemetry = {
            'inference_times': [],
            'frame_times': [],
            'region_times': {0: [], 1: [], 2: []}
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Load model once"""
        print(f"\n[INIT] Loading YOLO model...")
        start = time.perf_counter()
        
        self.model = YOLO(self.config.model_path)
        self.model.conf = self.config.conf_threshold
        self.model.iou = self.config.iou_threshold
        
        # Warmup
        warmup = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)
        for _ in range(2):
            self.model(warmup, verbose=False)
        
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[INIT] Model loaded in {elapsed:.0f}ms\n")
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict]:
        """Process single frame with optimized region inference"""
        frame_start = time.perf_counter()
        
        # Optional: Scale entire frame for faster processing
        if self.config.resize_input < 1.0:
            h, w = frame.shape[:2]
            new_w, new_h = int(w * self.config.resize_input), int(h * self.config.resize_input)
            frame = cv2.resize(frame, (new_w, new_h))
        
        # Extract regions
        regions = self.region_manager.get_regions_for_frame(frame)
        if not regions:
            return frame, {'boxes': [], 'scores': [], 'classes': []}
        
        # Process regions (sequential is faster on RPi5 than parallel)
        region_results = []
        region_times = []
        
        for region_data in regions:
            region_start = time.perf_counter()
            
            # Run inference on region
            results = self.model(region_data['region'], verbose=False)[0]
            
            region_ms = (time.perf_counter() - region_start) * 1000
            region_times.append(region_ms)
            
            # Extract detections
            boxes, scores, classes = [], [], []
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy().tolist()
                scores = results.boxes.conf.cpu().numpy().tolist()
                classes = results.boxes.cls.cpu().numpy().astype(int).tolist()
            
            region_results.append({
                'region_id': region_data['region_id'],
                'boxes': boxes,
                'scores': scores,
                'classes': classes,
                'region_x': region_data['x'],
                'region_y': region_data['y'],
                'scale_factor': region_data.get('scale_factor', 1.0),
                'region_name': region_data['region_name']
            })
            
            # Store telemetry
            self.telemetry['region_times'][region_data['region_id']].append(region_ms)
        
        # Merge detections
        merge_start = time.perf_counter()
        merged = self.region_manager.merge_detections(region_results, frame.shape[:2])
        merge_ms = (time.perf_counter() - merge_start) * 1000
        
        # Render detections
        annotated = self._render_detections(frame, merged)
        
        total_ms = (time.perf_counter() - frame_start) * 1000
        self.telemetry['frame_times'].append(total_ms)
        self.telemetry['inference_times'].append(np.mean(region_times) if region_times else 0)
        
        # Attach metadata
        merged['timing_ms'] = {
            'regions': region_times,
            'avg_region_ms': np.mean(region_times) if region_times else 0,
            'merge_ms': merge_ms,
            'total_ms': total_ms
        }
        merged['detection_count'] = len(merged.get('boxes', []))
        
        return annotated, merged
    
    def _render_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Render bounding boxes"""
        if not detections.get('boxes'):
            return frame
        
        annotated = frame.copy()
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
        
        for i, box in enumerate(detections['boxes']):
            x1, y1, x2, y2 = map(int, box)
            score = detections['scores'][i]
            class_id = detections['classes'][i]
            
            color = colors[class_id % len(colors)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"Class {class_id}: {score:.2f}"
            cv2.putText(annotated, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.telemetry['frame_times']:
            return {}
        
        return {
            'avg_frame_ms': np.mean(self.telemetry['frame_times']),
            'avg_inference_ms': np.mean(self.telemetry['inference_times']),
            'fps': 1000 / np.mean(self.telemetry['frame_times']),
            'region_stats': {
                name: np.mean(times) if times else 0 
                for name, times in self.telemetry['region_times'].items()
            }
        }

# ============ METRICS COLLECTOR ============
@dataclass
class FrameMetrics:
    frame_idx: int
    timestamp: float
    inference_time_ms: float
    merge_time_ms: float
    total_frame_time_ms: float
    num_regions: int
    num_detections: int
    cpu_percent: float
    cpu_temp_c: float
    ram_percent: float

class ResearchMetricsCollector:
    """Comprehensive metrics collection"""
    
    def __init__(self, config: ResearchConfig, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.frame_metrics = []
        self.start_time = None
        self.end_time = None
        
    def start(self):
        self.start_time = time.time()
    
    def log_frame(self, metrics: FrameMetrics):
        self.frame_metrics.append(metrics)
    
    def stop(self):
        self.end_time = time.time()
        self._save_metrics()
    
    def _save_metrics(self):
        if not self.frame_metrics:
            return
        
        # Save CSV
        csv_path = self.run_dir / 'frame_metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_idx', 'timestamp', 'inference_ms', 'merge_ms', 
                           'total_ms', 'num_regions', 'num_detections', 
                           'cpu_percent', 'cpu_temp_c', 'ram_percent'])
            
            for m in self.frame_metrics:
                writer.writerow([m.frame_idx, m.timestamp, m.inference_time_ms,
                               m.merge_time_ms, m.total_frame_time_ms, m.num_regions,
                               m.num_detections, m.cpu_percent, m.cpu_temp_c, m.ram_percent])
    
    def generate_report(self) -> Dict:
        total_time = self.end_time - self.start_time
        avg_fps = len(self.frame_metrics) / total_time if total_time > 0 else 0
        
        if self.frame_metrics:
            avg_inference = np.mean([m.inference_time_ms for m in self.frame_metrics])
            avg_cpu = np.mean([m.cpu_percent for m in self.frame_metrics])
            avg_temp = np.mean([m.cpu_temp_c for m in self.frame_metrics])
        else:
            avg_inference = avg_cpu = avg_temp = 0
        
        return {
            'total_frames': len(self.frame_metrics),
            'total_time_sec': total_time,
            'average_fps': avg_fps,
            'average_inference_ms': avg_inference,
            'average_cpu_percent': avg_cpu,
            'average_temp_c': avg_temp
        }

# ============ VIDEO WRITER ============
class VideoWriter:
    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.process = None
        try:
            cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
                   '-i', '-', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                   output_path]
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, 
                                           stderr=subprocess.DEVNULL)
        except:
            pass
    
    def write(self, frame: np.ndarray):
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(frame.tobytes())
            except:
                pass
    
    def close(self):
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=2)
            except:
                self.process.terminate()

# ============ RUN MANAGER ============
class RunManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.run_dir = None
    
    def create_run(self) -> Path:
        existing = []
        for d in self.base_dir.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                try:
                    existing.append(int(d.name.split("_")[1]))
                except:
                    pass
        
        run_num = max(existing) + 1 if existing else 1
        self.run_dir = self.base_dir / f"run_{run_num:03d}"
        self.run_dir.mkdir()
        
        print(f"\n📁 Run: {self.run_dir}\n")
        return self.run_dir

# ============ MAIN PROCESSOR ============
class ThreeRegionYoloProcessor:
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.run_manager = RunManager(config.base_output_dir)
        self.processor = None
        self.metrics = None
        self.video_writer = None
    
    def run(self):
        print("="*70)
        print("OPTIMIZED YOLO INFERENCE - 3-REGION TILING")
        print("="*70)
        
        # Check files
        if not Path(self.config.model_path).exists():
            print(f"ERROR: Model not found: {self.config.model_path}")
            return False
        
        if not Path(self.config.video_path).exists():
            print(f"ERROR: Video not found: {self.config.video_path}")
            return False
        
        # Setup
        run_dir = self.run_manager.create_run()
        self.processor = ThreeRegionFrameProcessor(self.config)
        self.metrics = ResearchMetricsCollector(self.config, run_dir)
        
        # Open video
        cap = cv2.VideoCapture(self.config.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps:.1f}fps")
        print(f"Regions: {self.config.tiling.region_names}")
        print(f"Region scale: {self.config.tiling.region_scale}x")
        
        # Video writer
        if self.config.save_video:
            self.video_writer = VideoWriter(str(run_dir / "output.mp4"), fps, width, height)
        
        # Process frames
        self.metrics.start()
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Frame stride
                if frame_count % self.config.frame_stride != 0:
                    frame_count += 1
                    continue
                
                # Process
                proc_start = time.perf_counter()
                annotated, detections = self.processor.process_frame(frame, frame_count)
                proc_ms = (time.perf_counter() - proc_start) * 1000
                
                # Collect metrics
                cpu_pct = psutil.cpu_percent()
                cpu_temp = self._get_temp()
                ram_pct = psutil.virtual_memory().percent
                
                self.metrics.log_frame(FrameMetrics(
                    frame_idx=frame_count,
                    timestamp=time.time(),
                    inference_time_ms=detections.get('timing_ms', {}).get('avg_region_ms', 0),
                    merge_time_ms=detections.get('timing_ms', {}).get('merge_ms', 0),
                    total_frame_time_ms=proc_ms,
                    num_regions=3,
                    num_detections=detections.get('detection_count', 0),
                    cpu_percent=cpu_pct,
                    cpu_temp_c=cpu_temp,
                    ram_percent=ram_pct
                ))
                
                # Save video
                if self.video_writer:
                    self.video_writer.write(annotated)
                
                # Progress
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    print(f"Frame {frame_count:5d}/{total_frames} | "
                          f"FPS: {fps_current:4.1f} | "
                          f"Inf: {detections.get('timing_ms', {}).get('avg_region_ms', 0):5.1f}ms | "
                          f"Dets: {detections.get('detection_count', 0):3d} | "
                          f"CPU: {cpu_pct:3.0f}% | Temp: {cpu_temp:4.1f}°C")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted")
        
        finally:
            cap.release()
            if self.video_writer:
                self.video_writer.close()
            
            self.metrics.stop()
            report = self.metrics.generate_report()
            
            # Print summary
            print("\n" + "="*70)
            print("COMPLETE")
            print("="*70)
            print(f"Frames: {report['total_frames']}")
            print(f"Time: {report['total_time_sec']:.1f}s")
            print(f"FPS: {report['average_fps']:.2f}")
            print(f"Avg inference: {report['average_inference_ms']:.1f}ms")
            print(f"CPU: {report['average_cpu_percent']:.1f}%")
            print(f"Temp: {report['average_temp_c']:.1f}°C")
            print(f"Run: {run_dir}")
            
            # Save config
            with open(run_dir / 'config.json', 'w') as f:
                config_dict = asdict(self.config)
                json.dump(config_dict, f, indent=2, default=str)
            
            return True
    
    def _get_temp(self) -> float:
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return int(f.read()) / 1000.0
        except:
            return 0.0

# ============ MAIN ============
def main():
    config = ResearchConfig()
    
    # Default paths
    config.model_path = "best2.pt"
    config.video_path = "test2.mp4"
    
    # Performance settings (adjust for your RPi5)
    config.tiling.region_scale = 0.5  # Scale regions by 50% (faster)
    config.resize_input = 0.75  # Scale input frame by 75%
    config.frame_stride = 1  # Process every frame
    
    # Optional: Faster but lower quality
    # config.tiling.region_scale = 0.35
    # config.resize_input = 0.5
    # config.frame_stride = 2
    
    processor = ThreeRegionYoloProcessor(config)
    success = processor.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
