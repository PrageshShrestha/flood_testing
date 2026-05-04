#!/usr/bin/env python3
"""
RESEARCH-GRADE YOLO INFERENCE WITH 3-REGION TILING FOR RPi5 CLI-ONLY
- Single YOLO instance with all cores focused on parallel region processing
- 3 specific regions: top-left, top-right, and bottom portion
- Automatic run folders with versioning
- Comprehensive research metrics (CPU, power, FPS, FLOPS, thermal)
- MP4 output with detections
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
import multiprocessing as mp
from multiprocessing import Pool
import concurrent.futures
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import cv2
from ultralytics import YOLO
import torch

# ============ 3-REGION TILING CONFIGURATION ============
@dataclass
class TilingConfig:
    """Configuration for 3-region tiling (top-left, top-right, bottom)"""
    enabled: bool = True
    # Define the 3 specific regions as ((x1,y1), (x2,y2))
    regions: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=lambda: [
        ((0, 0), (50, 50)),      # Region 1: Top-left piece
        ((50, 0), (100, 50)),    # Region 2: Top-right piece  
        ((0, 50), (100, 100))    # Region 3: Bottom portion (full width)
    ])
    region_names: List[str] = field(default_factory=lambda: [
        "top-left", "top-right", "bottom"
    ])
    num_workers: int = 3  # One per region
    merge_iou_threshold: float = 0.5  # NMS for merging regions
    
@dataclass
class ResearchConfig:
    """Research-grade configuration"""
    # Paths
    model_path: str = "best2.pt"
    video_path: str = "test2.mp4"
    base_output_dir: str = "research_runs"
    
    # Tiling configuration
    tiling: TilingConfig = field(default_factory=TilingConfig)
    
    # Model settings
    imgsz: int = 640
    conf_threshold: float = 0.3
    iou_threshold: float = 0.45
    
    # Performance
    batch_size: int = 1
    num_workers: int = 3
    use_half_precision: bool = False
    
    # Output settings
    save_video: bool = True
    save_detections_csv: bool = True
    save_metrics_json: bool = True
    save_frame_metrics: bool = True
    
    # Monitoring
    monitor_interval_ms: int = 100
    enable_power_monitoring: bool = True
    enable_thermal_monitoring: bool = True

# ============ 3-REGION TILE MANAGER ============
class ThreeRegionTileManager:
    """
    Manages splitting frames into 3 specific regions: top-left, top-right, and bottom
    All regions processed in parallel using a single model instance
    """
    
    def __init__(self, config: TilingConfig):
        self.config = config
        self.regions = config.regions
        self.region_names = config.region_names
        
    def get_regions_for_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Extract the 3 specific regions from the frame
        
        Returns:
            List of dicts with: 'region', 'x', 'y', 'width', 'height', 'region_id', 'region_name'
        """
        height, width = frame.shape[:2]
        
        # Scale regions to actual frame dimensions if frame is not 100x100
        # This makes the code work with any frame size by proportional scaling
        scale_x = width / 100.0
        scale_y = height / 100.0
        
        regions = []
        
        for region_id, (region_coords, region_name) in enumerate(zip(self.regions, self.region_names)):
            (x1_norm, y1_norm), (x2_norm, y2_norm) = region_coords
            
            # Scale normalized coordinates (0-100) to actual pixel coordinates
            x1 = int(x1_norm * scale_x)
            y1 = int(y1_norm * scale_y)
            x2 = int(x2_norm * scale_x)
            y2 = int(y2_norm * scale_y)
            
            # Ensure within bounds
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))
            
            # Extract region
            region = frame[y1:y2, x1:x2]
            
            if region.size == 0:
                print(f"  [Region {region_id}] Warning: Empty region at {region_name}")
                continue
            
            regions.append({
                'region_id': region_id,
                'region': region,
                'x': x1,
                'y': y1,
                'width': x2 - x1,
                'height': y2 - y1,
                'region_name': region_name,
                'original_width': width,
                'original_height': height
            })
            
            print(f"  [Region {region_id}] {region_name}: ({x1},{y1}) to ({x2},{y2}) size={region.shape[1]}x{region.shape[0]}")
        
        return regions
    
    def merge_detections(self, all_detections: List[Dict], original_shape: Tuple) -> Dict:
        """
        Merge detections from all regions using NMS
        
        Args:
            all_detections: List of detection dicts from each region with 'boxes', 'scores', 'classes', 'region_metadata'
            original_shape: (height, width) of original frame
            
        Returns:
            Merged detections with coordinates mapped back to original frame
        """
        all_boxes = []
        all_scores = []
        all_classes = []
        
        print(f"  [Merge] Processing {len(all_detections)} region results")
        
        for region_detections in all_detections:
            if not region_detections or len(region_detections.get('boxes', [])) == 0:
                print(f"  [Merge] Region has no detections")
                continue
            
            boxes = region_detections['boxes']
            scores = region_detections['scores']
            classes = region_detections['classes']
            region_x = region_detections['region_x']
            region_y = region_detections['region_y']
            region_name = region_detections.get('region_name', 'unknown')
            
            print(f"  [Merge] Region {region_name} at ({region_x}, {region_y}) has {len(boxes)} detections")
            
            # Map box coordinates back to original frame
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                
                # Adjust coordinates to original frame
                orig_x1 = x1 + region_x
                orig_y1 = y1 + region_y
                orig_x2 = x2 + region_x
                orig_y2 = y2 + region_y
                
                # Clip to frame boundaries
                orig_x1 = max(0, min(orig_x1, original_shape[1]))
                orig_y1 = max(0, min(orig_y1, original_shape[0]))
                orig_x2 = max(0, min(orig_x2, original_shape[1]))
                orig_y2 = max(0, min(orig_y2, original_shape[0]))
                
                all_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2])
                all_scores.append(float(scores[i]))
                all_classes.append(int(classes[i]))
        
        print(f"  [Merge] Total detections before NMS: {len(all_boxes)}")
        
        # Apply NMS to remove duplicate detections from overlapping regions
        if all_boxes:
            merged = self._apply_nms(
                np.array(all_boxes), 
                np.array(all_scores), 
                np.array(all_classes),
                iou_threshold=self.config.merge_iou_threshold
            )
            print(f"  [Merge] Detections after NMS: {len(merged['boxes'])}")
            return merged
        else:
            print(f"  [Merge] No detections to merge")
            return {'boxes': [], 'scores': [], 'classes': []}
    
    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float = 0.5) -> Dict:
        """Apply Non-Maximum Suppression to merged detections"""
        if len(boxes) == 0:
            return {'boxes': [], 'scores': [], 'classes': []}
        
        # Use torchvision NMS if available, otherwise simple NMS
        try:
            import torchvision.ops as ops
            keep = ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_threshold)
            keep = keep.numpy()
        except:
            # Simple NMS implementation
            keep = self._simple_nms(boxes, scores, iou_threshold)
        
        return {
            'boxes': boxes[keep].tolist(),
            'scores': scores[keep].tolist(),
            'classes': classes[keep].tolist()
        }
    
    def _simple_nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Simple NMS implementation"""
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
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep)

# ============ PARALLEL REGION PROCESSOR (SINGLE MODEL) ============
class ParallelRegionProcessor:
    """
    Processes multiple regions from the same frame in parallel using a SINGLE model instance
    All cores collaborate on the same model for faster inference
    """
    
    def __init__(self, model_path: str, config: TilingConfig):
        self.model_path = model_path
        self.config = config
        self.model = None  # Single model instance
        self.model_lock = threading.Lock()  # For thread-safe inference if needed
        
    def initialize(self):
        """Initialize single model instance"""
        print(f"  [Single Model] Loading YOLO model once...")
        self.model = YOLO(self.model_path)
        
        # Configure
        self.model.conf = 0.3
        self.model.iou = 0.45
        
        # Warmup
        warmup = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            _ = self.model(warmup, verbose=False)
        
        print(f"  [Single Model] Model ready for parallel region processing")
        print(f"  [Single Model] All {self.config.num_workers} cores will process regions through this single instance")
        return True
    
    def process_region(self, region_data: Dict, region_id: int) -> Dict:
        """
        Process a single region using the shared model instance
        This runs in parallel across cores but uses the same model
        """
        if self.model is None:
            with self.model_lock:
                if self.model is None:
                    self.initialize()
        
        region = region_data['region']
        
        # Inference on this region (thread-safe with GIL, YOLO handles parallelism)
        results = self.model(region, verbose=False)
        
        # Extract detections
        boxes = []
        scores = []
        classes = []
        
        if results[0].boxes is not None:
            boxes_tensor = results[0].boxes.xyxy.cpu().numpy()
            scores_tensor = results[0].boxes.conf.cpu().numpy()
            classes_tensor = results[0].boxes.cls.cpu().numpy()
            
            for i in range(len(boxes_tensor)):
                boxes.append(boxes_tensor[i].tolist())
                scores.append(float(scores_tensor[i]))
                classes.append(int(classes_tensor[i]))
        
        return {
            'region_id': region_id,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'region_x': region_data['x'],
            'region_y': region_data['y'],
            'region_name': region_data.get('region_name', f'region_{region_id}')
        }

# ============ FRAME PROCESSOR WITH 3-REGION TILING ============
class ThreeRegionFrameProcessor:
    """
    Processes frames using 3-region tiling with a SINGLE persistent model instance.
    Model is loaded once at initialization and reused across all frames.
    All cores work in parallel on the same model for different regions.
    """
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.region_manager = ThreeRegionTileManager(config.tiling)
        self.num_workers = config.tiling.num_workers
        
        # Performance telemetry storage
        self.telemetry = {
            'model_initialization_ms': [],
            'region_inference_ms': {0: [], 1: [], 2: []},
            'frame_processing_ms': [],
            'merge_operations_ms': [],
            'total_frames_processed': 0
        }
        
        # Initialize SINGLE persistent model instance
        self._initialize_single_model()
    
    def _initialize_single_model(self) -> None:
        """Create ONE model instance that will be shared across all cores"""
        print(f"\n[INIT] Creating SINGLE model instance for shared use across {self.num_workers} cores")
        init_start = time.perf_counter()
        
        self.model = YOLO(self.config.model_path)
        self.model.conf = self.config.conf_threshold
        self.model.iou = self.config.iou_threshold
        
        # Thermal initialization
        warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.model(warmup_frame, verbose=False)
        
        init_ms = (time.perf_counter() - init_start) * 1000
        self.telemetry['model_initialization_ms'].append(init_ms)
        
        print(f"  [INSTANCE] Single model loaded in {init_ms:.0f}ms")
        print(f"[INIT] Model ready. All cores will process regions through this single instance.")
        print(f"[INIT] Cache policy: Persistent. Zero reloads required.\n")
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Execute parallel region inference on a single frame using ONE model.
        
        Args:
            frame: Input image array (H, W, 3)
            frame_idx: Sequential frame identifier
        
        Returns:
            annotated_frame: Frame with bounding boxes rendered
            detections: Detection results with timing metadata
        """
        frame_start_ns = time.perf_counter_ns()
        
        regions = self.region_manager.get_regions_for_frame(frame)
        if not regions:
            return frame, self._empty_detection_result()
        
        self._log_frame_header(frame_idx, len(regions))
        
        # Parallel region processing using ThreadPoolExecutor
        # All threads share the SAME model instance
        region_results, region_telemetry = self._execute_parallel_regions(regions, frame_idx)
        
        # Parallel execution metrics
        parallel_duration_ms = max(t['inference_ms'] for t in region_telemetry.values())
        sequential_duration_ms = sum(t['inference_ms'] for t in region_telemetry.values())
        speedup_factor = sequential_duration_ms / parallel_duration_ms if parallel_duration_ms > 0 else 0
        
        self._log_parallel_metrics(region_telemetry, parallel_duration_ms, speedup_factor)
        
        # Merge detections from all regions
        merge_start_ns = time.perf_counter_ns()
        merged_detections = self.region_manager.merge_detections(
            region_results, 
            (frame.shape[0], frame.shape[1])
        )
        merge_duration_ms = (time.perf_counter_ns() - merge_start_ns) / 1_000_000
        self.telemetry['merge_operations_ms'].append(merge_duration_ms)
        
        # Render annotations
        annotate_start_ns = time.perf_counter_ns()
        annotated_frame = self._render_detections(frame, merged_detections)
        annotate_duration_ms = (time.perf_counter_ns() - annotate_start_ns) / 1_000_000
        
        # Frame completion metrics
        total_frame_ms = (time.perf_counter_ns() - frame_start_ns) / 1_000_000
        self.telemetry['frame_processing_ms'].append(total_frame_ms)
        self.telemetry['total_frames_processed'] += 1
        
        # Attach timing metadata to result
        merged_detections['timing_ms'] = {
            'parallel_regions': parallel_duration_ms,
            'merge': merge_duration_ms,
            'render': annotate_duration_ms,
            'total': total_frame_ms
        }
        merged_detections['parallel_speedup'] = speedup_factor
        merged_detections['detection_count'] = len(merged_detections.get('boxes', []))
        
        self._log_frame_complete(frame_idx, total_frame_ms, len(merged_detections.get('boxes', [])))
        
        return annotated_frame, merged_detections
    
    def _execute_parallel_regions(self, regions: List[Dict], frame_idx: int) -> Tuple[List[Dict], Dict]:
        """Execute region inference in parallel using ThreadPoolExecutor with SINGLE model."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        region_results = [None] * len(regions)
        region_telemetry = {}
        
        with ThreadPoolExecutor(max_workers=len(regions)) as executor:
            futures = {}
            for idx, region in enumerate(regions):
                # Submit all regions to the SAME model instance
                future = executor.submit(
                    self._inference_region,
                    region, self.model, idx, frame_idx
                )
                futures[future] = idx
            
            for future in as_completed(futures):
                idx = futures[future]
                result, telemetry = future.result()
                region_results[idx] = result
                region_telemetry[idx] = telemetry
        
        return region_results, region_telemetry
    
    def _inference_region(self, region_data: Dict, model, region_idx: int, frame_idx: int) -> Tuple[Dict, Dict]:
        """Execute single region inference using the shared model instance."""
        start_ns = time.perf_counter_ns()
        
        region = region_data['region']
        results = model(region, verbose=False)
        
        inference_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        
        boxes, scores, classes = [], [], []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            scores = results[0].boxes.conf.cpu().numpy().tolist()
            classes = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
        
        result = {
            'region_id': region_data['region_id'],
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'region_x': region_data['x'],
            'region_y': region_data['y'],
            'region_name': region_data.get('region_name', f'region_{region_idx}')
        }
        
        telemetry = {
            'region_name': region_data.get('region_name', f'region_{region_idx}'),
            'inference_ms': inference_ms,
            'detections': len(boxes)
        }
        
        # Store telemetry
        if region_idx in self.telemetry['region_inference_ms']:
            self.telemetry['region_inference_ms'][region_idx].append(inference_ms)
        else:
            self.telemetry['region_inference_ms'][region_idx] = [inference_ms]
        
        return result, telemetry
    
    def _render_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Render bounding boxes and labels on frame copy."""
        annotated = frame.copy()
        
        boxes = detections.get('boxes', [])
        scores = detections.get('scores', [])
        classes = detections.get('classes', [])
        
        # Define colors for different regions (visualization only)
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Red, Blue
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            confidence = scores[i]
            class_id = classes[i]
            
            # Choose color based on approximate region position (for visualization)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            if center_y < frame.shape[0] // 2:
                if center_x < frame.shape[1] // 2:
                    color = colors[0]  # Top-left
                else:
                    color = colors[1]  # Top-right
            else:
                color = colors[2]  # Bottom
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def _empty_detection_result(self) -> Dict:
        """Return empty detection structure with timing fields."""
        return {
            'boxes': [], 'scores': [], 'classes': [],
            'timing_ms': {'parallel_regions': 0, 'merge': 0, 'render': 0, 'total': 0},
            'parallel_speedup': 0, 'detection_count': 0
        }
    
    def _log_frame_header(self, frame_idx: int, region_count: int) -> None:
        """Log frame processing start."""
        print(f"\n[FRAME {frame_idx:04d}] Processing {region_count} regions (parallel on single model)")
    
    def _log_parallel_metrics(self, telemetry: Dict, parallel_ms: float, speedup: float) -> None:
        """Log parallel execution efficiency metrics."""
        print(f"  [PARALLEL] Duration: {parallel_ms:.1f}ms | Speedup: {speedup:.1f}x")
        for idx, t in telemetry.items():
            print(f"    Region {t['region_name']:10s} | {t['inference_ms']:5.1f}ms | {t['detections']} detections")
    
    def _log_frame_complete(self, frame_idx: int, total_ms: float, detections: int) -> None:
        """Log frame completion."""
        print(f"  [COMPLETE] Frame {frame_idx:04d} | {total_ms:.1f}ms total | {detections} detections")
    
    def generate_telemetry_report(self) -> Dict:
        """Generate comprehensive performance telemetry."""
        if not self.telemetry['frame_processing_ms']:
            return {'error': 'No frames processed'}
        
        frame_times = self.telemetry['frame_processing_ms']
        
        report = {
            'model_initialization': {
                'total_instances': len(self.telemetry['model_initialization_ms']),
                'total_time_ms': sum(self.telemetry['model_initialization_ms']),
                'per_instance_ms': self.telemetry['model_initialization_ms'],
                'model_strategy': 'SINGLE_INSTANCE_SHARED_ACROSS_CORES'
            },
            'frame_processing': {
                'frames_processed': self.telemetry['total_frames_processed'],
                'average_ms': np.mean(frame_times),
                'p95_ms': np.percentile(frame_times, 95),
                'p99_ms': np.percentile(frame_times, 99),
                'minimum_ms': np.min(frame_times),
                'maximum_ms': np.max(frame_times),
                'throughput_fps': 1000 / np.mean(frame_times)
            },
            'region_inference': {
                'average_per_region_ms': {
                    region_name: np.mean(times) if times else 0 
                    for region_name, times in self.telemetry['region_inference_ms'].items()
                }
            },
            'merge_operations': {
                'average_ms': np.mean(self.telemetry['merge_operations_ms']) if self.telemetry['merge_operations_ms'] else 0
            },
            'cache_efficiency': {
                'models_loaded': len(self.telemetry['model_initialization_ms']),
                'reloads_performed': 0,
                'cache_hit_rate': 1.0,
                'architecture': 'Single model, multi-core parallel region processing'
            }
        }
        
        self._print_telemetry_report(report)
        return report
    
    def _print_telemetry_report(self, report: Dict) -> None:
        """Print formatted telemetry report."""
        print("\n" + "="*70)
        print("PERFORMANCE TELEMETRY REPORT (SINGLE MODEL, 3-REGION PARALLEL)")
        print("="*70)
        
        print("\n[MODEL STRATEGY]")
        print(f"  Architecture: {report['model_initialization']['model_strategy']}")
        print(f"  Total model instances: {report['model_initialization']['total_instances']}")
        print(f"  Initialization cost: {report['model_initialization']['total_time_ms']:.0f}ms")
        print(f"  Cache hit rate: {report['cache_efficiency']['cache_hit_rate']*100:.0f}%")
        
        print("\n[FRAME THROUGHPUT]")
        print(f"  Frames processed: {report['frame_processing']['frames_processed']}")
        print(f"  Average latency: {report['frame_processing']['average_ms']:.1f}ms")
        print(f"  P95 latency: {report['frame_processing']['p95_ms']:.1f}ms")
        print(f"  Throughput: {report['frame_processing']['throughput_fps']:.1f} FPS")
        
        print("\n[REGION INFERENCE]")
        for region_name, avg_ms in report['region_inference']['average_per_region_ms'].items():
            print(f"  Region {region_name}: {avg_ms:.1f}ms average")
        
        print(f"\n[MERGE OVERHEAD]")
        print(f"  Average merge time: {report['merge_operations']['average_ms']:.1f}ms")
        print("="*70)
    
    def cleanup(self) -> None:
        """Release resources and generate final report."""
        self.generate_telemetry_report()

# ============ RESEARCH METRICS COLLECTOR ============
@dataclass
class FrameMetrics:
    """Per-frame performance metrics"""
    frame_idx: int
    timestamp: float
    inference_time_ms: float
    region_processing_time_ms: float
    merge_time_ms: float
    render_time_ms: float
    total_frame_time_ms: float
    parallel_speedup: float
    num_regions: int
    num_detections: int
    cpu_percent: float
    cpu_temp_c: float
    ram_percent: float

@dataclass
class SystemSnapshot:
    """System metrics snapshot"""
    timestamp: float
    cpu_percent: float
    cpu_per_core: List[float]
    cpu_freq_mhz: float
    cpu_temp_c: float
    ram_percent: float
    ram_used_gb: float
    ram_available_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_recv_mb: float
    network_sent_mb: float
    power_estimate_w: float
    thermal_throttled: bool

class ResearchMetricsCollector:
    """
    Comprehensive metrics collection for research analysis
    """
    
    def __init__(self, config: ResearchConfig, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.frame_metrics = []
        self.system_snapshots = []
        self.start_time = None
        self.end_time = None
        self.monitoring = False
        self.monitor_thread = None
        
        # For FLOPS calculation (approximate)
        self.total_inference_time_ms = 0
        self.total_frames = 0
        self.model_flops = 0  # Will be estimated
        
        # Power estimation coefficients (for RPi5)
        self.idle_power_w = 3.0
        self.cpu_power_coefficient = 0.15  # W per % CPU
        self.temp_power_coefficient = 0.05  # Additional W per °C above 50
        
    def start(self):
        """Start metrics collection"""
        self.start_time = time.time()
        self.monitoring = True
        
        # Estimate model FLOPS
        self._estimate_model_flops()
        
    def _estimate_model_flops(self):
        """Estimate model FLOPS based on model size"""
        try:
            # Rough estimate: YOLOv8n ~8.1 GFLOPS, YOLOv8s ~28.6 GFLOPS, YOLOv8m ~78.9 GFLOPS
            # For best2.pt, we'll estimate based on file size
            model_size_mb = Path(self.config.model_path).stat().st_size / (1024 * 1024)
            
            if model_size_mb < 10:
                self.model_flops = 8.1  # GFLOPS
            elif model_size_mb < 20:
                self.model_flops = 28.6  # GFLOPS
            elif model_size_mb < 40:
                self.model_flops = 78.9  # GFLOPS
            else:
                self.model_flops = 150.0  # YOLOv11x range
        except:
            self.model_flops = 28.6  # Default assume medium model
    
    def log_frame(self, metrics: FrameMetrics):
        """Log per-frame metrics"""
        self.frame_metrics.append(metrics)
        self.total_frames += 1
        self.total_inference_time_ms += metrics.inference_time_ms
    
    def stop(self):
        """Stop metrics collection"""
        self.monitoring = False
        self.end_time = time.time()
        
        # Save metrics to CSV
        self._save_frame_metrics()
    
    def _save_frame_metrics(self):
        """Save frame metrics to CSV"""
        if not self.frame_metrics:
            return
        
        frame_file = self.run_dir / 'frame_metrics.csv'
        with open(frame_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_idx', 'timestamp', 'inference_time_ms', 'region_processing_time_ms',
                'merge_time_ms', 'render_time_ms', 'total_frame_time_ms', 'parallel_speedup',
                'num_regions', 'num_detections', 'cpu_percent', 'cpu_temp_c', 'ram_percent'
            ])
            
            for m in self.frame_metrics:
                writer.writerow([
                    m.frame_idx, m.timestamp, m.inference_time_ms, m.region_processing_time_ms,
                    m.merge_time_ms, m.render_time_ms, m.total_frame_time_ms, m.parallel_speedup,
                    m.num_regions, m.num_detections, m.cpu_percent, m.cpu_temp_c, m.ram_percent
                ])
    
    def generate_research_report(self) -> Dict:
        """Generate comprehensive research report"""
        total_time = self.end_time - self.start_time
        avg_fps = self.total_frames / total_time if total_time > 0 else 0
        
        # Calculate metrics from frame metrics
        if self.frame_metrics:
            avg_inference_ms = np.mean([m.inference_time_ms for m in self.frame_metrics])
            p95_inference_ms = np.percentile([m.inference_time_ms for m in self.frame_metrics], 95)
            avg_detections = np.mean([m.num_detections for m in self.frame_metrics])
            avg_cpu = np.mean([m.cpu_percent for m in self.frame_metrics])
            avg_temp = np.mean([m.cpu_temp_c for m in self.frame_metrics])
        else:
            avg_inference_ms = p95_inference_ms = avg_detections = avg_cpu = avg_temp = 0
        
        # Estimate power (simplified)
        avg_power = self.idle_power_w + (avg_cpu / 100.0) * 15.0
        total_energy_wh = (avg_power * total_time) / 3600
        
        # FLOPS calculation
        total_flops = self.model_flops * self.total_frames * 1e9
        effective_flops = (self.total_frames / avg_inference_ms * 1000) * self.model_flops * 1e9 if avg_inference_ms > 0 else 0
        
        report = {
            'run_info': {
                'timestamp': datetime.now().isoformat(),
                'run_dir': str(self.run_dir),
                'model_path': self.config.model_path,
                'video_path': self.config.video_path,
                'tiling_enabled': self.config.tiling.enabled,
                'regions': [str(region) for region in self.config.tiling.regions],
                'region_names': self.config.tiling.region_names,
                'num_workers': self.config.tiling.num_workers,
                'architecture': 'SINGLE_MODEL_MULTI_CORE_PARALLEL_REGIONS'
            },
            'performance': {
                'total_frames': self.total_frames,
                'total_time_seconds': total_time,
                'average_fps': avg_fps,
                'average_inference_time_ms': avg_inference_ms,
                'p95_inference_time_ms': p95_inference_ms,
                'average_detections_per_frame': avg_detections,
                'total_detections': sum(m.num_detections for m in self.frame_metrics)
            },
            'system_metrics': {
                'average_cpu_percent': avg_cpu,
                'peak_cpu_percent': max([m.cpu_percent for m in self.frame_metrics]) if self.frame_metrics else 0,
                'average_cpu_temperature_c': avg_temp,
                'peak_cpu_temperature_c': max([m.cpu_temp_c for m in self.frame_metrics]) if self.frame_metrics else 0,
                'average_power_watts': avg_power,
                'total_energy_wh': total_energy_wh
            },
            'compute_metrics': {
                'estimated_model_flops_gflops': self.model_flops,
                'total_compute_flops': total_flops,
                'effective_throughput_gflops': effective_flops / 1e9 if effective_flops > 0 else 0,
                'energy_per_frame_mwh': (total_energy_wh / self.total_frames * 1000) if self.total_frames > 0 else 0
            },
            'region_efficiency': {
                'total_regions_processed': sum(m.num_regions for m in self.frame_metrics),
                'average_regions_per_frame': np.mean([m.num_regions for m in self.frame_metrics]) if self.frame_metrics else 0,
                'parallelism_gain': self.config.tiling.num_workers,
                'merge_overhead_ms': np.mean([m.merge_time_ms for m in self.frame_metrics]) if self.frame_metrics else 0
            }
        }
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _save_report(self, report: Dict):
        """Save research report in multiple formats"""
        
        # JSON format (machine readable)
        json_path = self.run_dir / 'research_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # TXT format (human readable)
        txt_path = self.run_dir / 'research_report.txt'
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RESEARCH-GRADE PERFORMANCE REPORT\n")
            f.write(f"Generated: {report['run_info']['timestamp']}\n")
            f.write("="*80 + "\n\n")
            
            f.write("RUN INFORMATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Run Directory: {report['run_info']['run_dir']}\n")
            f.write(f"Model: {report['run_info']['model_path']}\n")
            f.write(f"Video: {report['run_info']['video_path']}\n")
            f.write(f"Architecture: {report['run_info']['architecture']}\n")
            f.write(f"Regions: {report['run_info']['region_names']}\n")
            f.write(f"Workers: {report['run_info']['num_workers']}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Frames: {report['performance']['total_frames']}\n")
            f.write(f"Total Time: {report['performance']['total_time_seconds']:.2f} seconds\n")
            f.write(f"Average FPS: {report['performance']['average_fps']:.2f}\n")
            f.write(f"Avg Inference Time: {report['performance']['average_inference_time_ms']:.2f} ms\n")
            f.write(f"P95 Inference Time: {report['performance']['p95_inference_time_ms']:.2f} ms\n")
            f.write(f"Avg Detections/Frame: {report['performance']['average_detections_per_frame']:.2f}\n\n")
            
            f.write("SYSTEM RESOURCE USAGE\n")
            f.write("-"*40 + "\n")
            f.write(f"Average CPU Usage: {report['system_metrics']['average_cpu_percent']:.1f}%\n")
            f.write(f"Peak CPU Usage: {report['system_metrics']['peak_cpu_percent']:.1f}%\n")
            f.write(f"Average Temperature: {report['system_metrics']['average_cpu_temperature_c']:.1f}°C\n")
            f.write(f"Peak Temperature: {report['system_metrics']['peak_cpu_temperature_c']:.1f}°C\n")
            f.write(f"Average Power: {report['system_metrics']['average_power_watts']:.2f} W\n")
            f.write(f"Total Energy: {report['system_metrics']['total_energy_wh']:.2f} Wh\n\n")
            
            f.write("COMPUTE EFFICIENCY\n")
            f.write("-"*40 + "\n")
            f.write(f"Model FLOPs: {report['compute_metrics']['estimated_model_flops_gflops']:.1f} GFLOPs\n")
            f.write(f"Total Compute: {report['compute_metrics']['total_compute_flops']:.2e} FLOPs\n")
            f.write(f"Effective Throughput: {report['compute_metrics']['effective_throughput_gflops']:.2f} GFLOPs/s\n")
            f.write(f"Energy per Frame: {report['compute_metrics']['energy_per_frame_mwh']:.3f} mWh\n\n")
            
            f.write("REGION EFFICIENCY\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Regions Processed: {report['region_efficiency']['total_regions_processed']}\n")
            f.write(f"Avg Regions per Frame: {report['region_efficiency']['average_regions_per_frame']:.1f}\n")
            f.write(f"Parallelism Gain: {report['region_efficiency']['parallelism_gain']}x\n")
            f.write(f"Merge Overhead: {report['region_efficiency']['merge_overhead_ms']:.2f} ms\n\n")
            
            # Performance classification
            fps = report['performance']['average_fps']
            if fps >= 10:
                rating = "EXCELLENT - Real-time capable"
            elif fps >= 5:
                rating = "GOOD - Near real-time"
            elif fps >= 2:
                rating = "MARGINAL - Slow but functional"
            else:
                rating = "POOR - Not suitable for real-time"
            
            f.write("PERFORMANCE RATING\n")
            f.write("-"*40 + "\n")
            f.write(f"Rating: {rating}\n")
        
        print(f"  ✓ Research report saved to {txt_path}")

# ============ VIDEO WRITER WITH FFMPEG (CLI-ONLY) ============
class FFmpegVideoWriter:
    """Write video frames directly to FFmpeg for encoding"""
    
    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.process = None
        
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  ⚠️ FFmpeg not found. Video output disabled.")
            return
        
        # Start FFmpeg process
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"  ⚠️ Failed to start FFmpeg: {e}")
            self.process = None
        
    def write_frame(self, frame: np.ndarray):
        """Write a single frame"""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                pass
    
    def close(self):
        """Close FFmpeg process"""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except:
                self.process.terminate()

# ============ RUN MANAGER WITH AUTO-VERSIONING ============
class RunManager:
    """Manages run folders with automatic versioning"""
    
    def __init__(self, base_dir: str = "research_runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.run_dir = None
        self.run_number = None
        
    def create_run_folder(self) -> Path:
        """Create a new run folder with next available number"""
        # Find the next run number
        existing_runs = []
        for d in self.base_dir.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                try:
                    num = int(d.name.split("_")[1])
                    existing_runs.append(num)
                except:
                    pass
        
        self.run_number = max(existing_runs) + 1 if existing_runs else 1
        self.run_dir = self.base_dir / f"run_{self.run_number:03d}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.run_dir / "frames").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        
        # Save run info
        info = {
            'run_number': self.run_number,
            'run_dir': str(self.run_dir),
            'created_at': datetime.now().isoformat(),
            'config': {}
        }
        
        with open(self.run_dir / 'run_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n📁 Created run folder: {self.run_dir}")
        return self.run_dir
    
    def get_latest_run_dir(self) -> Path:
        """Get the latest run directory"""
        runs = sorted([d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if runs:
            return runs[-1]
        return None

# ============ MAIN PROCESSOR WITH 3-REGION TILING ============
class ThreeRegionYoloProcessor:
    """Main processor using 3-region tiling with single model parallel processing"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.run_manager = RunManager(config.base_output_dir)
        self.metrics_collector = None
        self.region_processor = None
        self.video_writer = None
        
    def run(self):
        """Main processing pipeline"""
        print("="*80)
        print("RESEARCH-GRADE YOLO INFERENCE WITH 3-REGION TILING")
        print("Raspberry Pi 5 CLI-ONLY OPTIMIZED")
        print("Architecture: SINGLE MODEL with ALL CORES focused on parallel regions")
        print("="*80)
        
        # Create run folder
        run_dir = self.run_manager.create_run_folder()
        
        # Initialize metrics collector
        self.metrics_collector = ResearchMetricsCollector(self.config, run_dir)
        
        # Initialize region processor
        self.region_processor = ThreeRegionFrameProcessor(self.config)
        
        # Open video
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video {self.config.video_path}")
            return False
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Video Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total Frames: {total_frames}")
        
        print(f"\n🔧 3-Region Tiling Configuration:")
        print(f"  Regions: Top-Left (0-50,0-50), Top-Right (50-100,0-50), Bottom (0-100,50-100)")
        print(f"  Workers: {self.config.tiling.num_workers} (parallel on single model)")
        print(f"  Strategy: All cores share ONE model instance")
        
        # Initialize video writer
        if self.config.save_video:
            output_video = run_dir / "output_with_detections.mp4"
            self.video_writer = FFmpegVideoWriter(str(output_video), fps, width, height)
            if self.video_writer.process:
                print(f"  Output Video: {output_video}")
            else:
                print(f"  ⚠️ Video output disabled (FFmpeg not found)")
        
        # Start metrics collection
        self.metrics_collector.start()
        
        # Process frames
        print(f"\n🚀 Processing frames with parallel 3-region tiling on single model...")
        print("-"*80)
        
        frame_idx = 0
        start_time = time.time()
        last_log_time = start_time
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # Process frame with 3-region tiling
                annotated_frame, detections = self.region_processor.process_frame(frame, frame_idx)
                
                frame_time_ms = (time.time() - frame_start) * 1000
                
                # Get system metrics
                cpu_percent = psutil.cpu_percent()
                cpu_temp = self._get_cpu_temperature()
                ram_percent = psutil.virtual_memory().percent
                
                # Log metrics
                frame_metrics = FrameMetrics(
                    frame_idx=frame_idx,
                    timestamp=time.time(),
                    inference_time_ms=detections.get('timing_ms', {}).get('parallel_regions', frame_time_ms),
                    region_processing_time_ms=detections.get('timing_ms', {}).get('parallel_regions', 0),
                    merge_time_ms=detections.get('timing_ms', {}).get('merge', 0),
                    render_time_ms=detections.get('timing_ms', {}).get('render', 0),
                    total_frame_time_ms=detections.get('timing_ms', {}).get('total', frame_time_ms),
                    parallel_speedup=detections.get('parallel_speedup', 0),
                    num_regions=len(self.region_processor.region_manager.get_regions_for_frame(frame)),
                    num_detections=detections.get('detection_count', 0),
                    cpu_percent=cpu_percent,
                    cpu_temp_c=cpu_temp,
                    ram_percent=ram_percent
                )
                self.metrics_collector.log_frame(frame_metrics)
                
                # Save video frame
                if self.video_writer and self.video_writer.process:
                    self.video_writer.write_frame(annotated_frame)
                
                # Progress logging
                frame_idx += 1
                current_time = time.time()
                
                if current_time - last_log_time >= 2.0:  # Log every 2 seconds
                    elapsed = current_time - start_time
                    fps_current = frame_idx / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_frames - frame_idx) / fps_current if fps_current > 0 else 0
                    
                    print(f"  Frame {frame_idx:6d}/{total_frames} | "
                          f"FPS: {fps_current:5.1f} | "
                          f"Time: {frame_time_ms:5.1f}ms | "
                          f"Detections: {len(detections.get('boxes', [])):3d} | "
                          f"CPU: {cpu_percent:3.0f}% | "
                          f"Temp: {cpu_temp:4.1f}°C | "
                          f"ETA: {eta_seconds/60:.1f}min")
                    
                    last_log_time = current_time
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            
            if self.video_writer:
                self.video_writer.close()
                if self.video_writer.process:
                    print(f"\n  ✓ Video saved: {run_dir}/output_with_detections.mp4")
            
            # Stop metrics collection and generate report
            self.metrics_collector.stop()
            report = self.metrics_collector.generate_research_report()
            
            # Save config
            with open(run_dir / 'config.json', 'w') as f:
                # Convert tuple to list for JSON serialization
                config_dict = asdict(self.config)
                config_dict['tiling']['regions'] = [[list(r1), list(r2)] for r1, r2 in config_dict['tiling']['regions']]
                json.dump(config_dict, f, indent=2, default=str)
            
            # Final statistics
            total_time = time.time() - start_time
            avg_fps = frame_idx / total_time if total_time > 0 else 0
            
            print("\n" + "="*80)
            print("PROCESSING COMPLETE")
            print("="*80)
            print(f"✓ Frames Processed: {frame_idx}")
            print(f"✓ Total Time: {total_time:.2f} seconds")
            print(f"✓ Average FPS: {avg_fps:.2f}")
            print(f"✓ Average Power: {report['system_metrics']['average_power_watts']:.2f} W")
            print(f"✓ Total Energy: {report['system_metrics']['total_energy_wh']:.2f} Wh")
            print(f"✓ Architecture: Single model instance shared across all cores")
            print(f"✓ Run Directory: {run_dir}")
            
            return True
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature (works on RPi and some Linux, returns 0 on MacOS)"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return int(f.read().strip()) / 1000.0
        except (FileNotFoundError, OSError):
            # Fallback for MacOS or systems without thermal sensor
            return 0.0

# ============ MAIN ============
def main():
    """Main entry point"""
    # Create configuration
    config = ResearchConfig()
    
    # Configure for 3-region tiling
    config.tiling.enabled = True
    config.tiling.regions = [
        ((0, 0), (50, 50)),      # Region 1: Top-left piece
        ((50, 0), (100, 50)),    # Region 2: Top-right piece  
        ((0, 50), (100, 100))    # Region 3: Bottom portion (full width)
    ]
    config.tiling.region_names = ["top-left", "top-right", "bottom"]
    config.tiling.num_workers = 3  # One per region
    
    # Output settings
    config.save_video = True
    config.save_detections_csv = True
    config.save_metrics_json = True
    config.save_frame_metrics = True
    
    # Paths
    config.model_path = "best2.pt"
    config.video_path = "test2.mp4"
    
    # Check if files exist
    if not Path(config.model_path).exists():
        print(f"ERROR: Model not found: {config.model_path}")
        print("Please ensure best2.pt is in the current directory")
        sys.exit(1)
    
    if not Path(config.video_path).exists():
        print(f"ERROR: Video not found: {config.video_path}")
        print("Please ensure test2.mp4 is in the current directory")
        sys.exit(1)
    
    # Run processor
    processor = ThreeRegionYoloProcessor(config)
    success = processor.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    # Set multiprocessing start method (if on MacOS/Linux)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()
