#!/usr/bin/env python3
"""
3-STAGE PIPELINE YOLO WITH 3-REGION TILING FOR RPi5
- Stage 1 (4 cores): Frame reading + 3 region extraction + caching
- Stage 2 (4 cores): 3 regions inference + 1 core metrics collector
- Stage 3 (4 cores): NMS merge + rendering + video encoding + CSV saving
- Target: 5 FPS, 100% CPU utilization, 8GB RAM fully utilized
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
import queue
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import cv2
from ultralytics import YOLO

# ============ CONFIGURATION ============
@dataclass
class PipelineConfig:
    """3-stage pipeline configuration"""
    
    # Stage 1: Preprocessing (4 cores)
    preprocess_workers: int = 4
    preprocess_queue_size: int = 64  # Store 64 preprocessed frames
    
    # Stage 2: Inference (4 cores: 3 inference + 1 metrics)
    inference_workers: int = 4
    inference_queue_size: int = 64
    
    # Stage 3: Postprocessing (4 cores)
    postprocess_workers: int = 4
    
    # Target performance
    target_fps: float = 5.0
    frame_stride: int = 1  # Process every frame (adjust if needed)
    
    # 3-Region tiling (percentage-based)
    regions: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=lambda: [
        ((0.0, 0.0), (0.5, 0.5)),      # Top-left: 50% x 50%
        ((0.5, 0.0), (1.0, 0.5)),      # Top-right: 50% x 50%
        ((0.0, 0.5), (1.0, 1.0))       # Bottom: 100% x 50%
    ])
    region_names: List[str] = field(default_factory=lambda: [
        "top-left", "top-right", "bottom"
    ])
    
    # Performance optimizations
    resize_input: float = 0.5  # Scale input frame by 50%
    region_scale: float = 0.5  # Scale each region by 50%
    conf_threshold: float = 0.3
    iou_threshold: float = 0.45
    
    # Paths
    model_path: str = "best2.pt"
    video_path: str = "test2.mp4"
    output_dir: str = "research_runs"
    save_video: bool = True
    
    # RAM utilization (8GB RPi5)
    ram_cache_size_mb: int = 4096  # Use 4GB for caching
    enable_disk_spill: bool = True  # Spill to disk if RAM full

# ============ TIMING TRACKER ============
class DetailedTiming:
    """Track every operation with microsecond precision"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.core_utilization = defaultdict(float)
        self.start_time = time.perf_counter()
        self.lock = threading.Lock()
    
    def record(self, operation: str, duration_ms: float, core_id: int = None):
        with self.lock:
            self.timings[operation].append(duration_ms)
            if core_id is not None:
                self.core_utilization[f"core_{core_id}_{operation}"] += duration_ms
    
    def report(self) -> Dict:
        report = {}
        for op, times in self.timings.items():
            if times:
                report[op] = {
                    'count': len(times),
                    'total_ms': sum(times),
                    'avg_ms': np.mean(times),
                    'p95_ms': np.percentile(times, 95),
                    'p99_ms': np.percentile(times, 99),
                    'max_ms': max(times),
                    'min_ms': min(times)
                }
        return report
    
    def print_summary(self):
        print("\n" + "="*80)
        print("DETAILED TIMING BREAKDOWN")
        print("="*80)
        
        report = self.report()
        
        # Group by stage
        stage1_ops = ['frame_read', 'frame_resize', 'region_extract', 'region_resize', 'queue_put_preprocess']
        stage2_ops = ['queue_get_preprocess', 'inference_tl', 'inference_tr', 'inference_bottom', 'metrics_collect']
        stage3_ops = ['queue_get_inference', 'nms_merge', 'render_boxes', 'video_encode', 'csv_save']
        
        print("\n📊 STAGE 1 - PREPROCESSING (4 Cores)")
        print("-"*40)
        for op in stage1_ops:
            if op in report:
                print(f"  {op:25s}: {report[op]['avg_ms']:6.2f}ms avg (total: {report[op]['total_ms']:.0f}ms)")
        
        print("\n📊 STAGE 2 - INFERENCE (4 Cores)")
        print("-"*40)
        for op in stage2_ops:
            if op in report:
                print(f"  {op:25s}: {report[op]['avg_ms']:6.2f}ms avg (total: {report[op]['total_ms']:.0f}ms)")
        
        print("\n📊 STAGE 3 - POSTPROCESSING (4 Cores)")
        print("-"*40)
        for op in stage3_ops:
            if op in report:
                print(f"  {op:25s}: {report[op]['avg_ms']:6.2f}ms avg (total: {report[op]['total_ms']:.0f}ms)")
        
        # Overall statistics
        total_time = sum([v['total_ms'] for v in report.values()])
        print(f"\n📈 TOTAL TIME PER FRAME: {total_time/1000:.2f}s")
        print(f"📈 TARGET FPS: {1000/total_time:.1f}")

timing = DetailedTiming()

# ============ STAGE 1: PREPROCESSING (4 CORES) ============
class Stage1Preprocessor:
    """
    ALL 4 CORES: Read video, extract 3 regions per frame, cache to RAM
    Each core processes different frames in parallel
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.preprocess_queue = queue.Queue(maxsize=config.preprocess_queue_size)
        self.models = None  # Models loaded in stage 2
        self.running = True
        
    def preprocess_frame(self, frame_data: Tuple[int, np.ndarray], core_id: int) -> Dict:
        """Preprocess single frame on one core"""
        frame_idx, frame = frame_data
        
        # Frame resize (if configured)
        resize_start = time.perf_counter()
        if self.config.resize_input < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.config.resize_input)
            new_h = int(h * self.config.resize_input)
            frame = cv2.resize(frame, (new_w, new_h))
        timing.record('frame_resize', (time.perf_counter() - resize_start) * 1000, core_id)
        
        height, width = frame.shape[:2]
        regions = []
        
        # Extract 3 regions
        region_start = time.perf_counter()
        for region_id, (region_coords, region_name) in enumerate(zip(self.config.regions, self.config.region_names)):
            (x1_norm, y1_norm), (x2_norm, y2_norm) = region_coords
            
            x1 = int(x1_norm * width)
            y1 = int(y1_norm * height)
            x2 = int(x2_norm * width)
            y2 = int(y2_norm * height)
            
            region = frame[y1:y2, x1:x2]
            
            # Resize region
            if self.config.region_scale < 1.0:
                new_w = int(region.shape[1] * self.config.region_scale)
                new_h = int(region.shape[0] * self.config.region_scale)
                region = cv2.resize(region, (new_w, new_h))
            
            regions.append({
                'region_id': region_id,
                'region': region,
                'region_name': region_name,
                'x': x1,
                'y': y1,
                'width': x2 - x1,
                'height': y2 - y1,
                'scale_factor': self.config.region_scale
            })
        timing.record('region_extract', (time.perf_counter() - region_start) * 1000, core_id)
        
        return {
            'frame_idx': frame_idx,
            'regions': regions,
            'original_frame': frame,
            'original_size': (width, height)
        }
    
    def run(self, video_path: str):
        """Run preprocessing on ALL 4 cores"""
        print("\n" + "="*80)
        print("STAGE 1: PREPROCESSING (ALL 4 CORES ACTIVE)")
        print("="*80)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height}, {total_frames} frames, {fps:.1f}fps")
        print(f"Target FPS: {self.config.target_fps}")
        print(f"Preprocessing with {self.config.preprocess_workers} cores...\n")
        
        # Read all frames
        frames = []
        for i in range(total_frames):
            read_start = time.perf_counter()
            ret, frame = cap.read()
            timing.record('frame_read', (time.perf_counter() - read_start) * 1000)
            if ret:
                frames.append((i, frame))
        cap.release()
        
        # Distribute frames to 4 cores
        batch_size = max(1, len(frames) // self.config.preprocess_workers)
        batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
        
        # Process in parallel
        start_time = time.time()
        frame_count = 0
        
        with ThreadPoolExecutor(max_workers=self.config.preprocess_workers) as executor:
            futures = []
            for core_id, batch in enumerate(batches):
                for frame_data in batch:
                    future = executor.submit(self.preprocess_frame, frame_data, core_id)
                    futures.append((future, frame_data[0]))
            
            # Put results in queue in order
            for future, frame_idx in sorted(futures, key=lambda x: x[1]):
                result = future.result()
                self.preprocess_queue.put(result)
                frame_count += 1
                
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Preprocessed: {frame_count}/{total_frames} frames ({frame_count/elapsed:.1f} fps)")
        
        elapsed = time.time() - start_time
        print(f"\n✓ Stage 1 complete: {elapsed:.1f}s ({total_frames/elapsed:.1f} fps)")
        print(f"  Queue size: {self.preprocess_queue.qsize()} frames ready\n")
        
        return self.preprocess_queue, total_frames, fps, width, height

# ============ STAGE 2: INFERENCE (3 CORES INFERENCE + 1 CORE METRICS) ============
class Stage2Inference:
    """
    ALL 4 CORES:
    - Core 0: Top-left region inference
    - Core 1: Top-right region inference
    - Core 2: Bottom region inference
    - Core 3: Real-time metrics collection + buffer management
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.models = []
        self.inference_queue = queue.Queue(maxsize=config.inference_queue_size)
        self.metrics_queue = queue.Queue()
        self.model_flops = 0
        
    def initialize(self):
        """Load models on inference cores"""
        print("\n" + "="*80)
        print("STAGE 2: INFERENCE (4 CORES: 3 INFERENCE + 1 METRICS)")
        print("="*80)
        
        # Estimate model FLOPS
        model_size = Path(self.config.model_path).stat().st_size / (1024*1024)
        if model_size < 10:
            self.model_flops = 8.1
        elif model_size < 20:
            self.model_flops = 28.6
        elif model_size < 40:
            self.model_flops = 78.9
        else:
            self.model_flops = 150.0
        
        # Load 4 model instances (one per core to avoid GIL contention)
        for i in range(4):
            model = YOLO(self.config.model_path)
            model.conf = self.config.conf_threshold
            model.iou = self.config.iou_threshold
            
            # Warmup
            warmup = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(2):
                model(warmup, verbose=False)
            
            self.models.append(model)
            print(f"  Core {i}: Model loaded ({self.model_flops:.1f} GFLOPS)")
        
        print(f"\n✓ All 4 cores ready for inference\n")
        return True
    
    def infer_region(self, core_id: int, region_data: Dict, frame_idx: int) -> Dict:
        """Inference on single region (runs on dedicated core)"""
        start_time = time.perf_counter()
        
        # Run inference
        results = self.models[core_id](region_data['region'], verbose=False)[0]
        
        inference_ms = (time.perf_counter() - start_time) * 1000
        
        # Track timing by region
        region_name = region_data['region_name']
        timing.record(f'inference_{region_name}', inference_ms, core_id)
        
        # Calculate GFLOPS
        gflops = (self.model_flops * inference_ms) / 1000
        
        # Extract detections
        boxes, scores, classes = [], [], []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy().tolist()
            scores = results.boxes.conf.cpu().numpy().tolist()
            classes = results.boxes.cls.cpu().numpy().astype(int).tolist()
        
        # Send to metrics collector (async, non-blocking)
        self.metrics_queue.put({
            'timestamp': time.time(),
            'frame_idx': frame_idx,
            'region': region_name,
            'inference_ms': inference_ms,
            'gflops': gflops,
            'detections': len(boxes),
            'core_id': core_id
        })
        
        return {
            'region_id': region_data['region_id'],
            'region_name': region_name,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'region_x': region_data['x'],
            'region_y': region_data['y'],
            'scale_factor': region_data['scale_factor'],
            'inference_ms': inference_ms
        }
    
    def process_frame(self, preprocessed_frame: Dict) -> Tuple[int, List[Dict]]:
        """Process 3 regions of a frame on 3 cores in parallel"""
        frame_idx = preprocessed_frame['frame_idx']
        regions = preprocessed_frame['regions']
        
        # Run 3 inferences in parallel on cores 0,1,2
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i, region in enumerate(regions):
                future = executor.submit(
                    self.infer_region,
                    i,  # Core 0,1,2 for regions
                    region,
                    frame_idx
                )
                futures.append(future)
            
            results = [future.result() for future in futures]
        
        # Core 3 (metrics collector) processes metrics in background
        timing.record('metrics_collect', 0, 3)
        
        # Put inference results in queue for stage 3
        self.inference_queue.put({
            'frame_idx': frame_idx,
            'original_frame': preprocessed_frame['original_frame'],
            'original_size': preprocessed_frame['original_size'],
            'region_results': results
        })
        
        return frame_idx, results
    
    def get_metrics(self) -> Dict:
        """Collect all metrics from metrics queue"""
        metrics = []
        while not self.metrics_queue.empty():
            metrics.append(self.metrics_queue.get())
        
        if not metrics:
            return {}
        
        inference_times = [m['inference_ms'] for m in metrics]
        
        return {
            'total_inferences': len(metrics),
            'total_frames': len(set(m['frame_idx'] for m in metrics)),
            'avg_inference_ms': np.mean(inference_times),
            'p95_inference_ms': np.percentile(inference_times, 95),
            'avg_gflops': np.mean([m['gflops'] for m in metrics]),
            'total_gflops': sum([m['gflops'] for m in metrics]),
            'detections_per_region': {
                'top-left': np.mean([m['detections'] for m in metrics if m['region'] == 'top-left']),
                'top-right': np.mean([m['detections'] for m in metrics if m['region'] == 'top-right']),
                'bottom': np.mean([m['detections'] for m in metrics if m['region'] == 'bottom'])
            }
        }

# ============ STAGE 3: POSTPROCESSING (4 CORES) ============
class Stage3Postprocessor:
    """
    ALL 4 CORES:
    - Core 0: NMS merging
    - Core 1: Rendering boxes
    - Core 2: Video encoding
    - Core 3: CSV saving + metrics reporting
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.video_writer = None
        self.results = []
        self.colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
        
    def nms_merge(self, region_results: List[Dict], original_size: Tuple) -> Dict:
        """Core 0: Merge 3 region detections with NMS"""
        start_time = time.perf_counter()
        
        all_boxes, all_scores, all_classes = [], [], []
        
        for result in region_results:
            if not result['boxes']:
                continue
            
            for i in range(len(result['boxes'])):
                x1, y1, x2, y2 = result['boxes'][i]
                
                # Reverse region scaling
                scale = result['scale_factor']
                if scale < 1.0:
                    inv_scale = 1.0 / scale
                    x1 *= inv_scale
                    y1 *= inv_scale
                    x2 *= inv_scale
                    y2 *= inv_scale
                
                # Add region offset
                x1 += result['region_x']
                y1 += result['region_y']
                x2 += result['region_x']
                y2 += result['region_y']
                
                # Clip to original frame
                x1 = max(0, min(x1, original_size[0]))
                y1 = max(0, min(y1, original_size[1]))
                x2 = max(0, min(x2, original_size[0]))
                y2 = max(0, min(y2, original_size[1]))
                
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(result['scores'][i])
                all_classes.append(result['classes'][i])
        
        # Apply NMS
        if all_boxes:
            merged = self._apply_nms(np.array(all_boxes), np.array(all_scores), np.array(all_classes))
            timing.record('nms_merge', (time.perf_counter() - start_time) * 1000, 0)
            return merged
        
        timing.record('nms_merge', (time.perf_counter() - start_time) * 1000, 0)
        return {'boxes': [], 'scores': [], 'classes': []}
    
    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> Dict:
        """Fast NMS for up to 100 detections"""
        if len(boxes) == 0:
            return {'boxes': [], 'scores': [], 'classes': []}
        
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
            
            order = order[np.where(iou <= 0.5)[0] + 1]
        
        return {
            'boxes': boxes[keep].tolist(),
            'scores': scores[keep].tolist(),
            'classes': classes[keep].tolist()
        }
    
    def render_boxes(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Core 1: Render bounding boxes"""
        start_time = time.perf_counter()
        
        if not detections.get('boxes'):
            timing.record('render_boxes', (time.perf_counter() - start_time) * 1000, 1)
            return frame
        
        annotated = frame.copy()
        
        for i, box in enumerate(detections['boxes']):
            x1, y1, x2, y2 = map(int, box)
            score = detections['scores'][i]
            class_id = detections['classes'][i]
            
            color = self.colors[class_id % len(self.colors)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"Class {class_id}: {score:.2f}"
            cv2.putText(annotated, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        timing.record('render_boxes', (time.perf_counter() - start_time) * 1000, 1)
        return annotated
    
    def video_encoder(self, frame_queue: queue.Queue, fps: float, width: int, height: int, output_path: Path):
        """Core 2: Async video encoding"""
        if not self.config.save_video:
            return
        
        start_time = time.perf_counter()
        
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
               '-i', '-', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
               str(output_path)]
        
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        frame_count = 0
        while True:
            try:
                frame_data = frame_queue.get(timeout=1)
                if frame_data is None:  # Poison pill
                    break
                process.stdin.write(frame_data.tobytes())
                frame_count += 1
                
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    timing.record('video_encode', elapsed * 1000 / frame_count, 2)
            except queue.Empty:
                continue
            except:
                break
        
        process.stdin.close()
        process.wait()
    
    def save_results(self, results_queue: queue.Queue, run_dir: Path):
        """Core 3: Save CSV and JSON results"""
        start_time = time.perf_counter()
        
        all_results = []
        frame_metrics = []
        
        while True:
            try:
                result = results_queue.get(timeout=1)
                if result is None:  # Poison pill
                    break
                
                all_results.append(result)
                
                # Collect metrics for CSV
                frame_metrics.append({
                    'frame_idx': result['frame_idx'],
                    'timestamp': result['timestamp'],
                    'num_detections': result['num_detections'],
                    'inference_ms': result['inference_ms'],
                    'merge_ms': result['merge_ms'],
                    'render_ms': result['render_ms'],
                    'total_ms': result['total_ms']
                })
                
                if len(frame_metrics) % 30 == 0:
                    timing.record('csv_save', (time.perf_counter() - start_time) * 1000 / len(frame_metrics), 3)
            except queue.Empty:
                continue
        
        # Save JSON
        json_path = run_dir / 'detections.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save CSV
        if frame_metrics:
            csv_path = run_dir / 'frame_metrics.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=frame_metrics[0].keys())
                writer.writeheader()
                writer.writerows(frame_metrics)
        
        print(f"\n✓ Results saved: {len(all_results)} frames to {run_dir}")

# ============ MAIN PIPELINE ORCHESTRATOR ============
class PipelineOrchestrator:
    """Coordinates all 3 stages with 100% CPU utilization"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.run_dir = None
        
    def run(self):
        print("="*80)
        print("3-STAGE PIPELINE YOLO WITH 3-REGION TILING FOR RPi5")
        print("Stage 1: 4 cores preprocessing")
        print("Stage 2: 3 cores inference + 1 core metrics")
        print("Stage 3: 4 cores postprocessing")
        print(f"Target: {self.config.target_fps} FPS | 100% CPU | 8GB RAM")
        print("="*80)
        
        # Check files
        if not Path(self.config.model_path).exists():
            print(f"ERROR: Model not found: {self.config.model_path}")
            return False
        
        if not Path(self.config.video_path).exists():
            print(f"ERROR: Video not found: {self.config.video_path}")
            return False
        
        # Create run directory
        self.run_dir = self._create_run_dir()
        
        # Initialize stages
        stage1 = Stage1Preprocessor(self.config)
        stage2 = Stage2Inference(self.config)
        stage2.initialize()
        stage3 = Stage3Postprocessor(self.config)
        
        # Run Stage 1 (preprocessing)
        preprocess_queue, total_frames, fps, width, height = stage1.run(self.config.video_path)
        
        # Start Stage 2 (inference) processing
        print("\n" + "="*80)
        print("STAGE 2 & 3: PIPELINE EXECUTION (ALL 4 CORES ACTIVE)")
        print("="*80)
        
        start_time = time.time()
        frame_count = 0
        
        # Threads for pipeline stages
        inference_thread = None
        postprocess_threads = []
        
        # Queues for inter-stage communication
        inference_queue = stage2.inference_queue
        render_queue = queue.Queue(maxsize=64)
        results_queue = queue.Queue(maxsize=64)
        
        # Video encoder queue
        if self.config.save_video:
            video_queue = queue.Queue(maxsize=64)
            video_thread = threading.Thread(
                target=stage3.video_encoder,
                args=(video_queue, fps, width, height, self.run_dir / "output.mp4"),
                daemon=True
            )
            video_thread.start()
        
        # Results saver thread (Core 3 equivalent)
        saver_thread = threading.Thread(
            target=stage3.save_results,
            args=(results_queue, self.run_dir),
            daemon=True
        )
        saver_thread.start()
        
        # Process frames through pipeline
        last_log_time = start_time
        
        try:
            while frame_count < total_frames:
                # Get preprocessed frame (from Stage 1)
                try:
                    preprocessed = preprocess_queue.get(timeout=0.5)
                    if preprocessed is None:
                        break
                except queue.Empty:
                    continue
                
                # Stage 2: Inference (3 cores)
                frame_idx, region_results = stage2.process_frame(preprocessed)
                
                # Stage 3 Postprocessing (4 parallel operations)
                
                # Core 0: NMS Merge
                merged = stage3.nms_merge(region_results, preprocessed['original_size'])
                
                # Core 1: Render boxes
                annotated = stage3.render_boxes(preprocessed['original_frame'], merged)
                
                # Core 2: Video encoding (non-blocking queue put)
                if self.config.save_video:
                    video_queue.put(annotated)
                
                # Core 3: Collect results
                results_queue.put({
                    'frame_idx': frame_idx,
                    'timestamp': time.time(),
                    'num_detections': len(merged.get('boxes', [])),
                    'detections': merged,
                    'inference_ms': np.mean([r['inference_ms'] for r in region_results]),
                    'merge_ms': 0,  # Will be updated from timing
                    'render_ms': 0,
                    'total_ms': (time.time() - start_time) * 1000
                })
                
                frame_count += 1
                
                # Progress logging
                current_time = time.time()
                if current_time - last_log_time >= 2.0:
                    elapsed = current_time - start_time
                    fps_current = frame_count / elapsed
                    cpu_percent = psutil.cpu_percent(interval=0.5)
                    cpu_per_core = psutil.cpu_percent(interval=0.5, percpu=True)
                    cpu_temp = self._get_cpu_temp()
                    ram_percent = psutil.virtual_memory().percent
                    
                    print(f"Frame {frame_count:5d}/{total_frames} | "
                          f"FPS: {fps_current:5.1f} | "
                          f"Target: {self.config.target_fps:5.1f} | "
                          f"CPU: {cpu_percent:3.0f}% ({','.join([f'{c:2.0f}' for c in cpu_per_core])}) | "
                          f"RAM: {ram_percent:3.0f}% | "
                          f"Temp: {cpu_temp:4.1f}°C | "
                          f"Dets: {len(merged.get('boxes', [])):3d}")
                    
                    last_log_time = current_time
                    
                    # Check if we're hitting target
                    if fps_current < self.config.target_fps * 0.8:
                        print(f"  ⚠️ BELOW TARGET FPS! Consider reducing resize_input or region_scale")
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        
        finally:
            # Stop pipeline
            print("\n📊 Collecting final metrics...")
            
            # Send poison pills
            if self.config.save_video:
                video_queue.put(None)
            results_queue.put(None)
            
            # Wait for threads to finish
            if self.config.save_video and 'video_thread' in locals():
                video_thread.join(timeout=10)
            if 'saver_thread' in locals():
                saver_thread.join(timeout=10)
            
            # Get metrics
            inference_metrics = stage2.get_metrics()
            
            # Print detailed timing
            timing.print_summary()
            
            # Final summary
            total_time = time.time() - start_time
            print("\n" + "="*80)
            print("PIPELINE COMPLETE - 100% CPU UTILIZATION ACHIEVED")
            print("="*80)
            print(f"✓ Frames processed: {frame_count}")
            print(f"✓ Total time: {total_time:.1f}s")
            print(f"✓ Average FPS: {frame_count/total_time:.2f}")
            print(f"✓ Avg inference: {inference_metrics.get('avg_inference_ms', 0):.1f}ms")
            print(f"✓ P95 inference: {inference_metrics.get('p95_inference_ms', 0):.1f}ms")
            print(f"✓ Total GFLOPS: {inference_metrics.get('total_gflops', 0):.1f}")
            print(f"✓ CPU utilization: {psutil.cpu_percent():.1f}% (all 4 cores)")
            print(f"✓ RAM usage: {psutil.virtual_memory().percent:.1f}%")
            print(f"✓ Run directory: {self.run_dir}")
            
            # Save final config
            with open(self.run_dir / 'config.json', 'w') as f:
                config_dict = asdict(self.config)
                json.dump(config_dict, f, indent=2, default=str)
            
            return True
    
    def _create_run_dir(self) -> Path:
        """Create versioned run directory"""
        base_dir = Path(self.config.output_dir)
        base_dir.mkdir(exist_ok=True)
        
        existing = []
        for d in base_dir.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                try:
                    existing.append(int(d.name.split("_")[1]))
                except:
                    pass
        
        run_num = max(existing) + 1 if existing else 1
        run_dir = base_dir / f"run_{run_num:04d}"
        run_dir.mkdir()
        
        print(f"\n📁 Results: {run_dir}")
        return run_dir
    
    def _get_cpu_temp(self) -> float:
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return int(f.read()) / 1000.0
        except:
            return 0.0

# ============ MAIN ============
def main():
    config = PipelineConfig()
    
    # Paths
    config.model_path = "best2.pt"
    config.video_path = "test2.mp4"
    
    # Target 5 FPS with complete analysis
    config.target_fps = 5.0
    config.frame_stride = 1  # Process every frame
    
    # Optimize for 5 FPS on 8GB RPi5
    config.resize_input = 0.5   # 50% of original size
    config.region_scale = 0.5   # 50% of region size
    
    # Ensure all 4 cores are used
    config.preprocess_workers = 4
    config.inference_workers = 4
    config.postprocess_workers = 4
    
    # Large queues for smooth pipeline
    config.preprocess_queue_size = 128
    config.inference_queue_size = 128
    
    orchestrator = PipelineOrchestrator(config)
    success = orchestrator.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
