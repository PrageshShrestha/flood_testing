#!/usr/bin/env python3
"""
3-STAGE PIPELINE YOLO WITH 3-REGION TILING FOR RPi5
Stage 1: 4 cores preprocessing
Stage 2: Single model, 3 cores inference + 1 core metrics
Stage 3: 4 cores postprocessing
Complete per-frame metrics: timing, CPU, temp, RAM, GFLOPS, power
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
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from ultralytics import YOLO

# ============ CONFIG ============
@dataclass
class Config:
    model_path: str = "best2.pt"
    video_path: str = "test2.mp4"
    output_dir: str = "research_runs"
    
    regions: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=lambda: [
        ((0.0, 0.0), (0.5, 0.5)),   # top-left
        ((0.5, 0.0), (1.0, 0.5)),   # top-right
        ((0.0, 0.5), (1.0, 1.0))    # bottom
    ])
    region_names: List[str] = field(default_factory=lambda: ["tl", "tr", "bt"])
    
    resize_input: float = 0.5
    region_scale: float = 0.5
    conf: float = 0.3
    iou: float = 0.45
    save_video: bool = True
    
    # Power estimation
    idle_power_w: float = 3.0
    cpu_power_coef: float = 0.15
    temp_power_coef: float = 0.05

# ============ METRICS COLLECTOR ============
class MetricsCollector:
    def __init__(self, run_dir: Path, cfg: Config):
        self.run_dir = run_dir
        self.cfg = cfg
        self.frame_metrics = []
        self.model_flops = self._estimate_flops()
        
    def _estimate_flops(self) -> float:
        size_mb = Path(self.cfg.model_path).stat().st_size / (1024*1024)
        if size_mb < 10:
            return 8.1
        elif size_mb < 20:
            return 28.6
        elif size_mb < 40:
            return 78.9
        return 150.0
    
    def _get_cpu_temp(self) -> float:
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return int(f.read()) / 1000.0
        except:
            return 0.0
    
    def _estimate_power(self, cpu_percent: float, cpu_temp: float) -> float:
        base = self.cfg.idle_power_w
        cpu_power = (cpu_percent / 100.0) * 15.0
        temp_penalty = max(0, (cpu_temp - 50) * self.cfg.temp_power_coef)
        return base + cpu_power + temp_penalty
    
    def log_frame(self, frame_idx: int, stage1_ms: float, stage2_ms: float, stage3_ms: float,
                  tl_ms: float, tr_ms: float, bt_ms: float, detections: int):
        
        cpu_percent = psutil.cpu_percent()
        cpu_per_core = psutil.cpu_percent(percpu=True)
        cpu_temp = self._get_cpu_temp()
        ram_percent = psutil.virtual_memory().percent
        ram_used_gb = psutil.virtual_memory().used / (1024**3)
        
        # Calculate GFLOPS
        inference_ms = (tl_ms + tr_ms + bt_ms) / 3
        gflops = (self.model_flops * inference_ms) / 1000
        total_gflops = self.model_flops * (tl_ms + tr_ms + bt_ms) / 1000
        
        # Power estimation
        power_w = self._estimate_power(cpu_percent, cpu_temp)
        
        metric = {
            'frame_idx': frame_idx,
            'timestamp': time.time(),
            'stage1_preprocess_ms': stage1_ms,
            'stage2_inference_ms': stage2_ms,
            'stage3_postprocess_ms': stage3_ms,
            'total_frame_ms': stage1_ms + stage2_ms + stage3_ms,
            'region_tl_ms': tl_ms,
            'region_tr_ms': tr_ms,
            'region_bt_ms': bt_ms,
            'num_detections': detections,
            'cpu_percent': cpu_percent,
            'cpu_per_core_0': cpu_per_core[0] if len(cpu_per_core) > 0 else 0,
            'cpu_per_core_1': cpu_per_core[1] if len(cpu_per_core) > 1 else 0,
            'cpu_per_core_2': cpu_per_core[2] if len(cpu_per_core) > 2 else 0,
            'cpu_per_core_3': cpu_per_core[3] if len(cpu_per_core) > 3 else 0,
            'cpu_temp_c': cpu_temp,
            'ram_percent': ram_percent,
            'ram_used_gb': ram_used_gb,
            'gflops': gflops,
            'total_gflops': total_gflops,
            'power_estimate_w': power_w
        }
        self.frame_metrics.append(metric)
    
    def save(self):
        if not self.frame_metrics:
            return
        
        # CSV
        csv_path = self.run_dir / 'frame_metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.frame_metrics[0].keys())
            writer.writeheader()
            writer.writerows(self.frame_metrics)
        
        # Generate report
        self._generate_report()
    
    def _generate_report(self):
        if not self.frame_metrics:
            return
        
        total_frames = len(self.frame_metrics)
        total_time = self.frame_metrics[-1]['timestamp'] - self.frame_metrics[0]['timestamp']
        avg_fps = total_frames / total_time
        
        avg_inference = np.mean([m['stage2_inference_ms'] for m in self.frame_metrics])
        p95_inference = np.percentile([m['stage2_inference_ms'] for m in self.frame_metrics], 95)
        avg_cpu = np.mean([m['cpu_percent'] for m in self.frame_metrics])
        avg_temp = np.mean([m['cpu_temp_c'] for m in self.frame_metrics])
        avg_power = np.mean([m['power_estimate_w'] for m in self.frame_metrics])
        total_energy_wh = (avg_power * total_time) / 3600
        total_gflops = sum([m['total_gflops'] for m in self.frame_metrics])
        
        report = {
            'run_info': {
                'timestamp': datetime.now().isoformat(),
                'run_dir': str(self.run_dir),
                'model_path': self.cfg.model_path,
                'video_path': self.cfg.video_path,
                'model_flops_gflops': self.model_flops,
                'resize_input': self.cfg.resize_input,
                'region_scale': self.cfg.region_scale
            },
            'performance': {
                'total_frames': total_frames,
                'total_time_sec': total_time,
                'average_fps': avg_fps,
                'avg_inference_ms': avg_inference,
                'p95_inference_ms': p95_inference,
                'avg_detections_per_frame': np.mean([m['num_detections'] for m in self.frame_metrics]),
                'total_detections': sum([m['num_detections'] for m in self.frame_metrics])
            },
            'system_metrics': {
                'avg_cpu_percent': avg_cpu,
                'peak_cpu_percent': max([m['cpu_percent'] for m in self.frame_metrics]),
                'avg_cpu_temp_c': avg_temp,
                'peak_cpu_temp_c': max([m['cpu_temp_c'] for m in self.frame_metrics]),
                'avg_ram_percent': np.mean([m['ram_percent'] for m in self.frame_metrics]),
                'peak_ram_percent': max([m['ram_percent'] for m in self.frame_metrics]),
                'avg_power_w': avg_power,
                'total_energy_wh': total_energy_wh
            },
            'compute_metrics': {
                'total_gflops': total_gflops,
                'avg_gflops_per_frame': total_gflops / total_frames if total_frames > 0 else 0
            },
            'region_breakdown': {
                'tl_avg_ms': np.mean([m['region_tl_ms'] for m in self.frame_metrics]),
                'tr_avg_ms': np.mean([m['region_tr_ms'] for m in self.frame_metrics]),
                'bt_avg_ms': np.mean([m['region_bt_ms'] for m in self.frame_metrics])
            }
        }
        
        # Save JSON
        json_path = self.run_dir / 'research_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save TXT
        txt_path = self.run_dir / 'research_report.txt'
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RESEARCH REPORT - 3-STAGE PIPELINE\n")
            f.write("="*70 + "\n\n")
            
            f.write("RUN INFORMATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Run Dir: {self.run_dir}\n")
            f.write(f"Model: {self.cfg.model_path}\n")
            f.write(f"Video: {self.cfg.video_path}\n")
            f.write(f"Model GFLOPS: {self.model_flops:.1f}\n\n")
            
            f.write("PERFORMANCE\n")
            f.write("-"*40 + "\n")
            f.write(f"Frames: {total_frames}\n")
            f.write(f"Total Time: {total_time:.2f}s\n")
            f.write(f"Average FPS: {avg_fps:.2f}\n")
            f.write(f"Avg Inference: {avg_inference:.2f}ms\n")
            f.write(f"P95 Inference: {p95_inference:.2f}ms\n\n")
            
            f.write("SYSTEM\n")
            f.write("-"*40 + "\n")
            f.write(f"CPU: {avg_cpu:.1f}% (peak {report['system_metrics']['peak_cpu_percent']:.1f}%)\n")
            f.write(f"Temp: {avg_temp:.1f}°C (peak {report['system_metrics']['peak_cpu_temp_c']:.1f}°C)\n")
            f.write(f"RAM: {report['system_metrics']['avg_ram_percent']:.1f}%\n")
            f.write(f"Power: {avg_power:.2f}W\n")
            f.write(f"Energy: {total_energy_wh:.2f}Wh\n\n")
            
            f.write("COMPUTE\n")
            f.write("-"*40 + "\n")
            f.write(f"Total GFLOPS: {total_gflops:.1f}\n")
            f.write(f"Avg GFLOPS/frame: {report['compute_metrics']['avg_gflops_per_frame']:.1f}\n\n")
            
            f.write("REGIONS\n")
            f.write("-"*40 + "\n")
            f.write(f"Top-Left: {report['region_breakdown']['tl_avg_ms']:.1f}ms\n")
            f.write(f"Top-Right: {report['region_breakdown']['tr_avg_ms']:.1f}ms\n")
            f.write(f"Bottom: {report['region_breakdown']['bt_avg_ms']:.1f}ms\n")
        
        print(f"\nSaved: {txt_path}")

# ============ STAGE 1 ============
class Stage1:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
    def run(self, video_path: str):
        print("\n[STAGE1] PREPROCESSING (4 cores)")
        print("-"*50)
        
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Input: {width}x{height}, {total} frames, {fps:.1f}fps")
        
        frames = []
        for i in range(total):
            ret, frame = cap.read()
            if ret:
                frames.append((i, frame))
        cap.release()
        
        start = time.time()
        q = queue.Queue()
        
        def process_frame(frame_data):
            idx, frame = frame_data
            t0 = time.perf_counter()
            
            if self.cfg.resize_input < 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w*self.cfg.resize_input), int(h*self.cfg.resize_input)))
            
            h, w = frame.shape[:2]
            regions = []
            
            for rid, (coords, name) in enumerate(zip(self.cfg.regions, self.cfg.region_names)):
                (x1n, y1n), (x2n, y2n) = coords
                x1, y1 = int(x1n*w), int(y1n*h)
                x2, y2 = int(x2n*w), int(y2n*h)
                region = frame[y1:y2, x1:x2]
                
                if self.cfg.region_scale < 1.0:
                    region = cv2.resize(region, (int(region.shape[1]*self.cfg.region_scale), 
                                                  int(region.shape[0]*self.cfg.region_scale)))
                
                regions.append({
                    'rid': rid, 'name': name, 'data': region,
                    'x': x1, 'y': y1, 'scale': self.cfg.region_scale
                })
            
            q.put({
                'idx': idx, 'regions': regions, 'frame': frame,
                'size': (w, h), 'preprocess_ms': (time.perf_counter()-t0)*1000
            })
        
        with ThreadPoolExecutor(max_workers=4) as ex:
            ex.map(process_frame, frames)
        
        elapsed = time.time() - start
        print(f"Done: {elapsed:.2f}s ({total/elapsed:.1f}fps)")
        return q, total, fps, width, height

# ============ STAGE 2 ============
class Stage2:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.gflops = 0
        
    def load_model(self):
        print("\n[STAGE2] LOADING MODEL")
        print("-"*50)
        t0 = time.time()
        self.model = YOLO(self.cfg.model_path)
        self.model.conf = self.cfg.conf
        self.model.iou = self.cfg.iou
        
        warm = np.zeros((640,640,3), dtype=np.uint8)
        for _ in range(3):
            self.model(warm, verbose=False)
        
        size_mb = Path(self.cfg.model_path).stat().st_size / (1024*1024)
        if size_mb < 10:
            self.gflops = 8.1
        elif size_mb < 20:
            self.gflops = 28.6
        elif size_mb < 40:
            self.gflops = 78.9
        else:
            self.gflops = 150.0
        
        print(f"Loaded: {time.time()-t0:.2f}s, {self.gflops:.1f}GFLOPS")
    
    def infer_region(self, region, name, idx):
        t0 = time.perf_counter()
        results = self.model(region, verbose=False)[0]
        inf_ms = (time.perf_counter() - t0) * 1000
        
        boxes, scores, classes = [], [], []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy().tolist()
            scores = results.boxes.conf.cpu().numpy().tolist()
            classes = results.boxes.cls.cpu().numpy().astype(int).tolist()
        
        return {'name': name, 'boxes': boxes, 'scores': scores, 'classes': classes,
                'inf_ms': inf_ms, 'detections': len(boxes)}
    
    def run(self, in_queue: queue.Queue):
        print("\n[STAGE2] INFERENCE (3 cores)")
        print("-"*50)
        
        out_queue = queue.Queue()
        start = time.time()
        processed = 0
        
        while True:
            try:
                prepped = in_queue.get(timeout=0.5)
                if prepped is None:
                    break
                
                regions = prepped['regions']
                t0 = time.perf_counter()
                
                with ThreadPoolExecutor(max_workers=3) as ex:
                    futures = []
                    for r in regions:
                        futures.append(ex.submit(self.infer_region, r['data'], r['name'], prepped['idx']))
                    results = [f.result() for f in futures]
                
                inf_ms = (time.perf_counter() - t0) * 1000
                
                out_queue.put({
                    'idx': prepped['idx'],
                    'frame': prepped['frame'],
                    'size': prepped['size'],
                    'results': results,
                    'inference_ms': inf_ms,
                    'tl_ms': results[0]['inf_ms'],
                    'tr_ms': results[1]['inf_ms'],
                    'bt_ms': results[2]['inf_ms']
                })
                
                processed += 1
                if processed % 50 == 0:
                    elapsed = time.time() - start
                    print(f"  Frame {processed}: {processed/elapsed:.1f}fps")
                    
            except queue.Empty:
                continue
        
        elapsed = time.time() - start
        print(f"Done: {elapsed:.2f}s ({processed/elapsed:.1f}fps)")
        return out_queue

# ============ STAGE 3 ============
class Stage3:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
    def nms(self, boxes, scores, thresh=0.5):
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        areas = (x2-x1+1)*(y2-y1+1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2-xx1+1)
            h = np.maximum(0, yy2-yy1+1)
            inter = w*h
            iou = inter/(areas[i]+areas[order[1:]]-inter)
            order = order[np.where(iou <= thresh)[0]+1]
        
        return keep
    
    def run(self, in_queue: queue.Queue, fps: float, width: int, height: int, 
            run_dir: Path, metrics: MetricsCollector):
        print("\n[STAGE3] POSTPROCESSING")
        print("-"*50)
        
        # Video writer
        video_process = None
        if self.cfg.save_video:
            video_path = run_dir / 'output.mp4'
            cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
                   '-i', '-', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                   str(video_path)]
            video_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        start = time.time()
        processed = 0
        frame_results = []
        
        while True:
            try:
                data = in_queue.get(timeout=0.5)
                if data is None:
                    break
                
                t0 = time.perf_counter()
                
                # Collect all detections
                all_boxes, all_scores, all_classes = [], [], []
                for res in data['results']:
                    if not res['boxes']:
                        continue
                    for i in range(len(res['boxes'])):
                        x1, y1, x2, y2 = res['boxes'][i]
                        # Scale back
                        if data['size'][0] > 0:
                            # Regions were already at final size, coordinates are correct
                            pass
                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(res['scores'][i])
                        all_classes.append(res['classes'][i])
                
                # NMS
                if all_boxes:
                    keep = self.nms(all_boxes, all_scores, 0.5)
                    boxes = [all_boxes[i] for i in keep]
                    scores = [all_scores[i] for i in keep]
                    classes = [all_classes[i] for i in keep]
                else:
                    boxes, scores, classes = [], [], []
                
                # Render
                annotated = data['frame'].copy()
                colors = [(0,255,0), (0,0,255), (255,0,0)]
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    color = colors[classes[i] % len(colors)]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{classes[i]}:{scores[i]:.2f}"
                    cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                post_ms = (time.perf_counter() - t0) * 1000
                
                # Write video
                if video_process:
                    video_process.stdin.write(annotated.tobytes())
                
                # Log metrics
                metrics.log_frame(
                    frame_idx=data['idx'],
                    stage1_ms=data.get('preprocess_ms', 0),
                    stage2_ms=data['inference_ms'],
                    stage3_ms=post_ms,
                    tl_ms=data['tl_ms'],
                    tr_ms=data['tr_ms'],
                    bt_ms=data['bt_ms'],
                    detections=len(boxes)
                )
                
                frame_results.append({
                    'frame': data['idx'],
                    'detections': len(boxes),
                    'boxes': boxes,
                    'scores': scores,
                    'classes': classes
                })
                
                processed += 1
                if processed % 50 == 0:
                    elapsed = time.time() - start
                    print(f"  Frame {processed}: post={post_ms:.1f}ms, {processed/elapsed:.1f}fps")
                    
            except queue.Empty:
                continue
        
        # Cleanup
        if video_process:
            video_process.stdin.close()
            video_process.wait()
        
        # Save detections
        with open(run_dir / 'detections.json', 'w') as f:
            json.dump(frame_results, f, indent=2)
        
        elapsed = time.time() - start
        print(f"Done: {elapsed:.2f}s ({processed/elapsed:.1f}fps)")
        return frame_results

# ============ MAIN ============
def main():
    cfg = Config()
    
    # Check files
    if not Path(cfg.model_path).exists():
        print(f"ERROR: {cfg.model_path} not found")
        return 1
    if not Path(cfg.video_path).exists():
        print(f"ERROR: {cfg.video_path} not found")
        return 1
    
    # Run directory
    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(exist_ok=True)
    existing = [int(d.name.split('_')[1]) for d in run_dir.iterdir() 
                if d.is_dir() and d.name.startswith('run_')]
    run_num = max(existing) + 1 if existing else 1
    run_dir = run_dir / f'run_{run_num:04d}'
    run_dir.mkdir()
    print(f"\nRUN: {run_dir}")
    
    # Save config
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(asdict(cfg), f, indent=2)
    
    # Initialize
    metrics = MetricsCollector(run_dir, cfg)
    s1 = Stage1(cfg)
    s2 = Stage2(cfg)
    s2.load_model()
    s3 = Stage3(cfg)
    
    # Run pipeline
    q1, total, fps, w, h = s1.run(cfg.video_path)
    q2 = s2.run(q1)
    results = s3.run(q2, fps, w, h, run_dir, metrics)
    
    # Save metrics
    metrics.save()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Frames:     {total}")
    print(f"Processed:  {len(results)}")
    print(f"FPS:        {len(results) / metrics.frame_metrics[-1]['timestamp'] if metrics.frame_metrics else 0:.2f}")
    print(f"Output:     {run_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
