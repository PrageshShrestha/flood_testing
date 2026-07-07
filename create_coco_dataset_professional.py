import os
import json
import random
import yaml
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
# MONKEY PATCH for transformers compatibility
import transformers
import sys

# Create the missing classes if they don't exist
if not hasattr(transformers, 'BackboneConfigMixin'):
    class BackboneConfigMixin:
        pass
    
    class BackboneMixin:
        pass
    
    transformers.BackboneConfigMixin = BackboneConfigMixin
    transformers.__dict__['BackboneConfigMixin'] = BackboneConfigMixin
    transformers.__dict__['BackboneMixin'] = BackboneMixin
    
    print(" Applied monkey patch for transformers Backbone classes")

# Now import RT-DETR
from ultralytics import YOLO
import supervision as sv
from datetime import datetime
from collections import defaultdict, Counter
import time
import psutil
import threading
import GPUtil
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Any
import shutil

# SAHI imports for small object detection enhancement
try:
    from sahi.utils.coco import Coco
    from sahi.utils.file import save_json
    from sahi.slicing import slice_coco
    from sahi.prediction import prediction_score
    from sahi.utils.cv import read_image
    from sahi.utils.yolov5 import export_coco_as_yolov5
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    print("Warning: SAHI not installed. Install with: pip install sahi[yolo]")

class ResourceMonitor:
    """Comprehensive resource monitoring class"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring in background"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def _monitor_loop(self):
        """Monitor system resources"""
        while self.monitoring:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # GPU metrics
                gpu_stats = []
                gpu_mem_stats = []
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_stats.append(gpu.load * 100)
                        gpu_mem_stats.append(gpu.memoryUtil * 100)
                except:
                    gpu_stats = [0.0]
                    gpu_mem_stats = [0.0]
                
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory.percent)
                self.gpu_usage.append(gpu_stats[0] if gpu_stats else 0.0)
                self.gpu_memory.append(gpu_mem_stats[0] if gpu_mem_stats else 0.0)
                self.timestamps.append(time.time())
                
                time.sleep(0.5)
            except Exception as e:
                print(f"Monitoring error: {e}")
                
    def stop_monitoring(self):
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
        return self.get_statistics()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive resource statistics"""
        if not self.cpu_usage:
            return {}
            
        stats = {
            'cpu': {
                'mean': np.mean(self.cpu_usage),
                'max': np.max(self.cpu_usage),
                'min': np.min(self.cpu_usage),
                'std': np.std(self.cpu_usage),
                'median': np.median(self.cpu_usage)
            },
            'memory': {
                'mean': np.mean(self.memory_usage),
                'max': np.max(self.memory_usage),
                'min': np.min(self.memory_usage),
                'std': np.std(self.memory_usage),
                'median': np.median(self.memory_usage)
            },
            'gpu': {
                'mean': np.mean(self.gpu_usage),
                'max': np.max(self.gpu_usage),
                'min': np.min(self.gpu_usage),
                'std': np.std(self.gpu_usage),
                'median': np.median(self.gpu_usage)
            },
            'gpu_memory': {
                'mean': np.mean(self.gpu_memory),
                'max': np.max(self.gpu_memory),
                'min': np.min(self.gpu_memory),
                'std': np.std(self.gpu_memory),
                'median': np.median(self.gpu_memory)
            },
            'duration': self.timestamps[-1] - self.timestamps[0] if self.timestamps else 0,
            'sample_count': len(self.cpu_usage)
        }
        
        return stats

class SAHIConfig:
    """SAHI slicing configuration for small object detection enhancement"""
    
    def __init__(self, 
                 slice_height: int = 512,
                 slice_width: int = 512,
                 overlap_height_ratio: float = 0.2,
                 overlap_width_ratio: float = 0.2,
                 min_area_ratio: float = 0.1,
                 min_bbox_area: int = 32*32,
                 verbose: bool = True):
        
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.min_area_ratio = min_area_ratio
        self.min_bbox_area = min_bbox_area
        self.verbose = verbose
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate SAHI configuration parameters"""
        if self.slice_height < 256 or self.slice_width < 256:
            raise ValueError("Slice dimensions should be at least 256x256")
        
        if not (0 < self.overlap_height_ratio < 1) or not (0 < self.overlap_width_ratio < 1):
            raise ValueError("Overlap ratios should be between 0 and 1")
        
        if self.min_area_ratio < 0 or self.min_area_ratio > 1:
            raise ValueError("Min area ratio should be between 0 and 1")
    
    def get_slice_params(self):
        """Get SAHI slice parameters dictionary"""
        return {
            "slice_height": self.slice_height,
            "slice_width": self.slice_width,
            "overlap_height_ratio": self.overlap_height_ratio,
            "overlap_width_ratio": self.overlap_width_ratio
        }
    
    def print_config(self):
        """Print SAHI configuration"""
        print("SAHI Configuration:")
        print(f"  • Slice size: {self.slice_width}x{self.slice_height}")
        print(f"  • Overlap ratio: {self.overlap_width_ratio:.2f}x{self.overlap_height_ratio:.2f}")
        print(f"  • Min area ratio: {self.min_area_ratio:.2f}")
        print(f"  • Min bbox area: {self.min_bbox_area} pixels²")

class DatasetCreator:
    def __init__(self, images_folder, model_path, output_dir="prepared_dataset", confidence_threshold=0.5, 
                 enable_sahi=True, sahi_config=None):
        self.images_folder = Path(images_folder)
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        self.enable_sahi = enable_sahi and SAHI_AVAILABLE
        
        # Filter to only keep person (class 0) and umbrella (class 25)
        self.target_classes = {0: "person", 25: "umbrella"}  # COCO class IDs
        
        # SAHI configuration
        if self.enable_sahi:
            self.sahi_config = sahi_config or SAHIConfig()
            self.sahi_config.print_config()
        else:
            self.sahi_config = None
            if enable_sahi and not SAHI_AVAILABLE:
                print("SAHI requested but not available. Install with: pip install sahi")
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        self.metrics = {
            'extraction_time': 0,
            'annotation_time': 0,
            'sahi_time': 0,
            'total_time': 0,
            'detection_stats': defaultdict(list),
            'category_counts': Counter(),
            'bbox_sizes': [],
            'confidence_scores': [],
            'sahi_stats': defaultdict(list) if self.enable_sahi else {},
            'filtered_count': 0,  # Track filtered annotations
            'total_original_detections': 0  # Track original detections before filtering
        }
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.plots_dir = self.output_dir / "analysis_plots"
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize model
        print("Loading RT-DETR model...")
        model_load_start = time.time()
        self.model = YOLO(model_path)
        model_load_time = time.time() - model_load_start
        print(f"Model loaded successfully in {model_load_time:.2f} seconds!")
        
        # YOLO dataset structure
        self.yolo_data = {
            "info": {
                "description": "Dataset created from images using RT-DETR - Only Person and Umbrella classes",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "RT-DETR Auto-Annotation",
                "date_created": datetime.now().isoformat()
            },
            "images": [],
            "annotations": []
        }
        
        # YOLO class names (Only person and umbrella)
        self.yolo_classes = ["person", "umbrella"]
        
        # Category mappings (remap original class IDs to 0 and 1)
        self.original_to_new_id = {0: 0, 25: 1}  # person:0->0, umbrella:25->1
        self.id_to_name = {0: "person", 1: "umbrella"}
        self.name_to_id = {"person": 0, "umbrella": 1}

    def rename_images(self):
        """Rename images sequentially"""
        print(f"Renaming images in {self.images_folder}...")
        
        if not self.images_folder.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_folder}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.images_folder.glob(f"*{ext}"))
            image_files.extend(self.images_folder.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {self.images_folder}")
        
        image_files.sort()
        print(f"Found {len(image_files)} images")
        
        # Rename files sequentially
        renamed_count = 0
        for i, image_path in enumerate(image_files, 1):
            # Get file extension
            ext = image_path.suffix.lower()
            if ext not in image_extensions:
                continue
                
            # New filename with zero-padding
            new_filename = f"{i:06d}.jpg"  # Always use .jpg for consistency
            new_path = self.images_folder / new_filename
            
            # Skip if file already has the correct name
            if image_path.name == new_filename:
                print(f"   Skipping {image_path.name} (already correctly named)")
                continue
            
            # Rename the file
            try:
                # If the target file exists, remove it first
                if new_path.exists():
                    new_path.unlink()
                
                image_path.rename(new_path)
                print(f"   Renamed {image_path.name} -> {new_filename}")
                renamed_count += 1
                
                # Convert to JPG if not already JPG (using PIL)
                if ext != '.jpg':
                    try:
                        img = Image.open(new_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(new_path, 'JPEG', quality=95)
                        print(f"   Converted {new_filename} to JPG format")
                    except Exception as e:
                        print(f"   Warning: Could not convert {new_filename} to JPG: {e}")
                        
            except Exception as e:
                print(f"   Error renaming {image_path.name}: {e}")
        
        print(f"Renamed {renamed_count} images")
        return len(image_files)

    def load_images(self):
        """Load images from folder"""
        print(f"Loading images...")
        
        self.resource_monitor.start_monitoring()
        loading_start = time.time()
        
        image_files = sorted(self.images_folder.glob("*.jpg"))
        
        if not image_files:
            raise ValueError(f"No JPG images found in {self.images_folder}")
        
        frames = []
        pbar = tqdm(total=len(image_files), desc="Loading", unit="img")
        
        for i, image_path in enumerate(image_files):
            # Get image dimensions without loading the full image
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                
                frames.append({
                    'filename': image_path.name,
                    'path': image_path,
                    'original_index': i,
                    'width': width,
                    'height': height
                })
            except Exception as e:
                print(f"   Warning: Could not read {image_path.name}: {e}")
                continue
            
            pbar.update(1)
        
        pbar.close()
        
        # Stop monitoring
        loading_stats = self.resource_monitor.stop_monitoring()
        self.metrics['extraction_time'] = time.time() - loading_start
        
        print(f"\nLoaded {len(frames)} images in {self.metrics['extraction_time']:.2f}s")
        print(f"Speed: {len(frames)/self.metrics['extraction_time']:.1f} img/s")
        print(f"CPU: {loading_stats.get('cpu', {}).get('mean', 0):.1f}% | Memory: {loading_stats.get('memory', {}).get('mean', 0):.1f}%")
        
        return frames

    def convert_to_coco(self, frames):
        """Convert YOLO annotations to COCO format"""
        print("Converting to COCO format...")
        
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories (only person and umbrella)
        coco_data["categories"].append({"id": 0, "name": "person", "supercategory": "object"})
        coco_data["categories"].append({"id": 1, "name": "umbrella", "supercategory": "object"})
        
        annotation_id = 0
        
        for i, frame_info in enumerate(tqdm(frames, desc="Converting to COCO", unit="images")):
            # Add image info
            coco_data["images"].append({
                "id": i + 1,
                "file_name": frame_info['filename'],
                "width": frame_info['width'],
                "height": frame_info['height']
            })
            
            # Convert YOLO annotations to COCO
            label_path = self.labels_dir / frame_info['filename'].replace('.jpg', '.txt')
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            bbox_width = float(parts[3])
                            bbox_height = float(parts[4])
                            
                            # Convert to absolute coordinates
                            abs_x_min = (x_center - bbox_width/2) * frame_info['width']
                            abs_y_min = (y_center - bbox_height/2) * frame_info['height']
                            abs_width = bbox_width * frame_info['width']
                            abs_height = bbox_height * frame_info['height']
                            
                            coco_data["annotations"].append({
                                "id": annotation_id,
                                "image_id": i + 1,
                                "category_id": class_id,
                                "bbox": [abs_x_min, abs_y_min, abs_width, abs_height],
                                "area": abs_width * abs_height,
                                "iscrowd": 0
                            })
                            annotation_id += 1
        
        return coco_data

    def apply_sahi(self, frames):
        """Apply SAHI slicing using proper COCO workflow"""
        if not self.enable_sahi:
            print("SAHI disabled")
            return frames
        
        print("Applying SAHI slicing...")
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        sahi_start = time.time()
        
        # Create temporary directories
        temp_dir = self.output_dir / "temp_sahi"
        temp_dir.mkdir(exist_ok=True)
        
        coco_file = temp_dir / "annotations.json"
        sliced_dir = temp_dir / "sliced"
        
        try:
            # Step 1: Convert to COCO format
            coco_data = self.convert_to_coco(frames)
            
            # Save COCO annotations
            with open(coco_file, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            # Step 2: Slice the COCO dataset
            print(f"Slicing with {self.sahi_config.slice_width}x{self.sahi_config.slice_height} slices...")
            
            slice_coco(
                coco_annotation_file_path=str(coco_file),
                image_dir=str(self.images_dir),
                slice_height=self.sahi_config.slice_height,
                slice_width=self.sahi_config.slice_width,
                overlap_height_ratio=self.sahi_config.overlap_height_ratio,
                overlap_width_ratio=self.sahi_config.overlap_width_ratio,
                min_area_ratio=self.sahi_config.min_area_ratio,
                out_dir=str(sliced_dir),
                verbose=False
            )
            
            # Step 3: Load sliced COCO data
            sliced_coco_file = sliced_dir / "sliced_coco.json"
            if not sliced_coco_file.exists():
                print("No sliced data generated")
                return frames
            
            sliced_coco = Coco.from_coco_dict_or_path(str(sliced_coco_file), str(sliced_dir / "images"))
            
            # Step 4: Export back to YOLO format
            print("Converting sliced data back to YOLO format...")
            
            yolo_output_dir = self.output_dir / "sahi_yolo"
            data_yml_path = export_coco_as_yolov5(
                output_dir=str(yolo_output_dir),
                train_coco=sliced_coco,
                val_coco=None,
                test_coco=None
            )
            
            # Step 5: Load sliced images and add to dataset
            sliced_images_dir = yolo_output_dir / "images"
            sliced_labels_dir = yolo_output_dir / "labels"
            
            sliced_frames = []
            slice_id = len(self.yolo_data['images']) + 1
            
            for img_path in tqdm(list(sliced_images_dir.glob("*.jpg")), desc="Loading sliced images", unit="img"):
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                    
                    sliced_frames.append({
                        'filename': img_path.name,
                        'path': img_path,
                        'width': width,
                        'height': height,
                        'is_slice': True
                    })
                    
                    # Add to YOLO data
                    self.yolo_data['images'].append({
                        "id": slice_id,
                        "width": width,
                        "height": height,
                        "file_name": img_path.name,
                        "path": str(img_path),
                        "is_slice": True
                    })
                    
                    # Process sliced annotations (filter for person/umbrella)
                    label_path = sliced_labels_dir / img_path.name.replace('.jpg', '.txt')
                    if label_path.exists():
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    original_class_id = int(parts[0])
                                    # Only keep person (0) and umbrella (25) - check original IDs
                                    if original_class_id in self.original_to_new_id:
                                        new_class_id = self.original_to_new_id[original_class_id]
                                        x_center = float(parts[1])
                                        y_center = float(parts[2])
                                        bbox_width = float(parts[3])
                                        bbox_height = float(parts[4])
                                        
                                        self.yolo_data['annotations'].append({
                                            "id": len(self.yolo_data['annotations']) + 1,
                                            "image_id": slice_id,
                                            "category_id": new_class_id,
                                            "bbox": [x_center, y_center, bbox_width, bbox_height],
                                            "area": bbox_width * bbox_height
                                        })
                    
                    slice_id += 1
                    
                except Exception as e:
                    print(f"Error loading sliced image {img_path.name}: {e}")
                    continue
            
            # Update metrics
            self.metrics['sahi_time'] = time.time() - sahi_start
            self.metrics['sahi_stats'] = {
                'total_slices': len(sliced_frames),
                'avg_slices_per_image': len(sliced_frames) / len(frames) if frames else 0,
                'small_objects_enhanced': 0,
                'total_objects': len(self.yolo_data['annotations']),
                'small_object_ratio': 0
            }
            
            # Stop monitoring
            sahi_stats = self.resource_monitor.stop_monitoring()
            
            print(f"\nSAHI: {len(sliced_frames)} slices from {len(frames)} images")
            print(f"Time: {self.metrics['sahi_time']:.2f}s | CPU: {sahi_stats.get('cpu', {}).get('mean', 0):.1f}% | Memory: {sahi_stats.get('memory', {}).get('mean', 0):.1f}%")
            
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return sliced_frames
            
        except Exception as e:
            print(f"SAHI processing failed: {e}")
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            return frames

    def shuffle_frames(self, frames):
        """Shuffle frames randomly"""
        print("Shuffling frames...")
        random.shuffle(frames)
        return frames

    def annotate_frames(self, frames):
        """Annotate frames using RF-DETR and filter for only person and umbrella"""
        print("Annotating frames...")
        print("Filtering for ONLY: Person (class 0) and Umbrella (class 25)")
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        annotation_start = time.time()
        
        detected_categories = set()
        inference_times = []
        total_annotations = 0
        filtered_out_count = 0
        
        # Progress bar
        pbar = tqdm(total=len(frames), desc="Annotating frames", unit="frames")
        
        for i, frame_info in enumerate(frames):
            frame_start = time.time()
            
            # Load and convert frame
            frame = cv2.imread(str(frame_info['path']))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Run inference
            results = self.model.predict(pil_image, conf=self.confidence_threshold, verbose=False)
            inference_time = time.time() - frame_start
            inference_times.append(inference_time)
            
            # Get image dimensions for YOLO format
            height, width = frame.shape[:2]
            
            # Add image info to YOLO data
            image_info = {
                "id": i + 1,
                "width": width,
                "height": height,
                "file_name": frame_info['filename'],
                "path": str(frame_info['path'])
            }
            self.yolo_data["images"].append(image_info)
            
            # Create YOLO annotation file
            label_filename = frame_info['filename'].replace('.jpg', '.txt')
            label_path = self.labels_dir / label_filename
            
            # Process detections and create YOLO format annotations
            frame_annotations = []
            
            # RT-DETR returns results as a list, take first result
            if results and len(results) > 0:
                detections = results[0]  # Get first result
                frame_detections = len(detections.boxes)
                self.metrics['detection_stats']['detections_per_frame'].append(frame_detections)
                
                for j in range(len(detections.boxes)):
                    box = detections.boxes.xyxy[j].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = detections.boxes.conf[j].cpu().numpy()
                    cls = int(detections.boxes.cls[j].cpu().numpy())
                    
                    # Track original detections
                    self.metrics['total_original_detections'] += 1
                    
                    # FILTER: Only keep person (class 0) and umbrella (class 25)
                    if cls not in self.original_to_new_id:
                        filtered_out_count += 1
                        continue  # Skip this detection
                    
                    # Remap to new class IDs (person:0->0, umbrella:25->1)
                    new_class_id = self.original_to_new_id[cls]
                    confidence = conf
                    
                    # Convert to YOLO format [class_id, x_center, y_center, width, height] (normalized)
                    x1, y1, x2, y2 = box
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    # Normalize coordinates
                    x_center = (x1 + x2) / 2.0 / width
                    y_center = (y1 + y2) / 2.0 / height
                    norm_width = bbox_width / width
                    norm_height = bbox_height / height
                    
                    # YOLO format: class_id x_center y_center width height (using remapped class ID)
                    yolo_annotation = f"{new_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                    frame_annotations.append(yolo_annotation)
                    
                    # Collect metrics
                    area = bbox_width * bbox_height
                    self.metrics['category_counts'][new_class_id] += 1
                    self.metrics['bbox_sizes'].append(area)
                    self.metrics['confidence_scores'].append(confidence)
                    
                    # Store annotation data for analysis
                    annotation_data = {
                        "image_id": i + 1,
                        "category_id": new_class_id,
                        "original_class_id": cls,
                        "category_name": self.id_to_name[new_class_id],
                        "bbox": [float(x1), float(y1), float(bbox_width), float(bbox_height)],
                        "area": float(area),
                        "confidence": float(confidence),
                        "yolo_format": [new_class_id, x_center, y_center, norm_width, norm_height]
                    }
                    self.yolo_data["annotations"].append(annotation_data)
                    total_annotations += 1
                    detected_categories.add(new_class_id)
            
            # Write YOLO annotation file (only with person and umbrella annotations)
            with open(label_path, 'w') as f:
                f.write('\n'.join(frame_annotations))
            
            pbar.update(1)
        
        pbar.close()
        
        # Stop monitoring
        annotation_stats = self.resource_monitor.stop_monitoring()
        self.metrics['annotation_time'] = time.time() - annotation_start
        self.metrics['filtered_count'] = filtered_out_count
        
        print(f"\n📊 Filtering Results:")
        print(f"   • Original detections: {self.metrics['total_original_detections']}")
        print(f"   • Filtered out: {filtered_out_count} ({filtered_out_count/self.metrics['total_original_detections']*100:.1f}%)")
        print(f"   • Kept (Person + Umbrella): {total_annotations} ({total_annotations/self.metrics['total_original_detections']*100:.1f}%)")
        print(f"\nAnnotated {total_annotations} objects in {len(detected_categories)} categories")
        print(f"   • Person: {self.metrics['category_counts'].get(0, 0)} annotations")
        print(f"   • Umbrella: {self.metrics['category_counts'].get(1, 0)} annotations")
        print(f"Time: {self.metrics['annotation_time']:.2f}s | Speed: {len(frames)/self.metrics['annotation_time']:.1f} FPS")
        print(f"Avg inference: {np.mean(inference_times):.3f}s | Detections/frame: {np.mean(self.metrics['detection_stats']['detections_per_frame']):.1f}")
        print(f"CPU: {annotation_stats.get('cpu', {}).get('mean', 0):.1f}% | Memory: {annotation_stats.get('memory', {}).get('mean', 0):.1f}%")
        
        return detected_categories

    def create_train_val_test_split(self, frames, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Create train/validation/test splits"""
        print(f"Creating train/val/test splits ({train_ratio*100:.0f}/{val_ratio*100:.0f}/{test_ratio*100:.0f})...")
        
        total_frames = len(frames)
        train_end = int(total_frames * train_ratio)
        val_end = train_end + int(total_frames * val_ratio)
        
        train_indices = list(range(1, train_end + 1))
        val_indices = list(range(train_end + 1, val_end + 1))
        test_indices = list(range(val_end + 1, total_frames + 1))
        
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        print(f"Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")
        
        return splits

    def save_yolo_annotations(self, splits):
        """Organize YOLO annotations and images for train/val/test splits"""
        print("Organizing dataset splits...")
        
        # Create subdirectories for each split
        for split_name in ['train', 'val', 'test']:
            split_images_dir = self.output_dir / split_name / "images"
            split_labels_dir = self.output_dir / split_name / "labels"
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Move files to appropriate split directories
        for split_name, image_ids in splits.items():
            split_images_dir = self.output_dir / split_name / "images"
            split_labels_dir = self.output_dir / split_name / "labels"
            
            split_images = [img for img in self.yolo_data["images"] if img["id"] in image_ids]
            moved_count = 0
            
            for img_info in split_images:
                if 'is_slice' in img_info and img_info['is_slice']:
                    src_image_path = self.sahi_images_dir / img_info["file_name"]
                    src_label_path = self.sahi_labels_dir / img_info["file_name"].replace('.jpg', '.txt')
                else:
                    src_image_path = Path(img_info["path"])
                    src_label_path = self.labels_dir / img_info["file_name"].replace('.jpg', '.txt')
                
                dst_image_path = split_images_dir / img_info["file_name"]
                dst_label_path = split_labels_dir / img_info["file_name"].replace('.jpg', '.txt')
                
                if src_image_path.exists():
                    shutil.copy2(src_image_path, dst_image_path)
                    moved_count += 1
                
                if src_label_path.exists():
                    shutil.copy2(src_label_path, dst_label_path)
            
            print(f"   {split_name}: {moved_count} images")
        
        # Remove empty original directories if they exist
        try:
            if self.images_dir.exists() and not any(self.images_dir.iterdir()):
                self.images_dir.rmdir()
            if self.labels_dir.exists() and not any(self.labels_dir.iterdir()):
                self.labels_dir.rmdir()
        except:
            pass

    def create_yolo_dataset_yaml(self, splits):
        """Create YOLO dataset configuration YAML"""
        print("Creating YOLO dataset configuration YAML...")
        
        # Create YOLO format configuration
        yolo_config = {
            'train': str(self.output_dir / 'train' / 'images'),
            'val': str(self.output_dir / 'val' / 'images'),
            'test': str(self.output_dir / 'test' / 'images'),
            'nc': 2,  # number of classes (person and umbrella)
            'names': ['person', 'umbrella']  # class names
        }
        
        # Add SAHI configuration if enabled
        if self.enable_sahi:
            yolo_config.update({
                'sahi_enabled': True,
                'sahi_config': {
                    'slice_height': self.sahi_config.slice_height,
                    'slice_width': self.sahi_config.slice_width,
                    'overlap_height_ratio': self.sahi_config.overlap_height_ratio,
                    'overlap_width_ratio': self.sahi_config.overlap_width_ratio,
                    'min_area_ratio': self.sahi_config.min_area_ratio,
                    'min_bbox_area': self.sahi_config.min_bbox_area
                }
            })
        
        # Save YOLO dataset.yaml
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yolo_config, f, default_flow_style=False, indent=2)
        
        # Also save a metadata file with dataset information
        metadata_config = {
            'dataset_info': {
                'name': 'Person-Umbrella Detection Dataset',
                'description': 'Dataset filtered to only include Person and Umbrella classes using RT-DETR auto-annotation',
                'total_images': len(self.yolo_data['images']),
                'total_annotations': len(self.yolo_data['annotations']),
                'categories': ['person', 'umbrella'],
                'created_date': datetime.now().isoformat(),
                'sahi_enabled': self.enable_sahi,
                'filter_stats': {
                    'original_detections': self.metrics.get('total_original_detections', 0),
                    'filtered_out': self.metrics.get('filtered_count', 0),
                    'kept_annotations': len(self.yolo_data['annotations'])
                }
            },
            'splits': {
                'train': {
                    'images': len([img for img in self.yolo_data['images'] if img['id'] in splits['train']]),
                    'annotations': len([ann for ann in self.yolo_data['annotations'] if ann['image_id'] in splits['train']])
                },
                'val': {
                    'images': len([img for img in self.yolo_data['images'] if img['id'] in splits['val']]),
                    'annotations': len([ann for ann in self.yolo_data['annotations'] if ann['image_id'] in splits['val']])
                },
                'test': {
                    'images': len([img for img in self.yolo_data['images'] if img['id'] in splits['test']]),
                    'annotations': len([ann for ann in self.yolo_data['annotations'] if ann['image_id'] in splits['test']])
                }
            },
            'categories': {0: 'person', 1: 'umbrella'},
            'training_config': {
                'input_format': 'yolo',
                'target_format': 'yolo',
                'confidence_threshold': self.confidence_threshold,
                'source_type': 'images_folder',
                'sahi_enabled': self.enable_sahi,
                'filtered_classes': ['person', 'umbrella']
            }
        }
        
        # Add SAHI-specific metadata if enabled
        if self.enable_sahi:
            metadata_config['sahi_stats'] = self.metrics.get('sahi_stats', {})
            metadata_config['sahi_config'] = {
                'slice_height': self.sahi_config.slice_height,
                'slice_width': self.sahi_config.slice_width,
                'overlap_height_ratio': self.sahi_config.overlap_height_ratio,
                'overlap_width_ratio': self.sahi_config.overlap_width_ratio,
                'min_area_ratio': self.sahi_config.min_area_ratio,
                'min_bbox_area': self.sahi_config.min_bbox_area
            }
        
        # Save metadata
        metadata_path = self.output_dir / 'dataset_metadata.yaml'
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata_config, f, default_flow_style=False, indent=2)
        
        print(f"   YOLO dataset configuration saved to: {yaml_path}")
        print(f"   Dataset metadata saved to: {metadata_path}")
        print(f"   • Classes: {yolo_config['names']}")
        print(f"   • Number of classes: {yolo_config['nc']}")

    def generate_eda_plots(self):
        """Generate comprehensive EDA plots"""
        print("Generating Exploratory Data Analysis plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Category Distribution (Person vs Umbrella)
        if self.metrics['category_counts']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Bar chart
            categories = ['Person', 'Umbrella']
            counts = [self.metrics['category_counts'].get(0, 0), self.metrics['category_counts'].get(1, 0)]
            colors = ['#2E86AB', '#A23B72']
            
            bars = ax1.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
            ax1.set_title('Annotation Distribution: Person vs Umbrella', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Number of Annotations')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')
            
            # Pie chart
            if sum(counts) > 0:
                ax2.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, colors=colors,
                       textprops={'fontsize': 12, 'fontweight': 'bold'})
                ax2.set_title('Category Distribution (Percentage)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'category_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Bounding Box Analysis
        if self.metrics['bbox_sizes']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Bbox size distribution
            ax1.hist(self.metrics['bbox_sizes'], bins=50, alpha=0.7, edgecolor='black', color='#2E86AB')
            ax1.set_title('Bounding Box Area Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Area (pixels²)')
            ax1.set_ylabel('Frequency')
            ax1.set_xscale('log')
            
            # Confidence score distribution
            ax2.hist(self.metrics['confidence_scores'], bins=50, alpha=0.7, edgecolor='black', color='#A23B72')
            ax2.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'bbox_confidence_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Detections per Frame
        if self.metrics['detection_stats']['detections_per_frame']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            detections_per_frame = self.metrics['detection_stats']['detections_per_frame']
            
            # Time series of detections
            ax1.plot(detections_per_frame, alpha=0.7, color='#2E86AB', linewidth=1)
            ax1.set_title('Detections per Frame (Time Series)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Frame Index')
            ax1.set_ylabel('Number of Detections')
            ax1.grid(True, alpha=0.3)
            
            # Distribution of detections per frame
            ax2.hist(detections_per_frame, bins=30, alpha=0.7, edgecolor='black', color='#A23B72')
            ax2.set_title('Detections per Frame Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Number of Detections')
            ax2.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'detections_per_frame.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("   EDA plots saved to analysis_plots/ directory")
        
        # SAHI-specific EDA plots if enabled
        if self.enable_sahi and 'sahi_stats' in self.metrics:
            self.generate_sahi_eda_plots()

    def generate_sahi_eda_plots(self):
        """Generate SAHI-specific EDA plots"""
        print("Generating SAHI-specific EDA plots...")
        
        sahi_stats = self.metrics['sahi_stats']
        
        # SAHI Performance Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Slices per Image Distribution
        if 'slices_per_image' in sahi_stats:
            ax1.hist(sahi_stats['slices_per_image'], bins=20, alpha=0.7, edgecolor='black', color='purple')
            ax1.set_title('SAHI Slices per Image Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Number of Slices')
            ax1.set_ylabel('Frequency')
            ax1.axvline(np.mean(sahi_stats['slices_per_image']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(sahi_stats["slices_per_image"]):.1f}')
            ax1.legend()
        
        # 2. Small Object Enhancement
        if 'small_objects_enhanced' in sahi_stats and 'total_objects' in sahi_stats:
            small_ratio = sahi_stats['small_objects_enhanced'] / max(sahi_stats['total_objects'], 1) * 100
            categories = ['Small Objects', 'Regular Objects']
            counts = [sahi_stats['small_objects_enhanced'], 
                     sahi_stats['total_objects'] - sahi_stats['small_objects_enhanced']]
            
            ax2.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, 
                   colors=['lightcoral', 'lightblue'])
            ax2.set_title(f'Small vs Regular Objects\n(Small: {small_ratio:.1f}%)', 
                         fontsize=14, fontweight='bold')
        
        # 3. SAHI Processing Time Analysis
        if 'sahi_time' in self.metrics:
            times = {
                'Loading': self.metrics.get('extraction_time', 0),
                'Annotation': self.metrics.get('annotation_time', 0),
                'SAHI': self.metrics.get('sahi_time', 0),
                'Other': self.metrics.get('total_time', 0) - self.metrics.get('extraction_time', 0) - 
                        self.metrics.get('annotation_time', 0) - self.metrics.get('sahi_time', 0)
            }
            
            ax3.bar(times.keys(), times.values(), color=['blue', 'green', 'purple', 'gray'])
            ax3.set_title('Processing Time Breakdown', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Time (seconds)')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. SAHI Enhancement Metrics
        if 'total_slices' in sahi_stats and 'slices_per_image' in sahi_stats:
            metrics_data = {
                'Total Images': len(self.yolo_data['images']) - sahi_stats.get('total_slices', 0),
                'SAHI Slices': sahi_stats.get('total_slices', 0),
                'Enhancement Factor': sahi_stats.get('total_slices', 0) / max(len(self.yolo_data['images']) - sahi_stats.get('total_slices', 0), 1)
            }
            
            # Create bar plot for enhancement
            ax4.bar(['Original Images', 'SAHI Slices'], 
                   [metrics_data['Total Images'], metrics_data['SAHI Slices']], 
                   color=['blue', 'purple'])
            ax4.set_title(f'Dataset Enhancement\nFactor: {metrics_data["Enhancement Factor"]:.2f}x', 
                         fontsize=14, fontweight='bold')
            ax4.set_ylabel('Number of Images')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'sahi_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   SAHI EDA plots saved to analysis_plots/sahi_analysis.png")

    def generate_comprehensive_report(self):
        """Generate detailed EDA report"""
        print("Generating comprehensive EDA report...")
        
        report = {
            "dataset_overview": {
                "total_images": len(self.yolo_data['images']),
                "total_annotations": len(self.yolo_data['annotations']),
                "unique_categories": len(self.metrics['category_counts']),
                "avg_annotations_per_image": len(self.yolo_data['annotations']) / len(self.yolo_data['images']) if self.yolo_data['images'] else 0,
                "source_type": "images_folder",
                "confidence_threshold": self.confidence_threshold,
                "filtered_classes": ["person", "umbrella"],
                "filter_stats": {
                    "original_detections": self.metrics.get('total_original_detections', 0),
                    "filtered_out": self.metrics.get('filtered_count', 0),
                    "kept_annotations": len(self.yolo_data['annotations'])
                }
            },
            "performance_metrics": {
                "loading_time_seconds": self.metrics.get('extraction_time', 0),
                "annotation_time_seconds": self.metrics.get('annotation_time', 0),
                "sahi_time_seconds": self.metrics.get('sahi_time', 0),
                "total_processing_time_seconds": self.metrics.get('extraction_time', 0) + self.metrics.get('annotation_time', 0) + self.metrics.get('sahi_time', 0),
                "loading_speed_images_per_second": len(self.yolo_data['images']) / self.metrics.get('extraction_time', 1) if self.metrics.get('extraction_time', 0) > 0 else 0,
                "annotation_speed_fps": len(self.yolo_data['images']) / self.metrics.get('annotation_time', 1) if self.metrics.get('annotation_time', 0) > 0 else 0,
                "sahi_speed_slices_per_second": self.metrics.get('sahi_stats', {}).get('total_slices', 0) / self.metrics.get('sahi_time', 1) if self.metrics.get('sahi_time', 0) > 0 else 0
            },
            "detection_statistics": {
                "avg_detections_per_frame": float(np.mean(self.metrics['detection_stats']['detections_per_frame'])) if self.metrics['detection_stats']['detections_per_frame'] else 0,
                "max_detections_per_frame": int(np.max(self.metrics['detection_stats']['detections_per_frame'])) if self.metrics['detection_stats']['detections_per_frame'] else 0,
                "min_detections_per_frame": int(np.min(self.metrics['detection_stats']['detections_per_frame'])) if self.metrics['detection_stats']['detections_per_frame'] else 0,
                "std_detections_per_frame": float(np.std(self.metrics['detection_stats']['detections_per_frame'])) if self.metrics['detection_stats']['detections_per_frame'] else 0,
                "avg_confidence_score": float(np.mean(self.metrics['confidence_scores'])) if self.metrics['confidence_scores'] else 0,
                "avg_bbox_area": float(np.mean(self.metrics['bbox_sizes'])) if self.metrics['bbox_sizes'] else 0,
                "median_bbox_area": float(np.median(self.metrics['bbox_sizes'])) if self.metrics['bbox_sizes'] else 0
            },
            "category_analysis": {
                "annotations_by_class": {
                    "person": self.metrics['category_counts'].get(0, 0),
                    "umbrella": self.metrics['category_counts'].get(1, 0)
                },
                "percentage_by_class": {
                    "person": (self.metrics['category_counts'].get(0, 0) / len(self.yolo_data['annotations']) * 100) if len(self.yolo_data['annotations']) > 0 else 0,
                    "umbrella": (self.metrics['category_counts'].get(1, 0) / len(self.yolo_data['annotations']) * 100) if len(self.yolo_data['annotations']) > 0 else 0
                }
            },
            "resource_utilization": {
                "system_info": {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                },
                "processing_efficiency": {
                    "frames_per_second_overall": len(self.yolo_data['images']) / (self.metrics.get('extraction_time', 0) + self.metrics.get('annotation_time', 0) + self.metrics.get('sahi_time', 0)) if (self.metrics.get('extraction_time', 0) + self.metrics.get('annotation_time', 0) + self.metrics.get('sahi_time', 0)) > 0 else 0,
                    "memory_efficiency_mb_per_frame": (psutil.virtual_memory().used / (1024**2)) / len(self.yolo_data['images']) if self.yolo_data['images'] else 0
                }
            }
        }
        
        # Add SAHI-specific metrics if enabled
        if self.enable_sahi and 'sahi_stats' in self.metrics:
            report["sahi_analysis"] = {
                "sahi_enabled": True,
                "sahi_config": {
                    "slice_height": self.sahi_config.slice_height,
                    "slice_width": self.sahi_config.slice_width,
                    "overlap_height_ratio": self.sahi_config.overlap_height_ratio,
                    "overlap_width_ratio": self.sahi_config.overlap_width_ratio,
                    "min_area_ratio": self.sahi_config.min_area_ratio,
                    "min_bbox_area": self.sahi_config.min_bbox_area
                },
                "sahi_statistics": {
                    "total_slices_created": self.metrics['sahi_stats'].get('total_slices', 0),
                    "avg_slices_per_image": self.metrics['sahi_stats'].get('avg_slices_per_image', 0),
                    "small_objects_enhanced": self.metrics['sahi_stats'].get('small_objects_enhanced', 0),
                    "total_objects_processed": self.metrics['sahi_stats'].get('total_objects', 0),
                    "small_object_ratio": self.metrics['sahi_stats'].get('small_object_ratio', 0),
                    "dataset_enhancement_factor": self.metrics['sahi_stats'].get('total_slices', 0) / max(len(self.yolo_data['images']) - self.metrics['sahi_stats'].get('total_slices', 0), 1)
                }
            }
        
        # Save report as JSON
        with open(self.output_dir / 'eda_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create human-readable summary
        total_annotations = len(self.yolo_data['annotations'])
        person_count = self.metrics['category_counts'].get(0, 0)
        umbrella_count = self.metrics['category_counts'].get(1, 0)
        filtered_out = self.metrics.get('filtered_count', 0)
        original_total = self.metrics.get('total_original_detections', 0)
        
        summary = f"""
#  Person & Umbrella Detection Dataset - EDA Report

##  Dataset Overview
- **Total Images**: {report['dataset_overview']['total_images']:,}
- **Total Annotations**: {total_annotations:,}
- **Classes**: Person, Umbrella
- **Avg Annotations per Image**: {report['dataset_overview']['avg_annotations_per_image']:.2f}
- **Confidence Threshold**: {report['dataset_overview']['confidence_threshold']}

##  Filtering Results
- **Original Detections**: {original_total:,}
- **Filtered Out (other classes)**: {filtered_out:,} ({filtered_out/original_total*100:.1f}% removed)
- **Kept (Person + Umbrella)**: {total_annotations:,} ({total_annotations/original_total*100:.1f}% kept)

##  Annotations by Class
| Class | Count | Percentage |
|-------|-------|------------|
|  Person | {person_count:,} | {person_count/total_annotations*100:.1f}% |
|  Umbrella | {umbrella_count:,} | {umbrella_count/total_annotations*100:.1f}% |
| **Total** | **{total_annotations:,}** | **100%** |

##  Performance Metrics
- **Loading Time**: {report['performance_metrics']['loading_time_seconds']:.2f} seconds
- **Annotation Time**: {report['performance_metrics']['annotation_time_seconds']:.2f} seconds
- **SAHI Processing Time**: {report['performance_metrics']['sahi_time_seconds']:.2f} seconds
- **Total Processing Time**: {report['performance_metrics']['total_processing_time_seconds']:.2f} seconds
- **Annotation Speed**: {report['performance_metrics']['annotation_speed_fps']:.2f} FPS

##  Detection Statistics
- **Avg Detections per Frame**: {report['detection_statistics']['avg_detections_per_frame']:.2f}
- **Max Detections per Frame**: {report['detection_statistics']['max_detections_per_frame']}
- **Min Detections per Frame**: {report['detection_statistics']['min_detections_per_frame']}
- **Avg Confidence Score**: {report['detection_statistics']['avg_confidence_score']:.3f}
- **Avg Bbox Area**: {report['detection_statistics']['avg_bbox_area']:.0f} pixels²"""

        # Add SAHI section if enabled
        if self.enable_sahi and 'sahi_analysis' in report:
            summary += f"""

##  SAHI Analysis
- **SAHI Enabled**: True
- **Total Slices Created**: {report['sahi_analysis']['sahi_statistics']['total_slices_created']:,}
- **Avg Slices per Image**: {report['sahi_analysis']['sahi_statistics']['avg_slices_per_image']:.1f}
- **Dataset Enhancement Factor**: {report['sahi_analysis']['sahi_statistics']['dataset_enhancement_factor']:.2f}x
- **Slice Configuration**: {report['sahi_analysis']['sahi_config']['slice_width']}x{report['sahi_analysis']['sahi_config']['slice_height']} with {report['sahi_analysis']['sahi_config']['overlap_width_ratio']*100:.0f}% overlap"""

        summary += f"""

##  System Information
- **CPU Cores**: {report['resource_utilization']['system_info']['cpu_count']}
- **Total Memory**: {report['resource_utilization']['system_info']['memory_total_gb']:.1f} GB
- **Python Version**: {report['resource_utilization']['system_info']['python_version']}

##  Processing Efficiency
- **Overall FPS**: {report['resource_utilization']['processing_efficiency']['frames_per_second_overall']:.2f}
- **Memory per Frame**: {report['resource_utilization']['processing_efficiency']['memory_efficiency_mb_per_frame']:.2f} MB

---
*Report generated on: {datetime.now().isoformat()}*
*Dataset filtered for: Person (class 0) and Umbrella (class 25 only)*
"""
        
        with open(self.output_dir / 'eda_summary.md', 'w') as f:
            f.write(summary)
        
        print("   EDA report saved as JSON and Markdown")

    def create(self):
        """Create dataset with SAHI enhancement"""
        print("=" * 60)
        print(" Creating Person & Umbrella Detection Dataset")
        print("=" * 60)
        total_start = time.time()
        
        # Step 1: Rename images sequentially
        total_images = self.rename_images()
        
        # Step 2: Load renamed images
        frames = self.load_images()
        
        # Step 3: Shuffle frames
        shuffled_frames = self.shuffle_frames(frames)
        
        # Step 4: Annotate frames (with filtering for person/umbrella)
        detected_categories = self.annotate_frames(shuffled_frames)
        
        # Step 5: Apply SAHI slicing for small object enhancement (if enabled)
        if self.enable_sahi:
            sliced_frames = self.apply_sahi(shuffled_frames)
            # Combine original and sliced frames for training
            enhanced_frames = shuffled_frames + sliced_frames
            print(f"\nEnhanced dataset: {len(shuffled_frames)} original + {len(sliced_frames)} SAHI slices = {len(enhanced_frames)} total images")
        else:
            enhanced_frames = shuffled_frames
        
        # Step 6: Create train/val/test splits
        splits = self.create_train_val_test_split(enhanced_frames)
        
        # Step 7: Save YOLO annotations and organize splits
        self.save_yolo_annotations(splits)
        
        # Step 8: Create YOLO dataset YAML
        self.create_yolo_dataset_yaml(splits)
        
        # Step 9: Generate EDA plots
        self.generate_eda_plots()
        
        # Step 10: Generate comprehensive report
        self.generate_comprehensive_report()
        
        self.metrics['total_time'] = time.time() - total_start
        
        print("\n" + "=" * 60)
        print(" Dataset created successfully!")
        print("=" * 60)
        print(f" Output: {self.output_dir}")
        print(f" Time: {self.metrics['total_time']:.2f}s")
        print(f" Images: {len(self.yolo_data['images']):,}")
        print(f" Annotations: {len(self.yolo_data['annotations']):,}")
        print(f" Classes: Person ({self.metrics['category_counts'].get(0, 0)}), Umbrella ({self.metrics['category_counts'].get(1, 0)})")
        print(f" EDA Plots: analysis_plots/")
        print(f" Report: eda_summary.md")
        print("=" * 60)
        
        return self.output_dir

if __name__ == "__main__":
    # Configuration
    img_dir = "video_images"
    model_path = "rtdetr-l.pt"
    out_dir = "prepared_dataset"
    conf_thresh = 0.25
    
    # SAHI Configuration
    enable_sahi = True
    sahi_cfg = SAHIConfig(
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        min_area_ratio=0.1,
        min_bbox_area=32*32,
        verbose=True
    )
    
    # Create dataset
    creator = DatasetCreator(
        images_folder=img_dir,
        model_path=model_path,
        output_dir=out_dir,
        confidence_threshold=conf_thresh,
        enable_sahi=enable_sahi,
        sahi_config=sahi_cfg
    )
    
    dataset_path = creator.create()
    print(f"\n🎉 Dataset ready: {dataset_path}")
    print(f"📊 Visualizations: analysis_plots/ | Report: eda_summary.md")