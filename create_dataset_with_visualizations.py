"""
Enhanced COCO Dataset Creator with Integrated Visualizations
Processes all *.mp4 files from a videos/ folder instead of a single video source.
"""

import os
import json
import argparse
import subprocess
import sys
from pathlib import Path


def get_video_files(videos_dir):
    """Return sorted list of .mp4 paths from the given directory. Exits if none found."""
    videos_dir = Path(videos_dir)
    if not videos_dir.exists():
        print(f"Videos directory not found: {videos_dir}")
        sys.exit(1)

    mp4_files = sorted(videos_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"No .mp4 files found in: {videos_dir}")
        sys.exit(1)

    return mp4_files


def run_dataset_creation(video_path, model_path, output_dir, fps=15, confidence_threshold=0.5):
    """Run dataset creation for a single video. Returns the output dataset path."""
    from create_coco_dataset_professional import COCODatasetCreator

    creator = COCODatasetCreator(
        video_path=video_path,
        model_path=model_path,
        output_dir=output_dir,
        fps=fps,
        confidence_threshold=confidence_threshold
    )

    dataset_path = creator.create_dataset()
    return dataset_path


def run_dataset_creation_from_folder(videos_dir, model_path, output_dir, fps=15, confidence_threshold=0.5):
    """
    Iterate over all *.mp4 files in videos_dir and run dataset creation for each.
    Each video writes into its own subdirectory under output_dir named after the video stem.
    Returns a list of (video_path, dataset_path) tuples.
    """
    video_files = get_video_files(videos_dir)
    print(f"Found {len(video_files)} video(s) in {videos_dir}")

    results = []
    for video_path in video_files:
        # Per-video output subdir keeps datasets isolated from each other
        video_output_dir = Path(output_dir) / video_path.stem
        print(f"\n-- Processing: {video_path.name} -> {video_output_dir}")

        dataset_path = run_dataset_creation(
            video_path=str(video_path),
            model_path=model_path,
            output_dir=str(video_output_dir),
            fps=fps,
            confidence_threshold=confidence_threshold
        )
        print(f"   Dataset ready: {dataset_path}")
        results.append((str(video_path), dataset_path))

    return results


def interactive_mode():
    """Prompt-driven mode letting the user pick what to run."""
    print("Interactive Dataset Creation Pipeline")
    print("=" * 50)

    videos_dir = input("Videos folder (default: videos): ").strip() or "videos"
    model_path = input("Model path (default: rf-detr-medium.pth): ").strip() or "rf-detr-medium.pth"
    output_dir = input("Output directory (default: prepared_dataset): ").strip() or "prepared_dataset"
    fps = int(input("FPS for frame extraction (default: 15): ").strip() or "15")
    confidence = float(input("Confidence threshold (default: 0.5): ").strip() or "0.5")

    run_dataset_creation_from_folder(videos_dir, model_path, output_dir, fps, confidence)


def main():
    parser = argparse.ArgumentParser(description="Enhanced COCO Dataset Creator with Visualizations")
    parser.add_argument("--videos_dir", type=str, default="videos",
                       help="Folder containing *.mp4 files to process")
    parser.add_argument("--model_path", type=str, default="rf-detr-medium.pth",
                       help="Path to RF-DETR model")
    parser.add_argument("--output_dir", type=str, default="prepared_dataset",
                       help="Root output directory; each video gets its own subdirectory here")
    parser.add_argument("--fps", type=int, default=15,
                       help="FPS for frame extraction")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold for detections")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--skip_dataset", action="store_true",
                       help="Skip dataset creation, use existing datasets under output_dir")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    print("Enhanced Dataset Creation Pipeline")
    print("=" * 50)

    dataset_results = []

    # Step 1: build datasets from every .mp4 in the videos folder, or reuse existing ones
    if not args.skip_dataset:
        dataset_results = run_dataset_creation_from_folder(
            args.videos_dir, args.model_path, args.output_dir,
            args.fps, args.confidence
        )
    else:
        # Collect the per-video subdirs that were created by a prior run
        output_root = Path(args.output_dir)
        if not output_root.exists():
            print(f"Dataset directory not found: {output_root}")
            return

        subdirs = [p for p in sorted(output_root.iterdir()) if p.is_dir()]
        if not subdirs:
            print(f"No subdirectories found under: {output_root}")
            return

        dataset_results = [("(existing)", str(d)) for d in subdirs]
        print(f"Using {len(dataset_results)} existing dataset(s) under {output_root}")

    print("\nPipeline completed.")
    print(f"Datasets: {args.output_dir}/")


if __name__ == "__main__":
    main()