import os
import subprocess

def extract_frames_to_single_folder(input_dir, output_dir, fps=5):
    os.makedirs(output_dir, exist_ok=True)

    video_extensions = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv")

    for file in os.listdir(input_dir):
        if file.lower().endswith(video_extensions):
            video_path = os.path.join(input_dir, file)

            # Get filename without extension
            name = os.path.splitext(file)[0]

            print(f"Processing: {file}")

            # Use video name as prefix to avoid overwriting frames
            output_pattern = os.path.join(output_dir, f"{name}_frame_%05d.png")

            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vf", f"fps={fps}",
                output_pattern
            ]

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            print(f"Done: {file}")

    print("All videos processed.")


if __name__ == "__main__":
    # 🔥 CHANGE THESE PATHS
    INPUT_DIR = "videos"
    OUTPUT_DIR = "video_images"

    extract_frames_to_single_folder(INPUT_DIR, OUTPUT_DIR, fps=15)