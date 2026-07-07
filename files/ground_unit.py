#!/usr/bin/env python3
"""
Ground Unit — receives the raw monitor-mode video link from air_unit.py
=========================================================================
Pipeline:
    [rx: raw 802.11 capture] -> [ffmpeg: H.264 -> rawvideo] -> [this script: cv2 display]

This is the counterpart to air_unit.py. It chains two subprocesses exactly
like the air side did, just in reverse:
    rx.stdout   -> ffmpeg.stdin   (bare H.264 elementary stream in)
    ffmpeg.stdout -> this script  (raw BGR frames out, read with cv2/numpy)

Usage:
    python3 ground_unit.py --rx_path ./rx --width 640 --height 344
    python3 ground_unit.py --rx_path ./rx --width 640 --height 344 --record out.mp4
    python3 ground_unit.py --rx_path ./rx --width 640 --height 344 --no-display
"""

import subprocess
import numpy as np
import cv2
import time
import sys
import signal
import argparse
from pathlib import Path


class GroundReceiver:
    def __init__(self, rx_path="./rx", width=640, height=344, fps=30, record=None):
        self.rx_path = rx_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_bytes = width * height * 3  # bgr24
        self.record = record

        self.rx_proc = None
        self.ffmpeg_proc = None
        self.writer = None

        self.frame_count = 0
        self.start_time = time.time()
        self.last_stats_time = time.time()
        self.last_frame_time = time.time()

    def start(self):
        # rx: sniffs the monitor-mode interface, strips 802.11/radiotap
        # headers, writes the raw payload bytes (the H.264 stream the air
        # unit injected) straight to stdout.
        self.rx_proc = subprocess.Popen(
            [self.rx_path], stdout=subprocess.PIPE, stderr=sys.stderr
        )

        # ffmpeg: takes that bare H.264 stream on stdin, decodes it, and
        # writes raw BGR frames to stdout -- same pixel format the air
        # unit's ffmpeg read frames in as, so no conversion needed here.
        ffmpeg_cmd = [
            'ffmpeg', '-loglevel', 'error',
            '-fflags', 'nobuffer', '-flags', 'low_delay',
            '-probesize', '32', '-analyzeduration', '0',
            '-f', 'h264', '-i', '-',
            '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-'
        ]
        self.ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=self.rx_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self.rx_proc.stdout.close()  # let rx get SIGPIPE if ffmpeg dies

        if self.record:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.record, fourcc, self.fps, (self.width, self.height))
            print(f"[Ground] Recording to {self.record}")

        print(f"[Ground] rx -> ffmpeg -> display  ({self.width}x{self.height})")

    def _read_exact(self, n):
        """Read exactly n bytes from ffmpeg's stdout, or None on EOF."""
        buf = bytearray()
        while len(buf) < n:
            chunk = self.ffmpeg_proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def read_frame(self):
        raw = self._read_exact(self.frame_bytes)
        if raw is None:
            return None
        now = time.time()
        gap = now - self.last_frame_time
        self.last_frame_time = now
        self.frame_count += 1

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))

        if time.time() - self.last_stats_time > 5:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"[STATS] Frames: {self.frame_count}, FPS: {fps:.1f}, last frame gap: {gap*1000:.0f}ms")
            self.last_stats_time = time.time()

        return frame

    def overlay(self, frame):
        display = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(display, f"GROUND | Frames: {self.frame_count} | FPS: {fps:.1f}",
                    (10, 30), font, 0.5, (0, 255, 255), 1)
        return display

    def write_record(self, frame):
        if self.writer:
            self.writer.write(frame)

    def stop(self):
        if self.writer:
            self.writer.release()
        for proc in (self.ffmpeg_proc, self.rx_proc):
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()


def main():
    parser = argparse.ArgumentParser(description='Ground Unit -- decode + display the raw monitor-mode video link')
    parser.add_argument('--rx_path', type=str, default='./rx', help='Path to the compiled rx raw-receiver binary')
    parser.add_argument('--width', type=int, default=640, help="Must match --stream_width used on the air unit")
    parser.add_argument('--height', type=int, default=344, help="Must match the air unit's computed stream height")
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--record', type=str, default=None, help='Optional path to save an .mp4 recording of the received feed')
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()

    if not Path(args.rx_path).exists():
        print(f"ERROR: rx binary not found at {args.rx_path} -- build it first: g++ -O2 -o rx rx.cpp")
        sys.exit(1)

    receiver = GroundReceiver(rx_path=args.rx_path, width=args.width, height=args.height,
                               fps=args.fps, record=args.record)
    receiver.start()

    def handler(sig, frame):
        receiver.stop()
        cv2.destroyAllWindows()
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)

    try:
        while True:
            frame = receiver.read_frame()
            if frame is None:
                print("[Ground] Stream ended (ffmpeg/rx exited)")
                break

            display = receiver.overlay(frame)
            receiver.write_record(frame)

            if not args.no_display:
                cv2.imshow('Ground Unit', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        receiver.stop()
        cv2.destroyAllWindows()
        print(f"\n[Ground] Total frames received: {receiver.frame_count}")


if __name__ == "__main__":
    main()
