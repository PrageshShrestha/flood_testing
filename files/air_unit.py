#!/usr/bin/env python3
"""
Air Unit — SAR Streamer over raw monitor-mode Wi-Fi
=====================================================
Same detection/telemetry architecture as the RTSP version (throttled
detection, decoupled background telemetry, non-blocking uplink), but the
OUTPUT SINK is different: instead of pushing to an RTSP server over normal
IP networking, this pipes H.264 bytes straight into the `tx` raw-socket
injector, which puts them on the air in monitor mode (WFB-ng style).

Pipeline:
    [cv2 capture + overlay/detection] -> [ffmpeg: rawvideo -> H.264] -> [tx: raw 802.11 injection]

Two subprocesses are chained:
    ffmpeg.stdin  <- frame bytes written by this script (same as before)
    ffmpeg.stdout -> tx.stdin (raw H.264 elementary stream)

Usage (unchanged flags from the RTSP version, minus --rtsp_url):
    python3 air_unit.py --input test2.mp4
    python3 air_unit.py --input test2.mp4 --detect --no-display
    python3 air_unit.py --input test2.mp4 --detect --telemetry_url http://localhost:8765
    python3 air_unit.py --iface wlan0mon --bitrate 2000
"""

import subprocess
import cv2
import time
import sys
import signal
import argparse
import json
import io
import threading
import queue
import urllib.request
import uuid
from pathlib import Path
from datetime import datetime

try:
    import serial
    import pynmea2
    _HAS_GPS_LIBS = True
except ImportError:
    _HAS_GPS_LIBS = False


# ============================================================
# TELEMETRY READER — unchanged: background thread, own timer,
# video path only ever reads a cached dict.
# ============================================================

class TelemetryReader:
    def __init__(self, interval=1.0, gps_port=None, gps_baud=9600):
        self.interval = interval
        self.lock = threading.Lock()
        self.latest = {
            'timestamp': time.time(),
            'gps_lat': 'N/A', 'gps_lng': 'N/A', 'gps_alt': 'N/A',
            'speed_kmh': 'N/A', 'heading_deg': 'N/A', 'satellites': 'N/A',
            'gps_fix': False, 'cpu_temp': 'N/A', 'battery': 'N/A',
        }
        self.running = False
        self.thread = None
        self._gps_serial = None
        self._override_file = Path("/tmp/telemetry_override.json")
        self._init_gps(gps_port, gps_baud)

    def _init_gps(self, port, baud):
        if not _HAS_GPS_LIBS or not port:
            return
        try:
            self._gps_serial = serial.Serial(port, baud, timeout=0.3)
        except Exception as e:
            print(f"[Telemetry] Could not open GPS port {port}: {e} -- GPS will read N/A")

    def _read_gps(self):
        out = {}
        if self._gps_serial is None:
            return out
        try:
            for _ in range(5):
                line = self._gps_serial.readline().decode("ascii", errors="ignore").strip()
                if not line:
                    break
                msg = pynmea2.parse(line)
                if isinstance(msg, pynmea2.types.talker.GGA):
                    if msg.latitude and msg.longitude:
                        out['gps_lat'] = msg.latitude
                        out['gps_lng'] = msg.longitude
                        out['gps_fix'] = int(msg.gps_qual) > 0
                    if msg.altitude is not None:
                        out['gps_alt'] = float(msg.altitude)
                    if msg.num_sats:
                        out['satellites'] = int(msg.num_sats)
                elif isinstance(msg, pynmea2.types.talker.RMC):
                    if msg.spd_over_grnd is not None:
                        out['speed_kmh'] = float(msg.spd_over_grnd) * 1.852
                    if msg.true_course is not None:
                        out['heading_deg'] = float(msg.true_course)
        except Exception:
            pass
        return out

    def _read_cpu_temp(self):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return round(int(f.read().strip()) / 1000.0, 1)
        except Exception:
            return 'N/A'

    def _read_override(self):
        if not self._override_file.exists():
            return {}
        try:
            with open(self._override_file) as f:
                data = json.load(f)
            mapped = {}
            if 'lat' in data: mapped['gps_lat'] = data['lat']
            if 'lon' in data: mapped['gps_lng'] = data['lon']
            if 'alt_m' in data: mapped['gps_alt'] = data['alt_m']
            if 'battery_pct' in data: mapped['battery'] = data['battery_pct']
            return mapped
        except Exception:
            return {}

    def _update_once(self):
        reading = {'timestamp': time.time()}
        reading.update(self._read_gps())
        reading['cpu_temp'] = self._read_cpu_temp()
        reading.update(self._read_override())
        with self.lock:
            self.latest.update(reading)

    def _loop(self):
        while self.running:
            self._update_once()
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def get_all(self):
        with self.lock:
            return dict(self.latest)


# ============================================================
# UPLINK — for telemetry/detection snapshots ONLY. This still
# goes over normal IP (e.g. a phone hotspot or second radio),
# NOT over the monitor-mode video link. The video link is
# one-way broadcast injection, same as WFB-ng's video channel.
# ============================================================

class Uplink:
    def __init__(self, base_url, maxsize=50):
        self.base_url = base_url.rstrip('/') if base_url else None
        self.q = queue.Queue(maxsize=maxsize)
        self.running = False
        self.thread = None
        self.dropped = 0

    def start(self):
        if not self.base_url:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def post_detection(self, jpeg_bytes, confidence, label='person'):
        if not self.base_url:
            return
        try:
            self.q.put_nowait(('detection', jpeg_bytes, confidence, label))
        except queue.Full:
            self.dropped += 1

    def _loop(self):
        while self.running:
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                _, jpeg_bytes, confidence, label = item
                self._post_multipart(jpeg_bytes, confidence, label)
            except Exception as e:
                print(f"[Uplink] Send failed (non-fatal): {e}")

    def _post_multipart(self, jpeg_bytes, confidence, label):
        boundary = uuid.uuid4().hex
        body = io.BytesIO()
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="confidence"\r\n\r\n{confidence or ""}\r\n'.encode())
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="label"\r\n\r\n{label}\r\n'.encode())
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="det.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode())
        body.write(jpeg_bytes)
        body.write(f'\r\n--{boundary}--\r\n'.encode())
        req = urllib.request.Request(
            f"{self.base_url}/detection", data=body.getvalue(),
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}, method='POST',
        )
        urllib.request.urlopen(req, timeout=3.0)


# ============================================================
# HUMAN DETECTOR — unchanged: throttled, resized, cached
# ============================================================

class HumanDetector:
    def __init__(self, detect_width=320):
        self.use_yolo = False
        self.model = None
        self.hog = None
        self.detection_count = 0
        self.detect_width = detect_width
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            self.use_yolo = True
        except Exception:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = self.detect_width / w if w > self.detect_width else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame
        detections = []
        if self.use_yolo and self.model:
            for r in self.model(small, verbose=False):
                if r.boxes is not None:
                    for box in r.boxes:
                        if box.cls == 0:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            detections.append({'bbox': [int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)],
                                                'confidence': conf, 'class': 'person'})
        elif self.hog:
            boxes, _ = self.hog.detectMultiScale(small, winStride=(8, 8))
            for (x, y, bw, bh) in boxes:
                detections.append({'bbox': [int(x/scale), int(y/scale), int((x+bw)/scale), int((y+bh)/scale)],
                                    'confidence': 0.5, 'class': 'person'})
        self.detection_count += len(detections)
        return detections


# ============================================================
# RAW-LINK STREAMER — the part that's different from the
# RTSP version: ffmpeg encodes to a bare H.264 stream on
# stdout, which is piped into `tx` (raw monitor-mode injector)
# instead of being handed to librtsp/RTSP muxing.
# ============================================================

class RawLinkStreamer:
    def __init__(self, tx_path="./tx", width=640, height=344, fps=30, bitrate=2000,
                 detect=False, save_detections=True, telemetry=True,
                 telemetry_interval=1.0, detect_every_n_frames=3, detect_width=320,
                 detection_cooldown=5.0, telemetry_url=None,
                 gps_port=None, gps_baud=9600):
        self.tx_path = tx_path
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.detect = detect
        self.save_detections = save_detections
        self.telemetry_enabled = telemetry

        self.ffmpeg_proc = None
        self.tx_proc = None

        self.telemetry = TelemetryReader(interval=telemetry_interval, gps_port=gps_port, gps_baud=gps_baud) if telemetry else None
        self.detector = HumanDetector(detect_width=detect_width) if detect else None
        self.uplink = Uplink(telemetry_url) if telemetry_url else None

        self.detect_every_n_frames = max(1, detect_every_n_frames)
        self.detection_cooldown = detection_cooldown
        self.last_detection_save_time = 0.0
        self.last_detections = []

        self.frame_count = 0
        self.last_stats_time = time.time()
        self.start_time = time.time()

        if detect and save_detections:
            self.detection_dir = Path(f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.detection_dir.mkdir(exist_ok=True)
            self.detection_log = self.detection_dir / "detections.jsonl"

    def start(self):
        # ffmpeg: raw BGR frames in on stdin -> bare H.264 elementary
        # stream out on stdout (no container, no RTSP -- tx.cpp expects
        # a raw byte stream it can chunk and inject as-is).
        ffmpeg_cmd = [
            'ffmpeg', '-loglevel', 'error', '-re',
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}', '-r', str(self.fps), '-i', '-',
            '-c:v', 'libx264', '-bf', '0', '-preset', 'ultrafast', '-tune', 'zerolatency',
            '-b:v', f'{self.bitrate}k',
            '-f', 'h264', '-'   # bare H.264 stream, not muxed into any container
        ]
        self.ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        # tx reads whatever bytes arrive on stdin and injects them as
        # raw 802.11 data frames -- feed it ffmpeg's H.264 stdout directly.
        self.tx_proc = subprocess.Popen(
            [self.tx_path], stdin=self.ffmpeg_proc.stdout, stderr=sys.stderr
        )
        # allow ffmpeg to receive SIGPIPE if tx dies, instead of hanging
        self.ffmpeg_proc.stdout.close()

        print(f"[RawLink] ffmpeg -> {self.tx_path} (monitor-mode injection)")
        print(f"[RawLink] {self.width}x{self.height} @ {self.fps}fps, {self.bitrate}kbps")
        print(f"[RawLink] Detection: {'every ' + str(self.detect_every_n_frames) + ' frames' if self.detect else 'disabled'}")
        print(f"[RawLink] Telemetry: {'background, ' + str(self.telemetry.interval) + 's interval' if self.telemetry_enabled else 'disabled'}")

        if self.telemetry:
            self.telemetry.start()
        if self.uplink:
            self.uplink.start()

    def _maybe_run_detection(self, frame):
        if self.frame_count % self.detect_every_n_frames == 0:
            self.last_detections = self.detector.detect(frame)
        return self.last_detections

    def _maybe_save_detection(self, frame, detections):
        if not detections or not self.save_detections:
            return
        now = time.time()
        if now - self.last_detection_save_time < self.detection_cooldown:
            return
        self.last_detection_save_time = now

        img = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imwrite(str(self.detection_dir / f"detection_{now:.0f}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        t = self.telemetry.get_all() if self.telemetry_enabled else {}
        with open(self.detection_log, 'a') as f:
            f.write(json.dumps({
                'timestamp': now, 'gps_lat': t.get('gps_lat', 'N/A'), 'gps_lng': t.get('gps_lng', 'N/A'),
                'gps_alt': t.get('gps_alt', 'N/A'), 'cpu_temp': t.get('cpu_temp', 'N/A'),
                'battery': t.get('battery', 'N/A'), 'detections': detections, 'frame_count': self.frame_count,
            }) + '\n')

        if self.uplink:
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                best_conf = max((d['confidence'] for d in detections), default=None)
                self.uplink.post_detection(buf.tobytes(), best_conf, detections[0]['class'])

    def process_frame(self, frame):
        self.frame_count += 1
        display = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.telemetry_enabled:
            t = self.telemetry.get_all()
            fmt = lambda v, s='': f"{v:.6f}{s}" if isinstance(v, float) else f"{v}{s}"
            cv2.putText(display, f"GPS: {fmt(t['gps_lat'])}, {fmt(t['gps_lng'])}", (10, 30), font, 0.5, (0, 255, 255), 1)
            cv2.putText(display, f"Alt: {fmt(t['gps_alt'],'m')} | Bat: {fmt(t['battery'],'%')} | CPU: {fmt(t['cpu_temp'],'C')}", (10, 55), font, 0.5, (0, 255, 255), 1)

        detections = []
        if self.detect and self.detector:
            detections = self._maybe_run_detection(frame)
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"PERSON {d['confidence']:.2f}", (x1, y1 - 10), font, 0.4, (0, 255, 0), 1)
            self._maybe_save_detection(frame, detections)

        if time.time() - self.last_stats_time > 5:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            msg = f"[STATS] Frames: {self.frame_count}, FPS: {fps:.1f}"
            if self.uplink and self.uplink.dropped:
                msg += f", Uplink dropped: {self.uplink.dropped}"
            print(msg)
            self.last_stats_time = time.time()

        return display

    def send_frame(self, frame):
        if frame is None or self.ffmpeg_proc is None:
            return False
        try:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self.ffmpeg_proc.stdin.write(frame.tobytes())
            return True
        except BrokenPipeError:
            print("[RawLink] Broken pipe -- ffmpeg or tx may have stopped")
            return False

    def stop(self):
        if self.telemetry:
            self.telemetry.stop()
        if self.uplink:
            self.uplink.stop()
        for proc in (self.ffmpeg_proc, self.tx_proc):
            if proc:
                try:
                    if proc.stdin:
                        proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()


def main():
    parser = argparse.ArgumentParser(description='Air Unit -- SAR streamer over raw monitor-mode Wi-Fi')
    parser.add_argument('--input', type=str, default='test2.mp4')
    parser.add_argument('--loop', action='store_true', default=True)
    parser.add_argument('--tx_path', type=str, default='./tx', help='Path to the compiled tx raw-injector binary')
    parser.add_argument('--bitrate', type=int, default=2000)
    parser.add_argument('--stream_width', type=int, default=640)
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--detect', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-telemetry', action='store_true')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--telemetry_interval', type=float, default=1.0)
    parser.add_argument('--detect_every_n_frames', type=int, default=3)
    parser.add_argument('--detect_width', type=int, default=320)
    parser.add_argument('--detection_cooldown', type=float, default=5.0)
    parser.add_argument('--telemetry_url', type=str, default=None)
    parser.add_argument('--gps_port', type=str, default=None)
    parser.add_argument('--gps_baud', type=int, default=9600)
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: Input not found: {args.input}")
        sys.exit(1)
    if not Path(args.tx_path).exists():
        print(f"ERROR: tx binary not found at {args.tx_path} -- build it first (see guide sec. 9)")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.input}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = args.fps or int(cap.get(cv2.CAP_PROP_FPS)) or 30

    stream_width = args.stream_width
    stream_height = int(height * (stream_width / width))
    stream_width -= stream_width % 2
    stream_height -= stream_height % 2

    streamer = RawLinkStreamer(
        tx_path=args.tx_path, width=stream_width, height=stream_height, fps=fps,
        bitrate=args.bitrate, detect=args.detect, save_detections=not args.no_save,
        telemetry=not args.no_telemetry, telemetry_interval=args.telemetry_interval,
        detect_every_n_frames=args.detect_every_n_frames, detect_width=args.detect_width,
        detection_cooldown=args.detection_cooldown, telemetry_url=args.telemetry_url,
        gps_port=args.gps_port, gps_baud=args.gps_baud,
    )
    streamer.start()

    def handler(sig, frame):
        streamer.stop()
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    streamer.frame_count = 0
                    streamer.start_time = time.time()
                    continue
                break
            display = streamer.process_frame(frame)
            streamer.send_frame(display)
            if not args.no_display:
                cv2.imshow('Air Unit Preview', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        streamer.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
