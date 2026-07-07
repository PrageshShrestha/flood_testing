# Raw Wi-Fi Monitor-Mode Video Link — Full Command Reference

A step-by-step reference for building a WFB-ng-style air-to-ground video link using
monitor mode + GStreamer + Python + C++ raw sockets. This is the manual/DIY approach
that gets you close to WFB-ng performance without the full WFB-ng stack.

---

## 0. Requirements

- Two Wi-Fi adapters that support monitor mode + packet injection (e.g. Atheros AR9271,
  Ralink RT3070, or most RTL8812AU-based cards with the right driver).
- One "Air" unit (drone/companion computer + camera) and one "Ground" unit (laptop/SBC).
- Same OS family (Linux) on both ends — Raspberry Pi OS, Ubuntu, etc.

Install dependencies on **both** units:

```bash
sudo apt update
sudo apt install -y python3-scapy gstreamer1.0-tools gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly aircrack-ng iw \
    build-essential libpcap-dev python3-pip
```

`aircrack-ng` gives you `airmon-ng`, which is the easiest way to manage monitor mode.

---

## 1. Starting Monitor Mode

### Option A — using `iw` (manual, works on most adapters)

```bash
# Bring interface down
sudo ip link set wlan0 down

# Switch to monitor mode
sudo iw wlan0 set monitor none

# Bring interface back up
sudo ip link set wlan0 up

# Lock both air and ground units to the SAME channel
sudo iw wlan0 set channel 149
```

Channel 149 is in the 5GHz band and commonly used for FPV because it's less congested.
Use `iw list` to see what channels/bands your adapter supports.

### Option B — using `airmon-ng` (kills interfering processes automatically)

```bash
# Check for and kill processes that might interfere (NetworkManager, wpa_supplicant)
sudo airmon-ng check kill

# Start monitor mode — this may create a new interface like wlan0mon
sudo airmon-ng start wlan0

# Set the channel
sudo iw wlan0mon set channel 149
```

### Verify monitor mode is active

```bash
iw dev wlan0 info
# Look for: type monitor
```

---

## 2. Stopping Monitor Mode / Reverting to Normal Wi-Fi

### If you used `iw`:

```bash
sudo ip link set wlan0 down
sudo iw wlan0 set type managed
sudo ip link set wlan0 up

# Restart network management so you get normal Wi-Fi back
sudo systemctl restart NetworkManager
```

### If you used `airmon-ng`:

```bash
sudo airmon-ng stop wlan0mon
sudo systemctl restart NetworkManager
```

---

## 3. The Air Unit — Transmitter Script (Python/Scapy version)

Save on the **air** unit as `tx.py`:

```python
import sys
from scapy.all import RadioTap, Dot11, sendp

# Configure your interface
interface = "wlan0"
# Dummy MAC addresses (monitor mode still requires valid frame structures)
src_mac = "00:11:22:33:44:55"
dst_mac = "66:77:88:99:aa:bb"

# Type 2, Subtype 0 = standard data frame
base_frame = RadioTap() / Dot11(type=2, subtype=0, addr1=dst_mac, addr2=src_mac, addr3=dst_mac)

print("Starting transmission stream... Press Ctrl+C to stop.")

try:
    while True:
        payload = sys.stdin.buffer.read(1024)
        if not payload:
            break
        packet = base_frame / payload
        sendp(packet, iface=interface, verbose=False)
except KeyboardInterrupt:
    print("\nTransmission stopped.")
```

### Run it — Raspberry Pi camera example:

```bash
gst-launch-1.0 libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 \
    ! videoconvert ! x264enc tune=zerolatency bitrate=1000 ! fdsink \
    | python3 tx.py
```

### Run it — USB/V4L2 camera example:

```bash
gst-launch-1.0 v4l2src device=/dev/video0 \
    ! video/x-raw,width=640,height=480,framerate=30/1 \
    ! videoconvert ! x264enc tune=zerolatency bitrate=1000 ! fdsink \
    | python3 tx.py
```

### Sanity-check with dummy data (no camera needed):

```bash
echo "Hello Ground" | python3 tx.py
```

---

## 4. The Ground Unit — Receiver Script (Python/Scapy version)

Save on the **ground** unit as `rx.py`:

```python
import sys
from scapy.all import sniff, Dot11

interface = "wlan0"
target_mac = "66:77:88:99:aa:bb"  # must match dst_mac in tx.py

def packet_handler(pkt):
    if pkt.haslayer(Dot11) and pkt.type == 2 and pkt.addr1 == target_mac:
        payload = bytes(pkt[Dot11].payload)
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()

print("Listening for raw frames...", file=sys.stderr)
sniff(iface=interface, prn=packet_handler, store=0)
```

### Run it and decode straight to a video window:

```bash
python3 rx.py | gst-launch-1.0 fdsrc ! h264parse ! avdec_h264 \
    ! videoconvert ! autovideosink
```

### Run it and also save to a file while displaying:

```bash
python3 rx.py | gst-launch-1.0 fdsrc ! tee name=t \
    t. ! queue ! h264parse ! avdec_h264 ! videoconvert ! autovideosink \
    t. ! queue ! filesink location=recording.h264
```

---

## 5. Full Startup Sequence (Both Ends) — Python Version

**On the Air unit:**

```bash
sudo airmon-ng check kill
sudo airmon-ng start wlan0
sudo iw wlan0mon set channel 149
gst-launch-1.0 libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 \
    ! videoconvert ! x264enc tune=zerolatency bitrate=1000 ! fdsink \
    | python3 tx.py
```

**On the Ground unit (in parallel):**

```bash
sudo airmon-ng check kill
sudo airmon-ng start wlan0
sudo iw wlan0mon set channel 149
python3 rx.py | gst-launch-1.0 fdsrc ! h264parse ! avdec_h264 \
    ! videoconvert ! autovideosink
```

---

## 6. Full Shutdown Sequence (Both Ends)

```bash
# Stop the pipeline: Ctrl+C in each terminal (tx.py / rx.py / gst-launch)

# Then revert the adapter
sudo airmon-ng stop wlan0mon
sudo systemctl restart NetworkManager
```

---

## 7. Debugging Tools

Confirm packets are actually being seen over the air:

```bash
# On the ground unit, before running rx.py, just watch raw traffic
sudo tcpdump -i wlan0 -e -s 0

# Or with tshark for more detail
sudo tshark -i wlan0
```

Check adapter capabilities (injection support, supported channels):

```bash
iw list
```

---

## 8. Why Python/Scapy Won't Match WFB-ng — and How to Close the Gap

| Feature | Python/Scapy | WFB-ng / OpenHD |
|---|---|---|
| FEC (Forward Error Correction) | None — video breaks on interference | Reconstructs lost packets |
| Encryption | None | AES-encrypted |
| Multi-card RX diversity | Not supported | Combines multiple antennas |
| Telemetry muxing (MAVLink) | Needs separate link | Muxed with video |
| Processing speed | Python/Scapy too slow for 30-60fps sustained | Optimized C |

To actually approach WFB-ng-level performance you'd need to:

1. **Rewrite `tx.py`/`rx.py` in C/C++** using raw sockets (`socket(AF_PACKET, SOCK_RAW)`)
   instead of Scapy — Python's per-packet overhead is the main bottleneck.
2. **Add Forward Error Correction** — e.g. link against `fec` (Reed-Solomon) so lost
   packets can be reconstructed instead of corrupting the frame.
3. **Add encryption** (e.g. ChaCha20/AES) if the link needs to be private.
4. **Add multi-card RX diversity** — listen on multiple monitor-mode interfaces
   simultaneously and merge/deduplicate frames.

---

## 9. C++ Version (Raw Sockets — Much Faster Than Python/Scapy)

This replaces `tx.py`/`rx.py`. It uses `AF_PACKET`/`SOCK_RAW` directly, avoiding Scapy's
per-packet Python overhead — the single biggest cause of latency/stutter in the DIY setup.

### Install build tools

```bash
sudo apt install -y build-essential libpcap-dev
```

### `tx.cpp` — Air unit transmitter

```cpp
// tx.cpp — reads raw video bytes from stdin (piped from GStreamer),
// wraps them in a minimal 802.11 data frame, injects via raw socket.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/if_packet.h>
#include <linux/if_ether.h>
#include <arpa/inet.h>

#define IFACE "wlan0"
#define CHUNK 1024

// Minimal radiotap header (8 bytes, no optional fields)
static const uint8_t radiotap_header[] = {
    0x00, 0x00,             // version, pad
    0x08, 0x00,             // header length (8 bytes)
    0x00, 0x00, 0x00, 0x00  // present flags (none)
};

// Minimal 802.11 data frame header
struct ieee80211_hdr {
    uint16_t frame_control;
    uint16_t duration;
    uint8_t addr1[6]; // dest
    uint8_t addr2[6]; // src
    uint8_t addr3[6]; // bssid
    uint16_t seq_ctrl;
} __attribute__((packed));

int main() {
    int sock = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sock < 0) { perror("socket"); return 1; }

    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, IFACE, IFNAMSIZ - 1);
    if (ioctl(sock, SIOCGIFINDEX, &ifr) < 0) { perror("ioctl"); return 1; }
    int ifindex = ifr.ifr_ifindex;

    struct sockaddr_ll sll;
    memset(&sll, 0, sizeof(sll));
    sll.sll_family = AF_PACKET;
    sll.sll_ifindex = ifindex;
    sll.sll_halen = ETH_ALEN;

    ieee80211_hdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.frame_control = htons(0x0800); // type=2 (data), subtype=0
    uint8_t dst_mac[6] = {0x66,0x77,0x88,0x99,0xaa,0xbb};
    uint8_t src_mac[6] = {0x00,0x11,0x22,0x33,0x44,0x55};
    memcpy(hdr.addr1, dst_mac, 6);
    memcpy(hdr.addr2, src_mac, 6);
    memcpy(hdr.addr3, dst_mac, 6);
    memcpy(sll.sll_addr, dst_mac, 6);

    uint8_t buf[sizeof(radiotap_header) + sizeof(hdr) + CHUNK];
    uint16_t seq = 0;

    fprintf(stderr, "Starting transmission... Ctrl+C to stop.\n");

    while (true) {
        uint8_t payload[CHUNK];
        ssize_t n = read(STDIN_FILENO, payload, CHUNK);
        if (n <= 0) break;

        hdr.seq_ctrl = htons(seq++ << 4);

        size_t off = 0;
        memcpy(buf + off, radiotap_header, sizeof(radiotap_header)); off += sizeof(radiotap_header);
        memcpy(buf + off, &hdr, sizeof(hdr)); off += sizeof(hdr);
        memcpy(buf + off, payload, n); off += n;

        sendto(sock, buf, off, 0, (struct sockaddr*)&sll, sizeof(sll));
    }

    close(sock);
    return 0;
}
```

### `rx.cpp` — Ground unit receiver

```cpp
// rx.cpp — sniffs raw 802.11 frames matching our MAC, strips headers,
// writes payload to stdout (pipe into GStreamer).
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/if_packet.h>
#include <linux/if_ether.h>
#include <arpa/inet.h>

#define IFACE "wlan0"
#define RADIOTAP_LEN 8

struct ieee80211_hdr {
    uint16_t frame_control;
    uint16_t duration;
    uint8_t addr1[6];
    uint8_t addr2[6];
    uint8_t addr3[6];
    uint16_t seq_ctrl;
} __attribute__((packed));

int main() {
    int sock = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sock < 0) { perror("socket"); return 1; }

    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, IFACE, IFNAMSIZ - 1);
    ioctl(sock, SIOCGIFINDEX, &ifr);

    struct sockaddr_ll sll;
    memset(&sll, 0, sizeof(sll));
    sll.sll_family = AF_PACKET;
    sll.sll_ifindex = ifr.ifr_ifindex;
    bind(sock, (struct sockaddr*)&sll, sizeof(sll));

    uint8_t target_mac[6] = {0x66,0x77,0x88,0x99,0xaa,0xbb};
    uint8_t buf[65536];

    fprintf(stderr, "Listening for raw frames...\n");

    while (true) {
        ssize_t n = recvfrom(sock, buf, sizeof(buf), 0, nullptr, nullptr);
        if (n < (ssize_t)(RADIOTAP_LEN + sizeof(ieee80211_hdr))) continue;

        ieee80211_hdr* hdr = (ieee80211_hdr*)(buf + RADIOTAP_LEN);
        if (memcmp(hdr->addr1, target_mac, 6) != 0) continue;

        size_t payload_off = RADIOTAP_LEN + sizeof(ieee80211_hdr);
        size_t payload_len = n - payload_off;

        write(STDOUT_FILENO, buf + payload_off, payload_len);
    }

    close(sock);
    return 0;
}
```

### Build

```bash
g++ -O2 -o tx tx.cpp
g++ -O2 -o rx rx.cpp
```

### Run — Air unit

```bash
sudo airmon-ng check kill
sudo airmon-ng start wlan0
sudo iw wlan0mon set channel 149

gst-launch-1.0 libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 \
    ! videoconvert ! x264enc tune=zerolatency bitrate=1000 ! fdsink \
    | sudo ./tx
```

### Run — Ground unit

```bash
sudo airmon-ng check kill
sudo airmon-ng start wlan0
sudo iw wlan0mon set channel 149

sudo ./rx | gst-launch-1.0 fdsrc ! h264parse ! avdec_h264 \
    ! videoconvert ! autovideosink
```

Both `tx` and `rx` need `sudo` (or `setcap cap_net_raw+ep ./tx`) since raw sockets
require `CAP_NET_RAW`.

### Why this is faster

- No Python interpreter overhead per packet — everything is compiled, fixed-size structs.
- Direct `sendto`/`recvfrom` on the raw socket, no Scapy packet-parsing/serialization layer.
- You can now realistically hit 30–60fps at low resolutions without the frame drops seen
  in the Python version.

### Still missing vs. WFB-ng (next steps if you want to go further)

- **FEC**: no packet-loss recovery yet. Add a Reed-Solomon library (e.g. `zfec` or the
  `fec` library WFB-ng itself uses) — group N data packets with K parity packets per block.
- **Encryption**: none yet — anyone with a monitor-mode card on the same channel can
  read your stream. Add ChaCha20-Poly1305 (e.g. via libsodium) before injection.
- **Multi-card RX diversity**: run multiple `rx` instances bound to different interfaces
  and merge/deduplicate frames by sequence number.
- **Adaptive bitrate / packet size tuning**: match your MTU-per-injection to what your
  driver/chipset handles best (some drivers fragment or drop oversized raw frames).

---

## 10. SAR-Streamer-Style Air/Ground Pair (Raw Radio Link, No ffmpeg/RTSP)

`air_unit.py` and `ground_unit.py` reproduce the SAR streamer's architecture
(decoupled telemetry thread, throttled/cached detection, background uplink
queue) but replace the ffmpeg/RTSP transport with the raw monitor-mode
802.11 injection built earlier. No Scapy in the hot path — both scripts use
`socket.AF_PACKET`/`SOCK_RAW` directly.

Install deps on both machines:

```bash
pip install opencv-python numpy
pip install pyserial pynmea2   # optional, only needed for real GPS
pip install ultralytics        # optional, only needed for --detect (YOLO); falls back to HOG if absent
```

### Air unit (drone / companion computer)

```bash
# 1. Monitor mode
sudo airmon-ng check kill
sudo airmon-ng start wlan0
sudo iw wlan0mon set channel 149

# 2. Run with a live camera
sudo python3 air_unit.py --iface wlan0mon --camera 0 --width 640 --height 480 --fps 30

# 2b. Or bench-test with a video file, no camera needed
sudo python3 air_unit.py --iface wlan0mon --input test2.mp4 --loop

# With real GPS attached:
sudo python3 air_unit.py --iface wlan0mon --camera 0 --gps_port /dev/serial0
```

### Ground unit (laptop)

```bash
# 1. Monitor mode (same channel as air unit)
sudo airmon-ng check kill
sudo airmon-ng start wlan0
sudo iw wlan0mon set channel 149

# 2. Plain video + telemetry overlay, no detection
sudo python3 ground_unit.py --iface wlan0mon

# 3. With human detection (throttled every 3rd frame by default)
sudo python3 ground_unit.py --iface wlan0mon --detect

# 4. Headless (e.g. running on a Pi ground station with no monitor)
sudo python3 ground_unit.py --iface wlan0mon --detect --no-display

# 5. With detections pushed to a dashboard
sudo python3 ground_unit.py --iface wlan0mon --detect --telemetry_url http://localhost:8765
```

### Shutdown (both ends)

```bash
# Ctrl+C the running script, then:
sudo airmon-ng stop wlan0mon
sudo systemctl restart NetworkManager
```

### What's different from the SAR streamer's ffmpeg/RTSP version

| | ffmpeg/RTSP version | This raw-radio version |
|---|---|---|
| Transport | ffmpeg -> RTSP over normal IP networking | Raw 802.11 frames, no IP stack at all |
| Requires an AP/router | Yes (or direct link config) | No — point-to-point, monitor mode only |
| Packet loss handling | TCP/RTSP handles it | None yet (occasional corrupt JPEG frames get silently dropped — see `cv2.imdecode` returning `None`) |
| Latency | ffmpeg encode/mux/RTSP overhead | Lower — JPEG chunks go straight to the radio |
| Range/interference behavior | Same as any Wi-Fi client link | Same broadcast-style behavior as WFB-ng, minus its FEC |

### Known gap vs. the earlier C++/WFB-ng discussion

This pair still has **no FEC**, so a lost chunk drops the *whole* JPEG frame
(you'll see an occasional skipped/frozen frame rather than a corrupted one —
`ground_unit.py`'s `cv2.imdecode` check protects against garbled JPEGs, it
just can't recover them). If you want to close that gap next, the highest-value
addition is Reed-Solomon FEC per frame (e.g. `zfec`), applied the same way
WFB-ng blocks N data + K parity packets — happy to add that as a `--fec`
option in `air_unit.py`/`ground_unit.py` if useful.

---

## 11. Air Unit — Full Script (`air_unit.py`)

This is the complete air unit script that includes:
- Video capture from file or camera
- Optional human detection (YOLO or HOG fallback)
- Background telemetry reader (GPS, CPU temp, battery)
- Uplink queue for detection snapshots (over separate IP)
- Raw 802.11 injection via `tx` binary

```python
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
```

---

## 12. Ground Unit — Full Script (`ground_unit.py`)

This is the complete ground unit script that matches the air unit's structure:

```python
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
```

---

## 13. Quick Reference — All Commands

### Start Monitor Mode

```bash
sudo airmon-ng check kill
sudo airmon-ng start wlan0
sudo iw wlan0mon set channel 149
```

### Stop Monitor Mode

```bash
sudo airmon-ng stop wlan0mon
sudo systemctl restart NetworkManager
```

### Build C++ Tools

```bash
g++ -O2 -o tx tx.cpp
g++ -O2 -o rx rx.cpp
```

### Run Air Unit

```bash
# Basic
sudo python3 air_unit.py --input test2.mp4 --tx_path ./tx

# With detection
sudo python3 air_unit.py --input test2.mp4 --tx_path ./tx --detect

# Headless (no display)
sudo python3 air_unit.py --input test2.mp4 --tx_path ./tx --detect --no-display

# With GPS and telemetry uplink
sudo python3 air_unit.py --input test2.mp4 --tx_path ./tx --detect \
    --gps_port /dev/serial0 --telemetry_url http://192.168.1.50:8765

# Live camera
sudo python3 air_unit.py --camera 0 --tx_path ./tx --width 640 --height 480 --fps 30
```

### Run Ground Unit

```bash
# Basic receive + display
sudo python3 ground_unit.py --rx_path ./rx --width 640 --height 344

# Record incoming stream
sudo python3 ground_unit.py --rx_path ./rx --width 640 --height 344 --record flight.mp4

# Headless (no display)
sudo python3 ground_unit.py --rx_path ./rx --width 640 --height 344 --no-display
```

---

## 14. Troubleshooting

### No video on ground unit

1. **Check monitor mode**: `iw dev wlan0 info` should show `type monitor`
2. **Check channel**: both units must be on the same channel
3. **Check MAC address**: `dst_mac` in tx.cpp must match `target_mac` in rx.cpp
4. **Check rx is capturing**: run `sudo ./rx | head -c 100 | xxd` — should see H.264 NAL units (starts with `00 00 00 01`)
5. **Check air unit is actually writing**: run `sudo python3 air_unit.py --input test2.mp4 --tx_path ./tx --no-display` and watch for `[STATS]` messages

### Laggy/stuttering video

- Reduce `--stream_width` to 480 or 320 on air unit
- Reduce `--bitrate` to 1000 or lower
- Use C++ tx/rx instead of Python/Scapy
- Check CPU usage: `top` or `htop`

### No detection boxes

- Install `ultralytics` for YOLO: `pip install ultralytics`
- Or use HOG fallback (slower but no extra deps)
- Check `--detect_width` — smaller values are faster but less accurate

### GPS not working

- Verify serial port: `ls -la /dev/ttyUSB*` or `/dev/serial0`
- Check permissions: `sudo chmod 666 /dev/serial0`
- Test GPS manually: `cat /dev/serial0` (should see NMEA sentences)
- Install pyserial and pynmea2: `pip install pyserial pynmea2`
