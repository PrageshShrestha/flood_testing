
# Raw Wi-Fi Monitor-Mode Video Link — Full Command Reference

A step-by-step reference for building a WFB-ng-style air-to-ground video link using
monitor mode + GStreamer + a Python (Scapy) injection/reception script. This is the
manual/DIY approach — it will **not** match WFB-ng's latency, reliability, or range
without the caveats noted at the end.

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
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly aircrack-ng iw
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

> Channel 149 is in the 5GHz band and commonly used for FPV because it's less congested.
> Use `iw list` to see what channels/bands your adapter supports.

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

## 3. The Air Unit — Transmitter Script (`tx.py`)

Save on the **air** unit:

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

## 4. The Ground Unit — Receiver Script (`rx.py`)

Save on the **ground** unit:

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

## 5. Full Startup Sequence (Both Ends)

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

## 8. Why This Won't Match WFB-ng — and How to Close the Gap

| Feature | This DIY script | WFB-ng / OpenHD |
|---|---|---|
| FEC (Forward Error Correction) | ❌ None — video breaks on interference | ✅ Reconstructs lost packets |
| Encryption | ❌ None | ✅ AES-encrypted |
| Multi-card RX diversity | ❌ Not supported | ✅ Combines multiple antennas |
| Telemetry muxing (MAVLink) | ❌ Needs separate link | ✅ Muxed with video |
| Processing speed | ❌ Python/Scapy too slow for 30-60fps sustained | ✅ Optimized C |

To actually approach WFB-ng-level performance you'd need to:

1. **Rewrite `tx.py`/`rx.py` in C/C++** using raw sockets (`socket(AF_PACKET, SOCK_RAW)`)
   instead of Scapy — Python's per-packet overhead is the main bottleneck.
2. **Add Forward Error Correction** — e.g. link against `fec` (Reed-Solomon) so lost
   packets can be reconstructed instead of corrupting the frame.
3. **Add encryption** (e.g. ChaCha20/AES) if the link needs to be private.
4. **Add multi-card RX diversity** — listen on multiple monitor-mode interfaces
   simultaneously and merge/deduplicate frames.

If you want, I can put together the alternative install/setup commands for
**OpenHD** or **Ruby FPV** instead, since both give you this feature set out of the box
without writing any custom code.
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

> Both `tx` and `rx` need `sudo` (or `setcap cap_net_raw+ep ./tx`) since raw sockets
> require `CAP_NET_RAW`.

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

## 10. Combining Both: SAR Streamer (Air) + Raw Monitor-Mode Link (Ground)

This section wires your SAR streamer (camera capture, throttled detection,
decoupled telemetry) into the raw monitor-mode pipeline from sections 3–9,
instead of its original RTSP output. Full file: `air_unit.py`.

**What changed vs. the original SAR streamer:**

| | Original (RTSP) | This version (raw link) |
|---|---|---|
| Output sink | `ffmpeg -f rtsp rtsp://...` | `ffmpeg -f h264 -` piped into `./tx` |
| Transport | Normal IP/UDP | Raw 802.11 injection, monitor mode |
| Telemetry uplink | Same `Uplink` class, same behavior | Unchanged — still goes over a **separate** normal IP link (hotspot/second radio), since the monitor-mode channel is one-way broadcast video only, same as WFB-ng |
| Detection/telemetry decoupling | Background thread + throttled detection | Identical — kept as-is, it's the fix that matters |

### Build the injector (from section 9)

```bash
g++ -O2 -o tx tx.cpp
```

Put `tx` in the same directory as `air_unit.py` (or pass `--tx_path`).

### Air unit — start monitor mode, then run the streamer

```bash
sudo airmon-ng check kill
sudo airmon-ng start wlan0
sudo iw wlan0mon set channel 149

# basic raw video, no detection
python3 air_unit.py --input test2.mp4 --tx_path ./tx

# with detection + telemetry, no local preview window (saves CPU)
python3 air_unit.py --input test2.mp4 --tx_path ./tx --detect --no-display \
    --detect_every_n_frames 3 --detect_width 320

# with a live GPS module and a telemetry uplink to a dashboard over a SEPARATE
# normal-IP link (e.g. phone hotspot) -- NOT over the monitor-mode video channel
python3 air_unit.py --input test2.mp4 --tx_path ./tx --detect \
    --gps_port /dev/serial0 --telemetry_url http://192.168.1.50:8765
```

### Ground unit — receive and decode

```bash
sudo airmon-ng check kill
sudo airmon-ng start wlan0
sudo iw wlan0mon set channel 149

# build the receiver (from section 9)
g++ -O2 -o rx rx.cpp

# run the actual ground-side script (ground_unit.py) -- NOT a bare ffplay pipe.
# --width/--height MUST match the air unit's --stream_width and its
# computed stream_height exactly, or the frame math breaks.
sudo python3 ground_unit.py --rx_path ./rx --width 640 --height 344

# optionally record the received feed to disk
sudo python3 ground_unit.py --rx_path ./rx --width 640 --height 344 --record flight.mp4

# headless (no display window, e.g. a Pi ground station)
sudo python3 ground_unit.py --rx_path ./rx --width 640 --height 344 --no-display
```

`ground_unit.py` internally chains `rx -> ffmpeg (H.264 decode, low-latency
flags) -> cv2 display`, the mirror image of what `air_unit.py` does on the
way out. If you just want a raw sanity check without Python, the bare
pipe still works too:

```bash
sudo ./rx | ffplay -f h264 -fflags nobuffer -flags low_delay -probesize 32 -
```

### Why this keeps the SAR streamer's speed fix intact

The bottleneck that dropped the original from ~200fps to ~12fps was per-frame
`vcgencmd` subprocess spawns and duplicated detection passes — none of that
is touched here. `RawLinkStreamer` is a straight rename/rewire of
`FFmpegStreamer`: same `TelemetryReader` (own thread, own timer, cached
reads), same `HumanDetector` (throttled + resized + cached), same `Uplink`
(fire-and-forget queue). The **only** thing that changed is what's on the
other end of the pipe — `./tx` (raw injection) instead of `-f rtsp` (normal
networking).

### Still open (same caveats as section 8/9)

- No FEC yet on the video channel — a corrupted packet will still glitch the
  H.264 decode on the ground unit until you add Reed-Solomon parity packets.
- No encryption on the video channel.
- Telemetry uplink assumes a second, ordinary IP link exists (hotspot, LTE
  modem, second radio) — if you don't have one, that part just won't send
  anywhere and the script keeps working fine as a local-only detector/logger.
## 11. File & Dependency Checklist — What Needs to Be Where

### Air unit (drone / companion computer)

**Files needed in the working directory:**

| File | Where it comes from | Required? |
|---|---|---|
| `tx.cpp` → compiled `tx` binary | Section 9 of this guide | Yes |
| `air_unit.py` | Section 10 | Yes |
| `test2.mp4` (or any test video) | Your own footage — only needed if you don't have a live camera attached yet | Optional — omit `--input` and use `--camera` flags instead if you wire up real capture (this script currently reads via `cv2.VideoCapture(args.input)`, so a live camera would need the `--input` path swapped for a device index like `cv2.VideoCapture(0)` — ask if you want that variant) |
| `yolov8n.pt` | Auto-downloaded by `ultralytics` on first run if `--detect` is used and internet is available; falls back to OpenCV's built-in HOG detector if `ultralytics` isn't installed | Optional |
| `/tmp/telemetry_override.json` | Written by any of your own scripts to feed GPS/battery values in without real hardware | Optional |

**System packages:**

```bash
sudo apt update
sudo apt install -y build-essential ffmpeg python3-pip iw aircrack-ng
```

**Python packages:**

```bash
pip install --break-system-packages opencv-python numpy
pip install --break-system-packages pyserial pynmea2   # only if using a real GPS module
pip install --break-system-packages ultralytics         # only if using --detect with YOLO
```

**Build:**

```bash
g++ -O2 -o tx tx.cpp
```

**Run (minimum working example):**

```bash
sudo airmon-ng check kill
sudo airmon-ng start wlan0
sudo iw wlan0mon set channel 149
python3 air_unit.py --input test2.mp4 --tx_path ./tx
```

---

### Ground unit (laptop)

**Files needed in the working directory:**

| File | Where it comes from | Required? |
|---|---|---|
| `rx.cpp` → compiled `rx` binary | Section 9 | Yes |
| `ground_unit.py` | Section 10/this session | Yes |

That's it — the ground side has no video file, no YOLO model, no GPS
libraries. It only decodes and displays whatever the air unit is injecting.

**System packages:**

```bash
sudo apt update
sudo apt install -y build-essential ffmpeg python3-pip iw aircrack-ng
```

**Python packages:**

```bash
pip install --break-system-packages opencv-python numpy
```

**Build:**

```bash
g++ -O2 -o rx rx.cpp
```

**Run (minimum working example):**

```bash
# 1. Kill interfering processes
sudo airmon-ng check kill

# 2. Start monitor mode on your interface
sudo airmon-ng start wlxd0374558ffd4

# 3. Set channel
sudo iw wlxd0374558ffd4mon set channel 149

# 4. Run the ground-side script
sudo python3 ground_unit.py --rx_path ./rx --width 640 --height 344 --iface wlxd0374558ffd4mon
```

> `--width`/`--height` must match what the air unit computed. With the
> default `--stream_width 640` and a 1280x720 source video, `air_unit.py`
> computes `stream_height = 640 * (720/1280) = 360`, then rounds down to
> the nearest even number → **344 only applies if your source is a
> 640x344-ish aspect ratio; check the console output of `air_unit.py`
> on startup — it doesn't currently print the final resolution, so add
> `print(stream_width, stream_height)` right after they're computed in
> `main()` if you want to confirm the exact numbers before starting the
> ground unit.**

---

### Optional third machine: the telemetry dashboard

Only needed if you pass `--telemetry_url` to `air_unit.py`. This is a
**separate normal-IP server** (not part of the monitor-mode link) that
receives POSTed detection JPEGs + GPS/battery data. Nothing in this guide
builds that server for you yet — `air_unit.py` just assumes something is
listening at `<url>/detection` and accepts a multipart POST. If you want,
I can write a minimal `telemetry_broadcaster.py` (Flask/FastAPI) to match
what `Uplink._post_multipart()` expects.

---

### Quick summary table

| | Air unit | Ground unit |
|---|---|---|
| Script | `air_unit.py` | `ground_unit.py` |
| Compiled binary | `tx` (from `tx.cpp`) | `rx` (from `rx.cpp`) |
| Test video | `test2.mp4` (or your own) | Not needed |
| YOLO model | Optional, auto-downloaded | Not needed |
| GPS libs | Optional (`pyserial`, `pynmea2`) | Not needed |
| Monitor mode required | Yes | Yes |
| Same Wi-Fi channel as other side | Yes | Yes |
