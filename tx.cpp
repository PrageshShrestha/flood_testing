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
