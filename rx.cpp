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
