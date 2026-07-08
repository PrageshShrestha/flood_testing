// rx_debug.cpp — with debug output
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

#define IFACE "wlxd0374558ffd4"  // CHANGE THIS TO YOUR INTERFACE
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
    if (ioctl(sock, SIOCGIFINDEX, &ifr) < 0) {
        fprintf(stderr, "Interface %s not found!\n", IFACE);
        return 1;
    }

    struct sockaddr_ll sll;
    memset(&sll, 0, sizeof(sll));
    sll.sll_family = AF_PACKET;
    sll.sll_ifindex = ifr.ifr_ifindex;
    if (bind(sock, (struct sockaddr*)&sll, sizeof(sll)) < 0) {
        perror("bind");
        return 1;
    }

    uint8_t target_mac[6] = {0x66,0x77,0x88,0x99,0xaa,0xbb};
    uint8_t buf[65536];

    fprintf(stderr, "Listening on %s for MAC %02x:%02x:%02x:%02x:%02x:%02x\n",
        IFACE,
        target_mac[0], target_mac[1], target_mac[2],
        target_mac[3], target_mac[4], target_mac[5]);

    while (true) {
        ssize_t n = recvfrom(sock, buf, sizeof(buf), 0, nullptr, nullptr);
        if (n < 0) { perror("recvfrom"); continue; }
        
        fprintf(stderr, "Received %ld bytes\n", n);
        
        if (n < (ssize_t)(RADIOTAP_LEN + sizeof(ieee80211_hdr))) {
            fprintf(stderr, "Packet too small: %ld bytes\n", n);
            continue;
        }

        ieee80211_hdr* hdr = (ieee80211_hdr*)(buf + RADIOTAP_LEN);
        
        fprintf(stderr, "Dest MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",
            hdr->addr1[0], hdr->addr1[1], hdr->addr1[2],
            hdr->addr1[3], hdr->addr1[4], hdr->addr1[5]);

        // TEMPORARILY REMOVED FILTER - accept ALL packets
        // if (memcmp(hdr->addr1, target_mac, 6) != 0) continue;

        size_t payload_off = RADIOTAP_LEN + sizeof(ieee80211_hdr);
        size_t payload_len = n - payload_off;

        fprintf(stderr, "Payload size: %ld bytes\n", payload_len);
        
        if (payload_len > 0) {
            write(STDOUT_FILENO, buf + payload_off, payload_len);
        }
    }

    close(sock);
    return 0;
}
