#!/usr/bin/env python3
"""
Real-time log monitor — watch Pi detector logs from your laptop.
Shows live detections and healing actions in colour.

Usage:
  # LAN cable:
  python monitor.py --pi-ip 192.168.100.1

  # WiFi:
  python monitor.py --pi-ip 192.168.1.42
"""

import paramiko
import argparse
import sys
import re
import time

# Terminal colours
RED    = '\033[91m'; GREEN  = '\033[92m'; YELLOW = '\033[93m'
CYAN   = '\033[96m'; BOLD   = '\033[1m';  RESET  = '\033[0m'
ORANGE = '\033[33m'

ATTACK_COLORS = {
    'DDoS_TCP': RED, 'DDoS_UDP': RED, 'DDoS_ICMP': RED,
    'DDoS_HTTP': RED, 'Ransomware': RED,
    'SQL_injection': YELLOW, 'MITM': YELLOW,
    'Backdoor': ORANGE, 'Password': ORANGE,
    'Port_Scanning': CYAN, 'Fingerprinting': CYAN,
    'Normal': GREEN,
}


def colorise(line: str) -> str:
    """Apply colour to log lines based on content."""
    if '🔴' in line or 'WARNING' in line:
        for cls, col in ATTACK_COLORS.items():
            if cls in line:
                return f'{col}{line}{RESET}'
        return f'{RED}{line}{RESET}'
    elif '🟢' in line:
        return f'{GREEN}{line}{RESET}'
    elif '📧' in line:
        return f'{CYAN}{line}{RESET}'
    elif '✅' in line:
        return f'{GREEN}{line}{RESET}'
    elif '❌' in line:
        return f'{RED}{line}{RESET}'
    elif 'BLOCK_IP' in line:
        return f'{RED}{BOLD}{line}{RESET}'
    elif 'ISOLATE' in line:
        return f'{RED}{BOLD}{line}{RESET}'
    elif 'RESTART' in line:
        return f'{ORANGE}{line}{RESET}'
    return line


def tail_log(pi_ip: str, pi_user: str, pi_pass: str,
             log_file: str, ssh_port: int = 22) -> None:
    """SSH into Pi and tail the detector log in real time."""
    print(f'{BOLD}{"="*60}{RESET}')
    print(f'{BOLD}  SecurityBERT Live Monitor{RESET}')
    print(f'  Pi: {pi_user}@{pi_ip}  →  {log_file}')
    print(f'{BOLD}{"="*60}{RESET}\n')

    while True:
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(pi_ip, port=ssh_port,
                           username=pi_user, password=pi_pass, timeout=10)
            print(f'{GREEN}✅ Connected to Pi at {pi_ip}{RESET}\n')

            _, stdout, _ = client.exec_command(
                f'tail -f {log_file}', get_pty=True
            )
            for line in stdout:
                line = line.rstrip()
                print(colorise(line))

        except KeyboardInterrupt:
            print(f'\n{YELLOW}Monitor stopped.{RESET}')
            sys.exit(0)
        except Exception as e:
            print(f'{RED}Connection lost: {e}  Retrying in 5s …{RESET}')
            time.sleep(5)
        finally:
            try: client.close()
            except: pass


def main():
    parser = argparse.ArgumentParser(
        description='SecurityBERT Pi Log Monitor'
    )
    parser.add_argument('--pi-ip',   required=True,
                        help='Pi IP address')
    parser.add_argument('--user',    default='pi',
                        help='Pi SSH username (default: pi)')
    parser.add_argument('--password',default='raspberry',
                        help='Pi SSH password')
    parser.add_argument('--log',
                        default='/home/pi/securitybert/logs/detector.log',
                        help='Log file path on Pi')
    parser.add_argument('--port',    type=int, default=22)
    args = parser.parse_args()

    tail_log(args.pi_ip, args.user, args.password,
             args.log, args.port)


if __name__ == '__main__':
    main()