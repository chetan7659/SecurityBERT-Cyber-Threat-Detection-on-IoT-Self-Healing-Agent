#!/usr/bin/env python3
"""
SecurityBERT + PPO Real-Time Detector for Raspberry Pi 3B
==========================================================
Full pipeline:
  Scapy capture → feature extraction → PPFLE encoding
  → BBPE tokenization → SecurityBERT → PPO Agent
  → SelfHealingManager → Email alert

Usage:
  sudo python3 detector.py                     # uses config.yaml defaults
  sudo python3 detector.py --interface eth0    # LAN cable
  sudo python3 detector.py --interface wlan0   # WiFi
  sudo python3 detector.py --simulate          # log commands, don't execute
"""

import os
import sys
import signal
import time
import logging
import argparse
import socket
import threading
from pathlib    import Path
from datetime   import datetime
from collections import deque

import yaml

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from feature_extractor import PacketWindowExtractor, extract_features
from ppfle_encoder     import encode_row
from model_inference   import SecurityBERTInference
from self_healing      import SelfHealingManager
from email_alert       import EmailAlerter

try:
    from scapy.all import sniff, IP, conf as scapy_conf
    scapy_conf.verb = 0
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False


def setup_logging(log_file: str, level: str, console: bool) -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_file)]
    if console:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level    = getattr(logging, level.upper(), logging.INFO),
        format   = '[%(asctime)s] %(levelname)s — %(message)s',
        datefmt  = '%Y-%m-%d %H:%M:%S',
        handlers = handlers,
        force    = True,
    )
    return logging.getLogger('detector')


class SecurityBERTDetector:
    """
    Main detector — orchestrates the full pipeline on Pi.

    Pipeline per cycle:
    1. Scapy captures packets for window_seconds
    2. PacketWindowExtractor → 46-feature row
    3. PPFLE encoder → MD5 token sequence
    4. SecurityBERTInference → attack_class + confidence + action
    5. SelfHealingManager → executes iptables / systemctl
    6. EmailAlerter → sends SMTP alert if attack detected
    """

    def __init__(self, config_path: str = 'config.yaml'):
        # ── Load config ───────────────────────────────────────────────────────
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # ── Logging ───────────────────────────────────────────────────────────
        self.log = setup_logging(
            log_file = self.cfg['logging']['log_file'],
            level    = self.cfg['logging']['level'],
            console  = self.cfg['logging']['console'],
        )
        self.log.info('=' * 60)
        self.log.info('  SecurityBERT Detector — Starting')
        self.log.info('=' * 60)

        # ── Inference pipeline ────────────────────────────────────────────────
        self.log.info('Loading SecurityBERT + PPO …')
        self.inference = SecurityBERTInference(
            bert_ckpt_path = self.cfg['model']['bert_checkpoint'],
            ppo_ckpt_path  = self.cfg['model']['ppo_checkpoint'],
            tokenizer_dir  = self.cfg['model']['tokenizer_dir'],
            conf_threshold = self.cfg['model']['conf_threshold'],
            device         = self.cfg['model']['device'],
        )
        self.log.info('✅ Inference pipeline ready.')

        # ── Self-healing manager ──────────────────────────────────────────────
        simulate = self.cfg['healing']['simulate']
        self.healer = SelfHealingManager(
            simulate = simulate,
            heal_log = self.cfg['logging']['heal_log'],
        )
        self.log.info(
            f'✅ SelfHealingManager ready '
            f'(SIMULATE={simulate})'
        )

        # ── Email alerter ─────────────────────────────────────────────────────
        ec = self.cfg['email']
        self.alerter = EmailAlerter(
            smtp_server     = ec['smtp_server'],
            smtp_port       = ec['smtp_port'],
            sender_email    = ec['sender_email'],
            sender_password = ec['sender_password'],
            receiver_email  = ec['receiver_email'],
            subject_prefix  = ec['subject_prefix'],
            cooldown_sec    = self.cfg['detection']['alert_cooldown_sec'],
        )

        # ── Packet window ─────────────────────────────────────────────────────
        self.window    = PacketWindowExtractor(
            window_seconds = self.cfg['detection']['window_seconds']
        )
        self.interface = self.cfg['detection']['interface']
        self.min_pkts  = self.cfg['detection']['min_packets']
        self.normal_cls= self.cfg['detection']['normal_class']

        # ── Stats ─────────────────────────────────────────────────────────────
        self.stats = {
            'packets'   : 0,
            'inferences': 0,
            'attacks'   : 0,
            'alerts'    : 0,
            'start_time': time.time(),
        }
        self.running = False

    # ── Packet callback ────────────────────────────────────────────────────────

    def _on_packet(self, pkt) -> None:
        """Called by Scapy for every captured packet."""
        self.stats['packets'] += 1
        self.window.add(pkt)

        if len(self.window) >= self.min_pkts:
            self._run_inference_cycle()

    # ── Inference cycle ────────────────────────────────────────────────────────

    def _run_inference_cycle(self) -> None:
        """One complete detection + healing cycle."""
        # Feature extraction
        feat_row = self.window.get_feature_row()
        if feat_row is None:
            return
        source_ip = self.window.get_source_ip()

        # PPFLE encode
        ppfle_seq = encode_row(feat_row)

        # SecurityBERT + PPO
        result = self.inference.predict(
            ppfle_sequence  = ppfle_seq,
            packet_features = feat_row,
        )
        self.stats['inferences'] += 1
        self.window.clear()

        pred_cls   = result['predicted_class']
        confidence = result['confidence']
        action_id  = result['action_id']
        action_name= result['action_name']
        total_ms   = result['total_ms']

        ts = datetime.now().strftime('%H:%M:%S')

        # ── Attack detected ────────────────────────────────────────────────────
        if pred_cls != self.normal_cls:
            self.stats['attacks'] += 1
            self.log.warning(
                f'[{ts}] 🔴 {pred_cls:<25} '
                f'conf={confidence*100:.1f}%  '
                f'src={source_ip}  '
                f'action={action_name}  '
                f'{total_ms:.0f}ms'
            )

            # Execute healing action
            self.healer.execute(
                action_id    = action_id,
                attack_class = pred_cls,
                confidence   = confidence,
                source_ip    = source_ip,
                all_probs    = result['all_probs'],
            )

            # Send email alert
            sent = self.alerter.send(
                attack_class = pred_cls,
                confidence   = confidence,
                source_ip    = source_ip,
                action_name  = action_name,
                all_probs    = result['all_probs'],
                packet_count = self.stats['packets'],
                total_ms     = total_ms,
            )
            if sent:
                self.stats['alerts'] += 1
                self.log.info(f'   📧 Alert sent for {pred_cls}')

        else:
            self.log.info(
                f'[{ts}] 🟢 Normal  '
                f'conf={confidence*100:.1f}%  '
                f'src={source_ip}  '
                f'{total_ms:.0f}ms'
            )

    # ── Stats printer ──────────────────────────────────────────────────────────

    def _print_stats(self) -> None:
        elapsed = time.time() - self.stats['start_time']
        self.log.info(
            f'\n📊 Stats ({elapsed/60:.1f} min):\n'
            f'   Packets    : {self.stats["packets"]:,}\n'
            f'   Inferences : {self.stats["inferences"]:,}\n'
            f'   Attacks    : {self.stats["attacks"]:,}\n'
            f'   Alerts sent: {self.stats["alerts"]:,}\n'
        )

    # ── Main ───────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the detector — blocking."""
        if not SCAPY_OK:
            self.log.error('scapy not installed. Run: pip install scapy')
            sys.exit(1)

        if os.geteuid() != 0:
            self.log.error('Must run as root: sudo python3 detector.py')
            sys.exit(1)

        # Graceful shutdown
        def _shutdown(sig, frame):
            self.log.info('\n🛑 Shutting down …')
            self._print_stats()
            self.running = False
            sys.exit(0)

        signal.signal(signal.SIGINT,  _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # Stats thread
        def _stats_thread():
            while self.running:
                time.sleep(60)
                self._print_stats()

        self.running = True
        t = threading.Thread(target=_stats_thread, daemon=True)
        t.start()

        pi_ip = self.alerter._get_ip()
        self.log.info(
            f'\n🚀 Detector STARTED\n'
            f'   Interface : {self.interface}\n'
            f'   Pi IP     : {pi_ip}\n'
            f'   Listening for packets … (Ctrl+C to stop)\n'
        )

        sniff(
            iface  = self.interface,
            prn    = self._on_packet,
            store  = False,
            filter = 'ip',
        )


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='SecurityBERT Real-Time IoT Threat Detector'
    )
    parser.add_argument('--config',    default='config.yaml')
    parser.add_argument('--interface', default=None,
                        help='Override network interface')
    parser.add_argument('--simulate',  action='store_true',
                        help='Simulate healing (log only, no real commands)')
    args = parser.parse_args()

    det = SecurityBERTDetector(config_path=args.config)

    if args.interface:
        det.interface = args.interface

    if args.simulate:
        det.healer.simulate = True
        det.log.info('Simulation mode: healing commands will be logged only')

    det.start()


if __name__ == '__main__':
    main()