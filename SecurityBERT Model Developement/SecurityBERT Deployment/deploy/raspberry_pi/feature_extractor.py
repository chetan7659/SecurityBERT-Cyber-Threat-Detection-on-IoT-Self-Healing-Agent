"""
Real-time feature extraction from live Scapy packets.
Maps captured packets → 46 SecurityBERT features.
"""

import time
import hashlib
from typing import Dict, Optional
from collections import defaultdict, deque

try:
    from scapy.all import IP, TCP, UDP, ICMP, ARP, DNS, DNSQR, Raw
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False


# ── UDP stream tracker ─────────────────────────────────────────────────────────
_udp_stream_map  = {}
_last_udp_time   = {}


def extract_features(packet) -> Optional[Dict]:
    """
    Extract all 46 features from one Scapy packet.
    Returns feature dict or None if packet not relevant.
    """
    if not packet.haslayer(IP):
        return None

    feat     = {col: 0 for col in _ALL_COLS}
    pkt_time = float(packet.time)

    # ── ARP ───────────────────────────────────────────────────────────────────
    if packet.haslayer(ARP):
        a = packet[ARP]
        feat['arp.opcode']  = int(a.op)
        feat['arp.hw.size'] = int(a.hwlen)

    # ── ICMP ──────────────────────────────────────────────────────────────────
    if packet.haslayer(ICMP):
        ic = packet[ICMP]
        feat['icmp.checksum'] = int(ic.chksum) if ic.chksum else 0
        feat['icmp.seq_le']   = int(ic.seq)    if hasattr(ic,'seq') else 0

    # ── TCP ───────────────────────────────────────────────────────────────────
    if packet.haslayer(TCP):
        tcp = packet[TCP]
        fl  = int(tcp.flags)

        feat['tcp.ack']              = int(tcp.ack)
        feat['tcp.ack_raw']          = int(tcp.ack)
        feat['tcp.checksum']         = int(tcp.chksum) if tcp.chksum else 0
        feat['tcp.connection.fin']   = 1 if fl & 0x01 else 0
        feat['tcp.connection.rst']   = 1 if fl & 0x04 else 0
        feat['tcp.connection.syn']   = 1 if fl & 0x02 else 0
        feat['tcp.connection.synack']= 1 if (fl & 0x12) == 0x12 else 0
        feat['tcp.flags']            = fl
        feat['tcp.flags.ack']        = 1 if fl & 0x10 else 0
        feat['tcp.len']              = len(tcp.payload)
        feat['tcp.seq']              = int(tcp.seq)

        if tcp.dport == 443 or tcp.sport == 443:
            feat['http.tls_port'] = 1

        # HTTP (port 80)
        if (tcp.dport == 80 or tcp.sport == 80) and packet.haslayer(Raw):
            try:
                raw = bytes(packet[Raw].load).decode('utf-8', errors='ignore')
                for method in ['GET','POST','PUT','DELETE','HEAD']:
                    if raw.startswith(method):
                        feat['http.request.method'] = hash(method) % 65535
                        break
                if 'HTTP/1.1' in raw: feat['http.request.version'] = 1
                elif 'HTTP/2'  in raw: feat['http.request.version'] = 2
                if 'Content-Length:' in raw:
                    try:
                        cl = raw.split('Content-Length:')[1].split('\r\n')[0].strip()
                        feat['http.content_length'] = int(cl)
                    except Exception:
                        pass
                if raw.startswith('HTTP/'): feat['http.response'] = 1
                if 'Referer:'  in raw: feat['http.referer'] = 1
            except Exception:
                pass

        # MQTT (port 1883)
        if (tcp.dport == 1883 or tcp.sport == 1883) and packet.haslayer(Raw):
            try:
                raw = bytes(packet[Raw].load)
                if len(raw) >= 2:
                    msg_type = (raw[0] & 0xF0) >> 4
                    feat['mqtt.msgtype']   = msg_type
                    feat['mqtt.hdrflags']  = raw[0] & 0x0F
                    feat['mqtt.len']       = raw[1] if len(raw) > 1 else 0
                    feat['mqtt.proto_len'] = len(raw)
                    feat['mqtt.ver']       = 4
                    if msg_type == 1 and len(raw) > 9:
                        feat['mqtt.conflags']          = raw[9]
                        feat['mqtt.conflag.cleansess'] = (raw[9] >> 1) & 1
                        feat['mqtt.protoname']         = 1
                    if msg_type == 2 and len(raw) > 3:
                        feat['mqtt.conack.flags'] = raw[2]
                    if msg_type == 3 and len(raw) > 4:
                        tl = (raw[2] << 8) | raw[3] if len(raw) > 3 else 0
                        feat['mqtt.topic_len']      = tl
                        feat['mqtt.msg_decoded_as'] = 1
            except Exception:
                pass

        # Modbus (port 502)
        if (tcp.dport == 502 or tcp.sport == 502) and packet.haslayer(Raw):
            try:
                raw = bytes(packet[Raw].load)
                if len(raw) >= 7:
                    feat['mbtcp.trans_id'] = (raw[0] << 8) | raw[1]
                    feat['mbtcp.len']      = (raw[4] << 8) | raw[5]
                    feat['mbtcp.unit_id']  = raw[6]
            except Exception:
                pass

    # ── UDP ───────────────────────────────────────────────────────────────────
    if packet.haslayer(UDP):
        udp = packet[UDP]
        key = (packet[IP].src, packet[IP].dst, udp.sport, udp.dport)
        if key not in _udp_stream_map:
            _udp_stream_map[key] = len(_udp_stream_map)
        feat['udp.stream'] = _udp_stream_map[key]
        if key in _last_udp_time:
            feat['udp.time_delta'] = pkt_time - _last_udp_time[key]
        _last_udp_time[key] = pkt_time

    # ── DNS ───────────────────────────────────────────────────────────────────
    if packet.haslayer(DNS):
        dns = packet[DNS]
        if dns.qr == 0 and packet.haslayer(DNSQR):
            try:
                qr = packet[DNSQR]
                name = qr.qname.decode('utf-8', errors='ignore').rstrip('.')
                feat['dns.qry.name']     = hash(name) % 65535
                feat['dns.qry.name.len'] = len(name)
                feat['dns.qry.qu']       = 1
                feat['dns.qry.type']     = int(qr.qtype)
            except Exception:
                pass
        if dns.qr == 1 and dns.rcode != 0:
            feat['dns.retransmission']    = 1
            feat['dns.retransmit_request']= 1

    return feat


# ── Column list (import from ppfle_encoder) ───────────────────────────────────
from ppfle_encoder import FEATURE_COLUMNS as _ALL_COLS


class PacketWindowExtractor:
    """
    Collects packets over a time window and returns
    aggregated feature row for SecurityBERT inference.
    """

    def __init__(self, window_seconds: float = 2.0):
        self.window_seconds = window_seconds
        self.packets        = deque()

    def add(self, packet) -> None:
        now = time.time()
        self.packets.append((now, packet))
        while self.packets and (now - self.packets[0][0]) > self.window_seconds:
            self.packets.popleft()

    def get_source_ip(self) -> str:
        """Return source IP of most recent packet."""
        for _, pkt in reversed(list(self.packets)):
            if pkt.haslayer(IP):
                return pkt[IP].src
        return '0.0.0.0'

    def get_feature_row(self) -> Optional[Dict]:
        """Aggregate features from window into one row."""
        if not self.packets:
            return None

        all_feats = []
        for _, pkt in list(self.packets):
            f = extract_features(pkt)
            if f:
                all_feats.append(f)

        if not all_feats:
            return None

        # Aggregate
        agg = {col: 0 for col in _ALL_COLS}
        flag_cols = {
            'tcp.connection.fin', 'tcp.connection.rst',
            'tcp.connection.syn', 'tcp.connection.synack',
            'tcp.flags.ack', 'http.response',
            'dns.retransmission', 'dns.retransmit_request',
        }
        for feat in all_feats:
            for col, val in feat.items():
                if col in flag_cols:
                    agg[col] = max(agg[col], val)
                else:
                    agg[col] += val

        n = len(all_feats)
        numeric_cols = [
            'tcp.ack', 'tcp.ack_raw', 'tcp.checksum',
            'tcp.flags', 'tcp.len', 'tcp.seq',
            'icmp.checksum', 'icmp.seq_le',
        ]
        for col in numeric_cols:
            agg[col] = agg[col] / n if n > 0 else 0

        return agg

    def clear(self) -> None:
        self.packets.clear()

    def __len__(self) -> int:
        return len(self.packets)