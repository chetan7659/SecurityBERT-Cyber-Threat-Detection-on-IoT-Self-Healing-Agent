"""
PPFLE — Privacy-Preserving Fixed-Length Encoding
Algorithm 1 from SecurityBERT paper Section III-C.
Standalone for Pi deployment.
"""

import hashlib
from typing import Dict, List

FEATURE_COLUMNS = [
    'arp.opcode', 'arp.hw.size',
    'icmp.checksum', 'icmp.seq_le', 'icmp.unused',
    'http.content_length', 'http.request.method', 'http.referer',
    'http.request.version', 'http.response', 'http.tls_port',
    'tcp.ack', 'tcp.ack_raw', 'tcp.checksum',
    'tcp.connection.fin', 'tcp.connection.rst',
    'tcp.connection.syn', 'tcp.connection.synack',
    'tcp.flags', 'tcp.flags.ack', 'tcp.len', 'tcp.seq',
    'udp.stream', 'udp.time_delta',
    'dns.qry.name', 'dns.qry.name.len', 'dns.qry.qu',
    'dns.qry.type', 'dns.retransmission',
    'dns.retransmit_request', 'dns.retransmit_request_in',
    'mqtt.conack.flags', 'mqtt.conflag.cleansess', 'mqtt.conflags',
    'mqtt.hdrflags', 'mqtt.len', 'mqtt.msg_decoded_as',
    'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.protoname',
    'mqtt.topic', 'mqtt.topic_len', 'mqtt.ver',
    'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id',
]


def H(x: str) -> str:
    """MD5 hash — 32-char fixed-length hex digest."""
    return hashlib.md5(x.encode('utf-8')).hexdigest()


def encode_row(feature_dict: Dict) -> str:
    """
    Apply PPFLE to one feature row.
    Returns space-separated MD5 tokens (one per feature).
    """
    tokens = [H(f'{col}${feature_dict.get(col, 0)}')
              for col in FEATURE_COLUMNS]
    return ' '.join(tokens)