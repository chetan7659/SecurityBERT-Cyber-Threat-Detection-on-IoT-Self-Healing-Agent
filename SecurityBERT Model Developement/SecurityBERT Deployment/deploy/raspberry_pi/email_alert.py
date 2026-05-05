"""
SMTP email alert with rich HTML body.
Called by detector.py when attack is detected.
"""

import smtplib
import time
import socket
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from datetime             import datetime
from typing               import Dict, Optional
import numpy as np


SEVERITY_COLOR = {
    'CRITICAL': '#c0392b', 'HIGH': '#e74c3c',
    'MEDIUM'  : '#e67e22', 'LOW' : '#27ae60',
}
CLASS_SEVERITY = {
    'Normal':0,'Fingerprinting':1,'Port_Scanning':2,'Vulnerability_scanner':3,
    'DDoS_ICMP':4,'DDoS_HTTP':4,'DDoS_UDP':4,'DDoS_TCP':5,'XSS':5,
    'SQL_injection':6,'MITM':7,'Password':7,'Uploading':7,
    'Backdoor':8,'Ransomware':10,
}
ACTION_NAMES = [
    'BLOCK_IP','RESET_CONNECTION','RESTART_SERVICE',
    'ISOLATE_DEVICE','LOG_AND_ALERT',
]
CLASS_NAMES = [
    'Backdoor','DDoS_HTTP','DDoS_ICMP','DDoS_TCP','DDoS_UDP',
    'Fingerprinting','MITM','Normal','Password','Port_Scanning',
    'Ransomware','SQL_injection','Uploading','Vulnerability_scanner','XSS',
]


def _severity(cls: str) -> str:
    s = CLASS_SEVERITY.get(cls, 5)
    if s >= 8: return 'CRITICAL'
    if s >= 5: return 'HIGH'
    if s >= 2: return 'MEDIUM'
    return 'LOW'


class EmailAlerter:
    def __init__(
        self,
        smtp_server    : str,
        smtp_port      : int,
        sender_email   : str,
        sender_password: str,
        receiver_email : str,
        subject_prefix : str = '[SecurityBERT]',
        cooldown_sec   : int = 30,
    ):
        self.smtp_server    = smtp_server
        self.smtp_port      = smtp_port
        self.sender_email   = sender_email
        self.sender_password= sender_password
        self.receiver_email = receiver_email
        self.subject_prefix = subject_prefix
        self.cooldown_sec   = cooldown_sec
        self._last_alert    : Dict[str, float] = {}
        self._pi_ip         = self._get_ip()

    def _get_ip(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return 'Unknown'

    def should_alert(self, attack_class: str) -> bool:
        if attack_class == 'Normal':
            return False
        last = self._last_alert.get(attack_class, 0)
        return (time.time() - last) >= self.cooldown_sec

    def send(
        self,
        attack_class  : str,
        confidence    : float,
        source_ip     : str,
        action_name   : str,
        all_probs     : Optional[np.ndarray] = None,
        packet_count  : int   = 0,
        total_ms      : float = 0.0,
    ) -> bool:
        if not self.should_alert(attack_class):
            return False

        sev     = _severity(attack_class)
        color   = SEVERITY_COLOR.get(sev, '#e74c3c')
        now     = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        subject = (
            f'{self.subject_prefix} [{sev}] '
            f'{attack_class} detected — {confidence*100:.0f}% conf'
        )

        # Top-5 probabilities table
        top5_rows = ''
        if all_probs is not None:
            top5 = sorted(
                zip(CLASS_NAMES, all_probs),
                key=lambda x: -x[1]
            )[:5]
            for cls, prob in top5:
                w   = int(prob * 180)
                c   = color if cls == attack_class else '#3498db'
                top5_rows += (
                    f'<tr><td style="padding:4px 8px;font-family:monospace">'
                    f'{cls}</td>'
                    f'<td style="padding:4px">'
                    f'<div style="background:#ecf0f1;width:180px;border-radius:3px">'
                    f'<div style="background:{c};width:{w}px;height:14px;'
                    f'border-radius:3px"></div></div></td>'
                    f'<td style="padding:4px 8px;font-weight:bold">'
                    f'{prob*100:.1f}%</td></tr>'
                )

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/></head>
<body style="font-family:Arial,sans-serif;background:#f5f5f5;padding:20px">
<div style="background:{color};color:white;padding:20px;border-radius:8px 8px 0 0">
  <h2 style="margin:0">🛡️ SecurityBERT Alert — {attack_class}</h2>
</div>
<div style="background:white;padding:24px;border-radius:0 0 8px 8px;
     box-shadow:0 2px 8px rgba(0,0,0,0.1)">
  <table style="width:100%;margin-bottom:16px">
    <tr>
      <td>
        <p style="margin:0;color:#777;font-size:12px">DETECTED</p>
        <h2 style="margin:4px 0;color:{color}">{attack_class}</h2>
      </td>
      <td style="text-align:right">
        <span style="background:{color};color:white;padding:6px 14px;
              border-radius:20px;font-weight:bold">{sev}</span>
        <p style="margin:8px 0 0;font-size:20px;font-weight:bold;
           color:{color}">{confidence*100:.1f}%</p>
      </td>
    </tr>
  </table>
  <table style="width:100%;border-collapse:collapse;margin-bottom:16px">
    <tr style="background:#f8f9fa">
      <td style="padding:8px 12px;color:#777">Time</td>
      <td style="padding:8px 12px;font-weight:bold">{now}</td>
    </tr>
    <tr>
      <td style="padding:8px 12px;color:#777">Source IP</td>
      <td style="padding:8px 12px;font-weight:bold;color:{color}">{source_ip}</td>
    </tr>
    <tr style="background:#f8f9fa">
      <td style="padding:8px 12px;color:#777">Pi IP</td>
      <td style="padding:8px 12px;font-weight:bold">{self._pi_ip}</td>
    </tr>
    <tr>
      <td style="padding:8px 12px;color:#777">Healing Action</td>
      <td style="padding:8px 12px;font-weight:bold;color:#27ae60">{action_name}</td>
    </tr>
    <tr style="background:#f8f9fa">
      <td style="padding:8px 12px;color:#777">Packets</td>
      <td style="padding:8px 12px">{packet_count}</td>
    </tr>
    <tr>
      <td style="padding:8px 12px;color:#777">Detection Time</td>
      <td style="padding:8px 12px">{total_ms:.0f} ms</td>
    </tr>
  </table>
  <h4 style="margin:0 0 8px">Probability Distribution</h4>
  <table>{top5_rows}</table>
  <p style="color:#bbb;font-size:11px;margin-top:20px;text-align:center">
    SecurityBERT v1.0 — IoT/IIoT Threat Detection + PPO Self-Healing
  </p>
</div></body></html>"""

        plain = (
            f'SecurityBERT Alert\n'
            f'Attack: {attack_class} ({sev})\n'
            f'Confidence: {confidence*100:.1f}%\n'
            f'Source: {source_ip}\n'
            f'Pi IP: {self._pi_ip}\n'
            f'Action: {action_name}\n'
            f'Time: {now}\n'
        )

        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = self.sender_email
        msg['To']      = self.receiver_email
        msg.attach(MIMEText(plain, 'plain'))
        msg.attach(MIMEText(html,  'html'))

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as s:
                s.starttls()
                s.login(self.sender_email, self.sender_password)
                s.sendmail(self.sender_email, self.receiver_email,
                           msg.as_string())
            self._last_alert[attack_class] = time.time()
            return True
        except Exception as e:
            return False