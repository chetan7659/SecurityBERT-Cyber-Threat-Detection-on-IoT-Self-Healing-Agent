"""
SelfHealingManager — executes healing actions on Raspberry Pi.
Ported from NB14 with SIMULATE=False for real Pi execution.
"""

import subprocess
import logging
import time
from typing     import Dict, Optional
from collections import defaultdict
from datetime   import datetime

logger = logging.getLogger(__name__)


class ActionResult:
    def __init__(self, action_id, action_name, attack_class,
                 source_ip, success, message,
                 command='', rollback_cmd=''):
        self.action_id    = action_id
        self.action_name  = action_name
        self.attack_class = attack_class
        self.source_ip    = source_ip
        self.success      = success
        self.message      = message
        self.command      = command
        self.rollback_cmd = rollback_cmd
        self.timestamp    = datetime.now().isoformat()

    def to_dict(self):
        return self.__dict__


def _run_cmd(cmd: str, simulate: bool = False):
    if simulate:
        return True, f'[SIM] {cmd}'
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=10)
        return r.returncode == 0, r.stdout + r.stderr
    except Exception as e:
        return False, str(e)


class SelfHealingManager:
    """
    Executes healing actions on Raspberry Pi.
    Set simulate=False for real iptables execution.
    """

    SERVICE_MAP = {
        'Backdoor'             : 'ssh',
        'Password'             : 'ssh',
        'Uploading'            : 'apache2',
        'Vulnerability_scanner': 'apache2',
        'SQL_injection'        : 'mysql',
        'XSS'                  : 'apache2',
    }

    def __init__(self, simulate: bool = False, heal_log: str = '/tmp/heal.log'):
        self.simulate      = simulate
        self.action_log    = []
        self.action_counts = defaultdict(int)
        self._heal_log     = open(heal_log, 'a', buffering=1)

    def execute(
        self,
        action_id   : int,
        attack_class: str,
        confidence  : float,
        source_ip   : str = '0.0.0.0',
        all_probs   : Optional[object] = None,
    ) -> ActionResult:

        dispatch = {
            0: self._block_ip,
            1: self._reset_connection,
            2: self._restart_service,
            3: self._isolate_device,
            4: self._log_and_alert,
        }
        fn     = dispatch.get(action_id, self._log_and_alert)
        result = fn(source_ip, attack_class, confidence)

        # Log
        self.action_log.append(result.to_dict())
        self.action_counts[result.action_name] += 1

        ts  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ok  = '✅' if result.success else '❌'
        log = (f'[{ts}] {ok} {result.action_name:20s} | '
               f'{attack_class:25s} | IP={source_ip} | '
               f'conf={confidence:.2f}')
        logger.info(log)
        self._heal_log.write(log + '\n')

        return result

    def rollback_last(self):
        """Undo the most recent blocking action."""
        for r in reversed(self.action_log):
            if r.get('rollback_cmd'):
                ok, msg = _run_cmd(r['rollback_cmd'], self.simulate)
                logger.info(f'Rollback {"✅" if ok else "❌"}: {r["rollback_cmd"]}')
                return ok
        return False

    def _block_ip(self, ip, cls, conf) -> ActionResult:
        cmd = f'sudo iptables -A INPUT -s {ip} -j DROP'
        rb  = f'sudo iptables -D INPUT -s {ip} -j DROP'
        ok, _ = _run_cmd(cmd, self.simulate)
        return ActionResult(0,'BLOCK_IP', cls, ip, ok,
                            f'Blocked {ip}', cmd, rb)

    def _reset_connection(self, ip, cls, conf) -> ActionResult:
        cmd   = f'sudo ss -K dst {ip}'
        ok, _ = _run_cmd(cmd, self.simulate)
        return ActionResult(1,'RESET_CONNECTION', cls, ip, ok,
                            f'Reset conn {ip}', cmd, '')

    def _restart_service(self, ip, cls, conf) -> ActionResult:
        svc   = self.SERVICE_MAP.get(cls, 'networking')
        cmd   = f'sudo systemctl restart {svc}'
        ok, _ = _run_cmd(cmd, self.simulate)
        return ActionResult(2,'RESTART_SERVICE', cls, ip, ok,
                            f'Restarted {svc}', cmd, cmd)

    def _isolate_device(self, ip, cls, conf) -> ActionResult:
        cmd_i = f'sudo iptables -I INPUT  -s {ip} -j DROP'
        cmd_o = f'sudo iptables -I OUTPUT -d {ip} -j DROP'
        rb_i  = f'sudo iptables -D INPUT  -s {ip} -j DROP'
        rb_o  = f'sudo iptables -D OUTPUT -d {ip} -j DROP'
        ok1,_ = _run_cmd(cmd_i, self.simulate)
        ok2,_ = _run_cmd(cmd_o, self.simulate)
        return ActionResult(3,'ISOLATE_DEVICE', cls, ip, ok1 and ok2,
                            f'Isolated {ip}',
                            f'{cmd_i} && {cmd_o}',
                            f'{rb_i} && {rb_o}')

    def _log_and_alert(self, ip, cls, conf) -> ActionResult:
        return ActionResult(4,'LOG_AND_ALERT', cls, ip, True,
                            f'Logged {cls} from {ip} conf={conf:.2f}',
                            'log', '')