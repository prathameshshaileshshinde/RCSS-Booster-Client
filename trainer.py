"""
trainer.py
----------
Monitor client that connects to RCSSServerMJ on the monitor port (60001) and
sends a kickoff command so the game transitions from BeforeKickOff automatically.

This means you never have to touch the GUI or manually trigger kickoff during
testing — just run this alongside start_team.py.

Usage:
    python trainer.py                        # auto-kickoff after --delay seconds
    python trainer.py --side Right           # kick off for the right team
    python trainer.py --delay 5              # wait 5 s before sending kickoff
    python trainer.py --interactive          # press Enter to send kickoff manually

Monitor protocol (RCSSServerMJ v0.2):
  - TCP port 60001 (same framing as agent port: 4-byte big-endian length + payload)
  - Commands are S-expressions; the server sends state each cycle
  - Kickoff command format: (kickOff <side>)  where side is Left or Right
    NOTE: the monitor protocol is marked TODO in the official docs.
    If (kickOff Left) doesn't work, try: (playMode KickOff_Left)
    Adjust KICKOFF_COMMANDS below and the server will log what it receives.
"""

import argparse
import re
import signal
import socket
import sys
import threading
import time
from types import FrameType


# ── Commands to try (in order) until one works ──────────────────────────────
# The monitor protocol is undocumented; these are based on rcssserver3d conventions.
# Extend this list if neither works — check the server log for accepted formats.
def _kickoff_commands(side: str) -> list[bytes]:
    s = side.capitalize()          # 'Left' or 'Right'
    return [
        f'(kickOff {s})'.encode(),
        f'(playMode KickOff_{s})'.encode(),
    ]


def _send(sock: socket.socket, payload: bytes) -> None:
    sock.send(len(payload).to_bytes(4, 'big') + payload)


def _recv(sock: socket.socket, buf: bytearray) -> bytes | None:
    """Read one length-prefixed message. Returns None on disconnect."""
    try:
        n = sock.recv_into(buf, nbytes=4, flags=socket.MSG_WAITALL)
    except OSError:
        return None
    if n != 4:
        return None
    size = int.from_bytes(buf[:4], 'big')
    if size > len(buf):
        buf[:] = bytearray(size)  # grow buffer
    try:
        n = sock.recv_into(buf, nbytes=size, flags=socket.MSG_WAITALL)
    except OSError:
        return None
    if n != size:
        return None
    return bytes(buf[:size])


def _parse_play_mode(msg: bytes) -> str | None:
    """Extract play mode from a GS S-expression, e.g. (GS (t 0.0) (pm BeforeKickOff) ...)"""
    m = re.search(rb'\(pm\s+(\w+)\)', msg)
    return m.group(1).decode() if m else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description='RCSSServerMJ trainer: auto-triggers game kickoff via the monitor port.')
    parser.add_argument('-s', '--host',        default='127.0.0.1',
                        help='Server host (default: 127.0.0.1)')
    parser.add_argument('-p', '--monitor-port', type=int, default=60001,
                        help='Monitor port (default: 60001)')
    parser.add_argument('--side',              default='Left', choices=['Left', 'Right'],
                        help='Which team kicks off (default: Left)')
    parser.add_argument('--delay',             type=float, default=3.0,
                        help='Seconds to wait after BeforeKickOff is detected before kicking off. '
                             'Ignored when --count is set (default: 3.0)')
    parser.add_argument('--count',             type=int, default=0,
                        help='Number of robots that must reach formation before kickoff. '
                             'When set, waits for this many entries in --ready-file instead of '
                             'using --delay. Match the value passed to start_team.py --count.')
    parser.add_argument('--ready-file',        type=str, default='formation_ready.txt',
                        help='Shared file agents write to when they reach formation '
                             '(default: formation_ready.txt)')
    parser.add_argument('--interactive', action='store_true',
                        help='Wait for Enter keypress instead of using --delay')
    args = parser.parse_args()

    # ── clear the ready file so stale entries from a previous run don't trigger early ──
    if args.count > 0:
        try:
            open(args.ready_file, 'w').close()
            print(f'[trainer] Cleared ready file "{args.ready_file}" — '
                  f'waiting for {args.count} robot(s) to reach formation.')
        except OSError as e:
            print(f'[trainer] WARNING: could not clear ready file: {e}')

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    print(f'[trainer] Connecting to monitor port {args.host}:{args.monitor_port} …')
    try:
        sock.connect((args.host, args.monitor_port))
    except ConnectionRefusedError:
        print(f'[trainer] ERROR: connection refused — is the server running?')
        sys.exit(1)
    print('[trainer] Connected.')

    buf = bytearray(4096)
    kicked_off = False
    saw_before_kickoff = False
    kickoff_at: float | None = None   # wall-clock time to fire kickoff

    def _shutdown(sig: int, frame: FrameType | None) -> None:
        print('\n[trainer] Shutting down.')
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    def _ready_count() -> int:
        """Count unique player numbers that have written to the ready file."""
        try:
            with open(args.ready_file) as f:
                lines = [l.strip() for l in f if l.strip()]
            return len(set(lines))
        except OSError:
            return 0

    # ── interactive mode: wait for Enter in a background thread ──────────────
    if args.interactive:
        print('[trainer] Interactive mode — press Enter to send kickoff.')

        def _wait_enter() -> None:
            input()
            nonlocal kickoff_at
            kickoff_at = time.monotonic()   # fire immediately

        threading.Thread(target=_wait_enter, daemon=True).start()

    # ── main receive loop ─────────────────────────────────────────────────────
    while True:
        msg = _recv(sock, buf)
        if msg is None:
            print('[trainer] Server disconnected.')
            break

        pm = _parse_play_mode(msg)
        if pm:
            if pm == 'BeforeKickOff' and not saw_before_kickoff:
                saw_before_kickoff = True
                if args.interactive:
                    print(f'[trainer] BeforeKickOff detected — press Enter to kick off.')
                elif args.count > 0:
                    print(f'[trainer] BeforeKickOff detected — waiting for '
                          f'{args.count} robot(s) to reach formation …')
                else:
                    kickoff_at = time.monotonic() + args.delay
                    print(f'[trainer] BeforeKickOff detected — kicking off in {args.delay:.1f} s '
                          f'({args.side} team).')

            if pm != 'BeforeKickOff' and not kicked_off:
                # Someone else already changed the play mode (e.g. GUI) — nothing to do
                print(f'[trainer] Play mode is already "{pm}" — no kickoff needed.')
                kicked_off = True

        # ── ready-file mode: trigger when all robots have arrived ─────────────
        if (args.count > 0 and saw_before_kickoff and not kicked_off
                and kickoff_at is None):
            n_ready = _ready_count()
            if n_ready >= args.count:
                print(f'[trainer] All {n_ready}/{args.count} robot(s) at formation — kicking off!')
                kickoff_at = time.monotonic()   # fire immediately

        # Fire kickoff when the timer expires
        if kickoff_at is not None and time.monotonic() >= kickoff_at and not kicked_off:
            kicked_off = True
            cmds = _kickoff_commands(args.side)
            for cmd in cmds:
                print(f'[trainer] Sending: {cmd.decode()}')
                _send(sock, cmd)
                time.sleep(0.1)   # brief gap between attempts
            print('[trainer] Kickoff command(s) sent. Watching for play mode change …')

        if kicked_off and pm and pm != 'BeforeKickOff':
            print(f'[trainer] Game is now in "{pm}" — done.')
            break

    sock.close()
    print('[trainer] Exiting.')


if __name__ == '__main__':
    main()
