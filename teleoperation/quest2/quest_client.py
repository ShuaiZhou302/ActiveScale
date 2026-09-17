#!/usr/bin/env python3
"""
Quest 2 pose relay (Windows side).

Reads HMD + left/right controller poses from SteamVR and streams them, RAW
(no axis remap, no clutch/safety logic -- that all lives on the robot),
over a TCP socket to quest_server.py running on the robot computer. Meant to be
reached through an SSH -L tunnel, so the default target is 127.0.0.1.

This process is a dumb, stateless sensor relay by design: if it crashes or
the network drops, the robot side notices via its own staleness watchdog
and freezes / ramps home on its own -- it does not depend on anything this
script decides.

Run:
    1. Meta Horizon app running, headset connected via Link/Air Link, SteamVR up.
    2. In a separate terminal, keep the SSH tunnel alive:
         powershell scripts/start_tunnel.ps1
    3. python quest_client.py
"""
import argparse
import socket
import time

from openvr_reader import QuestPoseReader
from protocol import encode, make_sample, DEFAULT_PORT


def connect(host, port, timeout=5.0):
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(None)  # sends are blocking; we don't read from this socket
    return sock


def main():
    ap = argparse.ArgumentParser(description="Stream Quest 2 poses to quest_server.py")
    ap.add_argument("--host", default="127.0.0.1",
                     help="Target host. Default 127.0.0.1 assumes an SSH -L tunnel "
                          "(see scripts/start_tunnel.ps1) forwards this to the robot.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--rate", type=float, default=50.0, help="Send rate in Hz.")
    ap.add_argument("--reconnect_delay", type=float, default=1.0,
                     help="Seconds to wait between reconnect attempts.")
    args = ap.parse_args()

    period = 1.0 / args.rate
    seq = 0
    print("Connecting to SteamVR ...")
    with QuestPoseReader() as reader:
        print("SteamVR connected.")
        sock = None
        last_status = 0.0
        sent_since_status = 0
        while True:
            if sock is None:
                try:
                    print("Connecting to %s:%d ..." % (args.host, args.port))
                    sock = connect(args.host, args.port)
                    print("Connected.")
                except OSError as e:
                    print("Connect failed (%s); retrying in %.1fs" % (e, args.reconnect_delay))
                    time.sleep(args.reconnect_delay)
                    continue

            t0 = time.monotonic()
            s = reader.get()
            msg = make_sample(seq, time.monotonic(), time.time(),
                               s["head"], s["left"], s["right"])
            seq += 1
            try:
                sock.sendall(encode(msg))
                sent_since_status += 1
            except OSError as e:
                print("Send failed (%s); reconnecting." % e)
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
                continue

            now = time.monotonic()
            if now - last_status > 1.0:
                l_ok = s["left"] is not None and s["left"]["valid"]
                r_ok = s["right"] is not None and s["right"]["valid"]
                h_ok = s["head"] is not None and s["head"]["valid"]
                print("seq=%d  hz=%.1f  head=%s left=%s right=%s" % (
                    seq, sent_since_status / (now - last_status),
                    "OK" if h_ok else "--",
                    "OK" if l_ok else "--",
                    "OK" if r_ok else "--",
                ))
                last_status = now
                sent_since_status = 0

            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped.")
