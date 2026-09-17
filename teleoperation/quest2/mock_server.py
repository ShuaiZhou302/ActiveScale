#!/usr/bin/env python3
"""
Dry-run stand-in for quest_server.py -- no ROS, no Pinocchio, no robot.

Listens like the real server (127.0.0.1 only, same FrameReader/protocol),
decodes each frame, applies the R_QUEST_TO_PIPER remap, and prints the
resulting hand poses + button state. Use this to validate the whole chain
(SteamVR -> quest_client.py -> SSH tunnel -> socket -> frame parsing ->
coordinate remap) end-to-end from Windows, before touching the real arms.

Run on the SAME machine as quest_client.py for a loopback test (no tunnel
needed), or through the real SSH tunnel to rehearse the full path if you
have any machine on the other end to run it on.

    python mock_server.py --port 8770
"""
import argparse
import socket
import time

import numpy as np

from protocol import FrameReader, DEFAULT_PORT, wire_to_pose

R_QUEST_TO_PIPER = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
], dtype=float)


def remap_to_piper(T_quest):
    T = np.eye(4)
    T[:3, 3] = R_QUEST_TO_PIPER @ T_quest[:3, 3]
    T[:3, :3] = R_QUEST_TO_PIPER @ T_quest[:3, :3] @ R_QUEST_TO_PIPER.T
    return T


def describe(name, d):
    if not d or not d.get("connected"):
        return "%-6s ---" % name
    if not d.get("valid"):
        return "%-6s invalid" % name
    T = remap_to_piper(wire_to_pose(d))
    x, y, z = T[:3, 3]
    s = "%-6s piper_xyz(%+.3f %+.3f %+.3f)" % (name, x, y, z)
    b = d.get("buttons")
    if b:
        s += " trig=%.2f grip=%.2f%s%s%s%s" % (
            b["trigger"], b["grip"],
            " GRIP" if b["grip_pressed"] else "",
            " AX" if b["button_ax"] else "",
            " BY" if b["button_by"] else "",
            " STICK" if b.get("stick_pressed") else "",
        )
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((args.host, args.port))
    s.listen(1)
    print("mock_server listening on %s:%d (Ctrl-C to stop)" % (args.host, args.port))

    while True:
        conn, addr = s.accept()
        print("client connected: %s" % (addr,))
        reader = FrameReader(conn)
        n = 0
        t_last_print = 0.0
        try:
            while True:
                frame = reader.read()
                if frame is None:
                    break
                n += 1
                now = time.monotonic()
                if now - t_last_print > 0.5:
                    print("seq=%-6d %s | %s | %s" % (
                        frame["seq"],
                        describe("head", frame.get("head")),
                        describe("left", frame.get("left")),
                        describe("right", frame.get("right")),
                    ))
                    t_last_print = now
        except (OSError, ValueError) as e:
            print("recv error: %s" % e)
        finally:
            conn.close()
            print("client disconnected (%d frames); waiting for reconnect" % n)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped.")
