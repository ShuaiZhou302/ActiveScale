#!/usr/bin/env python3
"""
WebXR bridge (Windows side): serves index.html over HTTPS+WSS so the Quest 2's
own browser can connect directly over LAN WiFi, then relays each incoming
pose/button sample straight through to quest_server.py on the robot over
the SSH -L tunnel, re-using the exact same length-prefixed wire protocol
quest_client.py used for the OpenVR path (see protocol.py) -- so
quest_server.py itself needs ZERO changes for this transport switch.

Also relays the OTHER direction: camera_streamer.py's JPEG frames (a
SEPARATE TCP connection/port from the pose stream on purpose, see
protocol.py) get broadcast to every connected browser WebSocket as-is, for
in-headset visual feedback while teleoperating (see CameraBridge). Status
frames (e.g. per-arm IK failure) that quest_server.py writes back on the
pose connection itself are relayed the same way (see RobotBridge).

Why this exists instead of reusing avp/tele_vision.py's Vuer wrapper: the
pinned vuer==0.0.31rc7's Python schema has a `Gamepads` scene component, but
the bundled client_build JS does NOT implement it (confirmed by grepping the
installed package -- every "gamepad" hit is Three.js's generic XR controller
-model rendering code, not an actual button/axis event emitter). Quest 2
Touch controllers are physical controllers, not AVP-style hand-tracking, so
we need real Gamepad button/axis data and Vuer's Hands component won't give
it to us. Rather than fight a half-implemented, version-pinned dependency,
this is a small hand-written WebXR page using the standard
`XRInputSource.gamepad` API directly (see index.html).

This process is a dumb relay, same philosophy as quest_client.py: it does
NOT parse pose data, apply any transform, or make any safety decision. It
only re-frames whatever JSON the browser sends as a length-prefixed message
on the outbound TCP socket. All of that logic stays in quest_server.py.

Run:
    1. Windows and the Quest 2 headset on the SAME WiFi network.
    2. python webxr_server.py
    3. In a separate terminal, keep the SSH tunnel alive:
         powershell ../scripts/start_tunnel.ps1
    4. On the Quest 2, open the Meta Quest Browser to:
         https://<this PC's LAN IP>:8443
       (first launch: accept the mkcert-issued cert warning if the Quest
       hasn't trusted the local CA -- see ../README.md)
    5. Tap "Enter VR".
"""
import argparse
import asyncio
import json
import os
import ssl
import struct
import sys
import time

import websockets
from websockets.asyncio.server import serve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from protocol import encode, DEFAULT_PORT, DEFAULT_CAMERA_PORT, MAX_FRAME_BYTES  # noqa: E402

PAGE_PATH = os.path.join(HERE, "index.html")
CERT = os.path.join(HERE, "cert.pem")
KEY = os.path.join(HERE, "key.pem")


class RobotBridge:
    """Best-effort TCP relay to quest_server.py, reconnecting in the background.

    A dropped or not-yet-established connection is not an error from this
    class's point of view -- send() just drops the sample. quest_server.py's
    own staleness watchdog is what keeps the robot safe when that happens;
    this class's job is only to reconnect promptly, not to buffer/retry.

    Uses asyncio streams (not a blocking socket) so a slow/failing connect
    attempt never stalls the HTTPS/WSS server's event loop -- start() only
    schedules the background reconnect loop, it does not await a connection.

    Also reads status frames quest_server.py writes back on this SAME
    connection (e.g. per-arm IK failure -- see quest_server.py's
    _send_status()) and relays them to every connected browser WebSocket,
    same as CameraBridge does for camera frames. The connection is
    browser-initiated (this class connects out to quest_server.py) but not
    one-way at the byte level; only the pose stream direction is fire-and-
    forget.
    """

    def __init__(self, host, port, clients, reconnect_delay=1.0):
        self.host = host
        self.port = port
        self.clients = clients  # set of currently-open websocket connections
        self.reconnect_delay = reconnect_delay
        self.writer = None
        self._task = None

    def start(self):
        # asyncio only holds a WEAK reference to a bare ensure_future() task --
        # with nothing else referencing it, it can be garbage-collected mid-run
        # (observed live: "Task was destroyed but it is pending!"). Keep a
        # strong reference on self for the process lifetime.
        self._task = asyncio.ensure_future(self._run())

    async def _run(self):
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.host, self.port)
            except OSError as e:
                print("[bridge] connect failed (%s); retrying in %.1fs" % (e, self.reconnect_delay))
                await asyncio.sleep(self.reconnect_delay)
                continue
            self.writer = writer
            print("[bridge] connected to %s:%d" % (self.host, self.port))
            try:
                while True:
                    frame = await _read_frame(reader)
                    if frame is None:
                        break
                    msg = json.dumps(frame)
                    for ws in list(self.clients):
                        try:
                            await ws.send(msg)
                        except Exception:
                            self.clients.discard(ws)
            except (OSError, ValueError) as e:
                print("[bridge] recv error: %s" % e)
            self.writer = None
            print("[bridge] disconnected from %s:%d; reconnecting" % (self.host, self.port))
            await asyncio.sleep(self.reconnect_delay)

    def send(self, data):
        writer = self.writer
        if writer is None:
            return
        try:
            writer.write(data)
        except OSError:
            self.writer = None


_CAM_HEADER = struct.Struct("!I")


async def _read_frame(reader):
    """Async counterpart of protocol.FrameReader, for asyncio streams. Returns
    None on clean EOF (peer closed)."""
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    (length,) = _CAM_HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ValueError("frame too large: %d bytes" % length)
    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        return None
    return json.loads(body.decode("utf-8"))


class CameraBridge:
    """Reads camera frames from camera_streamer.py (robot computer) and
    broadcasts each one to every currently connected browser WebSocket.

    Opposite data direction from RobotBridge (robot -> browser, not browser
    -> robot) but the same reconnect philosophy: a dropped/absent connection
    just means no camera frames arrive, it is never treated as fatal.
    """

    def __init__(self, host, port, clients, reconnect_delay=1.0):
        self.host = host
        self.port = port
        self.clients = clients  # set of currently-open websocket connections
        self.reconnect_delay = reconnect_delay
        self._task = None

    def start(self):
        # See RobotBridge.start()'s comment -- same GC pitfall, same fix.
        self._task = asyncio.ensure_future(self._run())

    async def _run(self):
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.host, self.port)
            except OSError as e:
                print("[cam] connect failed (%s); retrying in %.1fs" % (e, self.reconnect_delay))
                await asyncio.sleep(self.reconnect_delay)
                continue
            print("[cam] connected to %s:%d" % (self.host, self.port))
            try:
                while True:
                    frame = await _read_frame(reader)
                    if frame is None:
                        break
                    frame["type"] = "camera"
                    msg = json.dumps(frame)
                    for ws in list(self.clients):
                        try:
                            await ws.send(msg)
                        except Exception:
                            self.clients.discard(ws)
            except (OSError, ValueError) as e:
                print("[cam] recv error: %s" % e)
            writer.close()
            print("[cam] disconnected from %s:%d; reconnecting" % (self.host, self.port))
            await asyncio.sleep(self.reconnect_delay)


def make_process_request(page_bytes):
    def process_request(connection, request):
        if request.path in ("/", "/index.html"):
            response = connection.respond(200, page_bytes.decode("utf-8"))
            # respond() defaults to text/plain; Headers.__setitem__ APPENDS
            # rather than replaces (HTTP headers can be multi-valued), so
            # del the default first or the client sees a malformed
            # "text/plain; charset=utf-8,text/html; charset=utf-8" value.
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response
        return None  # anything else (expected: /ws) falls through to the WS handshake
    return process_request


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0", help="Bind address for the HTTPS/WSS server.")
    ap.add_argument("--port", type=int, default=8443, help="HTTPS/WSS port the Quest browser connects to.")
    ap.add_argument("--robot_host", default="127.0.0.1",
                     help="Where quest_server.py is reachable, via the SSH -L tunnel.")
    ap.add_argument("--robot_port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--camera_port", type=int, default=DEFAULT_CAMERA_PORT,
                     help="Where camera_streamer.py is reachable, via the SSH -L tunnel.")
    args = ap.parse_args()

    if not (os.path.isfile(CERT) and os.path.isfile(KEY)):
        raise SystemExit("cert.pem/key.pem not found next to this script; run mkcert first (see ../README.md).")
    with open(PAGE_PATH, "rb") as f:
        page_bytes = f.read()

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(CERT, KEY)

    clients = set()

    bridge = RobotBridge(args.robot_host, args.robot_port, clients)
    bridge.start()

    camera_bridge = CameraBridge(args.robot_host, args.camera_port, clients)
    camera_bridge.start()

    n_frames = 0
    t_last_status = 0.0

    async def handler(websocket):
        nonlocal n_frames, t_last_status
        print("[ws] client connected: %s" % (websocket.remote_address,))
        clients.add(websocket)
        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                bridge.send(encode(msg))
                n_frames += 1
                now = time.monotonic()
                if now - t_last_status > 1.0:
                    print("[ws] seq=%s frames=%d" % (msg.get("seq"), n_frames))
                    t_last_status = now
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            clients.discard(websocket)
            print("[ws] client disconnected")

    process_request = make_process_request(page_bytes)
    async with serve(handler, args.host, args.port, ssl=ssl_context, process_request=process_request):
        print("=" * 60)
        print("Quest WebXR bridge")
        print("  serving  https://<this PC's LAN IP>:%d" % args.port)
        print("  relaying pose to robot at %s:%d (expects an SSH -L tunnel)" % (args.robot_host, args.robot_port))
        print("  relaying camera from robot at %s:%d" % (args.robot_host, args.camera_port))
        print("=" * 60)
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped.")
