"""
Wire protocol between quest_client.py (Windows, OpenVR/SteamVR) and
quest_server.py (robot computer, ROS + Pinocchio IK).

One length-prefixed JSON frame per sample, sent over a TCP socket that is
carried through an SSH -L tunnel (see scripts/start_tunnel.ps1). Two
independent connections share this framing:
  - pose/button stream (DEFAULT_PORT): browser -> quest_server.py, one-way.
  - camera stream (DEFAULT_CAMERA_PORT): camera_streamer.py -> browser, the
    OTHER way, on a SEPARATE port/connection on purpose -- sharing one
    connection would let a large image frame sit in front of a time-critical
    pose frame in the same send queue.
Neither direction ever replies on the other's socket.

Poses are shipped RAW in the OpenVR "standing" world frame (+Y up, +Z back,
right-handed) -- the SAME convention documented in avp/Readme.md section 9
for the Apple Vision Pro / WebXR world frame. That means the existing
R_AVP_TO_PIPER-style axis remap applies unchanged; only the matrix name
differs (R_QUEST_TO_PIPER in quest_server.py). All frame math and all safety
policy (clutch, staleness watchdog, e-stop ramp-home) live server-side --
the client is a dumb sensor relay and must stay that way so the robot's
safety behavior does not depend on what's running on the Windows box.

Python 3.8 compatible on purpose: quest_server.py runs inside the `aloha`
robot-side Python environment. No walrus-free requirement, but
avoid PEP 604 / PEP 585 syntax (no `int | None`, no `list[int]`).
"""
import json
import struct

PROTOCOL_VERSION = 1
DEFAULT_PORT = 8770         # pose/button stream: browser -> quest_server.py
DEFAULT_CAMERA_PORT = 8771  # camera stream: camera_streamer.py -> browser (opposite
                             # direction, deliberately a SEPARATE TCP connection/port
                             # so a large image frame can never sit in front of a
                             # time-critical pose frame -- see camera_streamer.py)

_HEADER = struct.Struct("!I")  # 4-byte big-endian length prefix
MAX_FRAME_BYTES = 4 << 20      # 4 MiB sanity cap -- pose frames are ~1 KB, camera
                                # frames (3x base64 JPEG) are the ones that need headroom


def encode(msg):
    """dict -> length-prefixed JSON bytes, ready for sock.sendall()."""
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    return _HEADER.pack(len(body)) + body


class FrameReader(object):
    """Incremental length-prefixed JSON frame reader over a blocking socket.

    Call read() in a loop; it blocks (subject to the socket's own timeout)
    until a full frame arrives, and returns None on clean EOF (peer closed).
    Raises socket.timeout / OSError same as a raw recv() would.

    Explicitly tracks header-vs-body parsing state across calls so that a
    socket.timeout firing mid-body (not just mid-header) can't get the
    stream out of sync: read() is safe to call again after a timeout and
    resumes exactly where it left off, it will never reinterpret already
    -buffered body bytes as the next frame's length header.
    """

    def __init__(self, sock):
        self.sock = sock
        self._buf = b""
        self._stage = "header"  # "header" until length is parsed, then "body"
        self._body_len = 0

    def _fill(self, n):
        """Accumulate at least n bytes in self._buf. Returns False on clean EOF.
        May raise socket.timeout/OSError; already-buffered bytes are kept."""
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                return False
            self._buf += chunk
        return True

    def read(self):
        if self._stage == "header":
            if not self._fill(4):
                return None
            header, self._buf = self._buf[:4], self._buf[4:]
            (self._body_len,) = _HEADER.unpack(header)
            if self._body_len > MAX_FRAME_BYTES:
                raise ValueError("frame too large: %d bytes" % self._body_len)
            self._stage = "body"

        if not self._fill(self._body_len):
            return None
        body, self._buf = self._buf[:self._body_len], self._buf[self._body_len:]
        self._stage = "header"
        return json.loads(body.decode("utf-8"))


def pose_to_wire(mat4):
    """4x4 numpy homogeneous transform -> {'pos': [x,y,z], 'rot': [9 floats, row-major]}."""
    return {
        "pos": [float(mat4[0, 3]), float(mat4[1, 3]), float(mat4[2, 3])],
        "rot": [float(v) for v in mat4[:3, :3].flatten(order="C")],
    }


def wire_to_pose(d):
    """Inverse of pose_to_wire -> 4x4 numpy homogeneous transform."""
    import numpy as np
    T = np.eye(4)
    T[:3, 3] = d["pos"]
    T[:3, :3] = np.array(d["rot"], dtype=float).reshape(3, 3, order="C")
    return T


def device_to_wire(dev, with_buttons):
    """QuestPoseReader per-device dict -> wire dict, or {'connected': False}."""
    if dev is None:
        return {"connected": False}
    out = {
        "connected": True,
        "valid": bool(dev["valid"]),
        "tracking_ok": bool(dev["tracking_ok"]),
    }
    out.update(pose_to_wire(dev["matrix"]))
    if with_buttons and dev.get("buttons"):
        b = dev["buttons"]
        out["buttons"] = {
            "trigger": float(b["trigger"]),
            "grip": float(b["grip"]),
            "grip_pressed": bool(b["grip_pressed"]),
            "trigger_pressed": bool(b["trigger_pressed"]),
            "button_ax": bool(b["button_ax"]),
            "button_by": bool(b["button_by"]),
            # Thumbstick click. Physically separate from button_ax/button_by
            # (X/Y/A/B) on purpose -- quest_server.py uses button_ax/button_by
            # for base drive and needs an unrelated input for panic/e-stop.
            "stick_pressed": bool(b.get("stick_pressed", False)),
        }
    return out


def make_sample(seq, t_client_monotonic, t_client_wall, head, left, right):
    return {
        "v": PROTOCOL_VERSION,
        "seq": seq,
        "t_mono": t_client_monotonic,
        "t_wall": t_client_wall,
        "head": device_to_wire(head, with_buttons=False),
        "left": device_to_wire(left, with_buttons=True),
        "right": device_to_wire(right, with_buttons=True),
    }
