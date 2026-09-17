#!/usr/bin/env python3
"""
Camera passthrough for in-VR visual feedback. Run it on the robot alongside
quest_server.py as a separate process.

Subscribes to the 3 arm-mounted camera topics (same naming convention as
eef_avp_control_singlearm.py: /camera_l, /camera_f (mid/front), /camera_r),
downsamples + JPEG-encodes each at a low, configurable rate, and serves them
over their own TCP port to webxr_server.py, which relays them into the
headset's WebXR page as textured panels.

Deliberately a standalone process, not folded into quest_server.py: image
decode/resize/JPEG-encode is real CPU work with variable timing, and
quest_server.py's 50 Hz arm control loop must never share a thread/process
with anything that could jitter it. Also deliberately a SEPARATE TCP
connection/port from the pose stream (see protocol.py's module docstring)
for the same reason, one level up the stack.

This is read-only sensor passthrough -- it has no safety-relevant state and
no watchdog of its own beyond "keep retrying the TCP connection"; losing the
camera feed does not affect quest_server.py's ability to freeze/ramp-home
the arms, only the operator's visibility while teleoperating.

Run in a separate terminal from quest_server.py.
"""
import argparse
import base64
import io
import os
import socket
import sys
import threading
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from protocol import encode, DEFAULT_CAMERA_PORT  # noqa: E402


def _load_ros_dependencies():
    """Load ROS after argparse has handled ``--help``."""
    global rospy, ImageMsg
    try:
        import rospy as rospy_module
        from sensor_msgs.msg import Image as image_message
    except ImportError as error:
        raise SystemExit(
            "Missing ROS dependency. Run inside the robot ROS environment described "
            f"in teleoperation/quest2/README.md. Original error: {error}"
        ) from error
    rospy = rospy_module
    ImageMsg = image_message

# Matches ARM_CAM_SHORT / camera_topic defaults in eef_avp_control_singlearm.py.
DEFAULT_TOPICS = {
    "left": "/camera_l/color/image_raw",
    "mid": "/camera_f/color/image_raw",
    "right": "/camera_r/color/image_raw",
}


def decode_ros_image(msg):
    """ROS Image (rgb8/bgr8) -> RGB numpy array, or None on unsupported encoding.
    Same manual decode as eef_avp_control_singlearm.py's _decode_image (no
    cv_bridge dependency)."""
    if msg.encoding == "rgb8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)[:, :, ::-1]
    return None


class CameraChannel(object):
    def __init__(self, name, topic, max_size, jpeg_quality):
        self.name = name
        self.max_size = max_size
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._latest_arr = None
        rospy.Subscriber(topic, ImageMsg, self._cb, queue_size=1, buff_size=2 ** 24)

    def _cb(self, msg):
        arr = decode_ros_image(msg)
        if arr is None:
            return
        with self._lock:
            self._latest_arr = arr

    def latest_jpeg_b64(self):
        """Encode whatever the latest received frame is, or None if nothing yet."""
        with self._lock:
            arr = self._latest_arr
        if arr is None:
            return None
        img = Image.fromarray(arr)
        img.thumbnail(self.max_size, Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")


class CameraStreamServer(object):
    def __init__(self, args):
        self.args = args
        rospy.init_node("quest_camera_streamer", anonymous=True)
        self.channels = {
            name: CameraChannel(name, topic, (args.max_w, args.max_h), args.jpeg_quality)
            for name, topic in (
                ("left", args.left_topic), ("mid", args.mid_topic), ("right", args.right_topic),
            )
        }
        self._server_sock = None
        self._stop = False

    def _accept_loop(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.args.host, self.args.port))
        s.listen(1)
        self._server_sock = s
        print("[cam] listening on %s:%d" % (self.args.host, self.args.port))
        while not self._stop:
            try:
                conn, addr = s.accept()
            except OSError:
                break
            print("[cam] client connected: %s" % (addr,))
            conn.settimeout(2.0)
            self._send_loop(conn)
            conn.close()
            print("[cam] client disconnected; waiting for reconnect")

    def _send_loop(self, conn):
        period = 1.0 / self.args.rate
        while not self._stop:
            t0 = time.monotonic()
            frame = {}
            for name, chan in self.channels.items():
                b64 = chan.latest_jpeg_b64()
                if b64 is not None:
                    frame[name] = b64
            if frame:
                try:
                    conn.sendall(encode(frame))
                except OSError as e:
                    print("[cam] send failed (%s)" % e)
                    return
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def run(self):
        print("=" * 60)
        print("Quest camera streamer")
        print("  listen       = %s:%d" % (self.args.host, self.args.port))
        print("  rate         = %.1f Hz" % self.args.rate)
        print("  max size     = %dx%d, JPEG q=%d" % (self.args.max_w, self.args.max_h, self.args.jpeg_quality))
        for name, chan in self.channels.items():
            print("  %-5s topic  = %s" % (name, self.args.__dict__["%s_topic" % name]))
        print("=" * 60)
        try:
            self._accept_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop = True
            if self._server_sock is not None:
                try:
                    self._server_sock.close()
                except OSError:
                    pass


def get_args():
    p = argparse.ArgumentParser(description="Stream arm cameras to the Quest headset (robot side)")
    p.add_argument("--host", default="127.0.0.1",
                    help="Bind address. Keep this 127.0.0.1 -- reachability is meant to "
                         "come ONLY from the SSH -L tunnel, not the open network.")
    p.add_argument("--port", type=int, default=DEFAULT_CAMERA_PORT)
    p.add_argument("--rate", type=float, default=5.0,
                    help="Send rate in Hz. Kept low on purpose -- this is for situational "
                         "awareness while teleoperating, not a vision pipeline.")
    p.add_argument("--max_w", type=int, default=320)
    p.add_argument("--max_h", type=int, default=240)
    p.add_argument("--jpeg_quality", type=int, default=55)
    p.add_argument("--left_topic", default=DEFAULT_TOPICS["left"])
    p.add_argument("--mid_topic", default=DEFAULT_TOPICS["mid"])
    p.add_argument("--right_topic", default=DEFAULT_TOPICS["right"])
    return p.parse_args()


def main():
    args = get_args()
    _load_ros_dependencies()
    CameraStreamServer(args).run()


if __name__ == "__main__":
    main()
