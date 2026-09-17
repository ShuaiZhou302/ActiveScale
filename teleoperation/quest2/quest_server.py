#!/usr/bin/env python3
"""
Quest 2 tri-arm + base teleop (robot side).

Receives raw HMD + left/right controller poses from either quest_client.py
(Windows, OpenVR) or webxr_server.py (Windows, WebXR bridge) over a TCP
socket reached through an SSH -L tunnel -- this process only ever binds
127.0.0.1, it is not meant to be reachable except through that tunnel.

Mapping:
    head            -> MID arm end-effector  (same delta-pose-composition
                        math as eef_avp_control_singlearm.py's head->arm)
    left controller  -> LEFT arm end-effector, trigger -> gripper
    right controller -> RIGHT arm end-effector, trigger -> gripper
    left  X/Y (button_ax/button_by)  -> base turn left / turn right
    right A/B (button_ax/button_by)  -> base backward / forward
    thumbstick click (either hand)   -> panic: all 3 arms ramp home, base stops

Safety model (all enforced here, not on the Windows side -- the network
sensor relay is not trusted):
  - Clutch: pressing both grips together ONCE toggles all three arms
    engaged, latching the current head + left + right poses as the
    delta-tracking origins for all three at once, so none of them starts
    moving before the others. This is a TOGGLE, not hold-to-track --
    letting go of one/both grips after engaging does nothing by itself;
    pressing both grips together again disengages (freezes all three in
    place, holding last commanded joints, no drift, no auto-return).
    Re-engaging always relocks to the hands'/head's pose at that instant;
    small drift between engagements is expected, same as the AVP script's
    per-episode anchoring. Staleness/panic always force a disengage
    regardless of toggle state (see below) -- the toggle can only be
    trusted to hold state while the link is fresh.
  - Base drive (X/Y/A/B) is independent of the arm clutch -- squeezing grip
    (thumb-independent finger) doesn't block pressing X/Y/A/B with the
    thumb, so driving the base while the arms track is possible on purpose.
  - Staleness watchdog: if no fresh packet arrives within --stale_freeze_sec,
    all three arms are forced to the frozen (disengaged) state and the base
    is stopped, regardless of the last button reading (stale button state
    cannot be trusted).
  - E-stop ramp-home: if no fresh packet arrives within --stale_estop_sec
    (network dead, client crashed, ...), all three arms drive back to
    INITIAL_ARM_JOINTS at a capped joint speed (--return_speed_rad_s) and
    the base is stopped. Also triggered manually by a thumbstick click on
    either controller (accessible panic stop from inside the headset).
  - The base driver (interbotix_slate_driver) has its own independent
    300 ms cmd_vel timeout baked in (CMD_TIME_OUT in slate_base.h) -- this
    process's own staleness handling is a second, faster layer on top of
    that, not a replacement for it.

Run after starting the robot drivers and ROS topics described in README.md.
"""

# See eef_avp_control_singlearm.py: casadi (via PinocchioIKSolver) must be
# imported BEFORE rospy, or rospy's libstdc++ shadowing breaks casadi's import.
INITIAL_ARM_JOINTS = (0.0463, 0.5300, -0.5562, 0.000, 0.6500, 0.0000)
INITIAL_GRIPPER = 0.1  # fully open

import argparse
import os
import socket
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from protocol import FrameReader, DEFAULT_PORT, wire_to_pose, encode  # noqa: E402


def _load_robot_dependencies():
    """Load robot-only dependencies after argparse has handled ``--help``."""
    global PinocchioIKSolver, pin, rospy, Twist, JointState, Header
    global euler_matrix, euler_from_matrix

    try:
        from piper_ik import PinocchioIKSolver as solver
        import pinocchio as pin_module

        import rospy as rospy_module
        from geometry_msgs.msg import Twist as twist
        from sensor_msgs.msg import JointState as joint_state
        from std_msgs.msg import Header as header
        from tf.transformations import euler_matrix as make_euler_matrix
        from tf.transformations import euler_from_matrix as read_euler_matrix
    except ImportError as error:
        raise SystemExit(
            "Missing robot-side dependency. Install CasADi/Pinocchio and run "
            "inside the ROS environment described in teleoperation/quest2/README.md. "
            f"Original error: {error}"
        ) from error

    PinocchioIKSolver = solver
    pin = pin_module
    rospy = rospy_module
    Twist = twist
    JointState = joint_state
    Header = header
    euler_matrix = make_euler_matrix
    euler_from_matrix = read_euler_matrix


# Quest / OpenVR standing world (right / up / back) -> Piper world
# (forward / left / up). Numerically identical to R_AVP_TO_PIPER in
# eef_avp_control_singlearm.py: OpenVR's standing universe AND the WebXR
# world frame (see webxr/index.html) use the same right-handed, +Y-up,
# +Z-back convention as the AVP world frame (see avp/Readme.md section 9).
R_QUEST_TO_PIPER = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
], dtype=float)

DEFAULT_URDF = os.environ.get("PIPER_URDF")


def remap_to_piper(T_quest):
    """4x4 pose in OpenVR/WebXR world -> 4x4 pose in Piper world (rotation-only remap)."""
    T = np.eye(4)
    T[:3, 3] = R_QUEST_TO_PIPER @ T_quest[:3, 3]
    T[:3, :3] = R_QUEST_TO_PIPER @ T_quest[:3, :3] @ R_QUEST_TO_PIPER.T
    return T


class ArmChannel(object):
    """One arm's IK/state. Mirrors AvpEefController but driven by an
    externally-supplied lock/engaged state (see QuestTeleopServer's combined
    tri-arm clutch) and non-blocking (single step per main-loop tick instead
    of a dedicated blocking ramp loop)."""

    def __init__(self, name, urdf_path, joint_topic, cmd_topic, pub_queue_size=10):
        self.name = name
        self.ik = PinocchioIKSolver(urdf_path)
        self.joint = None  # latest /puppet feedback (JointState)
        self.pub = rospy.Publisher(cmd_topic, JointState, queue_size=pub_queue_size)
        rospy.Subscriber(joint_topic, JointState, self._joint_cb, queue_size=50)

        self.engaged = False
        self.lock_T = None            # (4,4) source pose (hand or head) in Piper world at engage
        self.target_q = None          # last commanded 6 joints
        self.gripper = INITIAL_GRIPPER
        self.initial_xyz = None
        self.initial_R = None
        self.returning_home = False

    def _joint_cb(self, msg):
        self.joint = msg

    def has_feedback(self):
        return self.joint is not None and len(self.joint.position) >= 6

    def fk(self, q6):
        q = np.asarray(q6, dtype=float).flatten()
        pin.framesForwardKinematics(self.ik.model, self.ik.data, q)
        se3 = self.ik.data.oMf[self.ik.ee_frame_id]
        xyz = np.asarray(se3.translation, dtype=float).flatten()
        rpy = np.array(euler_from_matrix(se3.rotation), dtype=float)
        return xyz, rpy

    def publish(self):
        names = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        m = JointState()
        m.header = Header(stamp=rospy.Time.now())
        m.name = names
        m.position = list(self.target_q) + [self.gripper]
        self.pub.publish(m)

    def boot_ramp_to_initial(self, duration, hz=30.0):
        target_q = np.asarray(INITIAL_ARM_JOINTS, dtype=float)
        current_q = np.asarray(self.joint.position[:6], dtype=float)
        steps = max(1, int(duration * hz))
        rate = rospy.Rate(hz)
        print("[%s] boot ramp (%.1fs): %s -> %s" %
              (self.name, duration, current_q.round(3), target_q.round(3)))
        for i in range(steps):
            if rospy.is_shutdown():
                return
            alpha = (i + 1) / steps
            interp = (1.0 - alpha) * current_q + alpha * target_q
            self.target_q = interp.tolist()
            self.gripper = INITIAL_GRIPPER
            self.publish()
            rate.sleep()
        self.target_q = target_q.tolist()
        fk_xyz, fk_rpy = self.fk(target_q)
        self.initial_xyz = fk_xyz
        self.initial_R = euler_matrix(*fk_rpy)[:3, :3]
        print("[%s] boot ramp done. FK anchor xyz=%s" % (self.name, fk_xyz.round(4)))

    def anchor_at_current_pose(self):
        """No motion: anchor this arm's delta-tracking origin at whatever
        joint state it's already in, instead of ramping to
        INITIAL_ARM_JOINTS. Only the mid/head channel has a fixed standby
        pose (it needs one -- head motion has no natural "current position"
        to anchor from); left/right start teleop from wherever they already
        are, no boot motion."""
        current_q = np.asarray(self.joint.position[:6], dtype=float)
        self.target_q = current_q.tolist()
        self.gripper = INITIAL_GRIPPER
        fk_xyz, fk_rpy = self.fk(current_q)
        self.initial_xyz = fk_xyz
        self.initial_R = euler_matrix(*fk_rpy)[:3, :3]
        print("[%s] anchored at current pose (no boot ramp). FK anchor xyz=%s" %
              (self.name, fk_xyz.round(4)))

    def lock(self, T_piper):
        """Latch T_piper as this arm's delta-tracking origin and start tracking.

        Also re-anchors the FK reference (initial_xyz/initial_R) from the
        CURRENT physical joint feedback. Without this, initial_xyz stayed
        fixed at whatever it was set to at boot; after any panic/stale-estop
        ramp-home (home_step drives target_q to INITIAL_ARM_JOINTS for all
        three arms but never touched initial_xyz), left/right's anchor went
        stale relative to the arm's real pose. The next lock() would then
        have track_step compute a target far from the current joints --
        clipped per-tick by max_joint_step, but visible as the arm briskly
        "running" to the wrong place right after re-engaging. Recomputing
        the anchor here every time matches the documented behavior
        ("re-engaging always relocks... same as the AVP script's
        per-episode anchoring") which previously only applied to lock_T.
        """
        current_q = np.asarray(self.joint.position[:6], dtype=float)
        fk_xyz, fk_rpy = self.fk(current_q)
        self.initial_xyz = fk_xyz
        self.initial_R = euler_matrix(*fk_rpy)[:3, :3]
        self.lock_T = T_piper.copy()
        self.engaged = True
        # Re-engaging cancels any pending ramp-home from a previous disengage
        # (see run()'s toggle-off handling) so tracking resumes immediately
        # instead of waiting for the arm to finish walking home first.
        self.returning_home = False

    def freeze(self):
        """Stop tracking; target_q (last commanded pose) is left untouched.
        Caller decides separately whether to also ramp home (see
        returning_home / home_step)."""
        self.engaged = False

    def track_step(self, T_piper, scale, max_joint_step, gripper_trigger,
                    gripper_min, gripper_max, respect_collision):
        """One IK-tracking tick while engaged. Returns an ik status string ('' = ok)."""
        delta_pos = T_piper[:3, 3] - self.lock_T[:3, 3]
        delta_R = T_piper[:3, :3] @ self.lock_T[:3, :3].T
        target_pos = self.initial_xyz + delta_pos * scale
        target_R = delta_R @ self.initial_R
        target_rpy = np.array(euler_from_matrix(target_R), dtype=float)

        seed = self.target_q if self.target_q is not None else (
            list(self.joint.position[:6]) if self.joint else None
        )
        sol, ok, msg = self.ik.solve(
            target_pos, target_rpy, gripper=INITIAL_GRIPPER, motorstate=seed,
            allow_collision=not respect_collision,
        )
        if not ok:
            return "IK fail: %s" % msg

        sol_arr = np.asarray(sol, dtype=float)
        if self.target_q is not None:
            prev = np.asarray(self.target_q, dtype=float)
            delta = sol_arr - prev
            if np.max(np.abs(delta)) > max_joint_step:
                sol_arr = np.clip(sol_arr, prev - max_joint_step, prev + max_joint_step)
        self.target_q = sol_arr.tolist()

        if gripper_trigger is None:
            # mid/head channel: head has no trigger, so its gripper just stays
            # fixed at INITIAL_GRIPPER -- same as eef_avp_control_singlearm.py.
            self.gripper = INITIAL_GRIPPER
        else:
            # Trigger: released (0, resting) = open (gripper_max), squeezed
            # (1) = closed (gripper_min) -- squeeze to grab, same as closing
            # a real hand around something.
            self.gripper = gripper_max - gripper_trigger * (gripper_max - gripper_min)
        return ""

    def home_step(self, max_step_rad):
        """One capped-speed step toward INITIAL_ARM_JOINTS. Returns True once arrived."""
        self.returning_home = True
        self.engaged = False
        target = np.asarray(INITIAL_ARM_JOINTS, dtype=float)
        cur = np.asarray(self.target_q, dtype=float)
        delta = target - cur
        step = np.clip(delta, -max_step_rad, max_step_rad)
        new_q = cur + step
        self.target_q = new_q.tolist()
        self.gripper = INITIAL_GRIPPER
        arrived = bool(np.max(np.abs(target - new_q)) < 1e-3)
        if arrived:
            self.returning_home = False
        return arrived


class QuestTeleopServer(object):
    def __init__(self, args):
        self.args = args
        rospy.init_node("quest_teleop", anonymous=True)

        self.mid = ArmChannel("mid", args.mid_urdf, args.mid_joint_topic, args.mid_cmd_topic)
        self.left = ArmChannel("left", args.left_urdf, args.left_joint_topic, args.left_cmd_topic)
        self.right = ArmChannel("right", args.right_urdf, args.right_joint_topic, args.right_cmd_topic)
        self.channels = (self.mid, self.left, self.right)

        self.cmd_vel_pub = rospy.Publisher(args.cmd_vel_topic, Twist, queue_size=1)
        self.prev_both_grip = False

        self._lock = threading.Lock()
        self._latest = None
        self._last_recv_mono = None
        self._current_conn = None  # same socket, written back to for status (see _send_status)
        self._server_sock = None
        self._stop = False

    # ---------- networking (background thread) ----------
    def _accept_loop(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.args.host, self.args.port))
        s.listen(1)
        s.settimeout(2.0)  # diagnostic: bare periodic heartbeat, see [net-hb] below
        self._server_sock = s
        print("[net] listening on %s:%d" % (self.args.host, self.args.port), flush=True)
        heartbeats = 0
        while not self._stop:
            try:
                conn, addr = s.accept()
            except socket.timeout:
                heartbeats += 1
                if heartbeats % 3 == 0:
                    print("[net-hb] accept_loop alive, still waiting (t=%ds)" % (heartbeats * 2), flush=True)
                continue
            except OSError:
                break
            print("[net] client connected: %s" % (addr,), flush=True)
            conn.settimeout(2.0)
            with self._lock:
                self._current_conn = conn
            reader = FrameReader(conn)
            try:
                while not self._stop:
                    try:
                        frame = reader.read()
                    except socket.timeout:
                        continue
                    if frame is None:
                        break
                    with self._lock:
                        self._latest = frame
                        self._last_recv_mono = time.monotonic()
            except (OSError, ValueError) as e:
                print("[net] recv error: %s" % e)
            finally:
                with self._lock:
                    if self._current_conn is conn:
                        self._current_conn = None
                conn.close()
                print("[net] client disconnected; waiting for reconnect")

    def _snapshot(self):
        with self._lock:
            return self._latest, self._last_recv_mono

    def _send_status(self, payload):
        """Best-effort write back on the SAME connection the browser is
        sending pose on -- webxr_server.py's RobotBridge relays whatever
        comes back here to the browser over WebSocket (see index.html's
        ik_status handling). Read (background thread) and write (this,
        called from the main control loop) happen on the same socket from
        different threads; that's fine, they're independent directions."""
        with self._lock:
            conn = self._current_conn
        if conn is None:
            return
        try:
            conn.sendall(encode(payload))
        except OSError:
            pass

    # ---------- boot ----------
    def wait_feedback(self, timeout_sec):
        deadline = rospy.Time.now() + rospy.Duration(timeout_sec)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if all(c.has_feedback() for c in self.channels):
                return True
            if rospy.Time.now() > deadline:
                return False
            rate.sleep()
        return False

    # ---------- main loop ----------
    def _device_from_frame(self, frame, key):
        """Returns (T_piper or None, buttons dict or None) for 'head'/'left'/'right'."""
        if frame is None:
            return None, None
        d = frame.get(key)
        if not d or not d.get("connected") or not d.get("valid"):
            return None, None
        T_piper = remap_to_piper(wire_to_pose(d))
        return T_piper, d.get("buttons")

    def _publish_cmd_vel(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_vel_pub.publish(msg)

    def run(self):
        print("=" * 60)
        print("Quest tri-arm + base teleop")
        print("  listen              = %s:%d" % (self.args.host, self.args.port))
        print("  mid   joint/cmd     = %s / %s" % (self.args.mid_joint_topic, self.args.mid_cmd_topic))
        print("  left  joint/cmd     = %s / %s" % (self.args.left_joint_topic, self.args.left_cmd_topic))
        print("  right joint/cmd     = %s / %s" % (self.args.right_joint_topic, self.args.right_cmd_topic))
        print("  cmd_vel_topic       = %s" % self.args.cmd_vel_topic)
        print("  base speed lin/ang  = %.2f m/s / %.2f rad/s" %
              (self.args.base_linear_speed, self.args.base_angular_speed))
        print("  stale_freeze_sec    = %.2f" % self.args.stale_freeze_sec)
        print("  stale_estop_sec     = %.2f" % self.args.stale_estop_sec)
        print("  return_speed_rad_s  = %.2f" % self.args.return_speed_rad_s)
        print("=" * 60)

        net_thread = threading.Thread(target=self._accept_loop, daemon=True)
        net_thread.start()

        print("[boot] waiting for /puppet joint feedback (10s timeout)...")
        if not self.wait_feedback(10.0):
            print("[boot] timed out waiting for joint feedback. Is the piper driver running?")
            return
        # Only mid has a fixed standby pose to ramp to (it needs one -- head
        # motion has no natural "current position" to anchor from). Left/right
        # anchor at whatever pose they're already in, no boot motion.
        self.mid.boot_ramp_to_initial(self.args.boot_duration)
        self.left.anchor_at_current_pose()
        self.right.anchor_at_current_pose()
        print("[teleop] ready. Press BOTH grips together ONCE to engage all 3 arms "
              "(head->mid, left->left, right->right); press together again to "
              "disengage (toggle, not hold). "
              "Left X/Y = base turn left/right, Right A/B = base back/fwd. "
              "Thumbstick click on either hand = panic ramp-home + base stop.")

        hz = self.args.rate
        rate = rospy.Rate(hz)
        home_step_rad = self.args.return_speed_rad_s / hz
        max_joint_step = self.args.max_joint_step
        last_print = 0.0

        while not rospy.is_shutdown():
            frame, last_recv = self._snapshot()
            recv_age = (time.monotonic() - last_recv) if last_recv is not None else float("inf")
            stale_freeze = recv_age > self.args.stale_freeze_sec
            stale_estop = recv_age > self.args.stale_estop_sec

            h_T, _ = (None, None) if stale_freeze else self._device_from_frame(frame, "head")
            l_T, l_btn = (None, None) if stale_freeze else self._device_from_frame(frame, "left")
            r_T, r_btn = (None, None) if stale_freeze else self._device_from_frame(frame, "right")

            panic = False
            for btn in (l_btn, r_btn):
                if btn and btn.get("stick_pressed"):
                    panic = True

            # ---- combined tri-arm clutch: both grips together TOGGLES all 3 ----
            # Press both grips together once to engage (lock all 3 origins),
            # press together again to disengage. NOT hold-to-track -- letting
            # go of one/both grips after engaging does nothing by itself.
            # Staleness/panic always force disengage regardless of toggle state.
            both_grip = bool(
                l_btn and l_btn.get("grip_pressed") and r_btn and r_btn.get("grip_pressed")
            )
            all_poses_ok = h_T is not None and l_T is not None and r_T is not None
            fresh = not (stale_freeze or stale_estop or panic)

            if not fresh:
                if self.mid.engaged:
                    print("[teleop] disengaged (stale/panic)")
                self.mid.freeze()
                self.left.freeze()
                self.right.freeze()
            elif both_grip and not self.prev_both_grip and all_poses_ok:
                if self.mid.engaged:
                    self.mid.freeze()
                    self.left.freeze()
                    self.right.freeze()
                    # Normal (non-panic) disengage now ramps home too, same
                    # speed/mechanism as the stale/panic path below -- set by
                    # user request after testing showed the arm just sitting
                    # wherever the hand last was felt wrong for ending a
                    # session. lock() clears this again if re-engaged mid-ramp.
                    self.mid.returning_home = True
                    self.left.returning_home = True
                    self.right.returning_home = True
                    print("[teleop] disengaged (toggle off) -- ramping home")
                else:
                    self.mid.lock(h_T)
                    self.left.lock(l_T)
                    self.right.lock(r_T)
                    print("[teleop] ENGAGED (all 3 arms, toggle on)")

            # Edge-tracking for next tick's both-grip rising-edge detection.
            # Only update it from a REAL button reading (l_btn/r_btn are None
            # only during stale_freeze, i.e. we truly have no idea what the
            # physical grip state is). Previously this was `both_grip and
            # fresh`, which forced prev_both_grip to False on every not-fresh
            # tick (staleness OR a panic click) regardless of the actual
            # button state -- if the user was still physically holding both
            # grips through a brief stale/panic blip, the first fresh frame
            # after recovery looked like a brand-new rising edge and silently
            # flipped engage<->disengage from what the user actually pressed
            # (observed live: a short stall mid-session inverted the toggle).
            # Leaving prev_both_grip untouched while we have no button data
            # means the next real reading is compared against the last
            # trusted one, not a forced False.
            if l_btn is not None and r_btn is not None:
                self.prev_both_grip = both_grip

            # ---- per-arm step ----
            # ik_status is sent back to the browser every tick (see
            # _send_status) so a stuck-in-place arm shows up as an explicit
            # "IK fail: ..." in the VR HUD instead of silently not following
            # -- IK failure (workspace/reach limit, etc.) intentionally does
            # NOT update target_q, so the arm just stops where it is with no
            # local signal to the operator that anything went wrong.
            ik_msgs = []
            ik_status = {"type": "ik_status", "mid": "", "left": "", "right": ""}
            for chan, T_piper, btn in (
                (self.mid, h_T, None), (self.left, l_T, l_btn), (self.right, r_T, r_btn),
            ):
                # stale_estop/panic force a ramp-home same as a normal
                # toggle-off disengage now does (chan.returning_home, set in
                # the toggle handling above) -- unify all three triggers
                # through the same home_step machinery/speed so a
                # deliberate disengage and a network drop both bring the
                # arm home the same slow way instead of just holding still.
                if stale_estop or panic or chan.returning_home:
                    chan.freeze()
                    if chan.target_q is not None:
                        chan.home_step(home_step_rad)
                    chan.publish()
                    continue

                if stale_freeze or T_piper is None:
                    if chan.target_q is not None:
                        chan.publish()  # hold last commanded pose
                    continue

                if chan.engaged:
                    trig = float(btn.get("trigger", 0.0)) if btn else None
                    msg = chan.track_step(
                        T_piper, self.args.scale, max_joint_step, trig,
                        self.args.gripper_min, self.args.gripper_max,
                        self.args.respect_collision,
                    )
                    if msg:
                        ik_msgs.append("%s: %s" % (chan.name, msg))
                        ik_status[chan.name] = msg

                if chan.target_q is not None:
                    chan.publish()

            self._send_status(ik_status)

            # ---- base drive: independent of arm clutch, gated by staleness/panic ----
            # Left X/Y = turn left/right, right A/B = backward/forward.
            lin = 0.0
            ang = 0.0
            if not (stale_freeze or stale_estop or panic):
                if l_btn:
                    if l_btn.get("button_ax"):  # left X: turn left (counter-clockwise)
                        ang += self.args.base_angular_speed
                    if l_btn.get("button_by"):  # left Y: turn right (clockwise)
                        ang -= self.args.base_angular_speed
                if r_btn:
                    if r_btn.get("button_ax"):  # right A: backward
                        lin -= self.args.base_linear_speed
                    if r_btn.get("button_by"):  # right B: forward
                        lin += self.args.base_linear_speed
            self._publish_cmd_vel(lin, ang)

            now = time.monotonic()
            if now - last_print > 1.0:
                status = "STALE(%.1fs)" % recv_age if stale_freeze else "live"

                def arm_status(c):
                    return "ENGAGED" if c.engaged else ("HOME" if c.returning_home else "idle")

                print("[teleop] %-16s M=%-7s L=%-7s R=%-7s base(lin=%+.2f ang=%+.2f)%s%s" % (
                    status, arm_status(self.mid), arm_status(self.left), arm_status(self.right),
                    lin, ang,
                    " PANIC" if panic else "",
                    ("  " + "; ".join(ik_msgs)) if ik_msgs else "",
                ))
                last_print = now

            rate.sleep()

        self._publish_cmd_vel(0.0, 0.0)
        self._stop = True
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass


def get_args():
    p = argparse.ArgumentParser(description="Quest 2 -> Piper tri-arm + base teleop (robot side)")
    p.add_argument("--host", default="127.0.0.1",
                    help="Bind address. Keep this 127.0.0.1 -- reachability is meant to "
                         "come ONLY from the SSH -L tunnel, not the open network.")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--rate", type=float, default=50.0, help="Control loop rate (Hz).")

    p.add_argument("--mid_urdf", default=DEFAULT_URDF)
    p.add_argument("--left_urdf", default=DEFAULT_URDF)
    p.add_argument("--right_urdf", default=DEFAULT_URDF)
    p.add_argument("--mid_joint_topic", default="/puppet/joint_mid")
    p.add_argument("--mid_cmd_topic", default="/master/joint_mid")
    p.add_argument("--left_joint_topic", default="/puppet/joint_left")
    p.add_argument("--left_cmd_topic", default="/master/joint_left")
    p.add_argument("--right_joint_topic", default="/puppet/joint_right")
    p.add_argument("--right_cmd_topic", default="/master/joint_right")

    p.add_argument("--cmd_vel_topic", default="/cmd_vel",
                    help="Twist topic for interbotix_slate_driver's slate_base_node.")
    p.add_argument("--base_linear_speed", type=float, default=0.15,
                    help="m/s while holding right B (forward) or A (backward). Conservative "
                         "default -- the driver's own software limit is 1.0 m/s.")
    p.add_argument("--base_angular_speed", type=float, default=0.3,
                    help="rad/s while holding left X (turn left) or Y (turn right).")

    p.add_argument("--scale", type=float, default=1.0,
                    help="Position-only scale factor: hand/head delta * scale = EE delta.")
    p.add_argument("--boot_duration", type=float, default=3.0)
    p.add_argument("--return_speed_rad_s", type=float, default=0.15,
                    help="Max joint speed while auto-returning home (e-stop / panic). "
                         "Slowed from an earlier 0.3 default after it felt too fast live.")
    p.add_argument("--max_joint_step", type=float, default=0.05,
                    help="Max per-joint change per control tick (rad) during normal tracking.")
    p.add_argument("--stale_freeze_sec", type=float, default=0.2,
                    help="No fresh packet for longer than this -> freeze all arms + stop base.")
    p.add_argument("--stale_estop_sec", type=float, default=1.0,
                    help="No fresh packet for longer than this -> ramp all arms home.")
    p.add_argument("--gripper_min", type=float, default=0.0)
    p.add_argument("--gripper_max", type=float, default=0.1)
    p.add_argument("--respect_collision", action="store_true")

    return p.parse_args()


def main():
    args = get_args()
    if not args.mid_urdf or not args.left_urdf or not args.right_urdf:
        raise SystemExit("Set PIPER_URDF or pass --mid_urdf, --left_urdf, and --right_urdf")
    _load_robot_dependencies()
    server = QuestTeleopServer(args)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[teleop] stopping.")


if __name__ == "__main__":
    main()
