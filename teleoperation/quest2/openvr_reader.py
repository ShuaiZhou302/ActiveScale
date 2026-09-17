"""
SteamVR / OpenVR pose reader for Quest 2 (Windows side).

Reads HMD + left/right controller poses and button state from SteamVR.
Coordinate frame: OpenVR "standing" world, right-handed, +Y up, +Z back
(-Z is the direction the operator faced when Guardian was set up).

This module does NOT do any robot-frame conversion or networking -- it is
the sensor layer only. quest_client.py wraps it and ships raw poses to
quest_server.py, which owns the axis remap and all safety/control logic
(see protocol.py's module docstring for why).

Button/trigger/grip state goes through OpenVR's Action system
(IVRInput), NOT the legacy IVRSystem::GetControllerState API. Confirmed
empirically on this machine: GetControllerState returns success=False for
an unregistered utility app talking to Quest 2 Touch controllers over the
Oculus->SteamVR bridge (SteamVR only routes legacy button state to the
focused scene application; a background pose-reader like this one never
has focus). GetControllerState is also Valve's own documented deprecated
path. The actions.json / openvr_actions/binding_oculus_touch.json next to
this file declare a "quest_teleop" action set with default bindings for
controller_type "oculus_touch" (input paths verified against the installed
driver's own profile: SteamVR/drivers/oculus/resources/input/touch_profile.json).

Also confirmed empirically: without an explicit app identity, SteamVR
auto-generates a shared app key ("system.generated.<exe name>") for ANY
process with that executable name -- e.g. every python.exe-based OpenVR
script on this machine collides under "system.generated.python.exe". Once
*any* of them is seen using the legacy GetControllerState API, SteamVR
appears to pin that shared app key to legacy-only mode, silently starving
the Action system for every other script sharing it (this is exactly what
happened while developing this file). openvr_actions/app.vrmanifest +
IVRApplications.identifyApplication() below give this script its own
stable app key ("quest.teleop.piper") so its input binding state can never
collide with some other unrelated openvr script run from the same
python.exe.

Prereqs: SteamVR running, headset connected via Link/Air Link and tracked.
    pip install openvr numpy
"""
import json
import os
import sys
import time

import numpy as np
import openvr

_HERE = os.path.dirname(os.path.abspath(__file__))
ACTIONS_MANIFEST = os.path.join(_HERE, "openvr_actions", "actions.json")
VR_MANIFEST = os.path.join(_HERE, "openvr_actions", "app.vrmanifest")
APP_KEY = "quest.teleop.piper"

GRIP_THRESHOLD = 0.5
TRIGGER_THRESHOLD = 0.5


def _write_vrmanifest():
    """(Re)generate app.vrmanifest with binary_path_windows pointing at the
    CURRENT interpreter's absolute path. A relative/wrong path here makes
    SteamVR silently skip the whole application entry (confirmed via
    vrserver.txt: "... binary_path ...python.exe is invalid. Skipping"),
    which then makes identifyApplication() fail with UnknownApplication --
    so this must be correct and absolute, and must be written fresh each
    run since sys.executable varies by machine/venv."""
    manifest = {
        "source": "builtin",
        "applications": [{
            "app_key": APP_KEY,
            "launch_type": "binary",
            "binary_path_windows": sys.executable,
            "is_dashboard_overlay": False,
            "action_manifest_path": ACTIONS_MANIFEST,
            "strings": {"en_us": {"name": "Quest Teleop (Piper)"}},
        }],
    }
    with open(VR_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)


def mat34_to_homogeneous(m):
    """HmdMatrix34_t -> 4x4 numpy homogeneous transform."""
    return np.array([
        [m[0][0], m[0][1], m[0][2], m[0][3]],
        [m[1][0], m[1][1], m[1][2], m[1][3]],
        [m[2][0], m[2][1], m[2][2], m[2][3]],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


class QuestPoseReader(object):
    """Reads HMD + left/right controller poses and button state from SteamVR."""

    def __init__(self, predict_seconds=0.0):
        # VRApplication_Other: pure pose reader, doesn't own the compositor.
        self.vr = openvr.init(openvr.VRApplication_Other)
        self.vrsys = openvr.VRSystem()
        self.predict_seconds = predict_seconds
        self._max_devices = openvr.k_unMaxTrackedDeviceCount

        # Claim our own app identity before touching input -- see module
        # docstring for why this matters (shared "system.generated.*" keys
        # across every python.exe-based OpenVR script otherwise collide).
        # temporary=False: a temporary manifest is accepted by
        # AddApplicationManifest but NOT added to the known-applications
        # list identifyApplication() checks against (confirmed empirically:
        # temporary=True raises ApplicationError_UnknownApplication on the
        # very next identifyApplication call). Persistent registration is
        # also simply correct here -- this is a real, recurring tool, not a
        # one-off script, so it should have a durable app identity.
        _write_vrmanifest()
        vrapps = openvr.VRApplications()
        vrapps.addApplicationManifest(VR_MANIFEST, False)
        vrapps.identifyApplication(0, APP_KEY)

        self.vrinput = openvr.VRInput()
        self.vrinput.setActionManifestPath(ACTIONS_MANIFEST)
        self._action_set = self.vrinput.getActionSetHandle("/actions/quest_teleop")
        self._act_trigger = self.vrinput.getActionHandle("/actions/quest_teleop/in/Trigger")
        self._act_grip = self.vrinput.getActionHandle("/actions/quest_teleop/in/Grip")
        self._act_primary = self.vrinput.getActionHandle("/actions/quest_teleop/in/ButtonPrimary")
        self._act_secondary = self.vrinput.getActionHandle("/actions/quest_teleop/in/ButtonSecondary")
        self._act_stick = self.vrinput.getActionHandle("/actions/quest_teleop/in/StickClick")
        self._src_left = self.vrinput.getInputSourceHandle("/user/hand/left")
        self._src_right = self.vrinput.getInputSourceHandle("/user/hand/right")

        self._active_sets = (openvr.VRActiveActionSet_t * 1)()
        self._active_sets[0].ulActionSet = self._action_set

    def _controller_index(self, role):
        idx = self.vrsys.getTrackedDeviceIndexForControllerRole(role)
        if idx == openvr.k_unTrackedDeviceIndexInvalid:
            return None
        return idx

    def _update_actions(self):
        self.vrinput.updateActionState(self._active_sets)

    def _read_buttons(self, hand_src):
        trig = self.vrinput.getAnalogActionData(self._act_trigger, hand_src)
        grip = self.vrinput.getAnalogActionData(self._act_grip, hand_src)
        primary = self.vrinput.getDigitalActionData(self._act_primary, hand_src)
        secondary = self.vrinput.getDigitalActionData(self._act_secondary, hand_src)
        stick = self.vrinput.getDigitalActionData(self._act_stick, hand_src)
        if not (trig.bActive or grip.bActive):
            return None
        return {
            "trigger": float(trig.x),
            "grip": float(grip.x),
            "grip_pressed": bool(grip.x > GRIP_THRESHOLD),
            "trigger_pressed": bool(trig.x > TRIGGER_THRESHOLD),
            "button_ax": bool(primary.bActive and primary.bState),
            "button_by": bool(secondary.bActive and secondary.bState),
            "stick_pressed": bool(stick.bActive and stick.bState),
        }

    def _pack(self, poses, idx, hand_src):
        if idx is None or idx >= self._max_devices:
            return None
        p = poses[idx]
        if not p.bDeviceIsConnected:
            return None
        T = mat34_to_homogeneous(p.mDeviceToAbsoluteTracking)
        out = {
            "index": int(idx),
            "valid": bool(p.bPoseIsValid),
            "tracking_ok": p.eTrackingResult == openvr.TrackingResult_Running_OK,
            "matrix": T,
        }
        if hand_src is not None:
            out["buttons"] = self._read_buttons(hand_src)
        return out

    def get(self):
        """One sample: {'head':..., 'left':..., 'right':...}, each a dict or None."""
        self._update_actions()
        poses = self.vrsys.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            self.predict_seconds,
            self._max_devices,
        )
        left_idx = self._controller_index(openvr.TrackedControllerRole_LeftHand)
        right_idx = self._controller_index(openvr.TrackedControllerRole_RightHand)
        return {
            "head": self._pack(poses, openvr.k_unTrackedDeviceIndex_Hmd, None),
            "left": self._pack(poses, left_idx, self._src_left),
            "right": self._pack(poses, right_idx, self._src_right),
        }

    def close(self):
        openvr.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    # Standalone smoke test: prints raw OpenVR poses, no networking.
    with QuestPoseReader() as reader:
        print("Connected to SteamVR. Ctrl-C to stop.")
        try:
            while True:
                s = reader.get()
                for name in ("head", "left", "right"):
                    d = s[name]
                    if d is None:
                        print("%-6s ---" % name, end="  ")
                    else:
                        x, y, z = d["matrix"][:3, 3]
                        line = "%-6s (%+.3f %+.3f %+.3f) valid=%d" % (name, x, y, z, d["valid"])
                        b = d.get("buttons")
                        if b:
                            line += " trig=%.2f grip=%.2f%s%s" % (
                                b["trigger"], b["grip"],
                                " GRIP" if b["grip_pressed"] else "",
                                " AX" if b["button_ax"] else "",
                            )
                        print(line, end="  ")
                print()
                time.sleep(1.0 / 20.0)
        except KeyboardInterrupt:
            print("\nstopped.")
