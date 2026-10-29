"""
Advanced teach / jog pendant for the workcell, built from Swift's own widgets.

TeachPendant adds a control panel to the Swift browser tab, next to the 3D
scene, that drives the robots already in the scene:
  - individual joint jogging: drag a joint slider, or step the selected joint
    with the Joint -/+ buttons
  - Cartesian jogging of the end-effector in world x, y and z, with the tool
    orientation held (closed-loop resolved-rate control, damped near
    singularities); each click moves one step, a gamepad jogs continuously
  - live state: status line (green / amber / red), end-effector pose, joint
    angles, manipulability, gamepad status and the most recent events
    (the full event log is also printed to the terminal)
  - a latching software E-stop with separate Reset and Resume actions, and
    taught poses (record / go to / save / load)
  - an optional gamepad (pygame) that mirrors the panel controls

Swift only runs the widget callbacks inside env.step(), so the pendant has to be
serviced from every env.step() call. Two ways to use it:

  Alongside an automatic sequence (non-blocking):

    pendant = TeachPendant(env, {"DoBot6": robot, "RS007N": nathanBot})
    pendant.attach()            # hooks env.step so the pendant is serviced by every step
    pendant.auto_busy = True    # manual jogging is refused while the sequence drives the robots
    try:
        ...                     # the sequence (execute_move etc.) -- E-stop freezes it in place
    finally:
        pendant.auto_busy = False
    pendant.run()               # then full manual jogging until Ctrl+C

  Stand-alone (blocking), in place of env.hold():

    pendant.run()
"""
import json
import os
import time

import numpy as np
import swift
from roboticstoolbox import jtraj
from spatialmath.base import vex

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ImportError:  # the gamepad is optional -- the panel still works
    pygame = None


TICK = 0.05                  # control / render period (s)
JOINT_SPEED = np.radians(30) # joint speed at 100% (rad/s)
CART_SPEED = 0.10            # Cartesian speed at 100% (m/s)
MAX_QDOT = np.radians(90)    # cap on any joint's speed during a Cartesian move (rad/s)
LIMIT_WARN = np.radians(5)   # warn when a joint is this close to a limit
MANIP_WARN = 1e-3            # warn below this manipulability (same threshold as move_arm_rmrc)
MAX_DAMPING = 0.05           # damped least squares near singularities (as in move_arm_rmrc)
LOG_LINES = 3                # recent events shown in the panel
SLIDER_STEP = 0.1            # joint slider resolution (deg)
ECHO_HISTORY = 20            # recent slider values pushed by the pendant, see _slider_moved

# Step sizes for one click of a jog button: (label, metres, degrees)
STEP_SIZES = [("1 mm / 1 deg", 0.001, 1), ("5 mm / 2 deg", 0.005, 2),
              ("10 mm / 5 deg", 0.010, 5), ("50 mm / 10 deg", 0.050, 10)]

# Gamepad layout (XInput order as reported by pygame on Windows, e.g. an Xbox pad)
PAD_DEADZONE = 0.15
AXIS_LX, AXIS_LY, AXIS_RY = 0, 1, 3
BTN_A, BTN_B, BTN_Y, BTN_LB, BTN_RB, BTN_START = 0, 1, 3, 4, 5, 7

OK, WARNING, FAULT = 0, 1, 2
LEVEL_ICON = {OK: "🟢", WARNING: "🟠", FAULT: "🔴"}
LEVEL_TAG = {OK: "INFO ", WARNING: "WARN ", FAULT: "FAULT"}


class TeachPendant:
    """Teach / jog panel in the Swift window for one or more robots that are already in `env`."""

    def __init__(self, env, robots, min_tool_z=0.6, poses_file="taught_poses.json"):
        """
        env        -- the Swift environment the robots were added to
        robots     -- dict of display name -> robot model
        min_tool_z -- safety plane: motion that takes the tool further below this
                      world z (e.g. into the table top) is blocked
        poses_file -- JSON file used by the Save / Load taught-pose buttons
        """
        self.env = env
        self.robots = dict(robots)
        self.names = list(self.robots)
        role_labels = {
            "DoBot6" : "Arm 1 | Wheel transfer",
            "RS007N" : "Arm 2 | Bearing insertion",
            "reBot B601" : "Arm 3 | Assembly storage"
        }
        self.robot_display_names = [role_labels.get(name, name) for name in self.names]
        self.min_tool_z = min_tool_z
        self.poses_file = poses_file
        self.start_q = {name: np.array(r.q, dtype=float) for name, r in self.robots.items()}

        self.robot_name = self.names[0]
        self.speed = 0.5            # fraction of full speed
        self.step_index = 2         # index into STEP_SIZES
        self.jog_joint = 0          # joint stepped by the Joint -/+ buttons
        self.estop = False
        self.joint_target = None    # joint-space goal (sliders, Joint -/+)
        self.cart_path = []         # queued end-effector positions (Cartesian buttons)
        self.cart_goal = None       # end-effector goal pose the arm is servoed onto
        self.playback = []          # queued joint configurations (go to taught / start pose)
        self.conditions = {}        # message -> level, rebuilt every tick
        self.prev_conditions = {}
        self.events = []            # recent log lines shown in the panel
        self.taught = []            # dicts: robot, name, q
        self.selected_pose = 0
        self.sliders_follow = True  # push the robot's joint angles back onto the sliders
        self.slider_pushed = []     # per slider: values recently pushed by the pendant (echo filter)

        self.auto_busy = False        # True while main()'s sequence is driving the robots
        self.awaiting_resume = False  # E-stop reset done, but not yet resumed
        self.collision = None         # message while an upcoming collision holds the sequence
        self._real_step = None        # the original env.step, once attach() has hooked it
        self._in_hook = False         # re-entry guard for the hook

        self.gamepad = None
        self.pad_mode = "cartesian"
        self.pad_joint = 0
        self.pad_buttons = ()
        self.pad_hat = (0, 0)

        self._build_panel()
        self._init_gamepad()
        self._select_robot(0)

    @property
    def robot(self):
        return self.robots[self.robot_name]

    def run(self):
        """Run the pendant on its own; blocks until Ctrl+C."""
        self._log(OK, "Pendant started - " + ", ".join(self.names))
        try:
            while True:
                if self._real_step is None:  # when hooked, env.step() already ticks the pendant
                    self.tick()
                self.env.step(TICK)  # renders, and runs the panel callbacks
        except KeyboardInterrupt:
            self._log(OK, "Pendant closed")
        finally:
            if pygame is not None:
                pygame.quit()

    # ------------------------------------------------------------------
    # Non-blocking operation: ride along on every env.step()
    # ------------------------------------------------------------------
    def attach(self):
        """Service the pendant on every env.step(), from wherever it is called."""
        if self._real_step is not None:
            return
        self._real_step = self.env.step
        self.env.step = self._step_hook
        self._log(OK, "Pendant attached to env.step")

    def _step_hook(self, *args, **kwargs):
        if self._in_hook:  # guard against re-entry
            return self._real_step(*args, **kwargs)
        self._in_hook = True
        try:
            self.tick()
            self._real_step(*args, **kwargs)
            # E-stop / awaiting resume: freeze the calling sequence right here.
            # Each pass still renders and polls the pendant, so buttons and the gamepad work.
            while self.estop or self.awaiting_resume:
                self.tick()
                self._real_step(TICK)
        finally:
            self._in_hook = False

    def _resume(self):
        if self.awaiting_resume and not self.estop:
            self.awaiting_resume = False
            self.collision = None
            self._log(OK, "Resumed")

    def hold_for_collision(self, message):
        """Called by execute_move when the path ahead is blocked: show a fault and freeze until Resume."""
        self.collision = message
        self.awaiting_resume = True

    def add_obstacle_sliders(self, obstacle, lo=(-1.5, -2.0, 0.6), hi=(1.5, 1.5, 1.5)):
        """Sliders that move the test obstacle live (the sliders keep working while the arm is held)."""
        self._add(swift.Label("Test Obstacle"))
        for axis, name in enumerate("XYZ"):
            self._add(swift.Slider(lambda v, a=axis: self._move_obstacle(obstacle, a, float(v)),
                                min=lo[axis], max=hi[axis], step=0.01,
                                value=float(obstacle.centre[axis]), desc=f"Test obstacle {name}", unit="m"))

    def _move_obstacle(self, obstacle, axis, value):
        centre = obstacle.centre.copy()
        centre[axis] = value
        obstacle.move_to(*centre)

    # ------------------------------------------------------------------
    # Panel layout (Swift shows elements in the order they are added)
    # ------------------------------------------------------------------
    def _add(self, element):
        self.env.add(element)
        return element

    def _build_panel(self):
        n_max = max(r.n for r in self.robots.values())

        self._add(swift.Label("WHEEL ASSEMBLY CELL")) #, compact=True))
        self.status_label = self._add(swift.Label(""))
        self._add(swift.Button(lambda _: self._press_estop(), "STOP - E-STOP"))
        self._add(swift.Button(lambda _: self._reset_estop(), "Reset E-stop"))
        self._add(swift.Button(lambda _: self._resume(), "Resume operation"))
        self.robot_select = self._add(swift.Select(self._robot_selected, "Selected robot", self.robot_display_names, 0))
        self._add(swift.Slider(self._set_speed, min=5, max=100, step=5, value=int(self.speed * 100),
                               desc="Manual motion speed", unit="%"))
        self._add(swift.Select(self._set_step, "Jog increment", [s[0] for s in STEP_SIZES], self.step_index))

        # Cartesian jog
        self._add(swift.Label("TOOL POSITION | World frame", compact=True))
        self.position_labels = [
            self._add(swift.Label(f"{axis}: — m", compact=True))
            for axis in ("X", "Y", "Z")
            ]
        self._add(swift.Label("TOOL ORIENTATION", compact=True))
        self.orientation_labels = [
            self._add(swift.Label(f"{axis}: — deg", compact=True))
            for axis in ("Roll", "Pitch", "Yaw")
            ]
        self._add(swift.Label("MOVE TOOL"))
        for axis, name in enumerate("XYZ"):
            for sign in (-1, 1):
                self._add(swift.Button(lambda _, a=axis, s=sign: self._step_cartesian(a, s),
                                       f"{name} {'−' if sign < 0 else '+'}"))

        # Joint jog
        self._add(swift.Label("MOVE JOINTS"))
        q0 = self._slider_values(self.robot)
        self.slider_pushed = [[v] for v in q0]
        self.joint_sliders = [
            self._add(swift.Slider(lambda v, j=j: self._slider_moved(j, v), min=-180, max=180, step=SLIDER_STEP,
                                   value=q0[j], desc=f"Joint {j + 1}", unit="°"))
            for j in range(n_max)]
        self._add(swift.Select(lambda i: setattr(self, "jog_joint", int(i)), "Joint to step",
                               [f"J{j + 1}" for j in range(n_max)], 0))
        self._add(swift.Button(lambda _: self._step_joint(-1), "Step selected joint -"))
        self._add(swift.Button(lambda _: self._step_joint(+1), "Step selected join +"))

        # State
        self._add(swift.Label("DIAGNOSTICS"))
        self.joint_label = self._add(swift.Label(""))
        self.manip_label = self._add(swift.Label(""))
        self.pad_label = self._add(swift.Label(""))
        self.pad_map_label = self._add(swift.Label(""))

        # Taught poses
        self._add(swift.Label("SAVED POSITIONS"))
        self.pose_select = self._add(swift.Select(lambda i: setattr(self, "selected_pose", int(i)),
                                                  "Saved position", ["(No positions recorded)"], 0))
        for desc, cmd in (("Record current position", self._record_pose), 
                          ("Move to selected position", self._goto_selected),
                          ("Return to initial position", self._goto_start),
                          ("Save positions to file", self._save_poses),
                          ("Load positions from file", self._load_poses),
                          ("Delete selected position", self._delete_pose)
                            ):
            self._add(swift.Button(lambda _, c=cmd: c(), desc))

        # Recent events
        self._add(swift.Label("RECENT EVENTS"))
        self.log_labels = [self._add(swift.Label("")) for _ in range(LOG_LINES)]

# ------------------------------------------------------------------
# Panel callbacks
# ------------------------------------------------------------------
    def _select_robot(self, index):
        self._stop_motion()
        self.robot_name = self.names[index]
        self.robot_select.value = index  # shows the change when it came from the gamepad
        self.pad_joint = 0
        qlim = np.degrees(self.robot.qlim)
        for j, slider in enumerate(self.joint_sliders):
            if j < self.robot.n:
                slider.min, slider.max = self._on_grid(qlim[0, j]), self._on_grid(qlim[1, j])
            else:  # this robot has fewer joints than the panel
                slider.min = slider.max = 0
        self.sliders_follow = True
        self._log(OK, f"Selected {self.robot_name}")

    def _robot_selected(self, index):
        # Swift also fires this when the dropdown first appears or the code sets it -- ignore no-op selections
        if int(index) != self.names.index(self.robot_name):
            self._select_robot(int(index))

    def _cycle_robot(self, step):
        self._select_robot((self.names.index(self.robot_name) + step) % len(self.names))

    def _set_speed(self, value):
        self.speed = float(value) / 100

    def _set_step(self, index):
        self.step_index = int(index)

    def _slider_moved(self, joint, value):
        # Swift reports every change of a slider's value, including the ones the pendant pushes
        # to make the sliders follow the robot; those echoes are not operator input
        value = float(value)
        if any(abs(value - pushed) < SLIDER_STEP / 2 for pushed in self.slider_pushed[joint]):
            return
        if self.estop or self.auto_busy or joint >= self.robot.n:
            return
        if self.joint_target is None:
            self._stop_motion()
            self.joint_target = np.array(self.robot.q, dtype=float)
        self.joint_target[joint] = np.radians(float(value))
        self.sliders_follow = False  # don't fight the operator's drag

    def _step_joint(self, sign):
        if self._refuse_if_estopped("Joint jog") or self.jog_joint >= self.robot.n:
            return
        r = self.robot
        target = np.array(r.q, dtype=float) if self.joint_target is None else self.joint_target
        self._stop_motion()
        j = self.jog_joint
        lo, hi = r.qlim[:, j]
        wanted = target[j] + sign * np.radians(STEP_SIZES[self.step_index][2])
        if not lo <= wanted <= hi:
            self._log(WARNING, f"J{j + 1} step stopped at its limit")
        target[j] = np.clip(wanted, lo, hi)
        self.joint_target = target
        self.sliders_follow = True

    def _step_cartesian(self, axis, sign):
        if self._refuse_if_estopped("Cartesian jog"):
            return
        if self.cart_goal is None:
            self._stop_motion()
            self.cart_goal = self.robot.fkine(self.robot.q).A.copy()  # holds the orientation and other axes
        start = self.cart_path[-1] if self.cart_path else self.cart_goal[:3, 3]
        end = start.copy()
        end[axis] += sign * STEP_SIZES[self.step_index][1]
        n = max(1, int(np.ceil(abs(end[axis] - start[axis]) / (CART_SPEED * self.speed * TICK))))
        self.cart_path += [start + (end - start) * k / n for k in range(1, n + 1)]
        self.sliders_follow = True

    def _refuse_if_estopped(self, what):
        if self.estop:
            self._log(WARNING, f"{what} refused - E-STOP active")
        elif self.auto_busy:
            self._log(WARNING, f"{what} refused - automatic sequence running")
        return self.estop or self.auto_busy

    def _stop_motion(self):
        self.joint_target = None
        self.cart_path = []
        self.cart_goal = None
        self.playback = []

    def _press_estop(self):
        if not self.estop:
            self.estop = True
            self._stop_motion()
            self._log(FAULT, "E-STOP pressed - all motion stopped")

    def _reset_estop(self):
        if self.estop:
            self.estop = False
            self.awaiting_resume = True
            self._log(OK, "E-STOP reset - press Resume to continue")

    def _record_pose(self):
        label = f"P{len(self.taught) + 1}"
        self.taught.append({"robot": self.robot_name, "name": label, "q": np.array(self.robot.q, dtype=float)})
        self.selected_pose = len(self.taught) - 1
        self._refresh_pose_select()
        self._log(OK, f"Recorded {label} for {self.robot_name}")

    def _refresh_pose_select(self):
        options = []
        for pose in self.taught:
            options.append(f"{pose['name']} | {pose['robot']}")
        self.selected_pose = min(self.selected_pose, max(len(options) - 1, 0))
        self.pose_select.options = options or ["No positions recorded"]
        self.pose_select.value = self.selected_pose

    def _goto_selected(self):
        if not self.taught:
            self._log(WARNING, "Go to: record or load a pose first")
            return
        pose = self.taught[self.selected_pose]
        if pose["robot"] not in self.robots:
            self._log(WARNING, f"Go to: robot {pose['robot']} is not in this workcell")
            return
        if pose["robot"] != self.robot_name:
            self._select_robot(self.names.index(pose["robot"]))
        self._plan_joint_move(pose["q"], pose["name"])

    def _goto_start(self):
        self._plan_joint_move(self.start_q[self.robot_name], "start pose")

    def _plan_joint_move(self, q_goal, label):
        if self._refuse_if_estopped(f"Go to {label}"):
            return
        q_now = np.array(self.robot.q, dtype=float)
        step = JOINT_SPEED * self.speed * TICK
        steps = max(2, int(np.ceil(np.abs(q_goal - q_now).max() / step * 1.5)))  # jtraj peaks ~1.5x mean speed
        self._stop_motion()
        self.playback = list(jtraj(q_now, q_goal, steps).q[1:])
        self.sliders_follow = True
        self._log(OK, f"Moving {self.robot_name} to {label}")

    def _delete_pose(self):
        if self.taught:
            pose = self.taught.pop(self.selected_pose)
            self._refresh_pose_select()
            self._log(OK, f"Deleted {pose['name']}")

    def _save_poses(self):
        data = [{"robot": p["robot"], "name": p["name"], "q_deg": np.degrees(p["q"]).round(3).tolist()}
                for p in self.taught]
        with open(self.poses_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        self._log(OK, f"Saved {len(data)} pose(s) to {self.poses_file}")

    def _load_poses(self):
        try:
            with open(self.poses_file, encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, ValueError) as err:
            self._log(WARNING, f"Could not load {self.poses_file}: {err}")
            return
        self.taught = [{"robot": d["robot"], "name": d["name"], "q": np.radians(d["q_deg"])} for d in data]
        self.selected_pose = 0
        self._refresh_pose_select()
        self._log(OK, f"Loaded {len(self.taught)} pose(s) from {self.poses_file}")

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    def tick(self):
        """One control cycle: gamepad, motion, monitoring, panel update. Call before each env.step()."""
        self.conditions = {}
        self._poll_gamepad()
        if not self.estop and not self.auto_busy:
            self._apply_motion()
        self._check_state()
        self._refresh_panel()

    def _apply_motion(self):
        if self.playback:
            if not self._try_set_q(self.playback.pop(0)):
                self.playback = []
            return

        pad = self._pad_command()
        if pad is not None:
            self.joint_target = None
            self.cart_path = []
            self.sliders_follow = True
            kind, direction = pad
            if kind == "joint":
                self.cart_goal = None
                self._jog_joints(direction)
            else:
                if self.cart_goal is None:
                    self.cart_goal = self.robot.fkine(self.robot.q).A.copy()
                self.cart_goal[:3, 3] += direction * CART_SPEED * self.speed * TICK
                self._servo_cartesian()
            return

        if self.cart_path:
            self.cart_goal[:3, 3] = self.cart_path.pop(0)
            if not self._servo_cartesian():
                self.cart_path = []
            return
        self.cart_goal = None

        if self.joint_target is not None:
            q = np.array(self.robot.q, dtype=float)
            step = JOINT_SPEED * self.speed * TICK
            dq = self.joint_target - q
            if np.abs(dq).max() <= step:
                moved = self._try_set_q(self.joint_target)
                self.joint_target = None
                self.sliders_follow = True
            else:
                moved = self._try_set_q(q + np.clip(dq, -step, step))
            if not moved:
                self.joint_target = None
                self.sliders_follow = True

    def _jog_joints(self, direction):
        r = self.robot
        q = np.array(r.q, dtype=float) + direction * JOINT_SPEED * self.speed * TICK
        lo, hi = r.qlim
        for i in np.flatnonzero((q < lo) | (q > hi)):
            self._flag(WARNING, f"J{i + 1} at its limit - jog in that direction blocked")
        self._try_set_q(np.clip(q, lo, hi))

    def _servo_cartesian(self):
        """Closed-loop resolved-rate step onto self.cart_goal. Returns False if the step was blocked."""
        r = self.robot
        q = np.array(r.q, dtype=float)
        T_actual = r.fkine(q)
        lin_vel = (self.cart_goal[:3, 3] - T_actual.t) / TICK
        ang_vel = vex((self.cart_goal[:3, :3] - T_actual.R) / TICK @ T_actual.R.T)
        xdot = np.concatenate([lin_vel, ang_vel])

        J = r.jacob0(q)
        m = self._manipulability(J)
        damping = MAX_DAMPING * (1 - m / MANIP_WARN) ** 2 if m < MANIP_WARN else 0.0
        qdot = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), xdot)
        qdot /= max(1.0, np.abs(qdot).max() / MAX_QDOT)
        q_new = q + qdot * TICK

        lo, hi = r.qlim
        blocked = np.flatnonzero((q_new < lo) | (q_new > hi))
        if blocked.size:
            joints = ", ".join(f"J{i + 1}" for i in blocked)
            self._flag(WARNING, f"Cartesian jog blocked - {joints} would pass its limit")
            self.cart_goal = None  # don't wind up a goal the arm can't reach
            return False
        if not self._try_set_q(q_new):
            self.cart_goal = None
            return False
        return True

    def _try_set_q(self, q):
        """Apply q unless it drives the tool further below the safety plane."""
        r = self.robot
        z_now = r.fkine(r.q).t[2]
        z_new = r.fkine(q).t[2]
        if z_new < self.min_tool_z and z_new < z_now:
            self._flag(FAULT, f"Motion blocked - tool would go below the safety plane (z < {self.min_tool_z:.3f} m)")
            return False
        r.q = q
        return True

    @staticmethod
    def _manipulability(J):
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    def _flag(self, level, message):
        self.conditions[message] = max(level, self.conditions.get(message, OK))

    def _check_state(self):
        r = self.robot
        q = np.array(r.q, dtype=float)
        if self.estop:
            self._flag(FAULT, "E-STOP active - press Reset E-stop to resume")
        if self.collision:
            self._flag(FAULT, self.collision)
        elif self.awaiting_resume:
            self._flag(WARNING, "Reset done - press Resume to continue")
        lo, hi = r.qlim
        for i in np.flatnonzero(np.minimum(q - lo, hi - q) < LIMIT_WARN):
            self._flag(WARNING, f"J{i + 1} near joint limit")
        if self._manipulability(r.jacob0(q)) < MANIP_WARN:
            self._flag(WARNING, "Near singularity - Cartesian jog is damped")
        if r.fkine(q).t[2] < self.min_tool_z:
            self._flag(WARNING, "Tool is below the safety plane - jog upwards")

        # Log each condition once, when it first appears
        for message, level in self.conditions.items():
            if message not in self.prev_conditions:
                self._log(level, message)
        self.prev_conditions = self.conditions

    def _set_label(self, label, text):
        if label.desc != text:  # only changed labels are resent to the browser
            label.desc = text

    def _refresh_panel(self):
        r = self.robot
        q = np.array(r.q, dtype=float)

        # Status line: the worst current condition, else the current activity
        if self.estop:
            text = "E-STOP ACTIVE | Reset then Resume"
        elif self.collision:
            text = "PATH BLOCKED | Clear obstacle then resume"
        elif self.awaiting_resume:
            text = "PAUSED | Awaiting resume"
        elif self.conditions:
            message, level = max(self.conditions.items(), key = lambda item: item[1])
            state = "FAULT" if level == FAULT else "WARNING"
            extra = len(self.conditions) - 1
            text = f"{LEVEL_ICON[level]} {state} | {message}"
            if extra:
                text += f" (+{extra} more)"
        elif self.auto_busy:
            text = "AUTOMATIC | Assembly sequence running"
        else:
            text = f"MANUAL | {self._activity()}"
        self._set_label(self.status_label, text)

        T = r.fkine(q)
        rpy = np.degrees(T.rpy(order="xyz"))
        m = self._manipulability(r.jacob0(q))
        lo, hi = r.qlim
        joints = []
        for i in range(r.n):
            near = min(q[i] - lo[i], hi[i] - q[i]) < LIMIT_WARN
            joints.append(f"J{i + 1} {np.degrees(q[i]):+.1f}{'⚠' if near else ''}")
        for label, axis, value in zip(self.position_labels, ("X", "Y", "Z"), T.t):
            self._set_label(label, f"{axis}: {value:+.3f} m")
        for label, axis, value in zip(self.orientation_labels, ("Roll", "Pitch", "Yaw"), rpy):
            self._set_label(label, f"{axis}: {value:+.1f} deg")
        self._set_label(self.joint_label, "Joints (°): " + "  ".join(joints))
        self._set_label(self.manip_label, f"Manipulability: {m:.2e} ({'LOW' if m < MANIP_WARN else 'ok'})"
                                          f" | Safety plane z = {self.min_tool_z:.3f} m")
        pad_status, pad_map = self._pad_summary()
        self._set_label(self.pad_label, pad_status)
        self._set_label(self.pad_map_label, pad_map)

        # Keep the joint sliders on the robot's actual angles unless the operator is dragging one
        if self.sliders_follow:
            for j, (slider, value) in enumerate(zip(self.joint_sliders, self._slider_values(r))):
                if slider.value != value:
                    slider.value = value
                    self.slider_pushed[j] = (self.slider_pushed[j] + [value])[-ECHO_HISTORY:]

    @staticmethod
    def _on_grid(degrees):
        return round(round(degrees / SLIDER_STEP) * SLIDER_STEP, 1)

    def _slider_values(self, robot):
        q = np.degrees(np.array(robot.q, dtype=float))
        n_max = max(r.n for r in self.robots.values())
        return [self._on_grid(q[j]) if j < robot.n else 0.0 for j in range(n_max)]

    def _activity(self):
        if self.auto_busy:
            return "automatic sequence running"
        if self.playback:
            return "moving to pose"
        pad = self._pad_command()
        if pad is not None:
            return f"gamepad {'joint' if pad[0] == 'joint' else 'Cartesian'} jog"
        if self.cart_path:
            return "Cartesian step"
        if self.joint_target is not None:
            return "joint move"
        return "Idle"

    def _log(self, level, message):
        line = f"{time.strftime('%H:%M:%S')}  {LEVEL_TAG[level]}  {message}"
        print("[pendant] " + line)
        self.events = (self.events + [f"{LEVEL_ICON[level]} {line}"])[-LOG_LINES:]
        if hasattr(self, "log_labels"):
            for label, text in zip(self.log_labels, reversed(self.events)):  # newest first
                self._set_label(label, text)

    # ------------------------------------------------------------------
    # Gamepad
    # ------------------------------------------------------------------
    def _init_gamepad(self):
        if pygame is None:
            return
        pygame.display.init()  # needed for the event queue; no window is opened
        pygame.joystick.init()

    def _poll_gamepad(self):
        if pygame is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEREMOVED and self.gamepad is not None \
                    and event.instance_id == self.gamepad.get_instance_id():
                self.gamepad = None
                self._log(WARNING, "Gamepad disconnected - gamepad motion stopped")
        if self.gamepad is None:
            if pygame.joystick.get_count() > 0:
                self.gamepad = pygame.joystick.Joystick(0)
                self.gamepad.init()
                self.pad_buttons = tuple(self.gamepad.get_button(b) for b in range(self.gamepad.get_numbuttons()))
                self._log(OK, f"Gamepad connected: {self.gamepad.get_name()}")
            return

        # Button presses act on the rising edge only
        buttons = tuple(self.gamepad.get_button(b) for b in range(self.gamepad.get_numbuttons()))
        pressed = {b for b, (now, before) in enumerate(zip(buttons, self.pad_buttons)) if now and not before}
        self.pad_buttons = buttons
        if BTN_B in pressed:
            self._press_estop()
        if BTN_START in pressed:
            # One button, two deliberate presses: first resets the E-stop, second resumes
            if self.estop:
                self._reset_estop()
            else:
                self._resume()
        if BTN_A in pressed:
            self.pad_mode = "joint" if self.pad_mode == "cartesian" else "cartesian"
            self._log(OK, f"Gamepad mode: {self.pad_mode}")
        if BTN_LB in pressed:
            self._cycle_robot(-1)
        if BTN_RB in pressed:
            self._cycle_robot(1)
        if BTN_Y in pressed:
            self._record_pose()

        # D-pad up/down selects the joint in joint mode
        hat = self.gamepad.get_hat(0) if self.gamepad.get_numhats() else (0, 0)
        if hat[1] != self.pad_hat[1] and hat[1] != 0:
            self.pad_joint = (self.pad_joint - hat[1]) % self.robot.n
        self.pad_hat = hat

    def _pad_axis(self, axis):
        if self.gamepad is None or axis >= self.gamepad.get_numaxes():
            return 0.0
        value = self.gamepad.get_axis(axis)
        if abs(value) < PAD_DEADZONE:
            return 0.0
        return float(np.sign(value) * (abs(value) - PAD_DEADZONE) / (1 - PAD_DEADZONE))

    def _pad_command(self):
        if self.gamepad is None or self.estop:
            return None
        if self.pad_mode == "cartesian":
            direction = np.array([self._pad_axis(AXIS_LX), -self._pad_axis(AXIS_LY), -self._pad_axis(AXIS_RY)])
            return ("cart", direction) if direction.any() else None
        value = -self._pad_axis(AXIS_LY)
        if value == 0:
            return None
        direction = np.zeros(self.robot.n)
        direction[self.pad_joint] = value
        return ("joint", direction)

    def _pad_summary(self):
        if pygame is None:
            return "🎮 Controller | Unavailable - use panel controls", ""
        if self.gamepad is None:
            return "🎮 Controller | Not connected", ""
        mapping = ("L-stick X/Y = world X/Y, R-stick Y = Z" if self.pad_mode == "cartesian"
                   else f"D-pad selects joint, L-stick Y jogs J{self.pad_joint + 1}")
        return (f"🎮 {self.gamepad.get_name()[:40]} - mode {self.pad_mode.upper()} (A toggles)",
                f"{mapping} | B = E-stop, Start = reset then resume (press twice), LB/RB = robot, Y = record")