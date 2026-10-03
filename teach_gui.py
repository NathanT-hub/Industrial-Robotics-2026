"""
Advanced teach / jog pendant for the workcell.

TeachPendant opens a tkinter window that drives the robots already in the
Swift scene:
  - individual joint jogging (hold the -/+ buttons, or drag the joint slider)
  - Cartesian jogging of the end-effector in world x, y and z, with the tool
    orientation held (closed-loop resolved-rate control, damped near
    singularities)
  - live state: end-effector pose, every joint against its limits,
    manipulability, gamepad status, a status banner and a timestamped log of
    warnings / faults
  - a latching software E-stop, and taught poses (record / go to / save / load)
  - an optional gamepad (pygame) that mirrors the mouse controls; everything it
    does is reflected in the window so the operator can always see the state

The pendant owns the main loop -- it steps Swift from a tkinter timer -- so
call run() last in main(), in place of env.hold():

    pendant = TeachPendant(env, {"DoBot6": robot, "RS007N": nathanBot})
    pendant.run()
"""
import json
import os
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from roboticstoolbox import jtraj
from spatialmath.base import vex

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ImportError:  # the gamepad is optional -- the mouse controls still work
    pygame = None


TICK = 0.05                  # control / render period (s)
JOINT_SPEED = np.radians(30) # joint jog speed at 100% (rad/s)
CART_SPEED = 0.10            # Cartesian jog speed at 100% (m/s)
MAX_QDOT = np.radians(90)    # cap on any joint's speed during a Cartesian jog (rad/s)
LIMIT_WARN = np.radians(5)   # warn when a joint is this close to a limit
MANIP_WARN = 1e-3            # warn below this manipulability (same threshold as move_arm_rmrc)
MAX_DAMPING = 0.05           # damped least squares near singularities (as in move_arm_rmrc)

# Gamepad layout (XInput order as reported by pygame on Windows, e.g. an Xbox pad)
PAD_DEADZONE = 0.15
AXIS_LX, AXIS_LY, AXIS_RY = 0, 1, 3
BTN_A, BTN_B, BTN_Y, BTN_LB, BTN_RB, BTN_START = 0, 1, 3, 4, 5, 7

OK, WARNING, FAULT = 0, 1, 2
BANNER_COLOURS = {OK: "#2e7d32", WARNING: "#f9a825", FAULT: "#c62828"}
ROW_COLOURS = {OK: "#d9d9d9", WARNING: "#ffd54f", FAULT: "#ef5350"}


class TeachPendant:
    """Teach / jog GUI for one or more robots that are already in `env`."""

    def __init__(self, env, robots, min_tool_z=0.6, poses_file="taught_poses.json"):
        """
        env        -- the Swift environment the robots were added to
        robots     -- dict of display name -> robot model
        min_tool_z -- safety plane: jogs that take the tool further below this
                      world z (e.g. into the table top) are blocked
        poses_file -- JSON file used by the Save / Load taught-pose buttons
        """
        self.env = env
        self.robots = dict(robots)
        self.min_tool_z = min_tool_z
        self.poses_file = poses_file
        self.start_q = {name: np.array(r.q, dtype=float) for name, r in self.robots.items()}

        self.estop = False
        self.mouse_jog = None       # ("joint", qdot direction) or ("cart", xyz direction) while a button is held
        self.slider_target = None   # joint target set by dragging a slider
        self.playback = []          # queued joint configurations (go to taught / start pose)
        self.cart_goal = None       # end-effector goal integrated during a Cartesian jog
        self.conditions = {}        # message -> level, rebuilt every tick
        self.prev_conditions = {}
        self.taught = []            # dicts: robot, name, q
        self.dragging = None        # joint index whose slider the mouse is holding
        self.syncing = False        # True while the code (not the user) moves the sliders

        self.gamepad = None
        self.pad_mode = "cartesian"
        self.pad_joint = 0
        self.pad_buttons = ()
        self.pad_hat = (0, 0)
        self.pad_scan_time = 0.0

        self.root = tk.Tk()
        self.root.title("Workcell Teach Pendant")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.robot_name = tk.StringVar(value=next(iter(self.robots)))
        self.speed = tk.IntVar(value=50)
        self._build_ui()
        self._init_gamepad()
        self._select_robot()

    @property
    def robot(self):
        return self.robots[self.robot_name.get()]

    def run(self):
        """Start the pendant; blocks until the window is closed."""
        self._log(OK, "Pendant started - " + ", ".join(self.robots))
        self.root.after(1, self._tick)
        self.root.mainloop()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = self.root
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)

        # Status banner + E-stop across the top
        top = tk.Frame(root)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        top.columnconfigure(0, weight=1)
        self.banner = tk.Label(top, text="READY", font=("Segoe UI", 14, "bold"), fg="white",
                               bg=BANNER_COLOURS[OK], anchor="w", padx=10, pady=8)
        self.banner.grid(row=0, column=0, sticky="ew")
        tk.Button(top, text="E-STOP", font=("Segoe UI", 14, "bold"), bg="#c62828", fg="white",
                  activebackground="#8e0000", width=9, command=self._press_estop).grid(row=0, column=1, padx=(6, 0))
        tk.Button(top, text="Reset", font=("Segoe UI", 11), width=7,
                  command=self._reset_estop).grid(row=0, column=2, padx=(6, 0), sticky="ns")

        left = ttk.Frame(root)
        left.grid(row=1, column=0, sticky="nsew", padx=6)
        right = ttk.Frame(root)
        right.grid(row=1, column=1, sticky="nsew", padx=6)

        # Robot selection + speed
        sel = ttk.LabelFrame(left, text="Robot")
        sel.pack(fill="x", pady=3)
        for name in self.robots:
            ttk.Radiobutton(sel, text=name, value=name, variable=self.robot_name,
                            command=self._select_robot).pack(side="left", padx=6, pady=3)
        spd = ttk.LabelFrame(left, text="Jog speed (%)")
        spd.pack(fill="x", pady=3)
        tk.Scale(spd, from_=5, to=100, orient="horizontal", variable=self.speed,
                 resolution=5, showvalue=True).pack(fill="x", padx=6)

        # Joint jog rows (rebuilt when the robot changes)
        self.joint_frame = ttk.LabelFrame(left, text="Joint jog (hold -/+ or drag)")
        self.joint_frame.pack(fill="x", pady=3)
        self.joint_rows = []

        # Cartesian jog
        cart = ttk.LabelFrame(left, text="Cartesian jog - world frame, tool orientation held (hold)")
        cart.pack(fill="x", pady=3)
        for axis, label in enumerate("XYZ"):
            for col, sign in enumerate((-1, 1)):
                b = ttk.Button(cart, text=f"{label} {'-' if sign < 0 else '+'}", width=8)
                b.grid(row=axis, column=col, padx=4, pady=2)
                direction = np.zeros(3)
                direction[axis] = sign
                self._bind_hold(b, ("cart", direction))
        self.cart_info = ttk.Label(cart, text="", justify="left")
        self.cart_info.grid(row=0, column=2, rowspan=3, padx=10, sticky="w")

        # State readout
        state = ttk.LabelFrame(right, text="State")
        state.pack(fill="x", pady=3)
        self.state_text = tk.Label(state, text="", font=("Consolas", 10), justify="left", anchor="w")
        self.state_text.pack(fill="x", padx=6, pady=3)

        # Gamepad
        pad = ttk.LabelFrame(right, text="Gamepad")
        pad.pack(fill="x", pady=3)
        self.pad_text = tk.Label(pad, text="", font=("Consolas", 9), justify="left", anchor="w")
        self.pad_text.pack(fill="x", padx=6, pady=3)

        # Taught poses
        teach = ttk.LabelFrame(right, text="Taught poses")
        teach.pack(fill="x", pady=3)
        self.pose_list = tk.Listbox(teach, height=6, font=("Consolas", 9))
        self.pose_list.pack(fill="x", padx=6, pady=3)
        btns = ttk.Frame(teach)
        btns.pack(fill="x", padx=6, pady=(0, 4))
        for text, cmd in (("Record", self._record_pose), ("Go to", self._goto_selected),
                          ("Delete", self._delete_pose), ("Start pose", self._goto_start),
                          ("Save", self._save_poses), ("Load", self._load_poses)):
            ttk.Button(btns, text=text, width=9, command=cmd).pack(side="left", padx=1)

        # Event / fault log
        logf = ttk.LabelFrame(root, text="Event log")
        logf.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)
        root.rowconfigure(2, weight=1)
        self.log = tk.Text(logf, height=8, font=("Consolas", 9), state="disabled")
        self.log.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(logf, command=self.log.yview)
        scroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scroll.set)
        for level, colour in ((WARNING, "#b26a00"), (FAULT, "#c62828")):
            self.log.tag_configure(str(level), foreground=colour)

    def _build_joint_rows(self):
        for row in self.joint_rows:
            row["frame"].destroy()
        self.joint_rows = []
        qlim = np.degrees(self.robot.qlim)
        for i in range(self.robot.n):
            f = tk.Frame(self.joint_frame)
            f.pack(fill="x", padx=4, pady=1)
            name = tk.Label(f, text=f"J{i + 1}", width=3)
            name.pack(side="left")
            minus = ttk.Button(f, text="-", width=3)
            minus.pack(side="left")
            scale = tk.Scale(f, from_=qlim[0, i], to=qlim[1, i], orient="horizontal", resolution=0.5,
                             showvalue=False, length=220, command=lambda v, j=i: self._slider_moved(j, v))
            scale.pack(side="left", padx=2)
            scale.bind("<ButtonPress-1>", lambda e, j=i: setattr(self, "dragging", j))
            scale.bind("<ButtonRelease-1>", lambda e: setattr(self, "dragging", None))
            plus = ttk.Button(f, text="+", width=3)
            plus.pack(side="left")
            value = tk.Label(f, text="", width=26, anchor="w", font=("Consolas", 9))
            value.pack(side="left", padx=4)
            for btn, sign in ((minus, -1), (plus, 1)):
                direction = np.zeros(self.robot.n)
                direction[i] = sign
                self._bind_hold(btn, ("joint", direction))
            self.joint_rows.append({"frame": f, "name": name, "scale": scale, "value": value})

    def _bind_hold(self, widget, command):
        """Jog continuously while the mouse button is held on `widget`."""
        widget.bind("<ButtonPress-1>", lambda e: self._start_mouse_jog(command))
        widget.bind("<ButtonRelease-1>", lambda e: self._stop_mouse_jog())

    # ------------------------------------------------------------------
    # Operator actions
    # ------------------------------------------------------------------
    def _select_robot(self):
        self._stop_motion()
        self.pad_joint = 0
        self._build_joint_rows()
        self._log(OK, f"Selected {self.robot_name.get()}")

    def _cycle_robot(self, step):
        names = list(self.robots)
        i = (names.index(self.robot_name.get()) + step) % len(names)
        self.robot_name.set(names[i])
        self._select_robot()

    def _start_mouse_jog(self, command):
        self.mouse_jog = command

    def _stop_mouse_jog(self):
        self.mouse_jog = None

    def _slider_moved(self, joint, value):
        if self.syncing or self.dragging != joint or self.estop:
            return
        if self.slider_target is None:
            self.slider_target = np.array(self.robot.q, dtype=float)
        self.slider_target[joint] = np.radians(float(value))

    def _stop_motion(self):
        self.mouse_jog = None
        self.slider_target = None
        self.playback = []
        self.cart_goal = None

    def _press_estop(self):
        if not self.estop:
            self.estop = True
            self._stop_motion()
            self._log(FAULT, "E-STOP pressed - all motion stopped")

    def _reset_estop(self):
        if self.estop:
            self.estop = False
            self._log(OK, "E-STOP reset - motion enabled")

    def _record_pose(self):
        name = self.robot_name.get()
        q = np.array(self.robot.q, dtype=float)
        label = f"P{len(self.taught) + 1}"
        self.taught.append({"robot": name, "name": label, "q": q})
        self._refresh_pose_list()
        self._log(OK, f"Recorded {label} for {name}")

    def _refresh_pose_list(self):
        self.pose_list.delete(0, "end")
        for pose in self.taught:
            r = self.robots.get(pose["robot"])
            xyz = r.fkine(pose["q"]).t if r is not None else (np.nan,) * 3
            self.pose_list.insert("end", f"{pose['name']:<4} {pose['robot']:<12} "
                                         f"x{xyz[0]:+.3f} y{xyz[1]:+.3f} z{xyz[2]:+.3f}")

    def _goto_selected(self):
        sel = self.pose_list.curselection()
        if not sel:
            self._log(WARNING, "Go to: select a taught pose first")
            return
        pose = self.taught[sel[0]]
        if pose["robot"] not in self.robots:
            self._log(WARNING, f"Go to: robot {pose['robot']} is not in this workcell")
            return
        if pose["robot"] != self.robot_name.get():
            self.robot_name.set(pose["robot"])
            self._select_robot()
        self._plan_joint_move(pose["q"], pose["name"])

    def _goto_start(self):
        self._plan_joint_move(self.start_q[self.robot_name.get()], "start pose")

    def _plan_joint_move(self, q_goal, label):
        if self.estop:
            self._log(WARNING, f"Go to {label} refused - E-STOP active")
            return
        q_now = np.array(self.robot.q, dtype=float)
        step = JOINT_SPEED * self.speed.get() / 100 * TICK
        steps = max(2, int(np.ceil(np.abs(q_goal - q_now).max() / step * 1.5)))  # jtraj peaks ~1.5x mean speed
        self._stop_motion()
        self.playback = list(jtraj(q_now, q_goal, steps).q[1:])
        self._log(OK, f"Moving {self.robot_name.get()} to {label}")

    def _delete_pose(self):
        sel = self.pose_list.curselection()
        if sel:
            pose = self.taught.pop(sel[0])
            self._refresh_pose_list()
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
        self._refresh_pose_list()
        self._log(OK, f"Loaded {len(self.taught)} pose(s) from {self.poses_file}")

    def _close(self):
        if pygame is not None:
            pygame.quit()
        self.root.destroy()

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    def _tick(self):
        self.conditions = {}
        self._poll_gamepad()
        if not self.estop:
            self._apply_motion()
        self._check_state()
        self._refresh_display()
        self.env.step(TICK)
        self.root.after(1, self._tick)

    def _apply_motion(self):
        if self.playback:
            if not self._try_set_q(self.playback.pop(0)):
                self.playback = []
            return

        command = self.mouse_jog or self._pad_command()
        if command is None:
            self.cart_goal = None
            if self.slider_target is not None:
                self._move_toward(self.slider_target)
            return

        self.slider_target = None
        kind, direction = command
        if kind == "joint":
            self.cart_goal = None
            self._jog_joints(direction)
        else:
            self._jog_cartesian(direction)

    def _move_toward(self, q_goal):
        q = np.array(self.robot.q, dtype=float)
        step = JOINT_SPEED * self.speed.get() / 100 * TICK
        dq = q_goal - q
        if np.abs(dq).max() <= step:
            self._try_set_q(q_goal)
            if self.dragging is None:
                self.slider_target = None
        else:
            self._try_set_q(q + np.clip(dq, -step, step))

    def _jog_joints(self, direction):
        r = self.robot
        q = np.array(r.q, dtype=float) + direction * JOINT_SPEED * self.speed.get() / 100 * TICK
        lo, hi = r.qlim
        for i in np.flatnonzero((q < lo) | (q > hi)):
            self._flag(WARNING, f"J{i + 1} at its limit - jog in that direction blocked")
        self._try_set_q(np.clip(q, lo, hi))

    def _jog_cartesian(self, direction):
        """Closed-loop resolved-rate jog: integrate a goal pose, then servo the arm onto it."""
        r = self.robot
        q = np.array(r.q, dtype=float)
        T_actual = r.fkine(q)
        if self.cart_goal is None:
            self.cart_goal = T_actual.A.copy()  # holds the orientation and the other axes
        self.cart_goal[:3, 3] += np.asarray(direction) * CART_SPEED * self.speed.get() / 100 * TICK

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
            return
        if not self._try_set_q(q_new):
            self.cart_goal = None

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
            self._flag(FAULT, "E-STOP active - press Reset to resume")
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

    def _refresh_display(self):
        r = self.robot
        q = np.array(r.q, dtype=float)
        lo, hi = r.qlim

        # Banner: the worst current condition, else the current activity
        if self.conditions:
            message, level = max(self.conditions.items(), key=lambda kv: kv[1])
            extra = len(self.conditions) - 1
            text = f"{'FAULT' if level == FAULT else 'WARNING'}: {message}" + (f"  (+{extra} more)" if extra else "")
        else:
            level, text = OK, f"READY - {self.robot_name.get()} - {self._activity()}"
        self.banner.config(text=text, bg=BANNER_COLOURS[level])

        # Joint rows
        self.syncing = True
        for i, row in enumerate(self.joint_rows):
            margin = min(q[i] - lo[i], hi[i] - q[i])
            level = FAULT if margin <= 1e-6 else WARNING if margin < LIMIT_WARN else OK
            row["scale"].config(troughcolor=ROW_COLOURS[level])
            if self.dragging != i:
                row["scale"].set(np.degrees(q[i]))
            row["value"].config(text=f"{np.degrees(q[i]):+8.1f} deg  [{np.degrees(lo[i]):+.0f}, {np.degrees(hi[i]):+.0f}]")
            selected = self.gamepad is not None and self.pad_mode == "joint" and i == self.pad_joint
            row["name"].config(bg="#64b5f6" if selected else row["frame"].cget("bg"))
        self.syncing = False

        # State panel
        T = r.fkine(q)
        rpy = np.degrees(T.rpy(order="xyz"))
        m = self._manipulability(r.jacob0(q))
        self.state_text.config(text=(
            f"Robot        {self.robot_name.get()}\n"
            f"Mode         {self._activity()}\n"
            f"Tool  x y z  {T.t[0]:+.3f} {T.t[1]:+.3f} {T.t[2]:+.3f} m\n"
            f"Tool  r p y  {rpy[0]:+7.1f} {rpy[1]:+7.1f} {rpy[2]:+7.1f} deg\n"
            f"Manipulab.   {m:.2e}  ({'LOW' if m < MANIP_WARN else 'ok'})\n"
            f"E-stop       {'ACTIVE' if self.estop else 'clear'}"))
        self.cart_info.config(text=f"Speed {CART_SPEED * self.speed.get():.1f} cm/s\n"
                                   f"Safety plane z = {self.min_tool_z:.3f} m")
        self.pad_text.config(text=self._pad_summary())

    def _activity(self):
        if self.estop:
            return "stopped"
        if self.playback:
            return "moving to pose"
        command = self.mouse_jog or self._pad_command()
        if command:
            return "joint jog" if command[0] == "joint" else "Cartesian jog"
        if self.slider_target is not None:
            return "moving to slider"
        return "idle"

    def _log(self, level, message):
        stamp = time.strftime("%H:%M:%S")
        tag = {OK: "INFO ", WARNING: "WARN ", FAULT: "FAULT"}[level]
        self.log.configure(state="normal")
        self.log.insert("end", f"{stamp}  {tag}  {message}\n", str(level))
        self.log.see("end")
        self.log.configure(state="disabled")

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
            self._reset_estop()
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
        if self.gamepad is None:
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
            return "pygame not installed - mouse control only"
        if self.gamepad is None:
            return "No gamepad connected (plug one in at any time)"
        mapping = ("Left stick X/Y -> world X/Y, right stick Y -> Z" if self.pad_mode == "cartesian"
                   else f"D-pad up/down select joint, left stick Y jogs J{self.pad_joint + 1}")
        return (f"{self.gamepad.get_name()[:40]}\n"
                f"Mode: {self.pad_mode.upper()}   (A toggles)\n"
                f"{mapping}\n"
                f"B = E-stop  Start = reset  LB/RB = robot  Y = record")
