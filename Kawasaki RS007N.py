# Nathan-Tyas-StudentRobot-Initialiser
"""
kawasaki_rs007n.py
 
Kawasaki RS007N -- 6-DOF industrial robot arm.
Individual parameter-based model for 41013 Assignment 2.
 
KINEMATIC SOURCE (real, cited, not fabricated)
-----------------------------------------------
Kawasaki does not publish a link-length table in its RS007N datasheet, so the
link geometry below is transcribed directly from Kawasaki's own official ROS
package on GitHub:
 
    Kawasaki-Robotics/khi_robot
    khi_rs_description/urdf/rs007n_macro.xacro
    https://github.com/Kawasaki-Robotics/khi_robot/blob/master/khi_rs_description/urdf/rs007n_macro.xacro
 
Joint range and velocity limits are taken from Kawasaki's official RS007N-B
datasheet:
    https://kawasakirobotics.com/uploads/sites/2/2022/01/specifications_robots_small-medium-payload-robots_rs_rs007n_en_01_2021.pdf
 
WHY THIS IS BUILT AS AN ETS CHAIN, NOT A CLASSIC (a, alpha, d, theta) TABLE
----------------------------------------------------------------------------
The xacro's own joint chain is built from translations plus fixed +/-90 deg
rotations about *Y* between consecutive joints. A classic Denavit-Hartenberg
table instead rotates about *X* between links, using the common-normal
construction -- converting one to the other is a non-trivial re-derivation
(different axis conventions, not just relabelled numbers). A first attempt at
that conversion here produced a table that did NOT reproduce the official
model when checked numerically, so rather than hand a table with a hidden
sign error to a real assignment submission, this file instead transcribes
Kawasaki's own chain directly as an RTB elementary transform sequence (ETS) --
still a fully parameter-based RTB model (RTB ships its own Panda and Puma560
models in this same ETS form, alongside their classic-DH versions).
 
VERIFICATION PERFORMED
-----------------------
1. This ETS chain was checked against a from-scratch re-implementation of the
   xacro's transform tree (independent of RTB) and matches to ~1e-15 over 300
   random joint configurations -- i.e. it reproduces the official model to
   floating-point precision.
2. Independent cross-check against the datasheet: this model's reach at full
   horizontal stretch is 0.808 m to the tool flange. Subtracting the final
   flange offset (0.078 m) gives 0.730 m -- an EXACT match to Kawasaki's
   quoted 730 mm maximum reach. This is strong evidence the transcription is
   correct (reach is normally quoted to the wrist reference point, not the
   flange tip).
 
KNOWN SOURCE DISCREPANCY (flag this to your tutor / mention it in your report)
-------------------------------------------------------------------------------
Kawasaki's own two official sources disagree on which physical joint is
"joint 5" vs "joint 6" for the last two wrist axes:
  - the ROS xacro's "joint5" carries a +/-125 deg range at 550 deg/s, and its
    "joint6" carries +/-360 deg at 1000 deg/s;
  - the RS007N-B datasheet's "JT5" (wrist swivel) is +/-360 deg at 1000 deg/s,
    and "JT6" (wrist twist) is +/-125 deg at 550 deg/s.
The two are simply swapped between sources. The limits below follow the ROS
package's own joint order (joint1..joint6 as used in the kinematic chain
above), since that's self-consistent with the geometry transcribed here.
"""
 
import numpy as np
from roboticstoolbox import ET, ERobot
from spatialmath import SE3
 
 
class RS007N(ERobot):
    """Kawasaki RS007N 6-DOF industrial robot arm."""
 
    def __init__(self, base: SE3 = None):
        # link geometry [m], from Kawasaki-Robotics/khi_robot rs007n_macro.xacro
        j0 = 0.360    # base_link -> J1 axis height
        j1 = 0.0      # J1 -> J2 (no offset)
        j2 = 0.355    # upper arm length (J2 -> J3)
        j3 = 0.0925   # elbow offset (J3 -> J4)
        j4 = 0.2825   # forearm length (J4 -> J5)
        j5 = 0.078    # wrist/flange offset (J5 -> J6 / tool)
 
        ets = (
            ET.tz(j0) * ET.Rz(flip=True)                            # joint 1
            * ET.tz(j1) * ET.Ry(-90, unit="deg") * ET.Rz()          # joint 2
            * ET.tx(j2) * ET.Rz(flip=True)                          # joint 3
            * ET.tx(j3) * ET.Ry(90, unit="deg") * ET.Rz()           # joint 4
            * ET.tz(j4) * ET.Ry(-90, unit="deg") * ET.Rz(flip=True)  # joint 5
            * ET.tx(j5) * ET.Ry(90, unit="deg") * ET.Rz()           # joint 6
        )
 
        super().__init__(ets, name="RS007N", base=base)
 
        deg = np.deg2rad
 
        # joint angle limits [rad], from Kawasaki-Robotics/khi_robot's rs007n_macro.xacro
        # (see the "KNOWN SOURCE DISCREPANCY" note above re: joint5/joint6 vs JT5/JT6)
        self.qlim = np.array([
            [-deg(180), deg(180)],   # joint 1 - arm rotation
            [-deg(135), deg(135)],   # joint 2 - arm up-down
            [-deg(155), deg(155)],   # joint 3 - arm out-in
            [-deg(200), deg(200)],   # joint 4 - wrist bend
            [-deg(125), deg(125)],   # joint 5
            [-deg(360), deg(360)],   # joint 6
        ]).T
 
        # max joint speeds [rad/s], same source -- informational only, RTB
        # doesn't enforce these automatically; apply them in your own
        # trajectory/velocity-limiting code.
        self.qdlim = deg(np.array([470, 380, 520, 550, 550, 1000]))
 
 
if __name__ == "__main__":
    robot = RS007N()
    print(robot)
    print("\nFK at q = 0:\n", robot.fkine([0, 0, 0, 0, 0, 0]))
 
    # quick reach sanity check against the datasheet's quoted 730 mm reach
    q_stretch = [0, np.deg2rad(-90), 0, 0, 0, 0]
    p = robot.fkine(q_stretch).t
    horiz_to_flange = np.hypot(p[0], p[1])
    print(f"\nHorizontal reach to flange: {horiz_to_flange:.3f} m")
    print(f"Minus flange offset (0.078 m): {horiz_to_flange - 0.078:.3f} m "
          f"(datasheet quotes 0.730 m max reach)")