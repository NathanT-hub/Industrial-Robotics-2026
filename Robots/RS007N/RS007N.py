import os
from math import pi
import numpy as np
import roboticstoolbox as rtb
 
from ir_support.robots.UTSMeshRobot import UTSMeshRobot
 
 
class RS007N(UTSMeshRobot):
    """
    Kawasaki RS007N 6-DOF industrial robot arm.
    """
 
    manufacturer_url = "https://kawasakirobotics.com/asia-oceania/products-robots/rs007n/"

     # Pose (world frame, at home_q = [0]*6) of each mesh file's own local
    # origin, taken from Kawasaki's official xacro. Link0 (base) then link1..link6.
    # The meshes are exported in Kawasaki's own per-link frame, which isn't the
    # same as our DH frames, so without this the meshes render misaligned.
    _MESH_HOME_POSES = [
        np.array([[1, 0, 0, 0.000], [0, 1, 0, 0], [0, 0, 1, 0.000], [0, 0, 0, 1]], dtype=float),
        np.array([[1, 0, 0, 0.000], [0, 1, 0, 0], [0, 0, 1, 0.360], [0, 0, 0, 1]], dtype=float),
        np.array([[1, 0, 0, 0.000], [0, 1, 0, 0], [0, 0, 1, 0.360], [0, 0, 0, 1]], dtype=float),
        np.array([[1, 0, 0, 0.000], [0, 1, 0, 0], [0, 0, 1, 0.715], [0, 0, 0, 1]], dtype=float),
        np.array([[1, 0, 0, 0.000], [0, 1, 0, 0], [0, 0, 1, 1.090], [0, 0, 0, 1]], dtype=float),
        np.array([[1, 0, 0, 0.000], [0, 1, 0, 0], [0, 0, 1, 1.090], [0, 0, 0, 1]], dtype=float),
        np.array([[1, 0, 0, 0.000], [0, 1, 0, 0], [0, 0, 1, 1.168], [0, 0, 0, 1]], dtype=float),
    ]
 
    def __init__(self, base=None):
        links = [
            rtb.RevoluteDH(d=0.360, a=0.0,   alpha=pi / 2, offset=-pi / 2, flip=True,
                            qlim=self._qlim(-180, 180)),
            rtb.RevoluteDH(d=0.0,   a=0.355, alpha=pi,     offset=pi / 2,
                            qlim=self._qlim(-135, 135)),
            rtb.RevoluteDH(d=0.0,   a=0.0,   alpha=pi / 2, offset=pi / 2,
                            qlim=self._qlim(-155, 155)),
            rtb.RevoluteDH(d=0.375, a=0.0,   alpha=pi / 2, offset=pi,
                            qlim=self._qlim(-200, 200)),
            rtb.RevoluteDH(d=0.0,   a=0.0,   alpha=pi / 2, offset=pi,
                            qlim=self._qlim(-125, 125)),
            rtb.RevoluteDH(d=0.078, a=0.0,   alpha=0.0,    offset=pi / 2,
                            qlim=self._qlim(-360, 360)),
        ]
 
        super().__init__(
            links=links,
            mesh_stem="RS007N",
            mesh_dir=os.path.abspath(os.path.dirname(__file__)),
            name="RS007N",
            home_q=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            base=base,
            qtest_transforms=self._MESH_HOME_POSES,   # <Creates accurate placement of meshes in world coordinate frame
        )

    def ikine_LM(self, Tep, q0=None, **kwargs):
        """
        ikine_LM with a fix for joint 1's flip=True.

        roboticstoolbox's ikine_LM solves on the ETS, which ignores flip=True,
        while fkine() and jacob0() respect it -- so the raw IK solution comes
        back with joint 1 mirrored. Negate flipped joints going in (q0) and
        coming out (sol.q) so the result agrees with fkine().
        """
        flip_signs = np.array([-1.0 if link.isflip else 1.0 for link in self.links])
        if q0 is not None:
            q0 = np.asarray(q0, dtype=float) * flip_signs
        sol = super().ikine_LM(Tep, q0=q0, **kwargs)
        sol.q = sol.q * flip_signs
        return sol
