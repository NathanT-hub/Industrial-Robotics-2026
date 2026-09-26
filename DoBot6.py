import os
from math import pi
from spatialmath import SE3
import roboticstoolbox as rtb
from ir_support.robots.UTSMeshRobot import UTSMeshRobot

class  DoBot6(UTSMeshRobot):
    def __init__(self, base=None):
        links = self._create_DH()

        # Provide non-standard file names
        link3d_names = {
            "link0" : "DoBot6_Base",
            "link1" : "DoBot6_Link1",
            "link2" : "DoBot6_Link2",
            "link3" : "DoBot6_Link3",
            "link4" : "DoBot6_Link4",
            "link5" : "DoBot6_Link5",
            "link6" : "DoBot6_Link6"
        }
        # File path to stl mesh files
        mesh_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes_DoBot6")

        super().__init__(
            links = links,
            mesh_stem = "DoBot6",
            mesh_dir = mesh_dir,
            name = "DoBot6",
            home_q = [0, 0, 0, 0, 0, 0],
            base = base,
            link3d_names = link3d_names,
            qtest_transforms=self._create_mesh_transforms()
        )

    # Creation of DH parameters for the DoBot6 Robot.
    # Offset creates standard starting position
    def _create_DH(self):
        links = [
            rtb.RevoluteDH(d = 0.1668, a = 0, alpha = pi/2, offset = -pi/2),
            rtb.RevoluteDH(d = 0, a = 0.18906, alpha = 0, offset = pi/2),
            rtb.RevoluteDH(d = 0, a = 0.1600, alpha = 0, offset = 0),
            rtb.RevoluteDH(d = 0.08600, a = 0, alpha = pi/2, offset = pi/2),
            rtb.RevoluteDH(d = 0.10100, a = 0, alpha = pi/2, offset = pi),
            rtb.RevoluteDH(d = 0.04700, a = 0, alpha = 0, offset = 0)
        ]
        return links

    # Create transforms to position mesh files around the arm
    def _create_mesh_transforms(self):
        # Standard rotation used by multiple pieces
        rotation = SE3.Rz(-pi/2) * SE3.Rx(pi/2)
        return[
            SE3(),
            SE3.Tz(0.1268),
            SE3.Trans(-0.046, 0, 0.16680) * rotation,
            SE3.Trans(-0.049, 0, 0.35586) * rotation,
            SE3.Trans(-0.054, 0, 0.51586) * rotation,
            SE3.Trans(-0.086, 0, 0.58286),
            SE3.Trans(-0.133, 0, 0.61686) * rotation
            ]
        