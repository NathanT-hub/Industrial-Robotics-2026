import os
from math import pi
import roboticstoolbox as rtb
from ir_support.robots.UTSMeshRobot import UTSMeshRobot

class  DoBot6(UTSMeshRobot):
    def __init__(self, base=None):
        links = self.create_DH()
        super().__init__(
            links = links,
            mesh_stem = "DoBot6",
            mesh_dir = os.path.abspath(os.path.dirname(__file__)),
            name = "DoBot6",
            home_q = [0, 0, 0, 0, 0, 0],
            base = base,
        )

    def _create_DH(self):
        links = [
            rtb.RevoluteDH(d = , a = , alpha = )
            rtb.RevoluteDH(d = , a = , alpha = )
            rtb.RevoluteDH(d = , a = , alpha = )
            rtb.RevoluteDH(d = , a = , alpha = )
            rtb.RevoluteDH(d = , a = , alpha = )
            rtb.RevoluteDH(d = , a = , alpha = )
        ]

        return links
