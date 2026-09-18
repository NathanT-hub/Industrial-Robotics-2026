import os
from math import pi
 
import roboticstoolbox as rtb
 
from ir_support.robots.UTSMeshRobot import UTSMeshRobot
 
 
class RS007N(UTSMeshRobot):
    """
    Kawasaki RS007N 6-DOF industrial robot arm.
    """
 
    manufacturer_url = "https://kawasakirobotics.com/asia-oceania/products-robots/rs007n/"
 
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
        )