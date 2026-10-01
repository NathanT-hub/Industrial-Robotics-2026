import json
import os
from math import pi

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from ir_support.robots.UTSMeshRobot import UTSMeshRobot

class ReBotB601(UTSMeshRobot):
    """Six-axis reBot B601-DM with official DH Data and Link Meshes"""

    manufacturer_url = "https://wiki.seeedstudio.com/rebot_arm_b601_dm_ros2_integration/"
    source_commit = "fbc769abd5c1335df309c2b2a5b172b240d5d369"

    def __init__(self, base=None):
        mesh_dir = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(mesh_dir, "dh_parameters.json"), encoding="utf-8") as file:
            model_data = json.load(file)

        links = [rtb.RevoluteDH(**row) for row in model_data["rows"]]
        for i, link in enumerate(links):
            link.qdlim = model_data["urdf_velocity"][i]
            link.tlim = model_data["urdf_effort"][i]

        home_q = np.array([0, -pi / 2, -pi / 2, 0, 0, 0])
        self._dh_base_offset = SE3(np.array(model_data["base"]))

        # UTSMeshRobot calibrates each mesh from its pose at home_q.  These
        # poses are derived from the official reBot URDF-to-DH conversion.
        dh_home = [np.eye(4)]
        for link, qi in zip(links, home_q):
            dh_home.append(dh_home[-1] @ link.A(qi).A)

        offsets = model_data["mesh_offsets"]
        mesh_home_poses = [
            np.linalg.inv(self._dh_base_offset.A),
            *[
                dh_home[i] @ np.array(offsets[f"link{i}"])
                for i in range(1, 7)
            ],
        ]

        mount = base if isinstance(base, SE3) else SE3() if base is None else SE3(base, check=False)
        super().__init__(
            links=links,
            mesh_stem="ReBotB601",
            mesh_dir=mesh_dir,
            name="ReBotB601-DM",
            home_q=home_q,
            base=mount * self._dh_base_offset,
            qtest_transforms=mesh_home_poses,
        )

        # Official transform from joint 6 to the end_link/TCP.
        self.tool = SE3(np.array(model_data["tool"]))

    def set_mount(self, mount_pose):
        """Place the physical base_link frame in the shared workcell."""
        mount_pose = mount_pose if isinstance(mount_pose, SE3) else SE3(mount_pose, check=False)
        self.base = mount_pose * self._dh_base_offset