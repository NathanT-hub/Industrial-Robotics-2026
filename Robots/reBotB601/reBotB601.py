import json
import os
from math import pi
from pathlib import Path

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as geometry
import swift
from spatialmath import SE3


class ReBotB601(rtb.DHRobot):
    """Six-axis reBot B601-DM with official link meshes and add_to_env()."""

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

        super().__init__(links, name="ReBotB601-DM")
        self.home_q = np.array([0, -pi / 2, -pi / 2, 0, 0, 0])
        self.tool = SE3(np.array(model_data["tool"]))
        self._dh_base_offset = SE3(np.array(model_data["base"]))

        # One mesh per DH frame, matching the UR3e support-folder layout.
        self.links_3d = [
            geometry.Mesh((Path(mesh_dir) / f"ReBotB601Link{i}.stl").as_posix())
            for i in range(self.n + 1)
        ]
        offsets = model_data["mesh_offsets"]
        self._relation_matrices = [
            np.linalg.inv(self._dh_base_offset.A),
            *[np.array(offsets[f"link{i}"]) for i in range(1, 7)],
        ]

        self.set_mount(base if base is not None else SE3())
        self.q = self.home_q.copy()
        self._update_3dmodel()

    def set_mount(self, mount_pose):
        """Place the physical base_link frame in the shared workcell."""
        mount_pose = mount_pose if isinstance(mount_pose, SE3) else SE3(mount_pose, check=False)
        self.base = mount_pose * self._dh_base_offset

    def _get_transforms(self, q):
        transforms = [self.base.A]
        for i, link in enumerate(self.links):
            transforms.append(transforms[i] @ link.A(q[i]).A)
        return transforms

    def _update_3dmodel(self):
        if not hasattr(self, "links_3d"):
            return
        transforms = self._get_transforms(self.q)
        for i, mesh in enumerate(self.links_3d):
            mesh.T = transforms[i] @ self._relation_matrices[i]

    def add_to_env(self, env):
        """Add all seven link meshes to an existing Swift environment."""
        if not isinstance(env, swift.Swift):
            raise TypeError("Environment must be Swift")
        self._update_3dmodel()
        for mesh in self.links_3d:
            env.add(mesh)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in {"q", "base"} and hasattr(self, "links_3d"):
            self._update_3dmodel()
