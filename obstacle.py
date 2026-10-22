"""
A movable box obstacle for the collision-response demo.

Obstacle keeps two things in step: the visible Swift box, and the vertices / faces /
face normals that check_collision.is_collision() needs. move_to() updates both, so
the collision data can never be left behind at the old position.

The collision box is inflated by `margin` on every side (is_collision only tests the
centre line of each link, so the margin stands in for the link's thickness). The visible
box is drawn at its true size.
"""
import numpy as np
from spatialgeometry import Cuboid
from spatialmath import SE3

# Box corners are indexed by their sign bits: bit 0 = x, bit 1 = y, bit 2 = z (0 = -, 1 = +)
_SIGNS = np.array([[(i & 1) * 2 - 1, ((i >> 1) & 1) * 2 - 1, ((i >> 2) & 1) * 2 - 1] for i in range(8)])
_FACES = np.array([[0, 2, 6, 4],   # -x
                   [1, 3, 7, 5],   # +x
                   [0, 1, 5, 4],   # -y
                   [2, 3, 7, 6],   # +y
                   [0, 1, 3, 2],   # -z
                   [4, 5, 7, 6]])  # +z
_NORMALS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], dtype=float)


class Obstacle:
    def __init__(self, env, size=(0.15, 0.15, 0.15), centre=(0.8, 0.8, 0.7), margin=0.03,
                 color=(1.0, 0.5, 0.0, 0.8)):
        """
        env    -- the Swift environment
        size   -- (x, y, z) side lengths of the visible box, metres
        centre -- initial box centre in world coordinates
        margin -- extra clearance added to the collision box on every side, metres
        """
        self.size = np.array(size, dtype=float)
        self.margin = margin
        self.visual = Cuboid(scale=list(self.size), color=list(color))
        env.add(self.visual)
        self.move_to(*centre)

    def move_to(self, x, y, z):
        """Move the box and rebuild its collision data."""
        self.centre = np.array([x, y, z], dtype=float)
        self.visual.T = SE3(x, y, z)
        half = self.size / 2 + self.margin
        self.vertices = self.centre + _SIGNS * half
        self.faces = _FACES
        self.face_normals = _NORMALS

    def get_data(self):
        """(vertices, faces, face_normals), in the form is_collision() expects."""
        return self.vertices, self.faces, self.face_normals