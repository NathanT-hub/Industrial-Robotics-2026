"""
Collision checking, adapted from the Week 5 lab solution. Fixed so it
correctly checks every link on any n-DOF robot (see note in is_collision).
"""
import numpy as np
from itertools import combinations
from spatialgeometry import Sphere
from spatialmath.base import transl
from ir_support import line_plane_intersection


def is_intersection_point_inside_triangle(intersect_p, triangle_verts):
    u = triangle_verts[1, :] - triangle_verts[0, :]
    v = triangle_verts[2, :] - triangle_verts[0, :]
    uu, uv, vv = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    w = intersect_p - triangle_verts[0, :]
    wu, wv = np.dot(w, u), np.dot(w, v)
    D = uv * uv - uu * vv

    s = (uv * wv - vv * wu) / D
    if s < 0.0 or s > 1.0:
        return False
    t = (uv * wu - uu * wv) / D
    if t < 0.0 or (s + t) > 1.0:
        return False
    return True


def get_link_poses(robot, q=None):
    """Transform of every joint frame (base through to the tool)."""
    if q is None:
        q = robot.q
    return robot.fkine_all(q).A


def is_collision(robot, q_matrix, faces, vertices, face_normals,
                  collisions=None, env=None, return_once_found=True):
    """
    Check whether `robot` collides with an obstacle (given as its mesh's
    faces/vertices/face_normals -- e.g. from RectangularPrism.get_data())
    at any configuration in q_matrix.

    If `env` is given, a red sphere is drawn in Swift at every collision
    point found. `collisions` tracks those spheres so you can remove/clear
    them later.
    """
    if collisions is None:
        collisions = []
    result = False

    for q in q_matrix:
        tr = get_link_poses(robot, q)

        for link_idx in range(len(tr) - 1):
            p1, p2 = tr[link_idx][:3, 3], tr[link_idx + 1][:3, 3]
            for j, face in enumerate(faces):
                vert_on_plane = vertices[face][0]
                intersect_p, check = line_plane_intersection(face_normals[j], vert_on_plane, p1, p2)
                if check == 1:
                    triangle_list = np.array(list(combinations(face, 3)), dtype=int)
                    for triangle in triangle_list:
                        if is_intersection_point_inside_triangle(intersect_p, vertices[triangle]):
                            if env is not None:
                                sphere = Sphere(radius=0.02, color=[1.0, 0.0, 0.0, 1.0])
                                sphere.T = transl(intersect_p[0], intersect_p[1], intersect_p[2])
                                env.add(sphere)
                                collisions.append(sphere)
                            result = True
                            if return_once_found:
                                return result
                            break
    return result