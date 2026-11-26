"""Orbit camera identical to src/Camera.h (same convention: Y up, looks along -Z)."""
import math

import numpy as np


class OrbitCamera:
    def __init__(self, target=(0, 0, 0), distance=4.0, yaw=0.6, pitch=0.35, fovy=math.radians(50)):
        self.target = np.array(target, dtype=np.float64)
        self.distance, self.yaw, self.pitch, self.fovy = distance, yaw, pitch, fovy

    @classmethod
    def from_c2w(cls, c2w, target, fovy):
        c2w = np.asarray(c2w, dtype=np.float64)
        pos, target = c2w[:3, 3], np.asarray(target, dtype=np.float64)
        v = pos - target
        dist = float(np.linalg.norm(v))
        return cls(target, dist, math.atan2(v[0], v[2]), math.asin(np.clip(v[1] / dist, -1, 1)), fovy)

    def position(self):
        return self.target + self.distance * np.array([
            math.cos(self.pitch) * math.sin(self.yaw), math.sin(self.pitch),
            math.cos(self.pitch) * math.cos(self.yaw)])

    def c2w(self):
        pos = self.position()
        back = pos - self.target
        back /= np.linalg.norm(back)
        right = np.cross([0, 1, 0], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, back, pos
        return m.astype(np.float32)

    def focal(self, h):
        return 0.5 * h / math.tan(0.5 * self.fovy)

    def orbit(self, dx, dy):
        self.yaw -= dx * 0.005
        self.pitch = float(np.clip(self.pitch + dy * 0.005, -1.5, 1.5))

    def zoom(self, s):
        self.distance = float(np.clip(self.distance * 0.9 ** s, 0.3, 50))

    def move(self, fwd, right, up):
        c = self.c2w()
        self.target += -c[:3, 2] * fwd + c[:3, 0] * right + np.array([0, 1, 0]) * up
