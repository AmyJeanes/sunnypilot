"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Lens models and the narrow->wide rotation as a rotation vector."""

import numpy as np


# measured lenses (3X bench + fleet, comma 4 fleet); f/cx/cy in px, angles in rad
X3_NARROW = {"f": 2600.85, "cx": 964.0, "cy": 604.0, "k1": -0.36400}  # pinhole with radial barrel: r=f·x(1+k1·x²)

# the lenses a real comma 4 has, not camera.py's (its 1141.5 narrow is from an early test unit): the wide is the 3X wide lens
# (bench) at the comma 4's 4 µm binned pixels, the narrow fitted against it from narrow<->wide matches on 17 comma 4s; the
# centers are not measurable from driving and stay at the table's
C4_WIDE = {"f": 447.502, "cx": 672.0, "cy": 380.0, "k": (-0.014942, -0.0023814, -0.00064424), "tc": 1.548}  # fisheye r=f·θ(1+k1θ²+k2θ⁴+k3θ⁶), linear past tc
C4_NARROW = {"f": 1156.5, "cx": 672.0, "cy": 380.0, "k1": -0.030}

# fleet median of 40 self-calibrated 3X units: the lens a unit gets when only its rotation is calibrated (the model's and the
# self-cal's rotations pair with this nominal center, not with a board-calibrated lens)
X3_WIDE_POP = {"f": 597.732, "cx": 963.936, "cy": 603.959, "k": (-0.011968, 0.024043, -0.0091132), "tc": 1.51354}

POP_ROTATION = (0.0154781, -0.0225994, -0.00068013)  # rotvec: the 3X wide points ~1° above the narrow

def calib_from_rotvec(rotvec):
  return {"wide": X3_WIDE_POP, "narrow": X3_NARROW, "R": tuple(float(v) for v in rotvec)}

def rotvec_from_wide_from_device_euler(euler):
  """The model's wide_from_device_euler (device frame: roll, pitch, yaw) as our camera-axes rotvec (x=pitch, y=yaw, z=roll)."""
  r, p, y = euler
  return (float(p), float(y), float(r))

def matrix_to_rotvec(M):
  a = np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))
  if a < 1e-9:
    return np.zeros(3)
  return a / (2 * np.sin(a)) * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])

def _dtheta_d(t, k):
  return 1 + 3 * k[0] * t**2 + 5 * k[1] * t**4 + 7 * k[2] * t**6

def _theta_d(th, L):
  k, tc = L["k"], L["tc"]

  def p(t):
    return t * (1 + k[0] * t**2 + k[1] * t**4 + k[2] * t**6)

  return np.where(th <= tc, p(th), p(tc) + _dtheta_d(tc, k) * (th - tc))

def unproject_fisheye(px, L):
  dx, dy = px[..., 0] - L["cx"], px[..., 1] - L["cy"]
  r = np.hypot(dx, dy)
  td = r / L["f"]
  th = td
  for _ in range(4):  # Newton from θ=θd: at double precision by the third step for every lens here
    th = th - (_theta_d(th, L) - td) / _dtheta_d(np.minimum(th, L["tc"]), L["k"])
  s = np.sin(th)
  rr = np.where(r > 0, r, 1)
  return np.stack([s * dx / rr, s * dy / rr, np.cos(th)], -1)

def unproject_pinhole_k1(px, L):
  xd, yd = (px[..., 0] - L["cx"]) / L["f"], (px[..., 1] - L["cy"]) / L["f"]
  rd = np.hypot(xd, yd)
  ru = rd
  for _ in range(5):  # Newton on r_u(1+k1·r_u²) = r_d from r_u=r_d: converged by the third step for the comma 4's mild barrel
    ru = ru - (ru * (1 + L["k1"] * ru * ru) - rd) / (1 + 3 * L["k1"] * ru * ru)
  s = np.where(rd > 0, ru / np.where(rd > 0, rd, 1), 1)
  d = np.stack([xd * s, yd * s, np.ones(px.shape[:-1])], -1)
  return d / np.linalg.norm(d, axis=-1, keepdims=True)

def project_fisheye(rays, L):
  rho = np.hypot(rays[..., 0], rays[..., 1])
  th = np.arctan2(rho, rays[..., 2])
  r = L["f"] * _theta_d(th, L)
  rr = np.where(rho > 0, rho, 1)
  return np.stack([L["cx"] + r * rays[..., 0] / rr, L["cy"] + r * rays[..., 1] / rr], -1)

def project_pinhole_k1(rays, L):
  x, y = rays[..., 0] / rays[..., 2], rays[..., 1] / rays[..., 2]
  s = 1 + L["k1"] * (x * x + y * y)
  return np.stack([L["cx"] + L["f"] * x * s, L["cy"] + L["f"] * y * s], -1)

def rotvec_to_matrix(v):
  v = np.asarray(v, dtype=np.float64)
  a = np.linalg.norm(v)
  if a == 0:
    return np.eye(3)
  k = v / a
  K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
  return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
