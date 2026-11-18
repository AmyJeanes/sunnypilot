"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Lens models, the narrow->wide rotation as a rotation vector, and where each comma 4 output pixel samples the 3X frames."""

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

ZMIN = np.cos(np.radians(92.0))  # a real comma 4 wide fades out at ~91-95° off-axis; the 3X lens is only measured to ~89°

FEATHER_PX = 50  # composite seam width in comma 4 narrow px (50 over 24, by eye on drive clips)

UV_FILL = 128

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

def sample_coords(out_cam, dst_w, dst_h, scale, calib):
  """Float 3X wide / narrow coordinates for every comma 4 `out_cam` pixel (scale 0.5 = the half-res chroma plane)."""
  xs, ys = np.meshgrid((np.arange(dst_w) + 0.5) / scale, (np.arange(dst_h) + 0.5) / scale)
  return sample_points(out_cam, np.stack([xs, ys], -1), scale, calib)

def sample_points(out_cam, px, scale, calib):
  """sample_coords for comma 4 `out_cam` pixel centers `px` (..., 2) given in full-resolution px."""
  if out_cam == "wide":
    rays = unproject_fisheye(px, C4_WIDE)
  else:
    rays = unproject_pinhole_k1(px, C4_NARROW)
  rays_w = rays @ rotvec_to_matrix(calib["R"]).T
  mw = project_fisheye(rays_w, calib["wide"]) * scale
  mw[rays_w[..., 2] < ZMIN] = -1
  mn = project_pinhole_k1(rays, calib["narrow"]) * scale
  mn[rays[..., 2] <= 0] = -1
  return mw, mn

def _nv12_index(xy, src_w, src_h, stride, uv_offset, chroma):
  """Nearest-neighbor byte index into a source NV12 buffer for float sample coords; chroma coords are in the half-res plane.
  Returns (index, valid). For chroma the caller adds +1 for the V byte."""
  xy = np.clip(np.nan_to_num(xy, nan=-1.0), -1e6, 1e6)  # grazing rays project to ±inf
  x = np.round(xy[..., 0] - 0.5).astype(np.int64)
  y = np.round(xy[..., 1] - 0.5).astype(np.int64)
  w, h = (src_w // 2, src_h // 2) if chroma else (src_w, src_h)
  valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
  x = np.clip(x, 0, w - 1)
  y = np.clip(y, 0, h - 1)
  idx = (uv_offset + y * stride + 2 * x) if chroma else (y * stride + x)
  return idx, valid

C4_CAM = (1344, 760)  # the comma 4 narrow/wide frame the stage produces, the size the big model's pkl is keyed by
