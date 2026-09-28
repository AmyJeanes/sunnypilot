"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The debug view's geometry: outlines of the blend band, the model inputs and the comma 4 frames in every frame the debug
clip can show."""
import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.sunnypilot.reproject_c4.geometry import (C4_NARROW, C4_WIDE, FEATHER_PX, project_fisheye, project_pinhole_k1,
                                                                rotvec_to_matrix, sample_coords, unproject_fisheye, unproject_pinhole_k1)


def _edge_points(corners, n=12):
  """A closed polygon's corners densified along its edges, so a curved mapping keeps the edge curved."""
  pts = []
  for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
    for t in np.linspace(0, 1, n, endpoint=False):
      pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
  return np.array(pts, float)


def _mask_outline(mask, step):
  """Boundary of a convex-ish mask on a coarse grid as a closed polygon in px, traced top, right, bottom, left."""
  rows, cols = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
  def first(v):
    return np.argmax(v)
  def last(v):
    return len(v) - 1 - np.argmax(v[::-1])

  top = [((c + 0.5) * step, (first(mask[:, c]) + 0.5) * step) for c in cols]
  right = [((last(mask[r]) + 0.5) * step, (r + 0.5) * step) for r in rows]
  bottom = [((c + 0.5) * step, (last(mask[:, c]) + 0.5) * step) for c in cols[::-1]]
  left = [((first(mask[r]) + 0.5) * step, (r + 0.5) * step) for r in rows[::-1]]
  pts = np.array(top + right + bottom + left, float)
  return pts[np.r_[True, (np.abs(np.diff(pts, axis=0)) > 1e-6).any(1)]]  # the corners are traced twice


def _erode(mask, n):
  """A boolean mask shrunk by n cells (4-neighbor)."""
  m = mask.copy()
  for _ in range(n):
    m[1:-1, 1:-1] &= m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
  return m


def _warp_points(M, px):
  """px through the 3x3 homography M."""
  p = np.c_[np.asarray(px, float), np.ones(len(px))] @ M.T
  return p[:, :2] / p[:, 2:3]


def _resample(poly, n):
  """A closed polygon as n points evenly spaced along its length, from its first point."""
  p = np.asarray(poly, float)
  q = np.r_[p, p[:1]]
  seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
  s = np.r_[0, np.cumsum(seg)]
  t = np.linspace(0, s[-1], n, endpoint=False)
  return np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], 1)


def debug_outlines(calib, device_from_calib_euler, dst_wh=(1344, 760), src_wh=(1928, 1208)):
  """The blend band, the model inputs and the comma 4 frames as polygons in the px of each frame the debug clip can show:
  device_narrow / device_wide (3X frames), c4_narrow / c4_wide (reprojected).
  A band is two edges of equal length. `visible` says whether any of it lands inside the frame (the legend)."""
  cams = DEVICE_CAMERAS[("mici", "os04c10")]
  Kn, Kw = cams.narrow_road.intrinsics, cams.wide_road.intrinsics
  dw, dh = dst_wh
  sw, sh = src_wh
  R = rotvec_to_matrix(calib["R"])

  def rays_n4(px):  # comma 4 narrow px -> device rays
    return unproject_pinhole_k1(np.asarray(px, float), C4_NARROW)

  def to_x3_narrow(px):
    return project_pinhole_k1(rays_n4(px), calib["narrow"])

  def to_x3_wide(rays):
    return project_fisheye(rays @ R.T, calib["wide"])

  def to_c4_wide(rays):
    return project_fisheye(rays, C4_WIDE)

  warp_n = get_warp_matrix(np.asarray(device_from_calib_euler, np.float32), Kn, False)  # model px -> comma 4 narrow px
  warp_w = get_warp_matrix(np.asarray(device_from_calib_euler, np.float32), Kw, True)

  # the inset (comma 4 narrow px that come from the 3X narrow) and the blend band just inside its edge, on a 2 px grid
  step = 2
  _, mn = sample_coords("narrow", dw // step, dh // step, 1.0 / step, calib)  # 3X coords come back scaled too
  inset = (mn[..., 0] >= 0) & (mn[..., 0] < sw / step) & (mn[..., 1] >= 0) & (mn[..., 1] < sh / step)
  # the band's two edges: the inset's outline and the outline FEATHER_PX further in, each outer point paired with its nearest inner
  inset_edge = _resample(_mask_outline(inset, step), 96)
  inner_dense = _resample(_mask_outline(_erode(inset, FEATHER_PX // step), step), 800)
  band_inner = inner_dense[np.linalg.norm(inset_edge[:, None, :] - inner_dense[None, :, :], axis=2).argmin(1)]
  model_corners = [(0, 0), (512, 0), (512, 256), (0, 256)]
  narrow_input = _warp_points(warp_n, _edge_points(model_corners))  # comma 4 narrow px
  wide_input = _warp_points(warp_w, _edge_points(model_corners))  # comma 4 wide px
  c4_frame = _edge_points([(0, 0), (dw, 0), (dw, dh), (0, dh)])  # either comma 4 frame's edge, in its own px
  wide_input_rays = unproject_fisheye(wide_input, C4_WIDE)

  def item(name, kind, pts, frame, inner=None):
    """A polygon, or with `inner` a band between two edges (same point count); `visible` if any of it lands in the frame.
    kind names what it outlines (narrow_cam, narrow_input, wide_input, c4_frame): its color, the same in every frame."""
    pts = np.asarray(pts, float)
    keep = np.isfinite(pts).all(1)
    if inner is not None:
      inner = np.asarray(inner, float)
      keep &= np.isfinite(inner).all(1)
    pts = pts[keep]
    allp = pts if inner is None else np.r_[pts, inner[keep]]
    visible = bool(((allp[:, 0] > 0) & (allp[:, 0] < frame[0]) & (allp[:, 1] > 0) & (allp[:, 1] < frame[1])).any())
    d = {"name": name, "kind": kind, "points": np.round(pts, 1).tolist(), "visible": visible}
    if inner is not None:
      d["inner"] = np.round(inner[keep], 1).tolist()
    return d

  return {
    "device_narrow": [item("Narrow model input", "narrow_input", to_x3_narrow(narrow_input), src_wh)],
    "device_wide": [item("Narrow cam coverage", "narrow_cam", to_x3_wide(rays_n4(inset_edge)), src_wh),
                    item("C4 wide cam frame", "c4_frame", to_x3_wide(unproject_fisheye(c4_frame, C4_WIDE)), src_wh),
                    item("Wide model input", "wide_input", to_x3_wide(wide_input_rays), src_wh)],
    "c4_narrow": [item("Narrow / wide cam blend", "narrow_cam", inset_edge, dst_wh, inner=band_inner),
                  item("Narrow model input", "narrow_input", narrow_input, dst_wh)],
    "c4_wide": [item("C4 narrow cam frame", "c4_frame", to_c4_wide(rays_n4(c4_frame)), dst_wh),
                item("Wide model input", "wide_input", wide_input, dst_wh),
                item("Narrow model input", "narrow_input", to_c4_wide(rays_n4(narrow_input)), dst_wh)],
  }
