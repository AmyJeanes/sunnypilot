"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The model's path, lane lines and road edges as the ui's onroad/model_renderer.py draws them, into any of the debug clip's
views."""
import colorsys
import copy

import numpy as np
from PIL import Image, ImageDraw

from openpilot.sunnypilot.tools.reproject_c4.view import solid


# onroad/model_renderer.py's path colors: bottom, middle and top of the view
THROTTLE = np.array([(13, 248, 122, 102), (114, 255, 92, 89), (114, 255, 92, 0)], np.float32)
NO_THROTTLE = np.array([(242, 242, 242, 102), (242, 242, 242, 89), (242, 242, 242, 0)], np.float32)
CLIP_MARGIN, MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE = 500, 10.0, 100.0


class ModelPath:
  """The model's path, lane lines and road edges drawn as onroad/model_renderer.py draws them, into any view: the path as a
  ribbon at the road's height with comma's gradient (green while the planner allows throttle, white while not; colored by
  acceleration in experimental mode), lane lines as wide as they are probable, road edges as opaque as they are certain."""
  def __init__(self):
    self.blend = 1.0

  def update(self, model: dict | None, lead_d: float | None, height: float, allow_throttle: bool, experimental: bool) -> 'ModelPath':
    """Step to this frame; returns this frame's path to draw (a copy, so frames can be drawn on other threads)."""
    self.model, self.lead_d, self.height, self.experimental = model, lead_d, height, experimental
    self.blend += (float(allow_throttle) - self.blend) * (0.05 / (0.25 + 0.05))  # the ui's FirstOrderFilter(rc 0.25) at 20 fps
    return copy.copy(self)

  def layer(self, size: tuple[int, int], T: np.ndarray) -> Image.Image:
    """On its own (RGBA) at full strength, to blend at any opacity. T: car space -> this view's px."""
    out = Image.new('RGBA', size, (0, 0, 0, 0))
    for color, alpha in self._shapes(*size, T):
      shape = color.convert('RGBA')
      shape.putalpha(alpha)
      out = Image.alpha_composite(out, shape)
    return out

  def _shapes(self, w: int, h: int, T: np.ndarray) -> list[tuple[Image.Image, Image.Image]]:
    """(color, alpha) at w x h in drawing order: the lane lines, the road edges, then the path."""
    if self.model is None:
      return []
    m = self.model
    path_x = m['path'][:, 0]
    max_d = float(np.clip(path_x[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
    idx = _length_idx(m['lanes'][0][:, 0], max_d)
    lanes = [(_polygon(ln, 0.025 * pr, 0.0, idx, max_d, T, w, h), np.clip(pr, 0, 0.7))
             for ln, pr in zip(m['lanes'], m['lane_probs'], strict=True)]
    edges = [(_polygon(e, 0.025, 0.0, idx, max_d, T, w, h), np.clip(1 - sd, 0, 1)) for e, sd in zip(m['edges'], m['edge_stds'], strict=True)]
    if self.lead_d is not None:
      lead_d = self.lead_d * 2.0
      max_d = float(np.clip(lead_d - min(lead_d * 0.35, 10.0), 0.0, max_d))
    path = _polygon(m['path'], 0.9, self.height, _length_idx(path_x, max_d), max_d, T, w, h, allow_invert=False)

    S = 2  # drawn at twice the size and scaled down: smooth edges
    out = []
    for polys, color in ((lanes, (255, 255, 255)), (edges, (255, 0, 0))):
      mask = Image.new('L', (w * S, h * S), 0)
      d = ImageDraw.Draw(mask)
      for pts, a in polys:
        if len(pts) >= 3:
          d.polygon([tuple(p) for p in (pts * S).tolist()], fill=int(a * 255))
      out.append((solid((w, h), color), mask.reduce(S)))
    if len(path) >= 3:
      mask = Image.new('L', (w * S, h * S), 0)
      ImageDraw.Draw(mask).polygon([tuple(p) for p in (path * S).tolist()], fill=255)
      rows = self._gradient(path, h, m['accel'])  # h x 4: the color on each row of the view
      color = Image.fromarray(np.ascontiguousarray(rows[:, None, :3]).astype(np.uint8)).resize((w, h), Image.NEAREST)
      alpha = np.asarray(mask.reduce(S), np.float32) * (rows[:, 3:4] / 255)
      out.append((color, Image.fromarray(alpha.astype(np.uint8))))
    return out

  def _gradient(self, path: np.ndarray, h: int, accel: np.ndarray) -> np.ndarray:
    t = 1 - (np.arange(h) + 0.5) / h  # 0 at the bottom of the view, 1 at the top
    if not self.experimental:
      cols = NO_THROTTLE + (THROTTLE - NO_THROTTLE) * round(self.blend * 100) / 100
      return np.stack([np.interp(t, [0, 0.5, 1], cols[:, c]) for c in range(4)], -1)
    stops, cols = [], []
    for i in range(0, min(len(path) // 2, len(accel)), 2):
      y = path[i, 1]
      if not 0 <= y <= h:
        continue
      p = 1 - y / h
      hue = np.clip(60 + accel[i] * 35, 0, 120)
      sat = min(abs(accel[i] * 1.5), 1)
      rgb = colorsys.hls_to_rgb(hue / 360, np.interp(sat, [0, 1], [0.95, 0.62]), sat)
      stops.append(p)
      cols.append([c * 255 for c in rgb] + [np.interp(p, [0.375, 0.75], [0.4, 0.0]) * 255])
    if len(stops) < 2:
      return np.tile(np.array([255, 255, 255, 30], np.float32), (h, 1))
    order = np.argsort(stops)
    s, c = np.asarray(stops)[order], np.asarray(cols)[order]
    return np.stack([np.interp(t, s, c[:, k]) for k in range(4)], -1)


def _length_idx(x: np.ndarray, d: float) -> int:
  i = np.where(x <= d)[0]
  return int(i[-1]) if i.size else 0


def _polygon(line: np.ndarray, y_off: float, z_off: float, max_idx: int, max_d: float, T: np.ndarray, w: int, h: int,
             allow_invert: bool = True) -> np.ndarray:
  """model_renderer's _map_line_to_polygon: a 3D line widened by y_off and raised by z_off, projected through T, kept where
  both sides land within 500 px of the view."""
  pts = line[:max_idx + 1]
  if 0 < max_idx < len(line) - 1:
    p0, p1 = line[max_idx], line[max_idx + 1]
    end = [max_d, np.interp(max_d, [p0[0], p1[0]], [p0[1], p1[1]]), np.interp(max_d, [p0[0], p1[0]], [p0[2], p1[2]])]
    pts = np.vstack([pts, np.asarray(end, pts.dtype)])
  pts = pts[pts[:, 0] >= 0]
  if not len(pts):
    return np.empty((0, 2))
  sides = [(T @ (pts + [0, dy, z_off]).T) for dy in (-y_off, y_off)]
  ok = (np.abs(sides[0][2]) >= 1e-6) & (np.abs(sides[1][2]) >= 1e-6)
  left, right = (s[:2, ok] / s[2, ok] for s in sides)
  def inside(s):
    return (s[0] >= -CLIP_MARGIN) & (s[0] <= w + CLIP_MARGIN) & (s[1] >= -CLIP_MARGIN) & (s[1] <= h + CLIP_MARGIN)
  keep = inside(left) & inside(right)
  left, right = left[:, keep], right[:, keep]
  if not allow_invert and left.shape[1] > 1:  # on a crest the far path would fold back down the view
    keep = left[1] == np.minimum.accumulate(left[1])
    left, right = left[:, keep], right[:, keep]
  return np.vstack([left.T, right[:, ::-1].T])
