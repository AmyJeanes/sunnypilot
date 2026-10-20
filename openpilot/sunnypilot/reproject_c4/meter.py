"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The seam exposure match: the wide surround's tone curve, shading and color against the narrow inset, measured live in the
seam ring, on top of the exposure model from the two camera states."""

import numpy as np

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from .geometry import C4_NARROW, UV_FILL, _nv12_index, sample_points


# encoded-domain luma ratio narrow/wide vs the sensors' exposure ratio (gain*integLines), fitted in the seam ring on a
# day-to-night drive (1607 frame pairs, 4 % residual): the exponent is the ISP tone curve, so a plain constant can't fit
# both dusk and night; the ring (not the whole overlap) because the wide lens falls off towards the seam
EXPOSURE_GAIN_A, EXPOSURE_GAIN_P = 0.868, 0.708

def exposure_gain(narrow_exposure, wide_exposure, lo=0.25, hi=4.0):
  """Gain to apply to the 3X wide surround so it matches the narrow inset; exposures are gain*integLines from the camera states."""
  if not (narrow_exposure > 0 and wide_exposure > 0):
    return 1.0
  return float(np.clip(EXPOSURE_GAIN_A * (narrow_exposure / wide_exposure) ** EXPOSURE_GAIN_P, lo, hi))


def _fields(pos):
  """The shading fields' terms at comma 4 positions (-1..1): x, y, xy, x^2, y^2."""
  x, y = pos[:, 0], pos[:, 1]
  return np.stack([x, y, x * y, x * x, y * y], 1).astype(np.float32)


def _hats_into(out, v, knots, inv):
  """The piecewise-linear basis on the knots (flat beyond the ends) written into out's columns."""
  r = np.clip((v[:, None] - knots[None, :-1]) * inv[None, :], 0, 1)
  out[:, 0] = 1 - r[:, 0]
  np.subtract(r[:, :-1], r[:, 1:], out=out[:, 1:-1])
  out[:, -1] = r[:, -1]


def _solve(A, b, w, ridge, prior):
  """Weighted least squares with a ridge toward prior, then once more with each pair's weight falling off with its residual
  (Tukey on the residuals' median: pairs on an edge or a misregistration are outliers). The normal equations, not lstsq:
  the general solver costs several times as much on the device CPU."""
  R = np.diag(ridge).astype(np.float32)
  Rp = R @ prior
  for _ in range(2):
    Aw = A * w[:, None]
    c = np.linalg.solve(A.T @ Aw + R, Aw.T @ b + Rp)
    r = b - A @ c
    a = np.abs(r[w > 0])
    s = 4.685 * 1.4826 * max(float(np.partition(a, len(a) // 2)[len(a) // 2]), 0.5)
    w = (w > 0) * np.clip(1 - (r / s) ** 2, 0, 1) ** 2
  return c, r


class SeamMeter:
  """Measures the narrow-inset / wide-surround match in the composite's seam ring straight from the two source NV12 buffers
  (host memory, a few thousand nearest-neighbor pixel pairs). The two cameras expose, tone-map and white-balance independently
  and differently per unit, so the match is measured live and low-pass filtered; the exposure model is the fallback when the
  ring is too dark/saturated.
  - luma: narrow = m * (curve(wide) + wide * mul_field(x, y) + add_field(x, y)). The curve is piecewise linear through
    KNOTS, in output luma rather than as gains, because the two ISPs differ by offsets in the shadows as well as by gain;
    the fields are quadratic (x, y, xy, x^2, y^2), multiplicative for the wide lens's shading and additive for flare.
  - chroma, per U/V: narrow = m * sat(Y) * wide + offset + gradient (x, y), the saturation a curve through KNOTS of the wide
    luma under the chroma sample: the wide's color is weaker by day, most in the bright sky, and stronger at night.
  - the fields are fitted on the ring only and held flat beyond its extent (the bounds): a quadratic extrapolated out to
    the frame edges darkens the corners.
  m is the exposure model's gain; the match is filtered relative to it, so a jump in either camera's exposure is followed
  the same frame."""
  KNOTS = (16.0, 50.0, 100.0, 150.0, 200.0, 237.0)  # the last at camerad's top: clipped wide pixels get their own value
  # [curve at KNOTS, multiplicative field, additive field, saturation at KNOTS, U/V offset, U/V x gradient, U/V y gradient,
  #  field bounds x_lo, x_hi, y_lo, y_hi]: the stage's parameters (compile_stage.py)
  IDENTITY = np.array([*KNOTS] + [0] * 10 + [1] * len(KNOTS) + [0] * 6 + [-1, 1, -1, 1], np.float32)
  N_LUMA, N_CHROMA = len(KNOTS) + 10, len(KNOTS) + 6
  CLIP = 235  # camerad's luma tops out at 237: a wide pixel at or above this is clipped
  REF = np.array([50.0, 100.0, 160.0])  # luma levels the step size watches

  def __init__(self, geometry, alpha=0.3, every=2):
    """geometry: the pixel pairs from `geometry`, kept with the tables."""
    self.y_w, self.y_n, self.cell, self.uv_w, self.uv_n = (geometry[k] for k in ("y_w", "y_n", "cell", "uv_w", "uv_n"))
    self.yc_w, self.yc_n = geometry["yc_w"], geometry["yc_n"]
    self.bounds = np.array(geometry["bounds"], np.float32)
    self.cell_bad = np.zeros(32 * 18, np.float32)
    self.CELL_ALPHA, self.CELL_LIMIT = 0.02, 0.2
    self.alpha, self.every, self.n_calls, self.moving = alpha, every, 0, False
    self.state = None  # the filtered match relative to the exposure model: luma coefficients and saturation divided by m
    # the design matrices' position columns are fixed by the geometry; only the luma-dependent columns change per frame
    self.knots = np.array(self.KNOTS, np.float32)
    self.inv = (1 / np.diff(self.knots)).astype(np.float32)
    self.P = _fields(geometry["pos"])
    self.A = np.zeros((len(self.y_w), self.N_LUMA), np.float32)
    k = len(self.KNOTS)
    self.A[:, k + 5:] = self.P
    n = len(self.uv_w)
    cx, cy = (np.concatenate([geometry["uv_pos"][:, k]] * 2) for k in (0, 1))
    off = np.stack([np.r_[np.ones(n), np.zeros(n)], np.r_[np.zeros(n), np.ones(n)]], 1)
    self.Ac = np.zeros((2 * n, self.N_CHROMA), np.float32)
    self.Ac[:, k:] = np.hstack([off, off * cx[:, None], off * cy[:, None]])
    self.H = np.empty((2 * n, len(self.KNOTS)), np.float32)

  @staticmethod
  def geometry(src_wh, dst_wh, calib, n_pairs=2048, ring=(30.0, 130.0), step=4) -> dict:
    """The NV12 byte indices of wide/narrow pixel pairs sampled in the seam ring (the narrow inset's band `ring` comma 4 px in
    from its edge, on a `step` px grid of the comma 4 narrow frame), their comma 4 positions and ring cells, the luma byte
    under each chroma pair (the top-left of its 2x2, as the stage reads it) and the ring's extent."""
    (sw, sh), (dw, dh) = src_wh, dst_wh
    stride, y_height, _, _ = get_nv12_info(sw, sh)
    sc = calib["narrow"]["f"] / C4_NARROW["f"]  # 3X narrow px per comma 4 narrow px

    def pairs(scale, n):
      st = max(1, round(step * scale))
      ys, xs = np.mgrid[0:int(dh * scale):st, 0:int(dw * scale):st]
      px = np.stack([(xs + 0.5) / scale, (ys + 0.5) / scale], -1).reshape(-1, 2)
      mw, mn = sample_points("narrow", px, scale, calib)
      iw, vw = _nv12_index(mw, sw, sh, stride, stride * y_height, scale < 1)
      inn, vn = _nv12_index(mn, sw, sh, stride, stride * y_height, scale < 1)
      d = np.minimum(np.minimum(mn[:, 0], sw * scale - mn[:, 0]), np.minimum(mn[:, 1], sh * scale - mn[:, 1])) / (scale * sc)
      sel = np.flatnonzero(vw & vn & (d > ring[0]) & (d < ring[1]))
      sel = sel[:: max(1, len(sel) // n)][:n]
      return iw[sel], inn[sel], (px[sel] / [dw, dh] * 2 - 1).astype(np.float32)

    def luma_under(uv):
      rel = uv - stride * y_height
      return (rel // stride) * 2 * stride + (rel % stride & ~1)

    g = {}
    g["y_w"], g["y_n"], g["pos"] = pairs(1.0, n_pairs)
    g["uv_w"], g["uv_n"], g["uv_pos"] = pairs(0.5, n_pairs // 2)  # U byte; V is the next one
    g["yc_w"], g["yc_n"] = luma_under(g["uv_w"]), luma_under(g["uv_n"])
    g["bounds"] = (float(g["pos"][:, 0].min()), float(g["pos"][:, 0].max()), float(g["pos"][:, 1].min()), float(g["pos"][:, 1].max()))
    # the ring in 32x18 cells: a cell whose pairs persistently disagree with the fit (an obstruction on one lens, dirt, a
    # wiper) is excluded from the fit rather than dragged along; passing objects average out before they count
    cell = np.floor((g["pos"] + 1) / 2 * [32, 18]).astype(int)
    g["cell"] = cell[:, 0] * 18 + cell[:, 1]
    return g

  def measure(self, wide, narrow, model_gain, prior):
    """One frame's raw match relative to the exposure model (the state's layout), or None when fewer than 256 usable luma
    pairs (night, glare). prior: the current state; the chroma is held to it where the ring says little (near-grey)."""
    yw = wide[self.y_w].astype(np.float32)
    yn = narrow[self.y_n].astype(np.float32)
    # a clipped wide pixel is still a pair: the curve's top knot learns what it maps to (in direct sun the wide clips across
    # sky the narrow still resolves); a clipped narrow says only "at least", so it counts only where the wide clipped too
    good = (yw > 16) & (yn > 16) & ((yn < self.CLIP) | (yw >= self.CLIP))
    n = int(good.sum())
    if n < 256:
      return None
    # excluded pairs get weight 0 rather than being dropped, so the position columns stay as built
    w = good & (self.cell_bad[self.cell] < self.CELL_LIMIT)
    w = (w if w.sum() >= 256 else good).astype(np.float32)  # too much excluded: fit everything rather than nothing
    A = self.A
    k = len(self.KNOTS)
    _hats_into(A[:, :k], yw, self.knots, self.inv)
    np.multiply(yw[:, None], self.P, out=A[:, k:k + 5])
    b = yn / model_gain
    # the luma is fitted with only a token ridge: held toward the last state, it lags the scene and biases the curve
    cl, r = _solve(A, b, w, np.full(self.N_LUMA, 1e-4 * n), prior[:self.N_LUMA])
    cells = self.cell[good]
    cnt = np.bincount(cells, minlength=len(self.cell_bad))
    bad = np.bincount(cells, weights=np.abs(r[good]) / np.maximum(b[good], 8), minlength=len(self.cell_bad))
    seen = cnt > 0
    self.cell_bad[seen] += self.CELL_ALPHA * (bad[seen] / cnt[seen] - self.cell_bad[seen])

    cc = prior[self.N_LUMA:]
    ycw, ycn = wide[self.yc_w].astype(np.float32), narrow[self.yc_n].astype(np.float32)
    cg = (ycw > 16) & (ycw < 235) & (ycn > 16) & (ycn < 235)
    if cg.sum() >= 128:
      cw = np.concatenate([wide[self.uv_w], wide[self.uv_w + 1]]).astype(np.float32) - UV_FILL
      cn = np.concatenate([narrow[self.uv_n], narrow[self.uv_n + 1]]).astype(np.float32) - UV_FILL
      wc = np.concatenate([cg, cg]).astype(np.float32)
      _hats_into(self.H, np.concatenate([ycw, ycw]), self.knots, self.inv)
      np.multiply(self.H, (cw * model_gain)[:, None], out=self.Ac[:, :k])
      m = float(wc.sum())
      # the saturation held toward its last value (on a near-grey ring it is barely measurable), the offsets lightly
      ridge = np.r_[np.full(k, 4.0 * m / k), np.full(6, 1e-3 * m)]
      cc, _ = _solve(self.Ac, cn, wc, ridge, prior[self.N_LUMA:])
    out = np.concatenate([cl, cc]).astype(np.float32)
    return out if np.isfinite(out).all() else None

  def update(self, wide, narrow, model_gain=1.0):
    """Filtered match for this frame as the Reprojector's parameter vector; measured when the ring is usable, else the tone
    curve decays to the exposure model while the shading fields and the color hold: those change slowly or not at all, and
    the ring is unmeasurable for minutes at a time at night."""
    self.n_calls += 1
    identity = self.IDENTITY[:self.N_LUMA + self.N_CHROMA]
    if self.state is None or self.moving or self.n_calls % self.every == 0:
      target = self.measure(wide, narrow, model_gain, identity if self.state is None else self.state)
      if target is None:
        target = identity.copy() if self.state is None else self.state.copy()
        target[:len(self.KNOTS)] = self.KNOTS
      if self.state is None:
        self.state = target
      else:
        # the two auto-exposures diverge for a second at a time (one brightens while the other darkens): when the match
        # moves, follow it almost at once and measure every frame
        k = len(self.KNOTS)
        a, s = np.interp(self.REF, self.KNOTS, target[:k]), np.interp(self.REF, self.KNOTS, self.state[:k])
        jump = float(np.mean(np.abs(a - s) / np.maximum(s, 8)))
        self.state = self.state + min(self.alpha + 3 * jump, 0.9) * (target - self.state)
        self.moving = jump > 0.03
    out = np.concatenate([self.state, self.bounds])
    out[:self.N_LUMA] *= model_gain
    out[self.N_LUMA:self.N_LUMA + len(self.KNOTS)] *= model_gain
    return out
