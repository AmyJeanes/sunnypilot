"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The seam exposure match: the wide surround's luma, tone bands, shading gradient and color offsets against the narrow inset,
measured live in the seam ring, on top of the exposure model from the two camera states."""

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

class SeamMeter:
  """Measures the narrow-inset / wide-surround match in the composite's seam ring straight from the two source NV12 buffers
  (host memory, a few thousand nearest-neighbor pixel pairs). The two cameras expose and white-balance independently and
  their constant differs per unit (fleet: +-8 % luma, +-2 U/V steps), so the match is measured live and low-pass filtered;
  the exposure model is the fallback when the ring is too dark/saturated. Beyond one luma gain: the wide lens's shading is
  uneven per unit (25 % side to side on some), measured as a gain gradient across the frame; and the two ISPs' tone curves
  differ (+-10 Y steps between shadows and highlights after the best single gain), measured as a gain per brightness band
  and applied as a lookup on the surround luma."""
  BANDS = ((16, 50), (50, 100), (100, 160), (160, 235))
  IDENTITY = np.array([1, 0, 0, 0, 0] + [1] * len(BANDS), np.float32)  # [gain, u_off, v_off, gx, gy, band gains]: no match

  def __init__(self, geometry, alpha=0.3, every=2):
    """The luma gains are filtered as a correction on top of the exposure model (from the camera states) and applied times
    the model's live value, so a jump in either camera's exposure is followed the same frame.
    geometry: the pixel pairs from `geometry`, kept with the tables."""
    self.y_w, self.y_n, self.pos, self.cell, self.uv_w, self.uv_n = (geometry[k] for k in ("y_w", "y_n", "pos", "cell", "uv_w", "uv_n"))
    self.cell_bad = np.zeros(32 * 18, np.float32)
    self.CELL_ALPHA, self.CELL_LIMIT = 0.02, 0.2
    self.alpha, self.every, self.n_calls, self.moving = alpha, every, 0, False  # measuring every frame is ~0.7 ms of CPU on a PC
    self.state = None  # filtered [gain_y, u_off, v_off, gx, gy, band gains...]

  @staticmethod
  def geometry(src_wh, dst_wh, calib, n_pairs=2048, ring=(30.0, 130.0), step=4) -> dict:
    """The NV12 byte indices of wide/narrow pixel pairs sampled in the seam ring (the narrow inset's band `ring` comma 4 px in
    from its edge, on a `step` px grid of the comma 4 narrow frame), their comma 4 positions and ring cells."""
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
      return iw[sel], inn[sel], px[sel]

    g = {}
    g["y_w"], g["y_n"], px = pairs(1.0, n_pairs)
    g["uv_w"], g["uv_n"], _ = pairs(0.5, n_pairs // 2)  # U byte; V is the next one
    g["pos"] = (px / [dw, dh] * 2 - 1).astype(np.float32)  # comma 4 px, -1..1
    # the ring in 32x18 cells: a cell whose pairs persistently disagree with the fit (an obstruction on one lens, dirt, a
    # wiper) is excluded from the fit rather than dragged along; passing objects average out before they count
    cell = np.floor((g["pos"] + 1) / 2 * [32, 18]).astype(int)
    g["cell"] = cell[:, 0] * 18 + cell[:, 1]
    return g

  def measure(self, wide, narrow):
    """One frame's raw state vector, or None when fewer than 256 usable luma pairs (night, glare)."""
    yw = wide[self.y_w].astype(np.float32)
    yn = narrow[self.y_n].astype(np.float32)
    good = (yw > 16) & (yw < 235) & (yn > 16) & (yn < 235)
    if good.sum() < 256:
      return None
    yw, yn, pos = yw[good], yn[good], self.pos[good]
    ratio = yn / yw
    gy = float(np.median(ratio))
    # tone (a gain per brightness band of the wide) and lens shading (a gain gradient over the frame) are confounded in
    # one frame - the sky is always at the top of the ring - so they are fitted jointly: log ratio = band gain + gx*x + gy*y
    band = np.searchsorted([hi for _, hi in self.BANDS[:-1]], yw)
    A = np.zeros((len(yw), len(self.BANDS) + 2), np.float32)
    A[np.arange(len(yw)), band] = 1
    A[:, -2:] = pos
    lr = np.log(np.clip(ratio, 0.25, 4.0))
    cells = self.cell[good]
    w = (self.cell_bad[cells] < self.CELL_LIMIT).astype(np.float32)
    if w.sum() < 256:
      w[:] = 1  # too much excluded: fit everything rather than nothing
    # weighted least squares by the normal equations (6x6, ridge 1e-3 so an empty band stays solvable): the general solver
    # costs 1.3 ms per call on the device CPU, this ~0.1 ms
    def solve(wt):
      Aw = A * wt[:, None]
      return np.linalg.solve(A.T @ Aw + 1e-3 * np.eye(A.shape[1], dtype=np.float32), Aw.T @ lr)
    c = solve(w)
    res = lr - A @ c
    wr = w * np.clip(1 - (res / 0.3) ** 2, 0, 1) ** 2  # reweight once: a pair's pull falls off with its residual (Tukey-like)
    if wr.sum() >= 256:
      c = solve(wr)
    n = np.bincount(cells, minlength=len(self.cell_bad))
    bad = np.bincount(cells, weights=np.abs(lr - A @ c), minlength=len(self.cell_bad))
    seen = n > 0
    self.cell_bad[seen] += self.CELL_ALPHA * (bad[seen] / n[seen] - self.cell_bad[seen])
    counts = np.bincount(band, minlength=len(self.BANDS))
    bands = [float(np.exp(c[i])) if counts[i] >= 100 else gy for i in range(len(self.BANDS))]
    gx, gyp = float(np.clip(c[-2], -0.5, 0.5)), float(np.clip(c[-1], -0.5, 0.5))
    uw, un = wide[self.uv_w].astype(np.float32), narrow[self.uv_n].astype(np.float32)
    vw, vn = wide[self.uv_w + 1].astype(np.float32), narrow[self.uv_n + 1].astype(np.float32)
    du, dv = float(np.median(un - UV_FILL - gy * (uw - UV_FILL))), float(np.median(vn - UV_FILL - gy * (vw - UV_FILL)))
    return np.array([gy, du, dv, gx, gyp, *bands], np.float32)

  def update(self, wide, narrow, model_gain=1.0):
    """Filtered match for this frame as the Reprojector's parameter vector; measured when the ring is usable, else the luma
    gains decay to the exposure model (a flat curve) while the color offsets and the shading gradient hold: those are
    per-unit constants, and the ring is unmeasurable for minutes at a time at night."""
    self.n_calls += 1
    if self.state is None or self.moving or self.n_calls % self.every == 0:
      target = self.measure(wide, narrow)
      if target is None:
        held = self.state[1:5] if self.state is not None else (0, 0, 0, 0)
        target = np.array([model_gain, *held] + [model_gain] * len(self.BANDS), np.float32)
      target = target.copy()
      target[0] /= model_gain
      target[5:] /= model_gain  # the luma entries are kept relative to the exposure model
      if self.state is None:
        self.state = target
      else:
        # the two auto-exposures diverge for a second at a time (one brightens while the other darkens): each measurement
        # is a median over thousands of pairs, so when the match moves, follow it almost at once and measure every frame
        jump = abs(float(target[0] - self.state[0])) / max(float(self.state[0]), 0.1)
        self.state += min(self.alpha + 3 * jump, 0.9) * (target - self.state)
        self.moving = jump > 0.03
    s = self.state.copy()
    s[0] *= model_gain
    s[5:] *= model_gain
    return s
