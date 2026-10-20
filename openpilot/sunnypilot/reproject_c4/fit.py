"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The rotation fit of one frame pair: phase-correlation matches between the narrow inset and the wide surround, Kabsch on
the matched rays. numpy only: it runs on the device CPU during the calibration phase."""

import numpy as np

from .geometry import C4_NARROW, calib_from_rotvec, matrix_to_rotvec, sample_coords, unproject_fisheye, unproject_pinhole_k1

MIN_MATCHES = 15  # matched patches a frame's fit must keep after kabsch's trimming
MAX_RMS_DEG = 0.25  # per-frame residual after trimming (good frames 0.10-0.17 on the 3X)
N_FRAMES = 12  # per fit: every logged session had converged by then, the median of 12 within 0.03 deg of the 40-frame mean

def render_layers(narrow_y, wide_y, calib, dst_wh=(1344, 760)):
  """The comma 4 narrow view's luma twice, nearest-sampled: inset from the 3X narrow, surround from the 3X wide, plus the
  wide sample coordinates and the mask where both exist. With the right rotation the two layers coincide."""
  sh, sw = narrow_y.shape
  mw, mn = sample_coords("narrow", dst_wh[0], dst_wh[1], 1.0, calib)
  xn = np.round(mn[..., 0] - 0.5).astype(int)
  yn = np.round(mn[..., 1] - 0.5).astype(int)
  xw = np.round(mw[..., 0] - 0.5).astype(int)
  yw = np.round(mw[..., 1] - 0.5).astype(int)
  vn = (xn >= 0) & (xn < sw) & (yn >= 0) & (yn < sh)
  vw = (xw >= 0) & (xw < sw) & (yw >= 0) & (yw < sh)
  inset = np.where(vn, narrow_y[yn.clip(0, sh - 1), xn.clip(0, sw - 1)], 0).astype(np.float32)
  surround = np.where(vw, wide_y[yw.clip(0, sh - 1), xw.clip(0, sw - 1)], 0).astype(np.float32)
  return inset, surround, mw, vn & vw

def phase_shift(a, b):
  """Shift of b relative to a (float32 patches, same shape) by phase correlation: (dx, dy, peak-to-sidelobe ratio)."""
  h, w = a.shape
  win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
  A = np.fft.rfft2((a - a.mean()) * win)
  B = np.fft.rfft2((b - b.mean()) * win)
  R = A * np.conj(B)
  R /= np.abs(R) + 1e-6
  r = np.fft.irfft2(R, s=(h, w))
  py, px = divmod(int(np.argmax(r)), w)
  peak = r[py, px]
  m = np.ones_like(r, bool)
  m[max(0, py - 5):py + 6, max(0, px - 5):px + 6] = False
  side = r[m]
  psr = (peak - side.mean()) / (side.std() + 1e-9)
  def sub(c, l, rr):  # parabolic sub-pixel peak
    d = l - 2 * c + rr
    return 0.0 if d >= 0 else float(0.5 * (l - rr) / d)
  dx = px + sub(peak, r[py, (px - 1) % w], r[py, (px + 1) % w])
  dy = py + sub(peak, r[(py - 1) % h, px], r[(py + 1) % h, px])
  if dx > w / 2:
    dx -= w
  if dy > h / 2:
    dy -= h
  return -dx, -dy, float(psr)

def match_rays(narrow_y, wide_y, calib, dst_wh=(1344, 760), patch=96, stride=64, min_psr=5.0, max_shift=None):
  """Correspondences between the inset and the surround (patch center -> center + its measured shift) as narrow and wide
  rays in the current calibration's geometry, or None when the frame has too few usable patches. min_psr 5: at dusk only
  a third of the textured patches reach 6 and frames fell under the match floor; 5 doubles them at the same per-frame
  accuracy (kabsch trims the rest)."""
  inset, surround, mw, valid = render_layers(narrow_y, wide_y, calib, dst_wh)
  dw, dh = dst_wh
  max_shift = max_shift or patch / 3
  pa, pb = [], []
  for y in range(0, dh - patch + 1, stride):
    for x in range(0, dw - patch + 1, stride):
      if valid[y:y + patch, x:x + patch].mean() < 0.98:
        continue
      a = inset[y:y + patch, x:x + patch]
      b = surround[y:y + patch, x:x + patch]
      if a.std() < 4 or b.std() < 4:  # flat: sky, bonnet, night
        continue
      dx, dy, psr = phase_shift(a, b)
      if psr < min_psr or abs(dx) > max_shift or abs(dy) > max_shift:
        continue
      pa.append((x + patch / 2, y + patch / 2))
      pb.append((x + patch / 2 + dx, y + patch / 2 + dy))
  if len(pa) < 4:
    return None
  pa, pb = np.array(pa, np.float32), np.array(pb, np.float32)
  rays_n = unproject_pinhole_k1(pa, C4_NARROW)
  xb, yb = np.round(pb[:, 0] - 0.5).astype(int).clip(0, dw - 1), np.round(pb[:, 1] - 0.5).astype(int).clip(0, dh - 1)
  rays_w = unproject_fisheye(mw[yb, xb], calib["wide"])
  return rays_n, rays_w

def kabsch(rays_n, rays_w):
  """Rotation taking narrow rays onto wide rays, trimmed twice at the 80th percentile: (rotvec, n_kept, rms_deg)."""
  for _ in range(2):
    U, _, Vt = np.linalg.svd(rays_w.T @ rays_n)
    d = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ np.diag([1, 1, d]) @ Vt
    res = np.degrees(np.arccos(np.clip((rays_n @ Rm.T * rays_w).sum(1), -1, 1)))
    keep = res <= np.percentile(res, 80)
    rays_n, rays_w = rays_n[keep], rays_w[keep]
  return matrix_to_rotvec(Rm), int(len(rays_n)), float(np.sqrt(np.mean(res[keep] ** 2)))

def fit_rotation(narrow_y, wide_y, calib0, dst_wh=(1344, 760), iters=2, coarse=True):
  """One frame pair -> (rotvec, n_matches, rms_deg), or None when too few patches matched. With `coarse` the first pass
  uses big patches (a fleet-median seed can be tens of px off) and the second re-renders with its fit and refines with
  small ones. From a good seed (the running estimate of earlier frames) coarse=False, iters=1 is enough."""
  calib = dict(calib0)
  R = np.asarray(calib0["R"], float)
  for it in range(iters):
    calib["R"] = tuple(R)
    m = match_rays(narrow_y, wide_y, calib, dst_wh, patch=192, stride=96, max_shift=64) if (it == 0 and coarse) else match_rays(narrow_y, wide_y, calib, dst_wh)
    if m is None:
      return None
    R, n, rms = kabsch(m[0], m[1])
  if n < MIN_MATCHES:
    return None
  return tuple(float(v) for v in R), n, rms


class Fit:
  """The running fit: the rotation it started from (what reprojectd applies), the accepted per-frame fits and their
  component-wise median, and the last frame it looked at."""
  def __init__(self, applied):
    self.applied = tuple(float(v) for v in applied)
    self.calib = calib_from_rotvec(self.applied)
    self.fits: list[tuple] = []
    self.mean = self.applied
    self.last_id = 0
    self.last_ok = False

  @property
  def n(self) -> int:
    return len(self.fits)

  @property
  def spread(self) -> float:
    return float(np.degrees(np.abs(np.array(self.fits) - self.mean).max())) if self.fits else 0.0

  def frame(self, narrow_y, wide_y, frame_id: int):
    """One frame pair: the first from the applied rotation with the coarse pass, later ones refined from the running median
    (half the work, ~1 frame/s on the 3X). Returns the frame's fit (rotvec, matches, rms) or None when rejected."""
    if not self.fits:
      r = fit_rotation(narrow_y, wide_y, self.calib)
    else:
      r = fit_rotation(narrow_y, wide_y, calib_from_rotvec(self.mean), iters=1, coarse=False)
    self.last_id = frame_id
    self.last_ok = r is not None and r[2] <= MAX_RMS_DEG
    if not self.last_ok:
      return None
    self.fits.append(r[0])
    self.mean = tuple(float(v) for v in np.median(self.fits, axis=0))
    return r
