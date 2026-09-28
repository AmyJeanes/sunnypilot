"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Where the debug clip's narrow->wide rotation comes from: the one the route's reprojectCalibrationSP says the device applied,
as it changed, or for a route that logged none (stock, or from before the fit) one fitted here before the clip, as
reprojectcalibd fits: the median of twelve frame pairs taken under calibrationd's conditions (straight road, above 15 mph),
about one a second, from the rotation it would seed from."""
import logging

import numpy as np

from openpilot.selfdrive.locationd.calibrationd import MAX_YAW_RATE_FILTER, MIN_SPEED_FILTER
from openpilot.sunnypilot import reproject_c4 as RC
from openpilot.sunnypilot.reproject_c4 import N_FRAMES, Fit
from openpilot.tools.clip.run import FRAMERATE
from openpilot.tools.lib.framereader import FrameReader
from openpilot.sunnypilot.tools.reproject_c4.route_data import SEG_FRAMES
PAIR_EVERY = FRAMERATE  # frames between pairs: about one a second, as the device manages

logger = logging.getLogger(__name__)


def seed(log, t: float) -> tuple:
  """The rotation reprojectcalibd starts from on a unit never fitted: the stock calibration's wideFromDeviceEuler, else the
  fleet median."""
  e = log.calib.at(t, ((), 0, ()))[2]
  if len(e) == 3 and np.isfinite(e).all():
    rot = RC.rotvec_from_wide_from_device_euler(e)
    logger.info(f"fit: starting from {deg(rot)} deg (the route's own calibration)")
    return rot
  logger.info(f'fit: starting from {deg(RC.POP_ROTATION)} deg (the fleet median: the route logged no calibration to start from)')
  return RC.POP_ROTATION


def deg(rot) -> str:
  return str(np.degrees(rot).round(2))


def clock(gidx: int) -> str:
  s = gidx // FRAMERATE
  return f'{s // 60}:{s % 60:02d}'


def usable(log, t: float) -> bool:
  """Whether reprojectcalibd would take a frame pair at t."""
  return (log.v_ego.at(t) or 0.0) > MIN_SPEED_FILTER and abs(log.yaw_rate.at(t, 0.0)) < MAX_YAW_RATE_FILTER


def describe(fit: Fit) -> str:
  return f'{fit.n} frame pairs, spread {fit.spread:.2f}°'


class Logged:
  name = "from the route's log"

  def __init__(self, log):
    self.log = log
    self.first: tuple = next(v for v in log.fit.v if len(v[0]) == 3)
    self.initial = self.first[0]  # what the stage starts from

  def at(self, t: float) -> tuple[tuple, tuple, str]:
    """(applied, target, status line) at log time t."""
    applied, target, status = self.log.fit.at(t, self.first)
    if len(applied) != 3:
      applied, target, status = self.first
    return applied, target if status == 'ready' else (), f"the device's fit: {status}"


class Fitted:
  """One fit for the whole clip, made before it plays: from the clip's frames, and the rest of its segments' when the clip
  alone has too few pairs."""
  name = 'fitted from this clip'

  def __init__(self, log, route, first: int, last: int, src_wh: tuple[int, int]):
    fit = Fit(seed(log, log.frame(*divmod(first, SEG_FRAMES))[0]))
    readers: dict = {}
    tried = 0
    seg_first, seg_last = first - first % SEG_FRAMES, last + (-last) % SEG_FRAMES
    near = [*range(first, last, PAIR_EVERY), *range(first - PAIR_EVERY, seg_first - 1, -PAIR_EVERY), *range(last, seg_last, PAIR_EVERY)]
    w, h = src_wh
    for gidx in near:
      seg, local = divmod(gidx, SEG_FRAMES)
      if not usable(log, log.frame(seg, local)[0]):
        continue
      for cam, paths in (('n', route.camera_paths()), ('w', route.ecamera_paths())):
        if (cam, seg) not in readers:
          readers[cam, seg] = FrameReader(paths[seg], pix_fmt='nv12')
      narrow, wide = (np.asarray(readers[c, seg].get(local), np.uint8).ravel()[:w * h].reshape(h, w) for c in ('n', 'w'))  # luma
      tried += 1
      r = fit.frame(narrow, wide, gidx)
      logger.info(f'fit: pair at {clock(gidx)} ' + (f'accepted, {deg(r[0])} deg' if r else 'rejected (too little matching detail)') + f'; {describe(fit)}')
      if fit.n >= N_FRAMES:
        break
    if not fit.fits:
      raise RuntimeError(f'no frame pair in the clip\'s segments fitted ({tried} tried, straight road above 15 mph): try a longer or daytime clip')
    if fit.n < N_FRAMES:
      logger.warning(f'fit: only {fit.n} of {N_FRAMES} pairs in the clip\'s segments ({tried} tried): the median of those')
    self.rotation = tuple(float(v) for v in fit.mean)
    self.status = f'median of {describe(fit)}'
    self.initial = self.rotation
    logger.info(f'fit: {deg(self.rotation)} deg, the median of {describe(fit)}')

  def at(self, t: float) -> tuple[tuple, tuple, str]:
    return self.rotation, (), self.status
