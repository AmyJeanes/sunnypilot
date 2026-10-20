#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Fits this unit's narrow->wide camera rotation for the 3X->comma 4 reprojection stage: a one-off, the median of twelve
frame pairs taken under calibrationd's own conditions (straight road, above 15 mph), about one frame a second.
The result is published as ready; reprojectd swaps it in and keeps it in ReprojectRotation, and only then is it
published as fitted (calibrationd holds until then, so the car cannot be engaged around the swap); afterwards this
process idles. The rotation is board geometry: it is fitted once and only a calibration reset, which clears it, fits it again. Numpy only, low
priority, one frame pair every few seconds."""
import os
import time

import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.locationd.calibrationd import MIN_SPEED_FILTER, MAX_YAW_RATE_FILTER
from openpilot.sunnypilot import reproject_c4 as RC
from openpilot.sunnypilot.reproject_c4 import N_FRAMES, Fit

NARROW, WIDE = VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD


def luma(buf) -> np.ndarray:
  return np.array(buf.data[:buf.uv_offset], dtype=np.uint8).reshape(-1, buf.stride)[:buf.height, :buf.width]


class FitState:
  """The fit as reprojectd, calibrationd, the alert and the log see it (reprojectCalibrationSP): published on every change and
  every 0.5 s in between."""
  def __init__(self):
    self.pm = messaging.PubMaster(['reprojectCalibrationSP'])
    self.last = None
    self.t = 0.0

  def publish(self, status: str, fit: Fit, why: str | None = None, target: tuple = ()) -> None:
    pct = 100 if status in ('ready', 'fitted') else 100 * fit.n // N_FRAMES
    mean = [float(v) for v in (fit.applied if status == 'fitted' else fit.mean)]
    key = (status, why, pct, tuple(mean), fit.last_id, fit.last_ok, fit.applied, target)
    if key == self.last and time.monotonic() - self.t < 0.5:
      return
    self.last, self.t = key, time.monotonic()
    msg = messaging.new_message('reprojectCalibrationSP', valid=True)
    f = msg.reprojectCalibrationSP
    f.status = status
    f.why = why or 'none'
    f.pct = pct
    f.mean = mean
    f.lastFrameId = fit.last_id
    f.lastAccepted = fit.last_ok
    f.applied = list(fit.applied)
    f.target = list(target)
    self.pm.send('reprojectCalibrationSP', msg)


def main():
  os.nice(10)
  sm = messaging.SubMaster(['carState', 'extrinsicsCalibration', 'cameraOdometry'])
  narrow = VisionIpcClient("camerad", NARROW, True)
  wide = VisionIpcClient("camerad", WIDE, False)
  params = Params()
  state = FitState()
  fit = Fit(RC.load_rotation())  # the rotation reprojectd applies: the same param, else the same seed
  fitted = bool(RC.read_rotation())
  if fitted and params.get("CalibrationParams") is None:
    # Reset Calibration (or the fit itself, see below) cleared the calibration: fit again first
    cloudlog.warning("reprojectcalibd: no calibration: refitting the rotation")
    fitted = False
  cloudlog.warning(f"reprojectcalibd: applied rotation {np.degrees(fit.applied).round(3)} deg, {'fitted' if fitted else 'not fitted yet'}")

  def done(final) -> None:
    while True:
      state.publish('fitted', Fit(final))
      time.sleep(0.5)

  if fitted:
    done(fit.applied)
  state.publish('waiting', fit)
  why = None
  why_t = log_t = time.monotonic()

  def waiting(reason: str | None) -> None:
    """What the fit is waiting for (cameras, model, speed, straight road, a matched frame pair), for the bar and the tile;
    a stretch of it longer than 5 s is logged, and one that never ends every 30 s, so a stalled bar explains itself in the rlog."""
    nonlocal why, why_t, log_t
    now = time.monotonic()
    if reason != why:
      if why is not None and now - why_t > 5.0:
        cloudlog.warning(f"reprojectcalibd: waited {now - why_t:.0f} s for {why} at {fit.n} fits")
      why, why_t, log_t = reason, now, now
    elif reason is not None and now - log_t > 30.0:
      cloudlog.warning(f"reprojectcalibd: still waiting for {reason} after {now - why_t:.0f} s at {fit.n} fits")
      log_t = now
    state.publish('fitting' if fit.fits else 'waiting', fit, why=reason)

  while True:
    sm.update(100)
    if not (narrow.is_connected() and wide.is_connected()):
      narrow.connect(False)
      wide.connect(False)
      waiting('cameras')
      continue
    if not (sm.alive['extrinsicsCalibration'] and sm.valid['extrinsicsCalibration']):
      # calibrationd is valid once modeld publishes cameraOdometry: the model is still loading. alive+valid, not
      # all_checks: its frequency check fails for a second after a 3.5 s coarse fit, which showed as 'Loading Model'
      waiting('model')
      continue
    if not all(sm.alive[k] and sm.valid[k] for k in ('carState', 'cameraOdometry')):
      waiting('model')
      continue
    if sm['carState'].vEgo <= MIN_SPEED_FILTER:
      waiting('speed')
      continue
    if abs(sm['cameraOdometry'].rot[2]) >= MAX_YAW_RATE_FILTER:
      waiting('straight')
      continue
    pair = RC.recv_pair(narrow, wide)
    if pair is None:
      waiting('pair')
      continue
    waiting(None)
    bn, bw = pair
    narrow_y, wide_y = luma(bn), luma(bw)
    t0 = time.monotonic()
    r = fit.frame(narrow_y, wide_y, narrow.frame_id)
    if r is None:
      cloudlog.warning(f"reprojectcalibd: frame {narrow.frame_id} rejected, {time.monotonic() - t0:.1f} s")
      # too few patches with texture (dusk, plain road): the bar says so instead of sitting at the same number
      state.publish('fitting' if fit.fits else 'waiting', fit, why='features')
      continue
    cloudlog.warning(f"reprojectcalibd: fit {fit.n}: {np.degrees(r[0]).round(3)} deg, {r[1]} matches, rms {r[2]:.3f} deg, {time.monotonic() - t0:.1f} s; "
                     + f"median {np.degrees(fit.mean).round(3)} spread {fit.spread:.3f} deg")
    state.publish('fitting', fit)
    if fit.n < N_FRAMES:
      continue
    final = tuple(float(v) for v in fit.mean)
    # calibrationd held at uncalibrated through the fit and calibrates afresh after the swap, but first writes that ~25 s
    # in: the calibration on disk is from other frames (stock, or another rotation) and must not outlive a reboot before then
    params.remove("CalibrationParams")
    while RC.read_rotation().get("rotvec") != list(final):  # reprojectd saves it once swapped in
      state.publish('ready', fit, target=final)
      time.sleep(0.1)
    cloudlog.warning(f"reprojectcalibd: rotation fitted {np.degrees(final).round(3)} deg (was {np.degrees(fit.applied).round(3)}, spread {fit.spread:.3f} deg)")
    done(final)


if __name__ == "__main__":
  main()
