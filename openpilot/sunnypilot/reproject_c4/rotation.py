"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The unit's fitted narrow->wide rotation, kept across boots in a param like CalibrationParams."""

import numpy as np

from .geometry import POP_ROTATION, rotvec_from_wide_from_device_euler


ROTATION_PARAM = "ReprojectRotation"  # the unit's fitted narrow->wide rotation, kept across boots like CalibrationParams

def read_rotation() -> dict:
  """{'rotvec', ...} as reprojectcalibd saved it, or {} when this unit's rotation has not been fitted."""
  try:
    from openpilot.common.params import Params
    d = Params().get(ROTATION_PARAM) or {}
    if len(d["rotvec"]) == 3 and np.isfinite(d["rotvec"]).all():
      return d
  except (KeyError, TypeError, ValueError):
    pass
  return {}

def load_rotation():
  """The unit's narrow->wide rotation: the fitted value, else seeded from the stock calibration's persisted
  wideFromDeviceEuler (calibrationd's block average of the model output), else the fleet median."""
  d = read_rotation()
  if d:
    return tuple(float(v) for v in d["rotvec"])
  try:
    from openpilot.cereal import log
    from openpilot.common.params import Params
    with log.Event.from_bytes(Params().get("CalibrationParams")) as msg:
      e = list(msg.extrinsicsCalibration.wideFromDeviceEuler)
    if len(e) == 3 and np.isfinite(e).all():
      return rotvec_from_wide_from_device_euler(e)
  except Exception:
    pass
  return POP_ROTATION

def save_rotation(rotvec, **extra) -> None:
  from openpilot.common.params import Params
  Params().put(ROTATION_PARAM, {'rotvec': [float(v) for v in rotvec], **extra}, block=True)
