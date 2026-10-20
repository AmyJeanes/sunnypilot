"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import openpilot.cereal.messaging as messaging
from openpilot.cereal import log, custom
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.reproject_c4.helpers import reproject_expected

# the reprojection message informs the hold only: absent without an eGPU, quiet while the fitter builds tables, and it may
# not make the calibration read invalid
REPROJECT_SERVICES = ['reprojectCalibrationSP']


class CalibratorExt:
  def reproject_ready(self, sm: messaging.SubMaster) -> bool:
    """False while the 3X->comma 4 reprojection stage (reprojectd) runs on a rotation reprojectcalibd has not fitted yet:
    the model output this calibration is built on depends on it, so the calibration is held at uncalibrated until then."""
    if not reproject_expected():
      return True
    fit = sm['reprojectCalibrationSP']
    ready = sm.seen['reprojectCalibrationSP'] and fit.status == custom.ReprojectCalibrationSP.Status.fitted
    if not ready and (self.valid_blocks or self.idx or self.cal_status != log.ExtrinsicsCalibration.Status.uncalibrated):
      # uncalibrated, not recalibrating: that status means the mount moved and raises the offroad mounting alert
      cloudlog.warning("calibrationd: holding until the reprojection rotation is fitted")
      self.reset()
      self.cal_status = log.ExtrinsicsCalibration.Status.uncalibrated
    return ready
