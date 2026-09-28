"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

reprojectd's stage on this PC, from the CPU builds of its compiled programs (scons builds them as on the device)."""
import os
os.environ.setdefault('DEV', 'CPU:LLVM')  # before tinygrad is imported: the programs are built for it
import logging

import numpy as np

from openpilot.sunnypilot import reproject_c4 as RC
from openpilot.sunnypilot.reproject_c4.tables import ALPHA_SHIFT
from openpilot.sunnypilot.tools.reproject_c4.draw import blend_weights

DEV = os.environ['DEV'].split(':')[0]

logger = logging.getLogger(__name__)


class Stage:
  """reprojectd's stage at a rotation, run three ways each frame: as on the device, and with the blend forced to all narrow and
  all wide inside the inset (the seam check's two sides, both matched as the device matches the wide). As on the device, each
  frame is matched with the seam match measured on the one before."""
  def __init__(self, src_wh, rotation):
    from openpilot.sunnypilot.reproject_c4.kernel import Reprojector
    self.src_wh = src_wh
    self.rp = {k: Reprojector(src_wh, DEV) for k in ('narrow', 'narrow_only', 'wide_only')}
    self.rotation: tuple = ()
    self.match: np.ndarray | None = None
    self.follow(rotation)

  def follow(self, rotation) -> None:
    """Swap to the rotation the log says is applied now."""
    from tinygrad import Tensor
    rot = tuple(float(v) for v in rotation)
    if rot == self.rotation:
      return
    self.rp['narrow'].set_rotation(rot)
    pw = self.rp['narrow'].tables['pw_narrow'].numpy().astype(np.int64)
    alpha = (pw >> ALPHA_SHIFT) & 0xff
    self.blend = blend_weights(pw)
    for k, a in (('narrow_only', np.where(alpha > 0, 255, 0)), ('wide_only', np.zeros_like(alpha))):
      for name, t in self.rp['narrow'].tables.items():
        v = ((pw & ~(0xff << ALPHA_SHIFT)) | (a << ALPHA_SHIFT)).astype(np.int32) if name == 'pw_narrow' else t.numpy()
        self.rp[k].tables[name].assign(Tensor(v, device=DEV)).realize()
    self.meter_geometry = RC.SeamMeter.geometry(self.src_wh, RC.C4_CAM, RC.calib_from_rotvec(rot))
    self.meter = RC.SeamMeter(self.meter_geometry)
    if self.rotation:
      logger.info(f'stage: now applying {np.degrees(rot).round(2)} deg')
    self.rotation = rot

  def run(self, wide: np.ndarray, narrow: np.ndarray, gain: float) -> dict[str, np.ndarray]:
    """The comma 4 narrow and wide frames and the narrow-only and wide-only narrow frames, as strided NV12."""
    from tinygrad import Tensor
    tw, tn = Tensor(wide, device=DEV).realize(), Tensor(narrow, device=DEV).realize()
    c4_wide, c4_narrow = self.rp['narrow'](tw, tn, self.match)
    out = {'narrow': c4_narrow.numpy(), 'wide': c4_wide.numpy()}
    for k in ('narrow_only', 'wide_only'):
      out[k] = self.rp[k](tw, tn, self.match)[1].numpy()
    self.match = self.meter.update(wide, narrow, gain)
    return out
