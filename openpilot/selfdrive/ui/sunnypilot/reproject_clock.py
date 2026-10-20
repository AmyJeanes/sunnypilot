"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time
from collections.abc import Callable

from msgq.visionipc import VisionIpcClient
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.selfdrive.ui.ui_state import ui_state

STAGE_TIMEOUT_MS = 50  # one camera period
STAGE_PRESENT_DELAY_S = 0.002  # let modeld copy the frames first; the swap's memory traffic slows that copy


class ReprojectFrameClock:
  """With the driving model on the chestnut, the reprojection stage is the 3X GPU's last job in each camera period; otherwise
  the fallback clock decides."""
  def __init__(self, fallback: Callable[[], bool]):
    self._fallback = fallback
    self._stage = VisionIpcClient("reproject", VisionStreamType.VISION_STREAM_WIDE_ROAD, True)  # reprojectd sends wide last

  def wait(self) -> bool:
    if not (ui_state.started and ui_state.sm['modelV2'].big and (self._stage.is_connected() or self._stage.connect(False))):
      return self._fallback()
    if self._stage.recv(0) is None and self._stage.recv(STAGE_TIMEOUT_MS) is None:
      return False
    time.sleep(STAGE_PRESENT_DELAY_S)
    return True
