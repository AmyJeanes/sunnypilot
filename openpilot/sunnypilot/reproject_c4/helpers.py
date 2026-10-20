"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import functools

from openpilot.cereal import custom
from openpilot.common.hardware import HARDWARE
from openpilot.common.hardware.usb import cable_connected
from openpilot.selfdrive.modeld.helpers import chestnut_compiled, chestnut_present
from openpilot.sunnypilot.models.helpers import get_active_model_runner


@functools.cache
def reproject_hardware() -> bool:
  """reprojectd can serve the 3X cameras as a comma 4's for the big model: a 3X with a chestnut and the big model and its warps in
  place. The chestnut test is modeld's, a cable counting before the chestnut enumerates, so the two always agree on the frames.
  Decided once per process, as modeld does, so a chestnut plugged in later takes a reboot."""
  return HARDWARE.get_device_type() == "tizi" and chestnut_compiled() and (chestnut_present() or cable_connected())


@functools.cache
def reproject_expected() -> bool:
  """reprojectd runs: the hardware allows it and the stock runner, the only one that reads its frames, is active. The runner
  only changes offroad, so onroad processes decide this once; the manager checks the runner itself each drive."""
  return reproject_hardware() and get_active_model_runner() == custom.ModelManagerSP.Runner.stock
