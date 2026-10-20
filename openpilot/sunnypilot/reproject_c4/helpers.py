"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import functools
from pathlib import Path

from openpilot.cereal import custom
from openpilot.common.hardware import HARDWARE
from openpilot.common.hardware.usb import cable_connected
from openpilot.common.params import Params
from openpilot.common.transformations.camera import DEVICE_CAMERAS, DeviceCameraConfig
from openpilot.selfdrive.modeld.helpers import chestnut_compiled, chestnut_present
from openpilot.sunnypilot.models.helpers import find_driving_pkl, get_active_bundle, get_active_model_runner

Runner = custom.ModelManagerSP.Runner
MODELD_V2_MODELS_DIR = Path(__file__).resolve().parents[1] / 'modeld_v2' / 'models'
C4_DEVICE_CAMERA = ("mici", "os04c10")  # the cameras reprojectd's frames are in


@functools.cache
def reproject_hardware() -> bool:
  """A 3X with a chestnut, by the modelds' own test: enumerated, or its cable connected before it enumerates. Decided once per
  process, as the modelds do, so a chestnut plugged in later takes a reboot."""
  return HARDWARE.get_device_type() == "tizi" and (chestnut_present() or cable_connected())


def big_model_ready(runner: int, params: Params | None = None) -> bool:
  """The runner will run a big model on the chestnut, by its own test: stock modeld's compiled big model, or modeld_v2's chestnut
  bundle downloaded, each with its 1344x760 warp."""
  if runner == Runner.stock:
    return chestnut_compiled()
  if runner == Runner.tinygrad:
    return (find_driving_pkl(get_active_bundle(params, chestnut=True)) is not None and
            (MODELD_V2_MODELS_DIR / 'big_driving_warp_1344x760_tinygrad.pkl').is_file())
  return False


def reproject_active(params: Params | None = None) -> bool:
  """reprojectd serves the 3X cameras as a comma 4's: every chestnut model is a comma 4 model, whichever runner runs it. The
  model choice only changes offroad; the manager asks each drive."""
  return reproject_hardware() and big_model_ready(get_active_model_runner(params), params)


@functools.cache
def reproject_expected() -> bool:
  """reproject_active() for onroad processes, which start after the model choice is fixed, so decide it once."""
  return reproject_active()


def camera_server() -> str:
  return "reproject" if reproject_expected() else "camerad"


def device_camera(device_type: str, sensor: str) -> DeviceCameraConfig:
  return DEVICE_CAMERAS[C4_DEVICE_CAMERA if reproject_expected() else (device_type, sensor)]
