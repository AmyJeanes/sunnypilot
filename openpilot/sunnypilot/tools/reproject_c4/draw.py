"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The debug clip's drawing processes: one frame's picture from the stage's renders, the 3X frames and what the main
process read from the log for it. Threads would fight the stage for the GIL, so this runs in worker processes."""
import signal

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.sunnypilot import reproject_c4 as RC
from openpilot.sunnypilot.reproject_c4.tables import ALPHA_SHIFT
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.sunnypilot.tools.reproject_c4 import stats
from openpilot.sunnypilot.tools.reproject_c4 import view as V
from openpilot.sunnypilot.tools.reproject_c4.outlines import debug_outlines

DW, DH = RC.C4_CAM
C4_CAMERA = DEVICE_CAMERAS['mici', 'os04c10']


def blend_weights(pw: np.ndarray) -> np.ndarray:
  """The stage's narrow/wide blend weights (0 all wide - 255 all narrow) over the comma 4 narrow frame's px, from its narrow
  table."""
  stride = get_nv12_info(DW, DH)[0]
  return ((pw[:stride * DH] >> ALPHA_SHIFT) & 0xff).reshape(DH, stride)[:, :DW].astype(np.uint8)


def fit_transform(native_wh: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
  """Native px -> px of the panel `bare` scales that frame into."""
  sc = min(size[0] / native_wh[0], size[1] / native_wh[1])
  return np.array([[sc, 0, (size[0] - round(native_wh[0] * sc)) / 2], [0, sc, (size[1] - round(native_wh[1] * sc)) / 2], [0, 0, 1]])


def input_coverage(alpha: np.ndarray, M: np.ndarray) -> tuple[float, float, float]:
  """Shares of the big model's narrow input (512x256 px, through modeld's warp M) that are pure 3X narrow, in the blend band
  and pure 3X wide."""
  ys, xs = np.mgrid[0:256, 0:512] + 0.5
  p = np.stack([xs, ys, np.ones_like(xs)], -1) @ M.T
  a = alpha[(p[..., 1] / p[..., 2]).astype(int).clip(0, DH - 1), (p[..., 0] / p[..., 2]).astype(int).clip(0, DW - 1)]
  return float((a == 255).mean()), float(((a > 0) & (a < 255)).mean()), float((a == 0).mean())


_W: dict = {}  # this process's slots and cameras, and the geometry for the rotation and calibration it last drew


def init_worker(cfg: dict) -> None:
  signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C reaches the whole process group; the main process stops the workers
  _W.update(cfg, geometry=None, alpha=None)


def _geometry(rotation: tuple, rpy: tuple, slot: int) -> dict:
  """The outlines as one layer over the whole picture (at full strength), the legends, the inset and the input coverage:
  they move only with the rotation and calibration."""
  if _W['geometry'] is None or _W['geometry']['key'] != (rotation, rpy):
    if _W['alpha'] is None or _W['alpha'][0] != rotation:
      slots = _W['slots']
      if tuple(slots.view(slot, 'blend_rotation').view(np.float64).tolist()) == rotation:  # the stage's, sent with the frame
        alpha = slots.view(slot, 'blend').reshape(DH, DW).copy()
      else:  # the slot has moved on to a later frame: numpy, as the drawing processes have no stage (~0.5 s)
        alpha = blend_weights(RC.build_tables(_W['src_wh'], RC.C4_CAM, RC.calib_from_rotvec(rotation))['narrow']['pw'])
      _W['alpha'] = (rotation, alpha)
    alpha = _W['alpha'][1]
    sets = debug_outlines(RC.calib_from_rotvec(rotation), np.asarray(rpy, np.float32))
    sizes = {'c4_narrow': ((DW, DH), (V.PW, V.PH)), 'c4_wide': ((DW, DH), (V.PW, V.PH)),
             'device_narrow': (_W['src_wh'], (V.XW, V.PH)), 'device_wide': (_W['src_wh'], (V.XW, V.PH))}
    _W['geometry'] = {'key': (rotation, rpy), 'masks': V.inset_masks(alpha, (V.PW, V.PH)),
                      'outlines': V.canvas_layer({k: V.outline_layer(sets[k], *wh) for k, wh in sizes.items()}),
                      'legends': {k: V.legend(sets[k]) for k in sizes},
                      'coverage': input_coverage(alpha, get_warp_matrix(np.asarray(rpy, np.float32), C4_CAMERA.narrow_road.intrinsics, False))}
  return _W['geometry']


def _panels(r: dict, device: dict, rpy: np.ndarray, wide_euler: tuple) -> tuple[dict, dict]:
  """The frame panels' pictures, from the stage's renders (r) and the 3X frames (packed NV12), and car space -> each
  panel's px, as the ui projects the path (the comma 4 frames are about the device axes)."""
  sw, sh = _W['src_wh']
  Kn, Kw = C4_CAMERA.narrow_road.intrinsics, C4_CAMERA.wide_road.intrinsics
  st, y_height, _, _ = get_nv12_info(DW, DH)
  c4 = {k: V.nv12_image(r[k.removeprefix('c4_')], DW, DH, st, st * y_height) for k in ('c4_narrow', 'c4_wide')}
  panels = {k: V.bare(img, V.PW, V.PH) for k, img in c4.items()}
  for k in ('device_narrow', 'device_wide'):
    panels[k] = V.nv12_image(device[k], sw, sh, sw, sw * sh, (V.XW, V.PH))
  S_in = np.diag([V.PH / 512, V.IH / 256, 1.0])
  M = {k: get_warp_matrix(rpy, K, k == 'input_wide') @ np.linalg.inv(S_in)  # panel px -> model px -> comma 4 px, as modeld warps
       for k, K in (('input_narrow', Kn), ('input_wide', Kw))}
  panels['input_narrow'] = V.perspective(c4['c4_narrow'], M['input_narrow'], (V.PH, V.IH))
  panels['input_wide'] = V.perspective(c4['c4_wide'], M['input_wide'], (V.PH, V.IH))

  view_from_calib = view_frame_from_device_frame @ rot_from_euler(rpy)
  view_from_wide = view_frame_from_device_frame @ rot_from_euler(wide_euler) @ rot_from_euler(rpy) if len(wide_euler) == 3 else view_from_calib
  x3 = _W['x3']
  T = {'c4_narrow': Kn @ view_from_calib, 'c4_wide': Kw @ view_from_calib,
       'device_narrow': x3.narrow_road.intrinsics @ view_from_calib, 'device_wide': x3.wide_road.intrinsics @ view_from_wide}
  S_c4, S_x3 = fit_transform((DW, DH), (V.PW, V.PH)), fit_transform((sw, sh), (V.XW, V.PH))
  to_panel = {k: (S_c4 if k.startswith('c4') else S_x3) @ T[k] for k in T}
  to_panel.update({k: np.linalg.inv(M[k]) @ T['c4_wide' if k == 'input_wide' else 'c4_narrow'] for k in M})
  return panels, to_panel


def frame(job: dict) -> None:
  """One frame's picture, into its slot. For the window, the path goes into a layer of its own, which it blends at its
  slider's opacity."""
  slots, slot, s = _W['slots'], job['slot'], job['settings']
  r = {k: slots.view(slot, k) for k in ('narrow', 'wide', 'narrow_only', 'wide_only')}
  g = _geometry(job['rotation'], tuple(np.round(job['rpy'], 3)), slot)
  panels, to_panel = _panels(r, {k: slots.view(slot, k) for k in ('device_narrow', 'device_wide')}, job['rpy'], job['wide_euler'])
  st = get_nv12_info(DW, DH)[0]
  n, n_only, w_only = (V.luma(r[k], DW, DH, st, (V.PW, V.PH)) for k in ('narrow', 'narrow_only', 'wide_only'))
  picture = V.compose(dict(panels, seam=V.seam_view(n, n_only, w_only, *g['masks']), legends=g['legends'], speed=job['speed'],
                           alert=job['alert'], stats=stats.stats_panel(*V.PANELS['stats'][2:], dict(job['stats'], coverage=g['coverage']))))
  def path_layer():
    return V.canvas_layer({k: job['path'].layer(img.size, to_panel[k]) for k, img in panels.items()})
  if job['window']:
    slots.view(slot, 'path')[:] = np.asarray(path_layer()).ravel()
  else:
    picture = picture.convert('RGBA')
    if s.path_opacity > 0:
      picture.alpha_composite(V.fade(path_layer(), s.path_opacity))
    if s.overlay_opacity > 0:
      picture.alpha_composite(V.fade(g['outlines'], s.overlay_opacity))
    picture = picture.convert('RGB')
  slots.view(slot, 'picture')[:] = np.asarray(picture).ravel()


def outline_canvas(rotation: tuple, rpy: tuple, slot: int) -> dict:
  """The window's outline layer for a rotation and calibration, as drawn for the frame in slot."""
  return {'key': (rotation, rpy), 'outlines': np.asarray(_geometry(rotation, rpy, slot)['outlines'])}
