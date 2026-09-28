"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The debug clip's frames in order, from the route to the drawn picture."""
import logging
import multiprocessing
import os
from collections import deque
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.sunnypilot import reproject_c4 as RC
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.clip.run import FRAMERATE, get_frame_dimensions
from openpilot.tools.lib.route import Route
from openpilot.sunnypilot.tools.reproject_c4 import draw, stats
from openpilot.sunnypilot.tools.reproject_c4 import rotations as R
from openpilot.sunnypilot.tools.reproject_c4 import view as V
from openpilot.sunnypilot.tools.reproject_c4.model_path import ModelPath
from openpilot.sunnypilot.tools.reproject_c4.route_data import SEG_FRAMES, CameradFrame, Frames, RouteLog, fetch
from openpilot.sunnypilot.tools.reproject_c4.stage import Stage

DW, DH = RC.C4_CAM
WINDOW_DEPTH = 6  # frames in flight in the window: enough to keep up, few enough that a seek answers quickly

logger = logging.getLogger(__name__)


class Slots:
  """Shared memory for the frames in flight: the stage's four renders and the two 3X frames go to the drawing processes, and
  their pictures come back, without pickling ~20 MB a frame through pipes (which holds the GIL the stage needs)."""
  def __init__(self, n: int, sizes: dict[str, int]):
    self.offsets = dict(zip(sizes, zip(np.cumsum([0, *sizes.values()])[:-1].tolist(), sizes.values(), strict=True), strict=True))
    self.stride = sum(sizes.values())
    self.shm = shared_memory.SharedMemory(create=True, size=n * self.stride)

  def view(self, slot: int, part: str) -> np.ndarray:
    off, size = self.offsets[part]
    return np.ndarray((size,), np.uint8, self.shm.buf, slot * self.stride + off)

  def close(self) -> None:
    self.shm.close()
    self.shm.unlink()


@dataclass(frozen=True)
class Settings:
  path_opacity: float = 1.0  # 1 = as the ui draws it
  overlay_opacity: float = 0.5


class Clip:
  """A route's frames between start and end, drawn in order from a position: the log, the stage on the GPU, the camera
  decoders and the drawing processes. submit() starts the next frame, take() hands back the oldest finished one."""
  def __init__(self, route: Route, start: int, end: int, window: bool = False):
    self.window = window  # draw for the window (the path apart), and loop
    self.first, self.last = start * FRAMERATE, end * FRAMERATE
    self.route, self.name = route, route.name.canonical_name.replace('|', '/')
    seg_start, seg_end = start // 60, (end - 1) // 60 + 1
    local = not any(p and p.startswith('http') for p in route.log_paths())
    logger.info(f'{self.name}: {start // 60}:{start % 60:02d}-{end // 60}:{end % 60:02d}, segment(s) {seg_start}-{seg_end - 1}, '
                + ('from this PC' if local else "from comma's servers (kept in its file cache after the first download)"))
    logs, cams = route.log_paths(), (route.camera_paths(), route.ecamera_paths())
    fetch([logs[s] for s in range(max(0, seg_start - 1), seg_end) if s < len(logs)] +
          [c[s] for c in cams for s in range(seg_start, seg_end) if s < len(c)])
    self.log = RouteLog(route, seg_start, seg_end)
    self.src_wh = get_frame_dimensions(next(p for p in route.camera_paths() if p))
    logger.info(f'device: {self.log.device} ({self.log.sensor}), cameras {self.src_wh[0]}x{self.src_wh[1]}')
    if self.src_wh != (1928, 1208):
      raise RuntimeError(f'{self.name} is from a {self.log.device or "device"} with {self.src_wh[0]}x{self.src_wh[1]} cameras: '
                         + 'only comma 3X routes (1928x1208) are reprojected')
    self.rotation: R.Logged | R.Fitted
    if any(len(v[0]) == 3 for v in self.log.fit.v):
      logger.info("rotation: the device's, as the route logged it")
      self.rotation = R.Logged(self.log)
    else:
      logger.info('rotation: the route logged none, so it is fitted here')
      self.rotation = R.Fitted(self.log, route, self.first, self.last, self.src_wh)

    workers = max(2, min(6, (os.cpu_count() or 4) // 2))  # past four the stage in this process sets the pace
    self.depth = WINDOW_DEPTH if window else 2 * workers + 1  # frames in flight
    self.n_slots = self.depth + 1  # and the one on screen
    st, y_height, uv_height, _ = get_nv12_info(DW, DH)
    body, frame = st * (y_height + uv_height), self.src_wh[0] * self.src_wh[1] * 3 // 2
    sizes = {'narrow': body, 'wide': body, 'narrow_only': body, 'wide_only': body, 'device_narrow': frame, 'device_wide': frame,
             'picture': V.W * V.H * 3}
    if window:
      sizes['path'] = V.W * V.H * 4
    self.slots = Slots(self.n_slots, sizes)
    # the traces' y scales, fixed for the clip so they hold still while it plays
    t0, t1 = (self.log.frame(*divmod(i, SEG_FRAMES))[0] for i in (self.first, self.last - 1))
    self.ranges = {}
    for k in ('e2e', 'model', 'dm', 'ui'):
      v = getattr(self.log, k).window(t0 - stats.WINDOW, t1)[:, 1]
      self.ranges[k] = (float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5))) if len(v) else None
    # forked before the stage opens the GPU and the decoders start their threads
    self.pool = multiprocessing.get_context('fork').Pool(workers, draw.init_worker, ({
      'slots': self.slots, 'src_wh': self.src_wh, 'x3': DEVICE_CAMERAS.get((self.log.device, self.log.sensor), DEVICE_CAMERAS['tici', 'ar0231'])},))
    self.stage = Stage(self.src_wh, self.rotation.initial)
    self.narrow_src, self.wide_src = CameradFrame(*self.src_wh), CameradFrame(*self.src_wh)
    self.pending: deque = deque()  # (job, result)
    self.decoders: tuple[Frames, Frames] | None = None
    self.count = 0  # frames submitted: slots are used in turn
    self.seek(self.first)

  def seek(self, gidx: int) -> None:
    """Draw on from frame gidx (of the route): the decoders restart at its keyframe and the seam match starts over."""
    self.discard()
    self._restart(gidx)

  def _restart(self, gidx: int) -> None:
    if self.decoders:  # not the first start
      for d in self.decoders:
        d.stop()
    self.next = max(self.first, min(gidx, self.last - 1))
    self.decoders = (Frames(self.route.camera_paths(), self.next, self.last, self.src_wh),
                     Frames(self.route.ecamera_paths(), self.next, self.last, self.src_wh))
    self.model_path = ModelPath()
    self.stage.meter = RC.SeamMeter(self.stage.meter_geometry)

  def submit(self, settings: Settings) -> bool:
    """Run the stage on the next frame and start drawing it; False at the end of the clip (in the window: back to its start)."""
    if self.next >= self.last:
      if not self.window:
        return False
      self._restart(self.first)
    assert self.decoders is not None and len(self.pending) < self.depth
    log, gidx = self.log, self.next
    seg, local = divmod(gidx, SEG_FRAMES)
    t, eof, narrow_exposure = log.frame(seg, local)
    narrow_bytes, wide_bytes = self.decoders[0].get(), self.decoders[1].get()

    applied, _, status = self.rotation.at(t)
    self.stage.follow(applied)
    r = self.stage.run(self.wide_src.load(wide_bytes), self.narrow_src.load(narrow_bytes),
                       RC.exposure_gain(narrow_exposure, log.wide_exposure.at(t, 0)))
    slot = self.count % self.n_slots
    for k, buf in (*r.items(), ('device_narrow', np.frombuffer(narrow_bytes, np.uint8)), ('device_wide', np.frombuffer(wide_bytes, np.uint8))):
      self.slots.view(slot, k)[:] = buf

    rpy, height, wide_euler = log.calib.at(t, ((0.0, 0.0, 0.0), 1.22, ()))
    alert, experimental = log.selfdrive.at(t, ('', False))
    path = self.model_path.update(log.models.get(eof) or log.model_at.at(t), log.lead.at(t), height,
                                  log.throttle.at(t, True) or not log.long_control.at(t, False), experimental)
    v_ego = log.v_ego.at(t)
    job = {'gidx': gidx, 'slot': slot, 'settings': settings, 'window': self.window, 'rotation': self.stage.rotation,
           'rpy': np.asarray(rpy if len(rpy) == 3 else (0, 0, 0), np.float32), 'wide_euler': wide_euler, 'path': path,
           'speed': f'{v_ego * 2.23694:.0f} mph' if v_ego is not None else None, 'alert': alert,
           'stats': {'series': {k: getattr(log, k).window(t - stats.WINDOW - 1, t) for k in ('e2e', 'model', 'dm', 'ui')},
                     'rotation': tuple(np.degrees(applied)), 'rotation_from': self.rotation.name, 'status': status, 'stage': log.stage.at(t),
                     'ranges': self.ranges, 'now': t,
                     'footer': f'{self.name} · segment {seg} · {R.clock(gidx)}'}}
    self.pending.append((job, self.pool.apply_async(draw.frame, (job,))))
    self.next += 1
    self.count += 1
    return True

  def ready(self) -> bool:
    return bool(self.pending) and self.pending[0][1].ready()

  def take(self) -> tuple[dict, np.ndarray]:
    """The oldest frame in flight, once drawn: its job and its picture (valid until `depth` more frames are submitted)."""
    job, result = self.pending.popleft()
    result.get()
    return job, self.slots.view(job['slot'], 'picture')

  def outlines(self, key: tuple):
    """Start drawing the window's outline layer for a (rotation, rounded calibration)."""
    return self.pool.apply_async(draw.outline_canvas, key)

  def discard(self) -> None:
    while self.pending:
      self.pending.popleft()[1].wait()

  def close(self) -> None:
    self.pool.terminate()  # frames in flight are dropped, not waited for
    self.slots.close()
    if self.decoders:
      for d in self.decoders:
        d.stop()
