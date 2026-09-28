"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

What the debug clip reads from a route: its logs as timelines and its cameras' frames, downloaded first if remote."""
import bisect
import json
import logging
import multiprocessing
import queue
import subprocess
import threading
import time
from functools import lru_cache

import numpy as np
import tqdm

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.clip.run import FRAMERATE
from openpilot.tools.lib.filereader import FileReader
from openpilot.tools.lib.framereader import FfmpegDecoder
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route
from openpilot.tools.lib.url_file import CHUNK_SIZE, URLFile

SEG_FRAMES = 60 * FRAMERATE

logger = logging.getLogger(__name__)


class Frames:
  """A camera's frames from one frame on, streamed from ffmpeg starting at the keyframe before it (the frame reader's index):
  the decoding stays out of this process's GIL, which the stage and the handoff to the drawing processes need."""
  def __init__(self, paths: list, first: int, last: int, wh: tuple[int, int]):
    self.size = wh[0] * wh[1] * 3 // 2
    self.q: queue.Queue = queue.Queue(maxsize=2 * FRAMERATE)
    self.stopped = threading.Event()
    threading.Thread(target=self._decode, args=(paths, first, last), daemon=True).start()

  def _decode(self, paths: list, first: int, last: int) -> None:
    try:
      for seg in range(first // SEG_FRAMES, (last - 1) // SEG_FRAMES + 1):
        if seg >= len(paths) or not paths[seg]:
          raise RuntimeError(f'no camera file for segment {seg}')
        dec = decoder_index(paths[seg])
        gop = int(dec.get_gop_start(max(first - seg * SEG_FRAMES, 0)))
        proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-flags2', 'showall', '-f', 'hevc', '-i', 'pipe:0', '-f', 'rawvideo', '-pix_fmt', 'nv12',
                                 '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)  # a stopped decoder complains
        assert proc.stdin is not None and proc.stdout is not None
        threading.Thread(target=self._feed, args=(proc.stdin, paths[seg], dec.prefix, int(dec.index[gop, 1])), daemon=True).start()
        try:
          for i in range(seg * SEG_FRAMES + gop, min(last, (seg + 1) * SEG_FRAMES)):
            buf = proc.stdout.read(self.size)
            if len(buf) < self.size:
              raise RuntimeError(f'{paths[seg]} ends at frame {i - seg * SEG_FRAMES}')
            if self.stopped.is_set():
              return
            if i >= first:
              self.q.put(buf)
        finally:
          proc.kill()
    except Exception as e:
      self.q.put(e)

  @staticmethod
  def _feed(pipe, path: str, prefix: bytes, offset: int) -> None:
    try:
      with FileReader(path) as f:
        f.seek(offset)
        pipe.write(prefix + f.read())
      pipe.close()
    except (BrokenPipeError, ValueError):  # the decoder was stopped
      pass

  def get(self) -> bytes:
    item = self.q.get(timeout=60)
    if isinstance(item, Exception):
      raise item
    return item

  def stop(self) -> None:
    self.stopped.set()
    while not self.q.empty():
      self.q.get_nowait()


@lru_cache(maxsize=16)
def decoder_index(path: str) -> FfmpegDecoder:
  """The frame reader's keyframe index of a camera file (~0.1 s to build), kept for seeks."""
  return FfmpegDecoder(path)


class CameradFrame:
  """A decoded frame (packed NV12) in camerad's strided buffer layout, which the stage's tables index."""
  def __init__(self, w: int, h: int):
    self.w, self.h = w, h
    stride, y_height, _, size = get_nv12_info(w, h)
    self.buf = np.zeros(size, np.uint8)
    self.y = self.buf[:stride * h].reshape(h, stride)[:, :w]
    self.uv = self.buf[stride * y_height:stride * (y_height + h // 2)].reshape(h // 2, stride)[:, :w]

  def load(self, packed: bytes) -> np.ndarray:
    a = np.frombuffer(packed, np.uint8)
    self.y[:] = a[:self.w * self.h].reshape(self.h, self.w)
    self.uv[:] = a[self.w * self.h:].reshape(self.h // 2, self.w)
    return self.buf


def read_segment(path: str) -> dict:
  """What the clip needs from one segment's log, as plain values (segments are parsed in worker processes)."""
  out: dict = {k: [] for k in ('cams', 'wide_exposure', 'e2e', 'model', 'dm', 'ui', 'models', 'fit', 'calib', 'selfdrive', 'stage',
                               'v_ego', 'throttle', 'lead', 'long_control', 'yaw_rate')}
  out['device'] = out['sensor'] = None
  for m in LogReader(path):
    try:
      w, t = m.which(), m.logMonoTime * 1e-9
      if w == 'narrowRoadCameraState':
        cs = m.narrowRoadCameraState
        out['cams'].append((t, cs.timestampEof, cs.gain * cs.integLines))
        out['sensor'] = str(cs.sensor)
      elif w == 'wideRoadCameraState':
        out['wide_exposure'].append((t, m.wideRoadCameraState.gain * m.wideRoadCameraState.integLines))
      elif w == 'modelV2':
        mv = m.modelV2
        out['e2e'].append((t, (m.logMonoTime - mv.timestampEof) * 1e-6))
        out['model'].append((t, mv.modelExecutionTime * 1e3))
        out['models'].append((t, mv.timestampEof, {
          'path': np.array([mv.position.x, mv.position.y, mv.position.z], np.float32).T,
          'lanes': [np.array([ll.x, ll.y, ll.z], np.float32).T for ll in mv.laneLines], 'lane_probs': list(mv.laneLineProbs),
          'edges': [np.array([e.x, e.y, e.z], np.float32).T for e in mv.roadEdges], 'edge_stds': list(mv.roadEdgeStds),
          'accel': np.array(mv.acceleration.x, np.float32)}))
      elif w == 'driverStateV2':
        out['dm'].append((t, m.driverStateV2.modelExecutionTime * 1e3))
      elif w == 'uiDebug' and m.uiDebug.frameTimeMillis > 0:  # some stock versions log zeros
        out['ui'].append((t, m.uiDebug.frameTimeMillis))
      elif w == 'reprojectCalibrationSP':
        f = m.reprojectCalibrationSP
        out['fit'].append((t, (tuple(f.applied), tuple(f.target), str(f.status))))
      elif w == 'extrinsicsCalibration':
        c = m.extrinsicsCalibration
        out['calib'].append((t, (tuple(c.rpyCalib), c.height[0] if len(c.height) else 1.22, tuple(c.wideFromDeviceEuler))))
      elif w == 'selfdriveState':
        out['selfdrive'].append((t, ((m.selfdriveState.alertText1 or '').strip(), m.selfdriveState.experimentalMode)))
      elif w == 'logMessage' and 'reprojectd.stage' in m.logMessage:
        out['stage'].append((t, json.loads(m.logMessage)['msg']))
      elif w == 'cameraOdometry':
        out['yaw_rate'].append((t, m.cameraOdometry.rot[2]))
      elif w == 'carState':
        out['v_ego'].append((t, m.carState.vEgo))
      elif w == 'longitudinalPlan':
        out['throttle'].append((t, m.longitudinalPlan.allowThrottle))
      elif w == 'radarState':
        lead = m.radarState.leadOne
        out['lead'].append((t, lead.dRel if lead.status else None))
      elif w == 'carParams':
        out['long_control'].append((t, m.carParams.openpilotLongitudinalControl))
      elif w == 'deviceState' and out['device'] is None:
        out['device'] = str(m.deviceState.deviceType)
    except Exception:  # a message or field this route's schema doesn't have
      continue
  return out


class Timeline:
  """One logged value: the latest at or before a time."""
  def __init__(self, rows: list):
    rows = sorted(rows, key=lambda r: r[0])
    self.t = [r[0] for r in rows]
    self.v = [r[1] for r in rows]

  def at(self, t: float, default=None):
    i = bisect.bisect_right(self.t, t) - 1
    return self.v[i] if i >= 0 else default

  def window(self, t0: float, t1: float) -> np.ndarray:
    """(time relative to t1, value) over [t0, t1]."""
    i, j = bisect.bisect_left(self.t, t0), bisect.bisect_right(self.t, t1)
    return np.array([(t - t1, v) for t, v in zip(self.t[i:j], self.v[i:j], strict=True)]).reshape(-1, 2)


def fetch(paths: list) -> None:
  """Download the remote ones into comma's file cache first, with a progress bar: read in 1 MB steps through the ordinary
  reader, so later reads (the logs, the frame reader, the decoders) find them local."""
  remote = [p for p in paths if p and p.startswith('http')]
  if not remote:
    return
  sizes = {p: URLFile(p).get_length() for p in remote}
  with tqdm.tqdm(total=sum(sizes.values()), unit='B', unit_scale=True, desc=f'Fetching {len(remote)} files') as bar:
    for p, size in sizes.items():
      with URLFile(p) as f:
        for pos in range(0, size, CHUNK_SIZE):
          f.seek(pos)
          bar.update(len(f.read(min(CHUNK_SIZE, size - pos))))


class RouteLog:
  """The clip's segments' logs (and the one before, for the stats' first 10 s) as timelines."""
  def __init__(self, route: Route, seg_start: int, seg_end: int):
    paths = route.log_paths()
    segs = [s for s in range(max(0, seg_start - 1), seg_end) if s < len(paths) and paths[s]]
    missing = [s for s in range(seg_start, seg_end) if s not in segs]
    if missing:
      raise RuntimeError(f'no log for segment(s) {missing}')
    t0 = time.monotonic()
    with multiprocessing.Pool(min(len(segs), 16)) as pool:
      parsed = dict(zip(segs, pool.map(read_segment, [paths[s] for s in segs]), strict=True))
    logger.info(f'logs: segments {segs[0]}-{segs[-1]} read in {time.monotonic() - t0:.1f} s')
    self.cams = {s: p['cams'] for s, p in parsed.items()}
    def rows(k):
      return [r for p in parsed.values() for r in p[k]]
    for k in ('wide_exposure', 'e2e', 'model', 'dm', 'ui', 'fit', 'calib', 'selfdrive', 'stage', 'v_ego', 'throttle', 'lead',
              'long_control', 'yaw_rate'):
      setattr(self, k, Timeline(rows(k)))
    self.models = {eof: m for _, eof, m in rows('models')}
    self.model_at = Timeline([(t, m) for t, _, m in rows('models')])
    self.device = next((p['device'] for p in parsed.values() if p['device']), None)
    self.sensor = next((p['sensor'] for p in parsed.values() if p['sensor']), None)

  def frame(self, seg: int, local: int) -> tuple[float, int, float]:
    """Log time, end-of-frame timestamp and exposure of the narrow camera's local-th frame of a segment."""
    cams = self.cams[seg]
    return cams[min(local, len(cams) - 1)]
