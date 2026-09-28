#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Play a route's 3X->comma 4 reprojection debug views in a window, off the device, or render them to an mp4 with -o.
  openpilot/sunnypilot/tools/reproject_c4/debug_clip.py <dongle/route> -s start -e end [-d data_dir] [-o out.mp4 [-f MB]]
      [--path-opacity 1.0] [--overlay-opacity 0.5]
The window plays the clip in real time with play/pause, a scrub bar and sliders for the two opacities, and records nothing.
Top row: the comma 4 narrow and wide frames the big model saw, rebuilt on this PC (reprojectd's compiled stage, CPU
build) from the route's fcamera and ecamera at a narrow->wide rotation (rotations.py: the one the route logged, else one
fitted here), and the seam check. Bottom row: the 3X frames, the model's two 512x256 inputs and the logged timings. Any
3X route with its full-resolution cameras (fcamera, ecamera) uploaded.

route_data.py reads the route, stage.py runs the reprojection, clip.py puts each frame through them and hands it to the
drawing processes (draw.py), which lay out the picture (view.py) with the model's path (model_path.py), the outlines
(outlines.py) and the stats (stats.py). window.py plays it."""
import logging
import os
import subprocess
import time
from argparse import ArgumentParser

import numpy as np
import tqdm

from openpilot.tools.clip.run import FRAMERATE
from openpilot.tools.lib.route import Route
from openpilot.sunnypilot.tools.reproject_c4 import view as V
from openpilot.sunnypilot.tools.reproject_c4.clip import Clip, Settings

logger = logging.getLogger(__name__)


def render(clip: Clip, output: str, target_mb: float, settings: Settings) -> None:
  ffmpeg = ['ffmpeg', '-v', 'warning', '-nostats', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{V.W}x{V.H}', '-r', str(FRAMERATE),
            '-i', 'pipe:0', '-vf', 'format=yuv420p', '-c:v', 'libx264', '-preset', 'veryfast']
  if target_mb > 0:
    rate = f'{int(target_mb * 8 * 1024 / ((clip.last - clip.first) / FRAMERATE))}k'
    ffmpeg += ['-b:v', rate, '-maxrate', rate, '-bufsize', rate]
  else:
    ffmpeg += ['-crf', '20']
  enc = subprocess.Popen(ffmpeg + ['-y', '-f', 'mp4', output], stdin=subprocess.PIPE)
  assert enc.stdin is not None
  with tqdm.tqdm(total=clip.last - clip.first, desc='Rendering', unit='frame') as bar:
    def write(keep: int) -> None:
      while clip.pending and (len(clip.pending) > keep or clip.ready()):
        enc.stdin.write(clip.take()[1])
        bar.update(1)
    try:
      while True:
        write(clip.depth - 1)
        if not clip.submit(settings):
          break
      write(0)
    finally:
      enc.stdin.close()
      enc.wait()


def main():
  logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
  ap = ArgumentParser(description="Play a route's reprojection debug views in a window, or render them to an mp4 with -o")
  ap.add_argument('route', help='Route ID (dongle/route or dongle/route/start/end)')
  ap.add_argument('-s', '--start', type=int, help='Start time in seconds')
  ap.add_argument('-e', '--end', type=int, help='End time in seconds')
  ap.add_argument('-o', '--output', help='Render to this mp4 instead of opening the window')
  ap.add_argument('-d', '--data-dir', help='Local directory with route data')
  ap.add_argument('-f', '--file-size', type=float, default=0, help="With -o, a target file size in MB (default: ffmpeg's quality)")
  ap.add_argument('--path-opacity', type=float, default=1.0, help="The model path's opacity, 1 = as the ui draws it")
  ap.add_argument('--overlay-opacity', type=float, default=0.5, help='The outlines, 0 hides them')
  a = ap.parse_args()
  if a.route.count('/') == 3:
    parts = a.route.split('/')
    a.route, a.start, a.end = '/'.join(parts[:2]), a.start or int(parts[2]), a.end or int(parts[3])
  if a.start is None or a.end is None:
    ap.error('--start and --end are required')
  if a.end <= a.start:
    ap.error(f'end ({a.end}) must be greater than start ({a.start})')
  if a.file_size and not a.output:
    ap.error('-f sets the size of a rendered file: give -o too')

  settings = Settings(float(np.clip(a.path_opacity, 0, 1)), float(np.clip(a.overlay_opacity, 0, 1)))
  t0 = time.monotonic()
  try:
    clip = Clip(Route(a.route, data_dir=a.data_dir), a.start, a.end, window=not a.output)
    try:
      if not a.output:
        from openpilot.sunnypilot.tools.reproject_c4.window import play
        logger.info('window: space pause, left/right 5 s, R reset; close it or Ctrl-C to stop')
        play(clip, settings)
      else:
        logger.info(f'rendering {a.end - a.start} s to {a.output}')
        render(clip, a.output, a.file_size, settings)
        logger.info(f'{os.path.abspath(a.output)} in {time.monotonic() - t0:.0f} s')
    finally:
      clip.close()
  except RuntimeError as e:
    raise SystemExit(f'error: {e}') from None
  except KeyboardInterrupt:
    raise SystemExit(130) from None


if __name__ == '__main__':
  main()
