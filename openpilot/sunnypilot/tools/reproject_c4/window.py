"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The debug clip in a window, in real time and looping: play/pause, a scrub bar and sliders for the path's and the outlines'
opacity under the picture. Nothing is recorded. The sliders act as the picture is shown: the path and outlines come from the
drawing processes as layers of their own, which the window blends at the sliders' opacity."""
import os
import time
from dataclasses import replace

import numpy as np
import pyray as rl

from openpilot.tools.clip.run import FRAMERATE
from openpilot.sunnypilot.tools.reproject_c4 import view as V
from openpilot.sunnypilot.tools.reproject_c4.clip import Clip, Settings
from openpilot.sunnypilot.tools.reproject_c4.rotations import clock

BAR = 64  # control bar height, window px
MIN_W = 1528  # the smallest the picture reads well at (1528 x 682 with the bar)
SEEK_S = 5
LABELS = {'path_opacity': 'model path', 'overlay_opacity': 'outlines'}


def rgb(c: tuple, a: int = 255) -> rl.Color:
  return rl.Color(c[0], c[1], c[2], a)


_mouse = {'down': False, 'pressed': False}


def poll_mouse() -> None:
  """A press is the button going down since the last frame (once per frame, before any clicked())."""
  down = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)
  _mouse['pressed'], _mouse['down'] = down and not _mouse['down'], down


def clicked(rect: rl.Rectangle) -> bool:
  return _mouse['pressed'] and rl.check_collision_point_rec(rl.get_mouse_position(), rect)


class Slider:
  """A horizontal slider: a track, the filled part and a knob, dragged with the mouse from on or near the track."""
  def __init__(self, lo: float, hi: float, color: tuple):
    self.lo, self.hi, self.color = lo, hi, color
    self.track = rl.Rectangle(0, 0, 0, 0)
    self.dragging = False

  def update(self, value: float) -> tuple[float, bool]:
    """The value after this frame's mouse input, and whether the drag just ended."""
    m = rl.get_mouse_position()
    if clicked(rl.Rectangle(self.track.x - 10, self.track.y - 14, self.track.width + 20, self.track.height + 28)):
      self.dragging = True
    if self.dragging:
      value = self.lo + float(np.clip((m.x - self.track.x) / max(self.track.width, 1), 0, 1)) * (self.hi - self.lo)
      if not _mouse['down']:
        self.dragging = False
        return value, True
    return value, False

  def draw(self, value: float) -> None:
    t = self.track
    frac = (value - self.lo) / (self.hi - self.lo)
    rl.draw_rectangle_rounded(t, 1.0, 6, rgb((48, 55, 66)))
    rl.draw_rectangle_rounded(rl.Rectangle(t.x, t.y, frac * t.width, t.height), 1.0, 6, rgb(self.color))
    rl.draw_circle(int(t.x + frac * t.width), int(t.y + t.height / 2), 9 if self.dragging else 8, rgb(V.INK))


def texture(w: int, h: int, fmt) -> rl.Texture:
  image = rl.gen_image_color(w, h, rl.BLANK)
  rl.image_format(image, fmt)
  tex = rl.load_texture_from_image(image)
  rl.unload_image(image)
  rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
  return tex


def upload(tex: rl.Texture, data: np.ndarray) -> None:
  rl.update_texture(tex, rl.ffi.cast('void *', data.ctypes.data))


def open_window() -> None:
  """Sized to the screen, centered, and no smaller than the picture reads well at (or what the screen allows)."""
  rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE | rl.ConfigFlags.FLAG_MSAA_4X_HINT)
  rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
  rl.init_window(1280, 600, 'reprojection debug clip')
  mon = rl.get_current_monitor()
  sc = min(1.0, 0.95 * rl.get_monitor_width(mon) / V.W, (0.9 * rl.get_monitor_height(mon) - BAR) / V.H)
  ww, wh = round(V.W * sc), round(V.H * sc) + BAR
  rl.set_window_size(ww, wh)
  rl.set_window_position((rl.get_monitor_width(mon) - ww) // 2, max(0, (rl.get_monitor_height(mon) - wh) // 2))
  min_w = min(MIN_W, ww)
  rl.set_window_min_size(min_w, round(min_w * V.H / V.W) + BAR)
  rl.set_target_fps(60)


class Player:
  def __init__(self, clip: Clip, settings: Settings):
    self.clip = clip
    self.settings = self.initial = settings  # reset goes back to the command line's, else the defaults
    F = rl.PixelFormat
    self.tex = {'picture': texture(V.W, V.H, F.PIXELFORMAT_UNCOMPRESSED_R8G8B8),
                'path': texture(V.W, V.H, F.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8),
                'outlines': texture(V.W, V.H, F.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)}
    self.fonts = {n: rl.load_font_ex(os.path.join(V.FONTS, f), 40, None, 0) for n, f in (('ui', 'Inter-Medium.ttf'), ('mono', 'JetBrainsMono-Medium.ttf'))}
    for f in self.fonts.values():
      rl.set_texture_filter(f.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    self.scrub = Slider(clip.first, clip.last - 1, (160, 170, 185))
    self.sliders = {'path_opacity': Slider(0, 1, (242, 242, 242)), 'overlay_opacity': Slider(0, 1, V.NARROW_IN)}
    self.shown: dict | None = None
    self.playing, self.want_frame = True, True
    self.due = time.monotonic()
    self.scrub_target: float = clip.first
    self.shown_times: list[float] = []
    self.fps, self.fps_at = 0, 0.0
    self.outlines_key, self.outlines_pending = None, None

  def width(self, s: str, size: float = 17, font: str = 'ui') -> float:
    return rl.measure_text_ex(self.fonts[font], s, size, 0).x  # noqa: TID251  # text_measure scales for the ui's fonts; this window draws unscaled

  def text(self, s: str, x: float, y: float, size: float = 17, color: tuple = V.INK, font: str = 'ui') -> float:
    rl.draw_text_ex(self.fonts[font], s, rl.Vector2(x, y - size / 2), size, 0, rgb(color))
    return self.width(s, size, font)

  def run(self) -> None:
    try:
      while not rl.window_should_close():
        now = time.monotonic()
        self.input()
        self.advance(now)
        rl.begin_drawing()
        self.draw_picture()
        self.draw_bar()
        rl.end_drawing()
    finally:
      for t in self.tex.values():
        rl.unload_texture(t)
      for f in self.fonts.values():
        rl.unload_font(f)
      rl.close_window()

  def seek(self, gidx: int) -> None:
    self.clip.seek(gidx)
    self.want_frame, self.due = True, time.monotonic()

  def toggle(self) -> None:
    self.playing, self.due = not self.playing, time.monotonic()

  def input(self) -> None:
    """Keys and the sliders' drags (the bar's buttons are clicked as it's drawn)."""
    poll_mouse()
    if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE):
      self.toggle()
    if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
      self.settings = self.initial
    for key, step in ((rl.KeyboardKey.KEY_RIGHT, SEEK_S), (rl.KeyboardKey.KEY_LEFT, -SEEK_S)):
      if rl.is_key_pressed(key) and self.shown:
        self.seek(self.shown['gidx'] + step * FRAMERATE)
    self.scrub_target, released = self.scrub.update(self.shown_gidx)
    if released:
      self.seek(int(self.scrub_target))
    for name, sl in self.sliders.items():
      value, done = sl.update(getattr(self.settings, name))
      if sl.dragging or done:
        self.settings = replace(self.settings, **{name: value})

  @property
  def shown_gidx(self) -> int:
    return self.shown['gidx'] if self.shown else self.clip.first

  def advance(self, now: float) -> None:
    """Keep the pipeline full while playing and show a frame every 50 ms; take up new outlines once drawn."""
    clip = self.clip
    if self.playing:
      while len(clip.pending) < clip.depth and clip.submit(self.settings):
        pass
    elif self.want_frame and not clip.pending:
      clip.submit(self.settings)
    if clip.pending and (self.want_frame or (self.playing and now >= self.due and clip.ready())):
      self.show(*clip.take())
      self.want_frame = False
      self.due = max(self.due + 1 / FRAMERATE, now - 1 / FRAMERATE)  # a late frame delays the rest rather than hurrying them
    if self.outlines_pending is not None and self.outlines_pending[1].ready():
      layer = self.outlines_pending[1].get()
      upload(self.tex['outlines'], layer['outlines'])
      self.outlines_key, self.outlines_pending = layer['key'], None
    self.shown_times = [t for t in self.shown_times if now - t < 2.0]
    if now - self.fps_at >= 1.0:  # a steady reading: the frames shown in the last 2 s, once a second
      self.fps, self.fps_at = round(len(self.shown_times) / 2), now

  def show(self, job: dict, picture: np.ndarray) -> None:
    self.shown = job
    upload(self.tex['picture'], picture)
    upload(self.tex['path'], self.clip.slots.view(job['slot'], 'path'))
    key = (job['rotation'], tuple(np.round(job['rpy'], 3)))
    if key != self.outlines_key and (self.outlines_pending is None or self.outlines_pending[0] != key):
      self.outlines_pending = (key, self.clip.outlines(key))
    self.shown_times.append(time.monotonic())

  def draw_picture(self) -> None:
    """The picture, then the path and outlines at the sliders' opacity, scaled to fit above the bar."""
    sw, pic_h = rl.get_screen_width(), rl.get_screen_height() - BAR
    s = min(sw / V.W, pic_h / V.H)
    whole = rl.Rectangle((sw - V.W * s) / 2, (pic_h - V.H * s) / 2, V.W * s, V.H * s)
    src = rl.Rectangle(0, 0, V.W, V.H)
    rl.clear_background(rgb(V.BG))
    rl.draw_texture_pro(self.tex['picture'], src, whole, rl.Vector2(0, 0), 0, rl.WHITE)
    for k, opacity in (('path', self.settings.path_opacity if self.shown else 0),
                       ('outlines', self.settings.overlay_opacity if self.outlines_key else 0)):
      if opacity > 0:
        rl.draw_texture_pro(self.tex[k], src, whole, rl.Vector2(0, 0), 0, rgb((255, 255, 255), round(255 * opacity)))

  def draw_bar(self) -> None:
    """Play/pause and the time, the scrub bar (three slider tracks long) and the end time, the sliders, then reset and the
    frame rate at the right."""
    sw, pic_h = rl.get_screen_width(), rl.get_screen_height() - BAR
    rl.draw_rectangle(0, pic_h, sw, BAR, rgb(V.PANEL))
    cy = pic_h + BAR / 2
    bx = 22.0
    if self.playing:
      rl.draw_rectangle(int(bx), int(cy - 10), 6, 20, rgb(V.INK))
      rl.draw_rectangle(int(bx + 11), int(cy - 10), 6, 20, rgb(V.INK))
    else:
      rl.draw_triangle(rl.Vector2(bx, cy - 11), rl.Vector2(bx, cy + 11), rl.Vector2(bx + 18, cy), rgb(V.INK))
    if clicked(rl.Rectangle(bx - 8, cy - 18, 34, 36)):
      self.toggle()
    bx += 42
    scrub_at = int(round(self.scrub_target)) if self.scrub.dragging else self.shown_gidx
    bx += self.text(clock(scrub_at), bx, cy, font='mono') + 16

    value_w, reset_w = self.width('100%', 16, 'mono'), self.width('reset', 15) + 24
    reset = rl.Rectangle(sw - 22 - self.width('20 fps', 15, 'mono') - 22 - reset_w, cy - 14, reset_w, 28)
    fixed = 14 + self.width(clock(self.clip.last), 17, 'mono') + 28 + sum(self.width(LABELS[n], 16) + 24 + value_w + 28 for n in self.sliders)
    track = max(40.0, (reset.x - bx - fixed) / (3 + len(self.sliders)))
    self.scrub.track = rl.Rectangle(bx, cy - 3, 3 * track, 6)
    self.scrub.draw(scrub_at)
    bx = self.scrub.track.x + self.scrub.track.width + 14
    bx += self.text(clock(self.clip.last), bx, cy, font='mono', color=V.MUTED) + 28
    for name, sl in self.sliders.items():
      bx += self.text(LABELS[name], bx, cy, 16, V.MUTED) + 12
      sl.track = rl.Rectangle(bx, cy - 3, track, 6)
      v = getattr(self.settings, name)
      sl.draw(v)
      bx += track + 12
      self.text(f'{v * 100:3.0f}%', bx, cy, 16, font='mono')
      bx += value_w + 28

    changed = self.settings != self.initial
    rl.draw_rectangle_rounded_lines_ex(reset, 0.4, 6, 1.5, rgb(V.INK if changed else (70, 78, 90)))
    self.text('reset', reset.x + 12, cy, 15, V.INK if changed else V.MUTED)
    if clicked(reset):
      self.settings = self.initial
    if self.playing:
      self.text(f'{self.fps:2d} fps', reset.x + reset.width + 22, cy, 15, V.MUTED, 'mono')


def play(clip: Clip, settings: Settings) -> None:
  open_window()
  Player(clip, settings).run()
