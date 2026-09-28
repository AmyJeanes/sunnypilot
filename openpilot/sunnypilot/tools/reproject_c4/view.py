"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The debug clip's picture: its layout, the panels' header bars and the image helpers the drawing shares."""
import os
from functools import cache, lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from openpilot.common.basedir import BASEDIR

FONTS = os.path.join(BASEDIR, 'openpilot/selfdrive/assets/fonts')
BG, PANEL, INK, MUTED, ALERT = (11, 13, 16), (20, 24, 30), (236, 239, 243), (142, 152, 166), (255, 200, 80)
# color follows what a thing is, the same in every panel: the two 3X cameras yellow / blue (complementary, so the seam's tint
# adds up to grey where they agree), the two model inputs cyan / magenta, the comma 4 frame edges lime (the model path is white)
NARROW_CAM, WIDE_CAM, NARROW_IN, WIDE_IN, C4_FRAME = (255, 196, 0), (60, 124, 255), (0, 216, 255), (255, 61, 220), (157, 255, 60)
KIND_COLORS = {'narrow_cam': NARROW_CAM, 'narrow_input': NARROW_IN, 'wide_input': WIDE_IN, 'c4_frame': C4_FRAME}
SEAM_KEY = (('3X narrow', NARROW_CAM), ('3X wide', WIDE_CAM), ('agree', (170, 170, 170)))

# two rows under 36 px headers: the comma 4 frames and the seam, then the 3X frames at their own shape, the model inputs
# stacked in a square and the stats in what's left
PW, PH, HB, GAP = 896, 507, 36, 8
XW = round(PH * 1928 / 1208)
W, H = 3 * PW + 2 * GAP, 2 * (HB + PH) + GAP
STW = W - 2 * XW - PH - 3 * GAP
SEAM_BLUR = round(16 * PW / 1344)  # the seam check's detail: anything finer than 16 comma 4 px

# limited-range video levels (Y 16-235, chroma 16-240) to the full range PIL's YCbCr conversion expects
Y_LUT = [min(255, max(0, round((v - 16) * 255 / 219))) for v in range(256)]
C_LUT = [min(255, max(0, round(128 + (v - 128) * 255 / 224))) for v in range(256)]


@cache
def font(size: int, weight: str = 'Medium') -> ImageFont.FreeTypeFont:
  return ImageFont.truetype(os.path.join(FONTS, f'Inter-{weight}.ttf'), size)


@cache
def mono(size: int) -> ImageFont.FreeTypeFont:
  return ImageFont.truetype(os.path.join(FONTS, 'JetBrainsMono-Medium.ttf'), size)


def luma(buf: np.ndarray, w: int, h: int, stride: int, size: tuple[int, int] | None = None) -> Image.Image:
  y = Image.fromarray(np.ascontiguousarray(buf[:stride * h].reshape(h, stride)[:, :w]))
  return (y if size in (None, (w, h)) else y.resize(size, Image.BILINEAR)).point(Y_LUT)


def nv12_image(buf: np.ndarray, w: int, h: int, stride: int, uv_off: int, size: tuple[int, int] | None = None) -> Image.Image:
  """The frame in RGB, scaled straight to `size` (default its own) from its planes."""
  size = size or (w, h)
  uv = buf[uv_off:uv_off + stride * (h // 2)].reshape(h // 2, stride)[:, :w]
  cb, cr = (Image.fromarray(np.ascontiguousarray(uv[:, i::2])).resize(size, Image.BILINEAR).point(C_LUT) for i in (0, 1))
  return Image.merge('YCbCr', (luma(buf, w, h, stride, size), cb, cr)).convert('RGB')


@lru_cache(maxsize=16)
def solid(size: tuple[int, int], color: tuple) -> Image.Image:
  return Image.new('RGB', size, color)


def perspective(img: Image.Image, M: np.ndarray, size: tuple[int, int]) -> Image.Image:
  """img resampled through M, which takes output px to img px."""
  return img.transform(size, Image.PERSPECTIVE, tuple((M / M[2, 2]).ravel()[:8]), Image.BILINEAR)


def bare(img: Image.Image, w: int, h: int, resample=Image.BILINEAR) -> Image.Image:
  """The frame scaled into w x h, letterboxed."""
  sc = min(w / img.width, h / img.height)
  size = (round(img.width * sc), round(img.height * sc))
  if size == (w, h):
    return img.resize(size, resample)
  out = Image.new(img.mode, (w, h), BG)
  out.paste(img.resize(size, resample), ((w - size[0]) // 2, (h - size[1]) // 2))
  return out


def seam_view(grey: Image.Image, narrow_only: Image.Image, wide_only: Image.Image, inset: np.ndarray, edge: np.ndarray) -> Image.Image:
  """Inside the inset, the stage's narrow-only and wide-only renders' fine detail tinted yellow and blue over their shared
  coarse brightness: grey where they line up, color fringes where they don't. Outside it, the frame in grey. All at the
  panel's size."""
  n, w = (np.asarray(im, np.float32) for im in (narrow_only, wide_only))
  bn, bw = (np.asarray(im.filter(ImageFilter.BoxBlur(SEAM_BLUR)), np.float32) for im in (narrow_only, wide_only))
  nh, wh = n + (bw - bn) / 2, w + (bn - bw) / 2  # each one's detail over the average of both's blur
  out = np.repeat(np.asarray(grey, np.float32)[..., None], 3, -1)
  out[inset] = np.stack([nh[inset], nh[inset], wh[inset]], -1)
  out[edge] = out[edge] * 0.3 + 255 * 0.7
  return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def inset_masks(alpha: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
  """Where the comma 4 narrow frame takes any 3X narrow (the stage's blend weights > 0) at a panel's size, and a 2 px line
  along its edge."""
  inset = np.asarray(Image.fromarray(alpha).resize(size, Image.NEAREST)) > 0
  edge = inset & ~(np.roll(inset, 1, 0) & np.roll(inset, -1, 0) & np.roll(inset, 1, 1) & np.roll(inset, -1, 1))
  return inset, edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)


def outline_layer(items: list, native_wh: tuple[int, int], size: tuple[int, int]) -> Image.Image:
  """The outlines as one RGBA layer at the panel's size, drawn at the frame's own size and scaled down for smooth edges.
  Fills, bands and strokes are layers of their own: on one layer a later shape replaces an earlier one's pixels."""
  fills, bands, strokes = (Image.new('RGBA', native_wh, (0, 0, 0, 0)) for _ in range(3))
  df, db, ds = ImageDraw.Draw(fills), ImageDraw.Draw(bands), ImageDraw.Draw(strokes)
  for it in items:
    if not it['visible'] or len(it['points']) < 3:
      continue
    col = KIND_COLORS[it['kind']]
    pts = [tuple(p) for p in it['points']]
    if it.get('inner'):
      inner = [tuple(p) for p in it['inner']]
      for i in range(len(pts)):
        j = (i + 1) % len(pts)
        db.polygon([pts[i], pts[j], inner[j], inner[i]], fill=col + (90,))
    else:
      if it['kind'] != 'c4_frame':  # frame edges are reference lines: no fill to tint the picture
        df.polygon(pts, fill=col + (32,))
      ds.line(pts + pts[:1], fill=(0, 0, 0, 150), width=6)  # a dark edge keeps every color readable over sky and road alike
      ds.line(pts + pts[:1], fill=col + (255,), width=3)
  out = Image.new('RGBA', native_wh, (0, 0, 0, 0))
  for layer in (fills, bands, strokes):
    out = Image.alpha_composite(out, layer)
  return bare(out.convert('RGBa'), *size, Image.LANCZOS).convert('RGBA')


def fade(layer: Image.Image, opacity: float) -> Image.Image:
  """An RGBA layer at a fraction of its strength."""
  if opacity >= 1:
    return layer
  out = layer.copy()
  out.putalpha(layer.getchannel('A').point([round(a * opacity) for a in range(256)]))
  return out


@lru_cache(maxsize=32)
def header(w: int, title: str, items: tuple = (), sub: str | None = None, right: str | None = None, alert: str = '') -> Image.Image:
  """A panel's header bar: its name, then a swatch and name per thing drawn on it; `right` at the far end, and an alert just
  before it (shortened to the room there is)."""
  p = Image.new('RGB', (w, HB), PANEL)
  d = ImageDraw.Draw(p)
  f, fi = font(17, 'SemiBold'), font(15)
  x, y = 10, (HB - 17) // 2 - 1
  d.text((x, y), title, font=f, fill=INK)
  x += d.textlength(title, font=f) + 16
  if sub:
    d.text((x, y + 1), sub, font=fi, fill=MUTED)
    x += d.textlength(sub, font=fi) + 16
  sw = 13
  for name, col in items:
    sy = (HB - sw) // 2
    d.rectangle([x, sy, x + sw, sy + sw], fill=tuple(int(c * 0.45) for c in col), outline=col, width=2)
    d.text((x + sw + 6, y + 1), name, font=fi, fill=INK)
    x += sw + 6 + d.textlength(name, font=fi) + 16
  end = w - 10
  if right:
    end -= d.textlength(right, font=mono(17))
    d.text((end, y), right, font=mono(17), fill=INK)
    end -= 16
  if alert:
    room = end - x
    while alert and d.textlength(alert, font=fi) > room:
      alert = alert[:-2] + '…'
    if len(alert) > 1:
      d.text((end - d.textlength(alert, font=fi), y + 1), alert, font=fi, fill=ALERT)
  return p


def legend(items: list) -> tuple:
  return tuple((it['name'], KIND_COLORS[it['kind']]) for it in items if it['visible'])


# where each panel's picture sits in the whole picture (x, y, w, h); its header bar is the HB px above it
IH = (PH - 2) // 2
_Y2 = HB + PH + GAP
PANELS = {'c4_narrow': (0, HB, PW, PH), 'c4_wide': (PW + GAP, HB, PW, PH), 'seam': (2 * (PW + GAP), HB, PW, PH),
          'device_narrow': (0, _Y2 + HB, XW, PH), 'device_wide': (XW + GAP, _Y2 + HB, XW, PH),
          'input_narrow': (2 * (XW + GAP), _Y2 + HB, PH, IH), 'input_wide': (2 * (XW + GAP), _Y2 + HB + IH + 2, PH, IH),
          'stats': (2 * (XW + GAP) + PH + GAP, _Y2 + HB, STW, PH)}
TITLES = {'c4_narrow': 'C4 NARROW', 'c4_wide': 'C4 WIDE', 'seam': 'SEAM', 'device_narrow': '3X NARROW', 'device_wide': '3X WIDE'}


def compose(v: dict) -> Image.Image:
  """The whole picture from the panels' pictures (RGB, at their panel sizes, the stats panel's too) and the headers'
  contents: v['legends'] per panel, v['speed'], v['alert']."""
  out = Image.new('RGB', (W, H), BG)
  for k, (x, y, _, _) in PANELS.items():
    out.paste(v[k], (x, y))
  for k, title in TITLES.items():
    x, y, w, _ = PANELS[k]
    out.paste(header(w, title, SEAM_KEY if k == 'seam' else v['legends'][k]), (x, y - HB))
  x, y, w, _ = PANELS['input_narrow']
  out.paste(header(w, 'MODEL INPUT', (), 'narrow above, wide below'), (x, y - HB))
  x, y, w, _ = PANELS['stats']
  out.paste(header(w, 'STATS', (), 'last 10 s', v['speed'], v['alert']), (x, y - HB))
  return out


def canvas_layer(panels: dict) -> Image.Image:
  """Panels' RGBA layers placed where their panels sit, transparent elsewhere."""
  out = Image.new('RGBA', (W, H), (0, 0, 0, 0))
  for k, img in panels.items():
    out.paste(img, PANELS[k][:2])
  return out
