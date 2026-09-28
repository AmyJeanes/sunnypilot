"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The debug clip's stats panel: the logged timings over the last 10 s, the applied rotation and where the narrow model
input comes from."""
import numpy as np
from PIL import Image, ImageDraw

from openpilot.sunnypilot.tools.reproject_c4.view import INK, MUTED, NARROW_CAM, PANEL, WIDE_CAM, font, mono

WINDOW = 10.0  # s the stats look back
TRACE = (200, 206, 214)


SPARK_BIN = 0.25  # s of the route's time per column of the trace's band
STAT_ROWS = (('camera frame → modelV2', 'e2e'), ('driving model', 'model'), ('driver monitoring', 'dm'), ('ui frame', 'ui'))


def sparkline(img: Image.Image, box: tuple, series: np.ndarray | None, color: tuple, rng: tuple | None, now: float) -> None:
  """A 10 s trace on a y scale fixed for the clip: the range and mean of each quarter second of route time, drawn 1.5 slices
  behind now so only whole slices show, and they slide as it scrolls instead of redrawing."""
  d = ImageDraw.Draw(img)
  x0, y0, x1, y1 = box
  d.rectangle(box, fill=(28, 33, 40))
  if series is None or not len(series) or rng is None:
    return
  lo, hi = rng
  step = next((st for st in (0.5, 1, 2, 5, 10, 20, 50) if (hi - lo) / st <= 3), 100)
  lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
  if hi - lo < step:
    hi = lo + step
  f = mono(11)
  grid = np.arange(lo, hi + step / 2, step)
  gx0 = round(x0 + max(d.textlength(f'{g:g}', font=f) for g in grid) + 6)
  right = now - 1.5 * SPARK_BIN  # route time at the right edge
  def gy(val):
    return y1 - 4 - (np.clip(val, lo, hi) - lo) / (hi - lo) * (y1 - y0 - 8)
  def gx(t):  # route time -> x in the trace's own image
    return (t - (right - WINDOW)) / WINDOW * (x1 - gx0)
  for g in grid:
    d.line([(gx0, gy(g)), (x1, gy(g))], fill=(48, 55, 66), width=1)
    d.text((x0 + 3, gy(g) - 7), f'{g:g}', font=f, fill=MUTED)
  xx = gx0 + gx(right - WINDOW / 2)
  d.line([(xx, y1 - 5), (xx, y1)], fill=(90, 98, 110), width=1)
  bins = np.floor((series[:, 0] + now) / SPARK_BIN)
  keys, first = np.unique(bins, return_index=True)
  parts = np.split(series[:, 1], first[1:])
  whole = (keys + 1) * SPARK_BIN <= now
  keys, parts = keys[whole], [p for p, k in zip(parts, whole, strict=True) if k]
  if len(keys) < 2:
    return
  xs = gx((keys + 0.5) * SPARK_BIN)
  top, bottom, mean = (gy(np.array([fn(p) for p in parts])) - y0 for fn in (np.max, np.min, np.mean))
  band = tuple(int(c * 0.45 + b * 0.55) for c, b in zip(color, (28, 33, 40), strict=True))
  trace = img.crop((gx0, y0, x1, y1))  # drawn on its own so what runs past either edge is cut off
  t = ImageDraw.Draw(trace)
  t.polygon(list(zip(xs.tolist(), top.tolist(), strict=True)) + list(zip(xs[::-1].tolist(), bottom[::-1].tolist(), strict=True)), fill=band)
  t.line(list(zip(xs.tolist(), mean.tolist(), strict=True)), fill=color, width=2)
  img.paste(trace, (gx0, y0))


def stats_panel(w: int, h: int, st: dict) -> Image.Image:
  """The logged timings over the last 10 s, the applied rotation and where the narrow model input comes from. Neutral: no
  pass/fail colors."""
  p = Image.new('RGB', (w, h), PANEL)
  d = ImageDraw.Draw(p)
  s = max(w / 900, 0.85)
  fk, fv = font(round(15 * s)), mono(round(26 * s))
  pad = round(22 * s)
  y = pad
  rh = round((h - y - pad - 215 * s) / len(STAT_ROWS))
  for label, k in STAT_ROWS:
    ser = st['series'].get(k)  # a second longer than the window: the trace's left edge runs past it
    has = ser is not None and len(ser)
    d.text((pad, y), label.upper(), font=fk, fill=MUTED)
    d.text((pad, y + round(20 * s)), f'{ser[-1, 1]:.1f} ms' if has else '—', font=fv, fill=INK)
    tx = pad + round(150 * s)
    if has:
      recent = ser[ser[:, 0] >= -WINDOW, 1]
      d.text((tx, y + round(20 * s)), f'p50 {np.percentile(recent, 50):.1f}', font=fk, fill=MUTED)
      d.text((tx, y + round(38 * s)), f'p99 {np.percentile(recent, 99):.1f}', font=fk, fill=MUTED)
    sparkline(p, (pad + round(245 * s), y + round(6 * s), w - pad, y + rh - round(14 * s)), ser, TRACE, st['ranges'].get(k), st['now'])
    y += rh
  p_, y_, r_ = st['rotation']
  stage = st['stage']
  line = st['status']
  if stage:  # the device's stage timing, when the route logged it and there's room
    more = f'  ·  stage {stage["p50_ms"]:.2f} ms p50, {stage["over_8ms"]} over 8 ms / min'
    line += more if d.textlength(line + more, font=fk) <= w - 2 * pad else ''
  d.text((pad, y), f'ROTATION  ·  {st["rotation_from"]}', font=fk, fill=MUTED)
  d.text((pad, y + round(20 * s)), f'pitch {p_:+.2f}°  yaw {y_:+.2f}°  roll {r_:+.2f}°', font=mono(round(21 * s)), fill=INK)
  d.text((pad, y + round(50 * s)), line, font=fk, fill=MUTED)
  if st['coverage'] is not None:
    y += round(80 * s)
    d.text((pad, y), 'NARROW MODEL INPUT FROM', font=fk, fill=MUTED)
    by0, by1 = y + round(22 * s), y + round(44 * s)
    parts = list(zip(st['coverage'], (NARROW_CAM, (150, 150, 150), WIDE_CAM), ('3X narrow', 'band', '3X wide'), strict=True))
    x, fs, sq = float(pad), font(round(17 * s)), round(13 * s)
    for frac, col, _ in parts:
      xe = x + frac * (w - 2 * pad)
      if xe - x >= 1:  # the colors as the outlines show them by default: half over a mid-grey frame
        d.rectangle([x, by0, xe, by1], fill=tuple(int(c * 0.5 + 56) for c in col))
      x = xe
    x = pad
    for frac, col, name in parts:
      lab = f'{name} {frac * 100:.0f}%'
      d.rectangle([x, by1 + round(12 * s), x + sq, by1 + round(12 * s) + sq], fill=tuple(int(c * 0.45) for c in col), outline=col, width=2)
      d.text((x + sq + round(7 * s), by1 + round(8 * s)), lab, font=fs, fill=INK)
      x += sq + round(7 * s) + d.textlength(lab, font=fs) + round(20 * s)
  footer, lh = st['footer'], fk.size + 5
  lines = ['']
  for word in footer.split():
    t = (lines[-1] + ' ' + word).strip()
    if d.textlength(t, font=fk) <= w - 2 * pad:
      lines[-1] = t
    else:
      lines.append(word)
  for i, line in enumerate(lines):
    d.text((pad, h - pad + 4 - (len(lines) - i) * lh), line, font=fk, fill=MUTED)
  return p
