"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The gather tables that map every comma 4 output byte to a 3X source byte. The device builds them on the GPU (compile_tables.py);
build_tables is the same in numpy, the reference the compiled program is checked against."""
import numpy as np

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from .geometry import C4_NARROW, FEATHER_PX, _nv12_index, sample_coords


IDX_BITS, ALPHA_SHIFT, INVALID_BIT = 0x3fffff, 22, 1 << 30  # one int32 per output byte: wide index | alpha << 22 | invalid

def build_tables(src_wh, dst_wh, calib):
  """Flat gather tables over the destination NV12 body (comma 4 layout). Byte b of the wide output = src[pw[b]] (the fill
  value where invalid); of the narrow composite, alpha[b]*narrow[pn[b]] + (1-alpha[b])*gain*wide[pw[b]]. The kernel is
  memory-bound, so the wide index, the blend weight and the validity share one int32 (`pw`)."""
  sw, sh = src_wh
  dw, dh = dst_wh
  s_stride, s_yh, s_uvh, _ = get_nv12_info(sw, sh)
  s_uv = s_stride * s_yh
  assert s_stride * (s_yh + s_uvh) <= IDX_BITS + 1
  d_stride, d_yh, d_uvh, _ = get_nv12_info(dw, dh)
  d_uv = d_stride * d_yh
  n_body = d_stride * (d_yh + d_uvh)
  sc = calib["narrow"]["f"] / C4_NARROW["f"]  # 3X narrow px per comma 4 narrow px
  out = {}
  for cam in ("wide", "narrow"):
    idx_w = np.zeros(n_body, np.int64)
    idx_n = np.zeros(n_body, np.int64)
    val_w = np.zeros(n_body, bool)
    val_n = np.zeros(n_body, bool)
    dist = np.zeros(n_body, np.float32)
    for chroma in (False, True):
      s = 0.5 if chroma else 1.0
      mw, mn = sample_coords(cam, dw // 2 if chroma else dw, dh // 2 if chroma else dh, s, calib)
      iw, vw = _nv12_index(mw, sw, sh, s_stride, s_uv, chroma)
      inn, vn = _nv12_index(mn, sw, sh, s_stride, s_uv, chroma)
      # distance to the narrow frame edge, in destination px: the composite feathers over FEATHER_PX of it
      d = np.minimum(np.minimum(mn[..., 0], sw * s - mn[..., 0]), np.minimum(mn[..., 1], sh * s - mn[..., 1])) / (s * sc)
      if chroma:
        rows = np.arange(dh // 2)[:, None]
        cols = np.arange(dw // 2)[None, :]
        for plane in (0, 1):
          flat = d_uv + rows * d_stride + 2 * cols + plane
          idx_w[flat] = iw + plane
          idx_n[flat] = inn + plane
          val_w[flat] = vw
          val_n[flat] = vn
          dist[flat] = d
      else:
        flat = np.arange(dh)[:, None] * d_stride + np.arange(dw)[None, :]
        idx_w[flat] = iw
        idx_n[flat] = inn
        val_w[flat] = vw
        val_n[flat] = vn
        dist[flat] = d
    alpha = np.round(np.clip(dist / FEATHER_PX, 0, 1) * val_n * 255).astype(np.int64)
    out[cam] = {"pw": (idx_w | (alpha << ALPHA_SHIFT) | (~val_w * INVALID_BIT)).astype(np.int32)}
  out["narrow"]["pn"] = idx_n.astype(np.int32)
  return out
