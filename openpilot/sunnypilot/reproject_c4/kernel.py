"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The GPU stage as the build compiled it (compile_tables.py, compile_stage.py; built next to modeld's warps)."""

import ctypes
import pickle

import numpy as np
from tinygrad import Tensor
from tinygrad.engine.realize import lower_and_compile

from openpilot.selfdrive.modeld.helpers import MODELS_DIR
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from .geometry import C4_CAM, rotvec_to_matrix
from .meter import SeamMeter

TABLES = ('pw_wide', 'pw_narrow', 'pn')


def _page_aligned(a: np.ndarray) -> np.ndarray:
  """A copy of `a` starting on a page boundary, as tinygrad's own buffers do: the stage may load a host buffer as wide
  vectors, and from an unaligned one those read the wrong bytes (the 32-float seam match from a 16-byte-aligned array)."""
  buf = np.empty(a.nbytes + 4096, np.uint8)
  start = -buf.ctypes.data % 4096
  out = buf[start:start + a.nbytes].view(a.dtype).reshape(a.shape)
  out[:] = a
  return out


def _load(name: str, src_wh: tuple[int, int]):
  with open(MODELS_DIR / f'reproject_c4_{name}_{src_wh[0]}x{src_wh[1]}_tinygrad.pkl', 'rb') as f:
    run = pickle.load(f)['run']
  run.captured._linear = lower_and_compile(run.captured._linear)
  return run


class Reprojector:
  """The tables for a rotation (set_rotation) and the per-frame gathers through them, on `device`."""

  def __init__(self, src_wh: tuple[int, int], device: str):
    stride, y_height, uv_height, _ = get_nv12_info(*C4_CAM)
    self.body = stride * (y_height + uv_height)
    self.device = device
    self._tables, self._stage = _load('tables', src_wh), _load('stage', src_wh)
    # uploaded rather than filled on the GPU: a fill is a kernel compiled at runtime (~0.5 s each on the 3X)
    self.tables = {k: Tensor(np.zeros(self.body, np.int32), device=device).realize() for k in TABLES}
    self.rotation = Tensor(np.eye(3, dtype=np.float32), device=device).realize()
    # the seam match: on QCOM the stage reads it straight from host memory (a per-call upload costs a scheduled copy)
    self.params_np = _page_aligned(SeamMeter.IDENTITY)
    self.host_params = device.startswith('QCOM')
    if self.host_params:
      self.gains = Tensor.from_blob(self.params_np.ctypes.data, self.params_np.shape, dtype='float32', device=device)
    else:
      self.gains = Tensor(self.params_np, device=device).contiguous().realize()
    self.out = {k: Tensor(np.zeros(self.body, np.uint8), device=device).realize() for k in ('wide', 'narrow')}

  def set_rotation(self, rotvec) -> None:
    """Build the tables of this narrow->wide rotation (~40 ms on the 3X GPU). The GPU runs it before the next stage call, so a
    stage never sees half-built tables."""
    self.rotation.assign(Tensor(rotvec_to_matrix(rotvec).astype(np.float32), device=self.device)).realize()
    self._tables(output_buffers=self.tables, R=self.rotation)

  def bind(self, host_wide: np.ndarray, host_narrow: np.ndarray) -> None:
    """Write the outputs straight into these host arrays (each `body` bytes, e.g. modeld's packed frame slots) instead of
    device tensors: a GPU->host readback costs ~40 ms on the 3X, a direct write ~10 ms. Bind once, to arrays that live as
    long as this: on QCOM, arrays freed and reallocated at the same address read back stale or zero bytes."""
    assert host_wide.nbytes == self.body and host_narrow.nbytes == self.body and host_wide.dtype == host_narrow.dtype == np.uint8
    self.out = {k: Tensor.from_blob(a.ctypes.data, (self.body,), dtype='uint8', device=self.device)
                for k, a in (('wide', host_wide), ('narrow', host_narrow))}

  def __call__(self, wide: Tensor, narrow: Tensor, match=None) -> tuple[Tensor, Tensor]:
    """wide/narrow: flat uint8 3X NV12 tensors. match: the seam match vector from SeamMeter.update() (None = no match).
    Returns (c4_wide, c4_narrow): flat uint8 NV12 tensors of `body` bytes; with bind() they are views of the bound arrays."""
    self.params_np[:] = SeamMeter.IDENTITY if match is None else match
    if self.host_params:
      # tinygrad cleaned this memory from the CPU cache only when it mapped it: without a clean the GPU reads stale matches
      from tinygrad.runtime.ops_qcom import dcache_flush
      start = self.params_np.ctypes.data & ~63
      dcache_flush().fxn(ctypes.c_uint64(start), (self.params_np.ctypes.data + self.params_np.nbytes - start + 63) // 64)
    else:
      self.gains.assign(Tensor(self.params_np, device=self.device)).realize()
    self._stage(output_buffers=self.out, wide=wide, narrow=narrow, gains=self.gains, **self.tables)
    return self.out['wide'], self.out['narrow']
