"""Compile the comma 3X -> comma 4 reprojection stage: the comma 4 wide and narrow frames gathered from the 3X frames through the
tables (compile_tables.py), the narrow inset composited over the wide surround with the seam match applied to the surround."""
import argparse
import pickle
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.engine.realize import lower_and_compile
from examples.openpilot.helpers import allocate_inputs, benchmark
from examples.openpilot.compile_warp import NV12Frame, parse_frame


def nv12_body(f: NV12Frame) -> int: return f.stride * (f.y_height + f.uv_height)  # the frame without the buffer's alignment tail


def pwl(v: Tensor, knots: list[float], g: Tensor, k0: int) -> Tensor:
  """Piecewise linear through g[k0 + j] at knots[j], flat beyond the ends: computed, not looked up (a 256-entry lookup on the
  Adreno doubles the stage)."""
  out = g[k0]
  for j in range(1, len(knots)):
    out = out + (g[k0 + j] - g[k0 + j - 1]) * ((v - knots[j - 1]) * (1 / (knots[j] - knots[j - 1]))).clip(0, 1)
  return out


def make_stage(src: NV12Frame, dst: NV12Frame, knots: list[float], uv_fill: int, idx_mask: int, alpha_shift: int, invalid_bit: int):
  body, uv_offset, src_uv = nv12_body(dst), dst.stride * dst.y_height, src.stride * src.y_height
  k = len(knots)
  mul, add, sat, off, bounds = k, k + 5, k + 10, 2 * k + 10, 2 * k + 16  # the match vector's layout (SeamMeter)

  def composite(w, n, pw, valid_fill, a):
    """The narrow inset over the matched surround w; where the wide has no pixel, the fill."""
    w = (pw < invalid_bit).where(w.clip(0, 255), valid_fill)
    return (a * n + (1 - a) * w).round().cast("uint8")

  def stage(wide, narrow, gains, pw_wide, pw_narrow, pn):
    i = Tensor.arange(body, dtype="int32")
    out_w = (pw_wide < invalid_bit).where(wide[pw_wide & idx_mask], (i >= uv_offset).cast("uint8") * uv_fill)
    # the composite's luma and chroma planes as two expressions, each written into its part of the output: under one where
    # every byte computed both planes' match, and the chroma's extra read of the luma under it would run on every byte
    planes = []
    for lo, hi in ((0, uv_offset), (uv_offset, body)):
      j = Tensor.arange(lo, hi, dtype="int32")
      pw, pnp = pw_narrow[lo:hi], pn[lo:hi]
      src_i = pw & idx_mask
      w = wide[src_i].float()
      # the byte's position from its index: computed, not read (a table read costs ~2 ms on the Adreno); the shading fields
      # are fitted on the seam ring and held flat beyond its extent
      row = j // dst.stride if lo == 0 else (j // dst.stride - dst.y_height) * 2
      x = ((j % dst.stride).float() * (2.0 / dst.width) - 1).clip(gains[bounds], gains[bounds + 1])
      y = (row.float() * (2.0 / dst.height) - 1).clip(gains[bounds + 2], gains[bounds + 3])
      if lo == 0:
        # luma: the tone curve plus multiplicative (lens shading) and additive (flare) quadratic fields
        q = (x, y, x * y, x * x, y * y)
        field_mul = sum((gains[mul + t] * q[t] for t in range(1, 5)), gains[mul] * q[0])
        field_add = sum((gains[add + t] * q[t] for t in range(1, 5)), gains[add] * q[0])
        matched, fill = pwl(w, knots, gains, 0) + w * field_mul + field_add, 0.0
      else:
        # chroma: a saturation curve of the wide luma under the sample (the top-left of its 2x2) and a U/V offset with a
        # gradient. U and V bytes alternate: picked by parity (a reshape/expand broadcast costs 0.8 ms on the 3X). Entries the
        # wide doesn't cover still read in range, and are filled below.
        rel = (src_i >= src_uv).where(src_i - src_uv, 0)
        yc = wide[(rel // src.stride) * (2 * src.stride) + ((rel % src.stride) & ~1)].float()
        u = (j & 1) == 0
        offset, grad_x, grad_y = (u.where(gains[t], gains[t + 1]) for t in (off, off + 2, off + 4))
        matched = (w - uv_fill) * pwl(yc, knots, gains, sat) + uv_fill + offset + grad_x * x + grad_y * y
        fill = float(uv_fill)
      a = ((pw >> alpha_shift) & 0xff).float() * (1 / 255)
      # the mask looks redundant but bounds the index; without it every gather is guarded (0.8 ms on the 3X)
      planes.append(composite(matched, narrow[pnp & idx_mask].float(), pw, fill, a))
    return out_w, planes

  return stage


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--src', type=parse_frame, required=True, help='3X camera: width,height,stride,y_height,uv_height,buffer_size')
  parser.add_argument('--dst', type=parse_frame, required=True, help='comma 4 camera: width,height,stride,y_height,uv_height,buffer_size')
  parser.add_argument('--knots', type=lambda v: [float(x) for x in v.split(',')], required=True, help="luma knots of the seam match's curves")
  parser.add_argument('--uv-fill', type=int, required=True)
  parser.add_argument('--idx-mask', type=int, required=True)
  parser.add_argument('--alpha-shift', type=int, required=True)
  parser.add_argument('--invalid-bit', type=int, required=True)
  parser.add_argument('--gains-device', help='device holding the seam match; defaults to DEV')
  parser.add_argument('--output', required=True)
  parser.add_argument('--benchmark-runs', type=int, default=20)
  args = parser.parse_args()

  stage = make_stage(args.src, args.dst, args.knots, args.uv_fill, args.idx_mask, args.alpha_shift, args.invalid_bit)
  identity = np.array(args.knots + [0] * 10 + [1] * len(args.knots) + [0] * 6 + [-1, 1, -1, 1], np.float32)
  n_gains = len(identity)
  specs = {'wide': ((args.src.size,), np.dtype(np.uint8).str, Device.DEFAULT),
           'narrow': ((args.src.size,), np.dtype(np.uint8).str, Device.DEFAULT),
           'gains': ((n_gains,), np.dtype(np.float32).str, args.gains_device or Device.DEFAULT)}
  table_specs = {k: ((nv12_body(args.dst),), np.dtype(np.int32).str, Device.DEFAULT) for k in ('pw_wide', 'pw_narrow', 'pn')}
  output_specs = {k: ((nv12_body(args.dst),), np.dtype(np.uint8).str, Device.DEFAULT) for k in ('wide', 'narrow')}

  def make_inputs(seed):
    rng = np.random.default_rng(seed)
    def initialize(arrays):
      for k in ('wide', 'narrow'):
        arrays[k][:] = rng.integers(0, 256, arrays[k].shape, dtype=np.uint8)
      arrays['gains'][:] = identity + rng.normal(0, 0.05, n_gains) * np.r_[[20] * len(args.knots), np.ones(n_gains - len(args.knots))]
    def initialize_tables(arrays):
      n = args.src.size
      for k in arrays:
        idx = rng.integers(0, n, arrays[k].shape)
        arrays[k][:] = idx if k == 'pn' else idx + (rng.integers(0, 256, idx.shape) << args.alpha_shift) + (rng.random(idx.shape) < 0.05) * args.invalid_bit
    return allocate_inputs(specs, initialize) | allocate_inputs(table_specs, initialize_tables) | {'output_buffers': allocate_inputs(output_specs)}

  @TinyJit(prune=True)
  def run(output_buffers, wide, narrow, gains, pw_wide, pw_narrow, pn):
    out_w, (luma, chroma) = stage(wide, narrow, gains.to(Device.DEFAULT), pw_wide, pw_narrow, pn)
    # each plane straight into its part of the buffer: joined first (cat), they cost a copy and a kernel more
    out_n, uv_offset = output_buffers['narrow'], args.dst.stride * args.dst.y_height
    Tensor.realize(output_buffers['wide'].assign(out_w), out_n[:uv_offset].assign(luma), out_n[uv_offset:].assign(chroma))

  expected = benchmark(run, **(inputs := make_inputs(42)))
  for _ in range(2):  # capture
    np.testing.assert_equal(benchmark(run, **inputs), expected)
  with np.testing.assert_raises(AssertionError):  # the outputs follow the inputs
    np.testing.assert_equal(benchmark(run, **make_inputs(43)), expected)
  for i in range(args.benchmark_runs):
    np.testing.assert_equal(benchmark(run, cb=lambda t, i=i: print(f"  [{i}/{args.benchmark_runs}] {t*1e3:.2f} ms"), **inputs), expected)

  artifact = {'run': run, 'input_specs': specs | table_specs, 'output_specs': output_specs}
  with open(args.output, 'wb') as f:
    pickle.dump(artifact, f)

  with open(args.output, 'rb') as f:
    loaded = pickle.load(f)
  loaded['run'].captured._linear = lower_and_compile(loaded['run'].captured._linear)
  np.testing.assert_equal(benchmark(loaded['run'], **make_inputs(42)), expected)
