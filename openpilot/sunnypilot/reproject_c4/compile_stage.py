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


def make_stage(dst: NV12Frame, bands: list[float], uv_fill: int, idx_mask: int, alpha_shift: int, invalid_bit: int):
  body, uv_offset = nv12_body(dst), dst.stride * dst.y_height
  centers = [0.5 * (lo + hi) for lo, hi in zip(bands[:-1], bands[1:], strict=True)]

  def stage(wide, narrow, gains, pw_wide, pw_narrow, pn):
    # the byte's position from its index: computed, not read (a table read costs ~2 ms on the Adreno)
    i = Tensor.arange(body, dtype="int32")
    chroma = i >= uv_offset
    out_w = (pw_wide < invalid_bit).where(wide[pw_wide & idx_mask], chroma.cast("uint8") * uv_fill)
    # match the wide surround to the narrow: luma times a gain per brightness band (the two ISPs' tone curves; piecewise
    # linear between the band centers, flat beyond) times a gain gradient over the frame (lens shading); chroma gain about
    # the midpoint plus a U/V offset (independent white balance), the offset a parity broadcast: U and V bytes alternate
    off = gains[1:3].reshape(1, 2).expand(body // 2, 2).reshape(body)
    w = wide[pw_narrow & idx_mask].float()
    tone = gains[5]
    for b in range(1, len(centers)):
      tone = tone + (gains[5 + b] - gains[4 + b]) * ((w - centers[b - 1]) * (1 / (centers[b] - centers[b - 1]))).clip(0, 1)
    x = (i % dst.stride).float() * (2.0 / dst.width) - 1
    y = (i // dst.stride).float() * (2.0 / dst.height) - 1
    w = chroma.where((w - uv_fill) * gains[0] + uv_fill + off, w * tone * (1 + gains[3] * x + gains[4] * y)).clip(0, 255)
    w = (pw_narrow < invalid_bit).where(w, chroma.cast("float32") * uv_fill)
    a = ((pw_narrow >> alpha_shift) & 0xff).float() * (1 / 255)
    # the mask looks redundant but bounds the index; without it every gather is guarded (0.8 ms on the 3X)
    out_n = (a * narrow[pn & idx_mask].float() + (1 - a) * w).round().cast("uint8")
    return out_w, out_n

  return stage


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--src', type=parse_frame, required=True, help='3X camera: width,height,stride,y_height,uv_height,buffer_size')
  parser.add_argument('--dst', type=parse_frame, required=True, help='comma 4 camera: width,height,stride,y_height,uv_height,buffer_size')
  parser.add_argument('--bands', type=lambda v: [float(x) for x in v.split(',')], required=True, help='luma band edges of the tone match')
  parser.add_argument('--uv-fill', type=int, required=True)
  parser.add_argument('--idx-mask', type=int, required=True)
  parser.add_argument('--alpha-shift', type=int, required=True)
  parser.add_argument('--invalid-bit', type=int, required=True)
  parser.add_argument('--gains-device', help='device holding the seam match; defaults to DEV')
  parser.add_argument('--output', required=True)
  parser.add_argument('--benchmark-runs', type=int, default=20)
  args = parser.parse_args()

  stage = make_stage(args.dst, args.bands, args.uv_fill, args.idx_mask, args.alpha_shift, args.invalid_bit)
  n_gains = 5 + len(args.bands) - 1
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
      arrays['gains'][:] = np.array([1, 0, 0, 0, 0] + [1] * (n_gains - 5)) + rng.normal(0, 0.05, n_gains)
    def initialize_tables(arrays):
      n = args.src.size
      for k in arrays:
        idx = rng.integers(0, n, arrays[k].shape)
        arrays[k][:] = idx if k == 'pn' else idx + (rng.integers(0, 256, idx.shape) << args.alpha_shift) + (rng.random(idx.shape) < 0.05) * args.invalid_bit
    return allocate_inputs(specs, initialize) | allocate_inputs(table_specs, initialize_tables) | {'output_buffers': allocate_inputs(output_specs)}

  @TinyJit(prune=True)
  def run(output_buffers, wide, narrow, gains, pw_wide, pw_narrow, pn):
    out_w, out_n = stage(wide, narrow, gains.to(Device.DEFAULT), pw_wide, pw_narrow, pn)
    Tensor.realize(output_buffers['wide'].assign(out_w), output_buffers['narrow'].assign(out_n))

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
