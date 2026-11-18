"""Compile the comma 3X -> comma 4 reprojection tables: for a narrow->wide rotation, which 3X NV12 byte each comma 4 output byte
samples, and the narrow/wide blend over the seam."""
import argparse
import json
import pickle
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.engine.realize import lower_and_compile
from examples.openpilot.helpers import allocate_inputs, benchmark
from examples.openpilot.compile_warp import NV12Frame, parse_frame


def nv12_body(f: NV12Frame) -> int: return f.stride * (f.y_height + f.uv_height)  # the frame without the buffer's alignment tail


def theta_d(th, L):
  k, tc = L["k"], L["tc"]
  def p(t): return t * (1 + k[0] * t * t + k[1] * t**4 + k[2] * t**6)
  return (th <= tc).where(p(th), p(tc) + (1 + 3 * k[0] * tc**2 + 5 * k[1] * tc**4 + 7 * k[2] * tc**6) * (th - tc))

def atan2_pos(y, x):
  """atan2 for y >= 0, with atan's argument kept within 1: its polynomial loses precision near the image center otherwise."""
  small = y <= x.abs()
  a = small.where(y / x.abs().maximum(1e-12), x / y.maximum(1e-12)).atan()
  return small.where((x >= 0).where(a, np.pi - a), np.pi / 2 - a)

def unproject_fisheye(px, py, L):
  k = L["k"]
  dx, dy = px - L["cx"], py - L["cy"]
  r = (dx * dx + dy * dy).sqrt()
  th = td = r / L["f"]
  for _ in range(4):  # Newton on θd(θ) = r/f from θ = θd
    tm = th.minimum(L["tc"])
    th = th - (theta_d(th, L) - td) / (1 + 3 * k[0] * tm * tm + 5 * k[1] * tm**4 + 7 * k[2] * tm**6)
  s, rr = th.sin(), (r > 0).where(r, 1.0)
  return s * dx / rr, s * dy / rr, th.cos()

def unproject_pinhole_k1(px, py, L):
  xd, yd = (px - L["cx"]) / L["f"], (py - L["cy"]) / L["f"]
  rd = (xd * xd + yd * yd).sqrt()
  ru = rd
  for _ in range(5):  # Newton on ru(1 + k1 ru²) = rd from ru = rd
    ru = ru - (ru * (1 + L["k1"] * ru * ru) - rd) / (1 + 3 * L["k1"] * ru * ru)
  s = (rd > 0).where(ru / (rd > 0).where(rd, 1.0), 1.0)
  n = ((xd * s) ** 2 + (yd * s) ** 2 + 1).sqrt()
  return xd * s / n, yd * s / n, 1 / n

def project_fisheye(x, y, z, L):
  rho = (x * x + y * y).sqrt()
  r = L["f"] * theta_d(atan2_pos(rho, z), L) / (rho > 0).where(rho, 1.0)
  return L["cx"] + r * x, L["cy"] + r * y

def project_pinhole_k1(x, y, z, L):
  zs = (z > 0).where(z, 1.0)
  s = 1 + L["k1"] * ((x / zs) ** 2 + (y / zs) ** 2)
  return L["cx"] + L["f"] * x / zs * s, L["cy"] + L["f"] * y / zs * s

def make_tables(src: NV12Frame, dst: NV12Frame, lenses: dict, zmin: float, feather: float, alpha_shift: int, invalid_bit: int):
  s_uv, d_uv, body = src.stride * src.y_height, dst.stride * dst.y_height, nv12_body(dst)
  sc = lenses["x3_narrow"]["f"] / lenses["c4_narrow"]["f"]  # 3X narrow px per comma 4 narrow px

  def nv12_index(mx, my, chroma, plane):
    """Nearest-neighbor source byte and whether the sample is inside the 3X frame; the index stays in the frame either way."""
    mx, my = (mx != mx).where(-1.0, mx), (my != my).where(-1.0, my)  # grazing rays project to nan
    xi, yi = (mx - 0.5).round(), (my - 0.5).round()
    w, h = chroma.where(src.width // 2, src.width).float(), chroma.where(src.height // 2, src.height).float()
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xc, yc = xi.maximum(0).minimum(w - 1).cast("int32"), yi.maximum(0).minimum(h - 1).cast("int32")
    return chroma.where(s_uv + yc * src.stride + 2 * xc + plane, yc * src.stride + xc), valid

  def tables(R, out_cam):
    i = Tensor.arange(body, dtype="int32")
    chroma = i >= d_uv
    j = chroma.where(i - d_uv, i)
    row, col = j // dst.stride, j % dst.stride
    pxcol, plane = chroma.where(col // 2, col), chroma.where(col % 2, 0)
    scale = chroma.where(0.5, 1.0)
    inside = (pxcol < chroma.where(dst.width // 2, dst.width)) & (row < chroma.where(dst.height // 2, dst.height))  # not padding
    px, py = (pxcol.float() + 0.5) / scale, (row.float() + 0.5) / scale
    x, y, z = (unproject_fisheye if out_cam == "wide" else unproject_pinhole_k1)(px, py, lenses[f"c4_{out_cam}"])
    xw, yw, zw = (R[r, 0] * x + R[r, 1] * y + R[r, 2] * z for r in range(3))
    mwx, mwy = project_fisheye(xw, yw, zw, lenses["x3_wide"])
    idx_w, val_w = nv12_index(mwx * scale, mwy * scale, chroma, plane)
    val_w = val_w & (zw >= zmin) & inside
    mnx, mny = project_pinhole_k1(x, y, z, lenses["x3_narrow"])
    mnx, mny = (z > 0).where(mnx * scale, -1.0), (z > 0).where(mny * scale, -1.0)
    idx_n, val_n = nv12_index(mnx, mny, chroma, plane)
    val_n = val_n & inside
    # distance to the narrow frame edge in comma 4 px: the composite feathers over `feather` px of it
    d = mnx.minimum(src.width * scale - mnx).minimum(mny).minimum(src.height * scale - mny) / (scale * sc)
    alpha = ((d / feather).maximum(0).minimum(1) * val_n.float() * 255).round().cast("int32")
    return idx_w + alpha * (1 << alpha_shift) + (1 - val_w.cast("int32")) * invalid_bit, idx_n

  return tables


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--src', type=parse_frame, required=True, help='3X camera: width,height,stride,y_height,uv_height,buffer_size')
  parser.add_argument('--dst', type=parse_frame, required=True, help='comma 4 camera: width,height,stride,y_height,uv_height,buffer_size')
  parser.add_argument('--lenses', type=json.loads, required=True, help='c4_wide, c4_narrow, x3_wide, x3_narrow lens models')
  parser.add_argument('--zmin', type=float, required=True, help="cosine of the widest angle the 3X wide is sampled at")
  parser.add_argument('--feather', type=float, required=True, help='seam width in comma 4 narrow px')
  parser.add_argument('--alpha-shift', type=int, required=True)
  parser.add_argument('--invalid-bit', type=int, required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--benchmark-runs', type=int, default=5)
  args = parser.parse_args()

  tables = make_tables(args.src, args.dst, args.lenses, args.zmin, args.feather, args.alpha_shift, args.invalid_bit)
  specs = {'R': ((3, 3), np.dtype(np.float32).str, Device.DEFAULT)}
  output_specs = {k: ((nv12_body(args.dst),), np.dtype(np.int32).str, Device.DEFAULT) for k in ('pw_wide', 'pw_narrow', 'pn')}

  def make_inputs(seed):
    rng = np.random.default_rng(seed)
    def initialize(arrays):
      v = rng.normal(0, 0.02, 3)  # a small rotation, as a unit's narrow->wide rotation is
      K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
      arrays['R'][:] = np.eye(3) + K + K @ K / 2
    return allocate_inputs(specs, initialize) | {'output_buffers': allocate_inputs(output_specs)}

  @TinyJit(prune=True)
  def run(output_buffers, R):
    (pw_w, _), (pw_n, pn) = tables(R, "wide"), tables(R, "narrow")
    Tensor.realize(output_buffers['pw_wide'].assign(pw_w), output_buffers['pw_narrow'].assign(pw_n), output_buffers['pn'].assign(pn))

  expected = benchmark(run, **(inputs := make_inputs(42)))
  for _ in range(2):  # capture
    np.testing.assert_equal(benchmark(run, **inputs), expected)
  with np.testing.assert_raises(AssertionError):  # the outputs follow the inputs
    np.testing.assert_equal(benchmark(run, **make_inputs(43)), expected)
  for i in range(args.benchmark_runs):
    np.testing.assert_equal(benchmark(run, cb=lambda t, i=i: print(f"  [{i}/{args.benchmark_runs}] {t*1e3:.2f} ms"), **inputs), expected)

  artifact = {'run': run, 'input_specs': specs, 'output_specs': output_specs}
  with open(args.output, 'wb') as f:
    pickle.dump(artifact, f)

  with open(args.output, 'rb') as f:
    loaded = pickle.load(f)
  loaded['run'].captured._linear = lower_and_compile(loaded['run'].captured._linear)
  np.testing.assert_equal(benchmark(loaded['run'], **make_inputs(42)), expected)
