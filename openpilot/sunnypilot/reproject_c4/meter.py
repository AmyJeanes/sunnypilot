"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The seam match: what the stage applies to the 3X wide surround so it matches the narrow inset, in the layout the
compiled stage reads, from the exposure model of the two camera states."""

import numpy as np


KNOTS = (16.0, 50.0, 100.0, 150.0, 200.0, 237.0)  # the last at camerad's top: clipped wide pixels get their own value
# [curve at KNOTS, multiplicative field, additive field, saturation at KNOTS, U/V offset, U/V x gradient, U/V y gradient,
#  field bounds x_lo, x_hi, y_lo, y_hi]: the stage's parameters (compile_stage.py); IDENTITY leaves the wide as it is
IDENTITY = np.array([*KNOTS] + [0] * 10 + [1] * len(KNOTS) + [0] * 6 + [-1, 1, -1, 1], np.float32)
N_LUMA, N_CHROMA = len(KNOTS) + 10, len(KNOTS) + 6

# encoded-domain luma ratio narrow/wide vs the sensors' exposure ratio (gain*integLines), fitted in the seam ring on a
# day-to-night drive (1607 frame pairs, 4 % residual): the exponent is the ISP tone curve, so a plain constant can't fit
# both dusk and night; the ring (not the whole overlap) because the wide lens falls off towards the seam
EXPOSURE_GAIN_A, EXPOSURE_GAIN_P = 0.868, 0.708

def exposure_gain(narrow_exposure, wide_exposure, lo=0.25, hi=4.0):
  """Gain to apply to the 3X wide surround so it matches the narrow inset; exposures are gain*integLines from the camera states."""
  if not (narrow_exposure > 0 and wide_exposure > 0):
    return 1.0
  return float(np.clip(EXPOSURE_GAIN_A * (narrow_exposure / wide_exposure) ** EXPOSURE_GAIN_P, lo, hi))


def exposure_match(gain: float) -> np.ndarray:
  """The match from the exposure model alone, in the stage's layout: the wide's tone curve and saturation scaled by gain."""
  out = IDENTITY.copy()
  out[:N_LUMA] *= gain
  out[N_LUMA:N_LUMA + len(KNOTS)] *= gain
  return out
