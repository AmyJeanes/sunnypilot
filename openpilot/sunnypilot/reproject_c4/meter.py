"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The seam match: what the stage applies to the 3X wide surround so it matches the narrow inset, in the layout the
compiled stage reads."""

import numpy as np


KNOTS = (16.0, 50.0, 100.0, 150.0, 200.0, 237.0)  # the last at camerad's top: clipped wide pixels get their own value
# [curve at KNOTS, multiplicative field, additive field, saturation at KNOTS, U/V offset, U/V x gradient, U/V y gradient,
#  field bounds x_lo, x_hi, y_lo, y_hi]: the stage's parameters (compile_stage.py); IDENTITY leaves the wide as it is
IDENTITY = np.array([*KNOTS] + [0] * 10 + [1] * len(KNOTS) + [0] * 6 + [-1, 1, -1, 1], np.float32)
N_LUMA, N_CHROMA = len(KNOTS) + 10, len(KNOTS) + 6
