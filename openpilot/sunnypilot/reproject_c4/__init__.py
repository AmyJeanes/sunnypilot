"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Reproject the comma 3X cameras into comma 4 camera geometry, on the GPU, in front of an untouched comma 4 model.

Every comma 4 output pixel is traced through the comma 4 lens to a ray, rotated onto the 3X device axis and looked up in
the 3X wide (fisheye) or, for the narrow output, composited from the 3X narrow inset and the 3X wide surround. None of it
depends on the calibration, only on the unit's narrow->wide rotation, so lookup tables are built once per rotation and the
per-frame work is one gather per output byte (two plus a blend for the composite) - the same cost as the model's own crop
warp.

geometry: lenses, rotations, sample coordinates. tables: the table layout and its numpy reference."""
from .geometry import *  # noqa: F403
from .tables import *  # noqa: F403
