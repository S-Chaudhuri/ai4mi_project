import tempfile
from pathlib import Path

import autoroot
import numpy as np
import SimpleITK as sitk

from src.postprocess import POSTPROCESSING, keep_largest_component, postprocess, postprocess_file

# --- LCC: stray heart / aorta blobs removed, esophagus pieces kept
pred = np.zeros((20, 30, 30), np.uint8)
pred[2:10, 2:12, 2:12] = 2  # heart
pred[15:17, 20:22, 20:22] = 2  # stray heart blob
pred[2:12, 15:18, 15:18] = 4  # aorta
pred[18, 28, 28] = 4  # single stray aorta voxel
pred[15:17, 2:4, 2:4] = 1  # esophagus piece 1
pred[2:4, 25:27, 25:27] = 1  # esophagus piece 2

out = keep_largest_component(pred)
assert (out[15:17, 20:22, 20:22] == 0).all() and (out[2:10, 2:12, 2:12] == 2).all()
assert out[18, 28, 28] == 0 and (out[2:12, 15:18, 15:18] == 4).all()
assert (out == 1).sum() == (pred == 1).sum()  # esophagus untouched
assert (pred[15:17, 20:22, 20:22] == 2).all()  # input not modified

# --- Connectivity: cubes touching only at a corner are 2 components with
# 6-connectivity (the smaller is removed), 1 component with 26-connectivity
corner = np.zeros((10, 10, 10), np.uint8)
corner[1:4, 1:4, 1:4] = 2
corner[4:6, 4:6, 4:6] = 2
assert (postprocess(corner, ["lcc"]) == 2).sum() == 27
assert (postprocess(corner, ["lcc26"]) == 2).sum() == 27 + 8

# --- Registry: no steps is a no-op, steps run in order, unknown names fail
assert np.array_equal(postprocess(pred, []), pred)
assert np.array_equal(postprocess(pred, ["lcc"]), out)
try:
    postprocess(pred, ["lcc", "not_a_method"])
    raise AssertionError("unknown step should raise")
except ValueError as e:
    assert "not_a_method" in str(e)
assert set(POSTPROCESSING) >= {"lcc", "lcc26"}

# --- On a NIfTI file: geometry kept, removed voxels reported per class
with tempfile.TemporaryDirectory() as tmp:
    img = sitk.GetImageFromArray(pred)
    img.SetSpacing((0.98, 0.98, 2.5))
    img.SetOrigin((-250.0, -200.0, 100.0))
    src, dest = Path(tmp) / "Patient_01.nii.gz", Path(tmp) / "pp" / "Patient_01.nii.gz"
    sitk.WriteImage(img, str(src))

    removed = postprocess_file(src, dest, ["lcc"])
    assert removed == {1: 0, 2: 8, 4: 1}, removed

    # Compared with the source as read from disk: NIfTI stores float32 geometry
    res, orig = sitk.ReadImage(str(dest)), sitk.ReadImage(str(src))
    assert res.GetSpacing() == orig.GetSpacing() and res.GetOrigin() == orig.GetOrigin()
    assert np.array_equal(sitk.GetArrayFromImage(res), out)

print("OK")
