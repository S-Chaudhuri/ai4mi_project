import gc
import json
import tempfile
from functools import partial
from pathlib import Path

import autoroot
import numpy as np

from src.train import gt_transform_3d, img_transform_3d
from src.utils.dataset import NpyBoxDataset


def write_fake(root: Path, channels: int, shape=(40, 50, 60)):
    """Two patients in preprocess_3d.py's layout."""
    rng = np.random.default_rng(0)
    for pid in ["Patient_01", "Patient_02"]:
        pdir = root / "train" / pid
        pdir.mkdir(parents=True)
        gt = np.zeros(shape, np.uint8)
        for k in range(1, 5):
            gt[5 * k : 5 * k + 4, 10:20, 10:20] = k
        img = rng.integers(0, 256, (channels, *shape), dtype=np.uint8)
        np.save(pdir / "img.npy", img)
        np.save(pdir / "gt.npy", gt)
        (pdir / "meta.json").write_text(json.dumps({"spacing_zyx": [2.5, 1.5, 1.5]}))


for channels in [1, 3]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_fake(root, channels)
        kw = dict(img_transform=img_transform_3d, gt_transform=partial(gt_transform_3d, 5))

        ds = NpyBoxDataset("train", root, sub_box_size=(32, 32, 32), fg_prob=1.0, **kw)
        assert ds.in_channels == channels and ds.spacing == (2.5, 1.5, 1.5)
        s = ds[0]
        assert s["images"].shape == (channels, 32, 32, 32), s["images"].shape
        assert s["gts"].shape == (5, 32, 32, 32), s["gts"].shape
        assert 0 <= s["images"].min() and s["images"].max() <= 1
        assert (s["gts"].sum(0) == 1).all()  # one-hot

        # Box larger than the volume along W: padded, padding is background
        ds_pad = NpyBoxDataset("train", root, sub_box_size=(32, 32, 64), **kw)
        s = ds_pad[1]
        assert s["images"].shape == (channels, 32, 32, 64), s["images"].shape
        assert (s["gts"][0, :, :, 60:] == 1).all()

        # No box: the whole volume
        ds_full = NpyBoxDataset("train", root, **kw)
        full = ds_full[0]
        assert full["images"].shape == (channels, 40, 50, 60), full["images"].shape

        # The volumes are memory-mapped: close them before the temp dir is
        # removed, Windows can't delete files that are still mapped
        del ds, ds_pad, ds_full, s, full
        gc.collect()

print("OK")
