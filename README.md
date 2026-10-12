# SegTHOR organ segmentation

Course project for AI for medical imaging: segmenting the esophagus, heart,
trachea and aorta (plus background) on chest CT from the
[SegTHOR challenge](https://competitions.codalab.org/competitions/19854).

Two pipelines share one training entrypoint:

- **2D**: NIfTI volumes sliced to PNG, `ShallowNet` or `ENet`, one prediction
  per slice.
- **3D**: whole volumes stored as `.npy`, `UNet3D`/`VNet3D`/`ResUNet3D`/`MedNeXt3D`,
  trained on random sub-boxes and evaluated with sliding-window inference back
  on the native NIfTI grid.

All options are dataclasses in `src/utils/config.py`: pass a yaml with
`--config`, override individual fields with dotted flags.

## Setup

Python 3.12.4 (see `.python-version`). We use
[uv](https://docs.astral.sh/uv/getting-started/installation/) with `uv.lock`:

```bash
uv sync
```

That creates `.venv` and installs the locked versions. Plain pip works too,
`requirements.txt` holds the same direct dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The Makefile calls `python` directly, so go through the
environment (`uv run make ...`) or activate `.venv` first.

## Data

Datasets live in `data/<dataset.name>`, the name is set in the config.

The 3D pipeline reads one `.npy` volume per patient instead of PNGs. The prep
script `src/preprocessing/preprocess_3d.py` builds them from the NIfTIs:

```bash
uv run python src/preprocessing/preprocess_3d.py --source_dir data/segthor_full \
    --dest_dir data/SEGTHOR_3D_1ch --windows soft --variants 2
uv run python src/preprocessing/preprocess_3d.py --source_dir data/segthor_full \
    --dest_dir data/SEGTHOR_3D_3ch --windows soft lung blood --variants 2
```

The same script adds the label-free test patients (unzipped `test.zip`) to an
already-built dataset: `--dest_dir data/SEGTHOR_3D_1ch --test_source_dir data/test`
(see [Submission](#submission)).

`1ch` is a single soft-tissue HU window, `3ch` adds lung and blood windows;
`--model.in_channels` must match the dataset, training checks this. The
original NIfTIs with `GT.nii.gz` live in `data/segthor_full` (`segthor_train_full`
on Snellius) and are only needed for 3D evaluation, which maps predictions back
to the native grid and scores there.

## Training

```bash
uv run python src/train.py --config configs/2d.yaml --paths.data_path data
uv run python src/train.py --config configs/3d.yaml --paths.data_path data
```

Both configs point `paths.data_path` at Snellius scratch paths, hence the
override. Anything else can be overridden the same way:

```bash
uv run python src/train.py --config configs/3d.yaml --paths.data_path data \
    --model.name ResUNet3D --training.epochs 100 --training.batch_size 4
```

`uv run python src/train.py --help` lists every field, the comments in
`src/utils/config.py` explain them. Worth knowing:

- `--runtime.debug`: keep only 10 samples per dataset, to smoke-test the loop.
- `--training.gpu`: on by default, silently falls back to CPU without CUDA.
- `--runtime.amp_dtype bf16|fp16`: mixed precision on GPU.
- `--model.deep_supervision`: UNet3D/VNet3D only.
- `--runtime.hd95_in_train`: also compute HD95 on training batches (slow).
- `-O`: `uv run python -O src/train.py ...` disables the sanity asserts for a
  bit of speed, once a setup is known to work.

Each run gets `results/<dataset>/<model>_<timestamp>/` containing `config.yaml`
(the resolved config), `bestweights.pt` (best validation Dice), `best_epoch.txt`
and the wandb files. Metrics go to wandb, the project defaults to
`<dataset>-<model>`.

Sweeps: any scalar in a yaml can be a spec like `<loguniform 1e-5 5e-4>`,
`<choice 8 16 32>`, `<uniform 0 0.1>`, `<intuniform 1 4>` or `<gaussian 0 0.1>`.
Each run draws one value and logs the spec; `SWEEP_SEED=<int>` makes the draw
reproducible. See `configs/sweep_example.yaml`.

## Evaluation

3D evaluation: sliding-window inference, predictions mapped back to the native
grid, Dice/HD95/NSD per organ and patient, written to `eval/metrics.csv`:

```bash
uv run python src/eval_3D.py --config configs/3d.yaml \
    --paths.data_path data \
    --paths.raw_dataset segthor_full \
    --eval.weights results/SEGTHOR_3D_1ch/<run>/bestweights.pt
```

With `eval.enabled: true` in the config, training runs this automatically on
its own `bestweights.pt` after the last epoch. Other options:

- `--eval.split train|val|test`, `--eval.patient_id 2` for a single patient.
- `--eval.overlap 0.5`: overlap between inference windows (0.5 = half).
- `--eval.submission`: also write per-patient prediction NIfTIs under
  `eval/submission/`.
- `--eval.postprocess lcc`: largest connected component for heart, trachea and
  aorta before scoring and writing (`src/postprocess.py`, several steps allowed).

`src/eval.py` is the older 2D single-patient inference script.

## Submission

The course submission is a `group-12/` folder plus a tarball (the required
layout is in `docs/original/readme.md`). `scripts/make_submission.py`
assembles it from a results run directory, once that run has produced the
eval outputs. On a machine with the data:

1. Checksum and extract the test set (Patients 41+, course SharePoint), then
   add the label-less test volumes to the 3D dataset:

   ```bash
   sha256sum -c data/test.zip.sha256
   unzip -q data/test.zip -d data/test
   uv run python src/preprocessing/preprocess_3d.py --dest_dir data/SEGTHOR_3D_1ch \
       --test_source_dir data/test
   ```

   Spacing and windows must match how the dataset was built.

2. Evaluate the best model on the val split (metrics + submission files) and
   the test split (predictions only):

   ```bash
   RUN=results/SEGTHOR_3D_1ch/<run>
   uv run python src/eval_3D.py --config $RUN/config.yaml \
       --paths.data_path data --paths.raw_dataset segthor_full \
       --eval.weights $RUN/bestweights.pt --eval.submission \
       --runtime.amp_dtype bf16
   uv run python src/eval_3D.py --config $RUN/config.yaml \
       --paths.data_path data --paths.raw_dataset segthor_full \
       --eval.weights $RUN/bestweights.pt --eval.split test \
       --runtime.amp_dtype bf16
   ```

   `--eval.split val` is the default. This writes `eval/metrics.csv`, the
   `3d_{dice,hd95,nsd}.npz` metric files (one float32 array per patient, five
   classes, background NaN) and `eval/submission/{val,test}/pred/` (plus
   `val/gt/`). The `--runtime.amp_dtype` override is only needed when the run's
   config has the old `none` value.

3. Assemble the folder and tarball:

   ```bash
   uv run python scripts/make_submission.py --run $RUN
   ```

   This rebuilds the model as a full torch module
   (`bestmodel-group-12.pkl`, the course's `torch.save(net, ...)` form),
   copies the val predictions/labels and metric npz files and the test
   predictions, bundles the `submission` branch into `group-12.bundle`, and
   writes `submission/group-12.tar.zst`. Run it with a clean working tree,
   the bundle only contains committed code.

Note: our HD95 is the 95th percentile of the directed surface-to-surface
distance (the worse of the two directions) and our NSD uses a 2 mm tau, which
differs slightly from the course reference definitions. The npz files hold
the values we report.

## Tests and lint

The tests are assert-based scripts, no pytest required. These pass on this
branch:

```bash
uv run python tests/test_metrics.py
uv run python tests/test_postprocess.py
uv run python tests/test_VNet3d.py
```

`tests/test_sweep.py` and `tests/test_3d_dataset_emma.py` (needs
`data/SEGTHOR`) are stale after the recent refactors and currently fail.

```bash
uv run ruff check
uv run ruff format
```

## Slurm (Snellius)

`jobs/3d_npy.job` streams the dataset tars to the node's NVMe and runs 3D
training + evaluation. The first argument is the model, everything after it is
forwarded to `train.py`:

```bash
sbatch -J ResUNet3D jobs/3d_npy.job ResUNet3D
sbatch -J UNet3D    jobs/3d_npy.job UNet3D --training.epochs 1   # quick check
sbatch --export=ALL,CHANNELS=3 -J ResUNet3D_3ch jobs/3d_npy.job ResUNet3D
```

`CHANNELS=3` picks the 3-channel dataset and `--model.in_channels 3` together.
Logs go to `logs/<job-name>/<jobid>/`. The 2D jobs (`jobs/2d.job`,
`jobs/2d_png.job`) still assume the original Snellius paths
(`/home/scur0088/project`), adjust them before use.

## Repository layout

```
configs/            yaml configs (2d, 3d, boxes/coarse variants, sweep example)
jobs/               Slurm scripts
src/train.py        training entrypoint (2D + 3D)
src/eval_3D.py      3D sliding-window inference + metrics
src/eval.py         2D single-patient inference
src/postprocess.py  postprocessing steps for evaluation
src/models/         ShallowNet, ENet, UNet3D, VNet3D, ResUNet3D, MedNeXt3D
src/preprocessing/  slicing, HU windows, offline augmentation, resampling
src/utils/          config, datasets, losses, metrics, augmentation
scripts/            data generation, stitching, plotting
tests/              assert-based test scripts
docs/               course reference (docs/original) and workflow notes
```

`docs/original/readme.md` is the course-provided readme with the assignment
details, submission layout and scoring; `docs/workflow/` has our notes on git
and environments.
