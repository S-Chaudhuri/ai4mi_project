# Submission runbook

How to produce the course submission (`group-12/` + tarball) for a given model
— a `config.yaml` plus its `bestweights.pt` — on a clean machine.

## What you need

- The repo, at the commit you want to submit (this branch: `submission`).
- Data on the machine:
  - `data/SEGTHOR_3D_1ch/` — preprocessed 3D dataset (`train/`, `val/`);
  - `data/segthor_full/` — raw NIfTIs (for the GT and the native grid);
  - `data/test.zip` — the course test set (Patients 41+), unzipped to `data/test/`
    as flat `Patient_XX.nii.gz` files.
- The run you want to submit: its `config.yaml` and `bestweights.pt`.

## 1. Clean environment

```bash
git clone <repo> && cd <repo>
git checkout submission
uv sync
```

## 2. Stage the run

`make_submission.py` expects a run directory with `config.yaml`,
`bestweights.pt` and (after step 4) `eval/`:

```bash
RUN=results/SEGTHOR_3D_1ch/best_vnet
mkdir -p $RUN
cp <path>/config.yaml   $RUN/config.yaml
cp <path>/bestweights.pt $RUN/bestweights.pt
```

The `config.yaml` must describe the model in the weights
(`model.name` VNet3D/UNet3D/... and `model.in_channels` must match the
dataset). The best model's config is `best_run.yaml` in the repo root
(VNet3D, 1 channel). A mismatch shows up as a `state_dict` error.

## 3. Add the test set to the dataset

```bash
sha256sum -c data/test.zip.sha256
unzip -q data/test.zip -d data/test
uv run python src/preprocessing/preprocess_3d.py --dest_dir data/SEGTHOR_3D_1ch \
    --test_source_dir data/test
```

Spacing/windows default to the values the dataset was built with — check
`data/SEGTHOR_3D_1ch/dataset.json` if yours differ.

## 4. Evaluate (val + test)

```bash
uv run python src/eval_3D.py --config $RUN/config.yaml \
    --paths.data_path data --paths.raw_dataset segthor_full --paths.results_dir $RUN \
    --eval.weights $RUN/bestweights.pt --eval.submission --runtime.amp_dtype bf16
uv run python src/eval_3D.py --config $RUN/config.yaml \
    --paths.data_path data --paths.raw_dataset segthor_full --paths.results_dir $RUN \
    --eval.weights $RUN/bestweights.pt --eval.split test --runtime.amp_dtype bf16
```

The first command (val, the default split) writes `eval/metrics.csv`, the
`3d_{dice,hd95,nsd}.npz` metric files and `eval/submission/val/{pred,gt}/`.
The second writes `eval/submission/test/pred/` (no labels, no metrics).
`--paths.raw_dataset` is the raw NIfTI folder name (`segthor_train_full` on
Snellius). `--runtime.amp_dtype bf16` is only needed when the config has the
old `none` value.

**If you rerun the eval after any change, delete `$RUN/eval` first** — old
prediction files survive and break the npz key check in step 5.

## 5. Assemble

```bash
uv run python scripts/make_submission.py --run $RUN
```

Produces `submission/group-12/` and `submission/group-12.tar.zst`:
the model as a full torch module (`bestmodel-group-12.pkl`), val
predictions/labels/metrics, test predictions and a git bundle of the
`submission` branch. Run it with a **clean working tree** and all commits
synced: the bundle only contains committed code, and the pkl must load with
the bundled code.

## Verify

```bash
find submission/group-12 -type f | sort
git bundle verify submission/group-12/group-12.bundle
uv run python -c "import torch; net = torch.load('submission/group-12/bestmodel-group-12.pkl', weights_only=False); print(type(net).__name__)"
```

The last line rebuilds the submitted model from the pkl — it should print the
model name (e.g. `VNet3D`).
