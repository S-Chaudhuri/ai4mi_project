"""Collect every training run from the Slurm logs into one table + CSV (for graphs).

    python jobs/collect_runs.py              all logs under logs/ and jobs/logs/
    python jobs/collect_runs.py skel         only runs whose name or model contains "skel"

Per run: Slurm state (sacct), config (from the config the log prints), progress,
the error if it failed, and the final whole-volume metrics (eval/metrics.csv of
the run's results folder, or the per-patient lines in the log if that is gone).
Writes results/all_runs_<date>_<time>.csv (never overwrites).
"""

import ast
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ORGANS = ["esophagus", "heart", "trachea", "aorta"]
LOG_GLOBS = ["logs/**/*.log", "logs/**/*.out", "jobs/logs/**/*.log", "jobs/logs/**/*.out", "slurm-*.out"]
ERROR_LINE = re.compile(r"^([\w.]*(Error|Exception)\b.*|.*\bCANCELLED\b.*|.*DUE TO TIME LIMIT.*|ERROR: .*)$")


def job_id(log: Path) -> str:
    for part in [log.stem, *reversed(log.parent.parts)]:
        if m := re.search(r"(\d{5,})", part):
            return m.group(1)
    return "?"


def run_name(log: Path) -> str:
    # logs/<name>/<jobid>/out.log (new jobs); older jobs had logs/<jobid>/out.log
    parts = log.parts
    if len(parts) >= 4 and parts[-2].isdigit() and parts[-3] != "logs":
        return parts[-3]
    return ""


def parse_config(text: str) -> dict:
    m = re.search(r"^\{'augmentation'.*?\}\}\s*$", text, re.S | re.M)
    if not m:
        return {}
    block = re.sub(r"\w*Path\('([^']*)'\)", r"'\1'", m.group(0))
    try:
        return ast.literal_eval(block)
    except (ValueError, SyntaxError):
        return {}


def metrics_from_csv(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["patient"] == "MEAN":
                for k in ("dice", "hd95", "nsd"):
                    out[f"{row['organ']}_{k}"] = float(row[k])
    return out


def metrics_from_log(text: str) -> dict:
    per = {}
    for m in re.finditer(r"^Patient_\d+ (\{.*\})\s*$", text, re.M):
        try:
            d = ast.literal_eval(m.group(1))
        except (ValueError, SyntaxError):
            continue
        for organ, vals in d.items():
            for k, v in vals.items():
                per.setdefault(f"{organ}_{k}", []).append(float(v))
    return {k: sum(v) / len(v) for k, v in per.items()}


def sacct(ids: list[str]) -> dict:
    out = {}
    ids = [i for i in ids if i.isdigit()]
    for i in range(0, len(ids), 200):
        try:
            res = subprocess.run(
                ["sacct", "-X", "-n", "-P", "-j", ",".join(ids[i:i + 200]),
                 "-o", "JobID,JobName,State,Elapsed,Partition"],
                capture_output=True, text=True, timeout=60,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return out
        for line in res.stdout.splitlines():
            jid, name, state, elapsed, part = (line.split("|") + [""] * 5)[:5]
            out[jid] = {"slurm_name": name, "state": state.split()[0] if state else "",
                        "elapsed": elapsed, "partition": part}
    return out


def collect(log: Path) -> dict:
    text = log.read_text(errors="replace").replace("\r", "\n")
    cfg = parse_config(text)
    g = lambda sec, key, default="": cfg.get(sec, {}).get(key, default)  # noqa: E731

    epochs = [int(e) for e in re.findall(r"Validation \(\s*(\d+)\)", text)]
    improved = re.findall(r"Improved dice at epoch (\d+): [\d.]+->([\d.]+)", text)
    errors = [ln.strip() for ln in text.splitlines() if ERROR_LINE.match(ln.strip())]
    errors = [e for e in errors if "wandb" not in e]

    results_dir = g("paths", "results_dir")
    metrics, source = {}, ""
    if results_dir and (Path(results_dir) / "eval" / "metrics.csv").is_file():
        metrics, source = metrics_from_csv(Path(results_dir) / "eval" / "metrics.csv"), "csv"
    elif m := metrics_from_log(text):
        metrics, source = m, "log"

    row = {
        "job": job_id(log), "name": run_name(log), "log": str(log),
        "model": g("model", "name"), "dataset": g("dataset", "name"),
        "channels": g("model", "in_channels", 1 if cfg else ""),
        "loss": g("training", "loss"), "skeleton_weight": g("training", "skeleton_weight", 0 if cfg else ""),
        "offline_aug": (not g("dataset", "skip_augmented", False)) if cfg else "",
        "online_aug": g("augmentation", "enabled"),
        "deep_supervision": g("model", "deep_supervision"),
        "mode": g("training", "mode"),
        "epochs_cfg": g("training", "epochs"), "batch_size": g("training", "batch_size"),
        "epochs_done": max(epochs) + 1 if epochs else 0,
        "best_val_dice": float(improved[-1][1]) if improved else "",
        "best_epoch": int(improved[-1][0]) if improved else "",
        "metrics_from": source,
        "error": errors[-1][:120] if errors else "",
        "results_dir": results_dir,
    }
    for k in ("dice", "hd95", "nsd"):
        vals = [metrics[f"{o}_{k}"] for o in ORGANS if f"{o}_{k}" in metrics]
        row[f"mean_{k}"] = round(sum(vals) / len(vals), 4) if len(vals) == len(ORGANS) else ""
    for o in ORGANS:
        for k in ("dice", "hd95", "nsd"):
            v = metrics.get(f"{o}_{k}")
            row[f"{o}_{k}"] = round(v, 4) if v is not None else ""
    return row


def guess_state(row: dict) -> str:
    err = row["error"]
    if "TIME LIMIT" in err:
        return "TIMEOUT"
    if "CANCELLED" in err:
        return "CANCELLED"
    if row["metrics_from"] or (row["epochs_done"] and row["epochs_done"] == row["epochs_cfg"] and not err):
        return "COMPLETED?"
    return "FAILED?" if err else "?"


def main() -> None:
    if not Path("src/train.py").is_file():
        sys.exit("Run this from the project root (cd ~/project)")
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    logs = sorted({p for pat in LOG_GLOBS for p in Path(".").glob(pat) if p.is_file()})
    rows = [collect(p) for p in logs]

    slurm = sacct([r["job"] for r in rows])
    for r in rows:
        s = slurm.get(r["job"], {})
        r["name"] = r["name"] or s.get("slurm_name", "")
        r["state"] = s.get("state") or guess_state(r)
        r["elapsed"] = s.get("elapsed", "")
        r["partition"] = s.get("partition", "")
    if flt:
        rows = [r for r in rows if flt in r["name"] or flt in str(r["model"])]
    rows.sort(key=lambda r: r["job"])
    if not rows:
        sys.exit("No logs found")

    fmt = "{:<9} {:<26} {:<10} {:<9} {:<10} {:<15} {:<7} {:>4} {:>5} {:>5} {:>6} {:>7} {:>7} {:>7}  {}"
    print(fmt.format("JOB", "NAME", "STATE", "ELAPSED", "MODEL", "DATASET", "LOSS", "SKEL", "AUG", "EP",
                     "VAL", "DICE", "HD95", "ESO", "ERROR"))
    for r in rows:
        aug = ("off" if r["offline_aug"] is True else "") + ("on" if r["online_aug"] is True else "")
        aug = {"": "-", "off": "offl", "on": "onl", "offon": "both"}[aug]
        print(fmt.format(
            r["job"], r["name"][:26], r["state"][:10], r["elapsed"], str(r["model"])[:10],
            str(r["dataset"]).replace("SEGTHOR", "S")[:15], str(r["loss"])[:7], str(r["skeleton_weight"]), aug,
            f"{r['epochs_done']}/{r['epochs_cfg']}", r["best_val_dice"], r["mean_dice"], r["mean_hd95"],
            r["esophagus_dice"], r["error"][:60],
        ))

    base = Path("results") / f"all_runs_{datetime.now():%Y%m%d_%H%M%S}"
    base.parent.mkdir(exist_ok=True)
    out, i = base.with_suffix(".csv"), 1
    while out.exists():
        out, i = Path(f"{base}_{i}.csv"), i + 1
    with open(out, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} runs. VAL = best training-time val Dice, DICE/HD95/ESO = final whole-volume eval "
          f"(mean of 4 organs; esophagus Dice). AUG: offl = offline copies, onl = online, both.\n>> wrote {out}")


if __name__ == "__main__":
    main()
