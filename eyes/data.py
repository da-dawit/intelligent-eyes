"""Build clips (bursts of 1-10 frames) and the frame table every later step indexes.

frames.csv has one row per frame, in clip order:
    row, path, clip_id, split, camera, frame_idx, y_frame, y_clip
Features, votes and predictions are all stored by this row number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _empty_label(root: Path, md: pd.DataFrame) -> int | None:
    """WILDS label of the 'empty' (no animal) class, from categories.csv when present."""
    cat = root / "categories.csv"
    if cat.exists():
        c = pd.read_csv(cat)
        name_col = "name" if "name" in c.columns else c.columns[1]
        hit = c[c[name_col].astype(str).str.lower() == "empty"]
        if len(hit) and "y" in c.columns:
            return int(hit["y"].iloc[0])
    return 0 if (md["y"] == 0).any() else None


def build_iwildcam(root: str | Path, splits=("train", "test"), max_frames: int = 10,
                   max_clips: dict | None = None, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """WILDS iWildCam v2.0 -> frame table of clips.

    A clip is one WILDS sequence (a motion-triggered burst) within one split, frames in
    time order, at most `max_frames`. Assumption A1 (one species per clip): clips whose
    frames name two different species are dropped, as are clips with no animal at all.
    Frames labelled 'empty' inside an animal clip are kept -- the animal is out of frame
    there, which is failure mode 3.
    max_clips: optional {split: n} to subsample clips (seeded) for quick runs.
    """
    root = Path(root)
    md = pd.read_csv(root / "metadata.csv")
    md = md[md["split"].isin(splits)].copy()
    empty = _empty_label(root, md)
    md["dt"] = pd.to_datetime(md["datetime"], format="mixed")
    md = md.sort_values(["split", "sequence_remapped", "dt", "filename"])
    md["frame_idx"] = md.groupby(["split", "sequence_remapped"]).cumcount()
    md = md[md["frame_idx"] < max_frames]

    species = md[md["y"] != empty].groupby(["split", "sequence_remapped"])["y"].agg(lambda s: set(s))
    n_species = species.apply(len)
    stats = dict(empty_label=empty,
                 clips_total=int(md.groupby(["split", "sequence_remapped"]).ngroups),
                 clips_all_empty=int(md.groupby(["split", "sequence_remapped"]).ngroups - len(species)),
                 clips_multi_species=int((n_species > 1).sum()))
    one = species[n_species == 1].apply(lambda s: next(iter(s)))
    one.name = "y_clip"
    md = md.join(one, on=["split", "sequence_remapped"], how="inner")

    if max_clips:
        rng = np.random.default_rng(seed)
        keep = []
        for split, n in max_clips.items():
            seqs = md.loc[md["split"] == split, "sequence_remapped"].unique()
            if n is not None and n < len(seqs):
                seqs = rng.choice(seqs, size=n, replace=False)
            keep.append(md[(md["split"] == split) & md["sequence_remapped"].isin(seqs)])
        md = pd.concat(keep + [md[~md["split"].isin(max_clips)]])

    md = md.sort_values(["split", "sequence_remapped", "frame_idx"]).reset_index(drop=True)
    frames = pd.DataFrame({
        "row": np.arange(len(md)),
        "path": ("train/" + md["filename"]).values,
        "clip_id": (md["split"] + "-" + md["sequence_remapped"].astype(str)).values,
        "split": md["split"].values,
        "camera": md["location_remapped"].values,
        "frame_idx": md["frame_idx"].values,
        "y_frame": md["y"].values,
        "y_clip": md["y_clip"].values,
    })
    stats.update(frames=len(frames), clips=int(frames["clip_id"].nunique()),
                 frames_per_clip=float(len(frames) / max(frames["clip_id"].nunique(), 1)),
                 by_split=frames.groupby("split")["clip_id"].nunique().to_dict())
    return frames, stats


def build_ddr(root: str | Path, max_clips: dict | None = None, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """DDR grading set -> one-frame clips. Grade 5 (ungradable) is dropped.

    Expects root/<folder>/{train,valid,test}.txt with lines '<file> <grade>' and the
    images in root/<folder>/<split>/, where <folder> is DR_grading (the Hugging Face copy)
    or DDR_grading (the original archive). 'valid' is not used (fitting uses 10% of train).
    """
    folder = next(f for f in ("DR_grading", "DDR_grading") if (Path(root) / f).is_dir())
    root = Path(root) / folder
    rows = []
    for split in ("train", "test"):
        lab = pd.read_csv(root / f"{split}.txt", sep=r"\s+", header=None, names=["file", "grade"])
        lab = lab[lab["grade"] != 5]
        if max_clips and max_clips.get(split):
            lab = lab.sample(n=min(max_clips[split], len(lab)), random_state=seed)
        for f, g in zip(lab["file"], lab["grade"]):
            rows.append((f"{folder}/{split}/{f}", f"{split}-{f}", split, g))
    df = pd.DataFrame(rows, columns=["path", "clip_id", "split", "y"])
    frames = pd.DataFrame({
        "row": np.arange(len(df)), "path": df["path"], "clip_id": df["clip_id"], "split": df["split"],
        "camera": np.arange(len(df)),          # no cameras: each photo is its own bootstrap unit
        "frame_idx": 0, "y_frame": df["y"], "y_clip": df["y"],
    })
    grades = frames.groupby(["split", "y_clip"]).size().unstack(0).fillna(0).astype(int)
    return frames, dict(frames=len(frames), by_split=frames.groupby("split").size().to_dict(),
                        grades={str(k): v for k, v in grades.to_dict().items()})
