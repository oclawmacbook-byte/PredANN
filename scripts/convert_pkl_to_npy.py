"""
Convert existing preprocessed NMED-T pkl/wav data to npy format for train.py.

The Preprocessed directory contains:
  DS_EEG_pkl/{subject}_{song_trigger}_{trial}.pkl  — EEG at 125Hz (128, T)
  Audio/{song_trigger}.wav                          — audio at 44100Hz

This script windows them and saves train_eeg.npy / val_eeg.npy / etc.

Usage:
    python scripts/convert_pkl_to_npy.py \
        --preprocessed_dir /Volumes/Untitled/Dataset/NMED-T/Preprocessed
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    DELAY_SAMPLES,
    TARGET_SR,
    TRAIN_STRIDE,
    WINDOW_SIZE,
    AUDIO_WINDOW_SIZE,
    SONG_TRIGGERS,
    extract_windows,
    normalize_eeg,
    split_dataset,
    truncate_to_max_duration,
)

AUDIO_SR = 22050


def load_audio_dict(audio_dir: Path) -> dict[int, np.ndarray]:
    """Load wav files named {trigger_id}.wav → {song_idx (0-9): waveform}."""
    import librosa
    songs: dict[int, np.ndarray] = {}
    for idx, trigger_id in enumerate(SONG_TRIGGERS):
        wav_path = audio_dir / f"{trigger_id}.wav"
        if not wav_path.exists():
            print(f"  Warning: {wav_path} not found, skipping song {idx}")
            continue
        audio, _ = librosa.load(str(wav_path), sr=AUDIO_SR, mono=True)
        songs[idx] = audio
    return songs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preprocessed_dir",
                   default="/Volumes/Untitled/Dataset/NMED-T/Preprocessed")
    p.add_argument("--delay_ms", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    base = Path(args.preprocessed_dir)
    pkl_dir = base / "DS_EEG_pkl"
    audio_dir = base / "Audio"
    out_dir = base

    if not pkl_dir.exists():
        print(f"DS_EEG_pkl not found in {base}")
        return
    if not audio_dir.exists():
        print(f"Audio dir not found in {base}")
        return

    delay_samples = int(args.delay_ms * TARGET_SR / 1000)
    audio_dict = load_audio_dict(audio_dir)

    # Group pkl files by subject
    pkl_files = sorted(glob.glob(str(pkl_dir / "*.pkl")))
    by_subject: dict[str, list[Path]] = {}
    for f in pkl_files:
        name = Path(f).stem  # e.g. "10_21_1"
        subject_id = name.split("_")[0]
        by_subject.setdefault(subject_id, []).append(Path(f))

    all_train_eeg, all_train_audio, all_train_labels = [], [], []
    all_val_eeg, all_val_audio, all_val_labels = [], [], []
    subject_ids_train, subject_ids_val = [], []

    for subject_id, files in sorted(by_subject.items()):
        print(f"Processing subject {subject_id}...")
        all_eeg, all_audio, all_labels = [], [], []

        for pkl_path in sorted(files):
            name = pkl_path.stem  # "{subject}_{trigger}_{trial}"
            parts = name.split("_")
            trigger_id = int(parts[1])
            if trigger_id not in SONG_TRIGGERS:
                continue
            song_idx = SONG_TRIGGERS.index(trigger_id)
            if song_idx not in audio_dict:
                continue

            with open(pkl_path, "rb") as f:
                eeg = pickle.load(f)
            if hasattr(eeg, "numpy"):
                eeg = eeg.numpy()
            eeg = np.array(eeg, dtype=np.float32)

            eeg = normalize_eeg(eeg)
            eeg = truncate_to_max_duration(eeg)

            eeg_w, audio_w, lbl = extract_windows(
                eeg, audio_dict[song_idx], song_idx,
                stride=TRAIN_STRIDE, delay_samples=delay_samples,
            )
            if len(eeg_w) > 0:
                all_eeg.append(eeg_w)
                all_audio.append(audio_w)
                all_labels.append(lbl)

        if not all_eeg:
            print(f"  No windows for subject {subject_id}, skipping.")
            continue

        eeg_all = np.concatenate(all_eeg)
        audio_all = np.concatenate(all_audio)
        labels_all = np.concatenate(all_labels)

        eeg_tr, eeg_v, aud_tr, aud_v, lbl_tr, lbl_v = split_dataset(
            eeg_all, audio_all, labels_all, seed=args.seed
        )

        all_train_eeg.append(eeg_tr)
        all_train_audio.append(aud_tr)
        all_train_labels.append(lbl_tr)
        subject_ids_train.append(np.full(len(lbl_tr), int(subject_id), dtype=np.int32))

        all_val_eeg.append(eeg_v)
        all_val_audio.append(aud_v)
        all_val_labels.append(lbl_v)
        subject_ids_val.append(np.full(len(lbl_v), int(subject_id), dtype=np.int32))

    if not all_train_eeg:
        print("No data converted.")
        return

    np.save(out_dir / "train_eeg.npy", np.concatenate(all_train_eeg))
    np.save(out_dir / "train_audio.npy", np.concatenate(all_train_audio))
    np.save(out_dir / "train_labels.npy", np.concatenate(all_train_labels))
    np.save(out_dir / "train_subject_ids.npy", np.concatenate(subject_ids_train))

    np.save(out_dir / "val_eeg.npy", np.concatenate(all_val_eeg))
    np.save(out_dir / "val_audio.npy", np.concatenate(all_val_audio))
    np.save(out_dir / "val_labels.npy", np.concatenate(all_val_labels))
    np.save(out_dir / "val_subject_ids.npy", np.concatenate(subject_ids_val))

    train_size = int(np.concatenate(all_train_labels).shape[0])
    val_size = int(np.concatenate(all_val_labels).shape[0])
    meta = {
        "delay_ms": args.delay_ms,
        "delay_samples": delay_samples,
        "seed": args.seed,
        "train_size": train_size,
        "val_size": val_size,
        "source": "converted from DS_EEG_pkl + Audio",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {out_dir}")
    print(f"  Train: {train_size} windows")
    print(f"  Val:   {val_size} windows")


if __name__ == "__main__":
    main()
