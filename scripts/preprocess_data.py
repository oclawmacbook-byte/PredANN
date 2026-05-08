"""
Preprocess raw NMED-T data into windowed numpy arrays ready for training.

Usage:
    python scripts/preprocess_data.py \
        --raw_dir /Volumes/Untitled/Dataset/NMED-T/Original \
        --out_dir /Volumes/Untitled/Dataset/NMED-T/Preprocessed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    DELAY_SAMPLES,
    ORIG_SR,
    TARGET_SR,
    TRAIN_STRIDE,
    downsample_eeg,
    extract_windows,
    load_nmedt_audio,
    load_nmedt_eeg,
    normalize_eeg,
    split_dataset,
    truncate_to_max_duration,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="/Volumes/Untitled/Dataset/NMED-T/Original",
                   help="Path to raw NMED-T directory (contains XX_Y_raw.mat files)")
    p.add_argument("--out_dir", default="/Volumes/Untitled/Dataset/NMED-T/Preprocessed",
                   help="Output directory for .npy files used by train.py")
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 21)),
                   help="Clean subject IDs to process (default: all 20)")
    p.add_argument("--delay_ms", type=int, default=200,
                   help="EEG-to-audio delay in ms (default: 200)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delay_samples = int(args.delay_ms * TARGET_SR / 1000)
    audio_dict = load_nmedt_audio(args.raw_dir)

    all_train_eeg, all_train_audio, all_train_labels = [], [], []
    all_val_eeg, all_val_audio, all_val_labels = [], [], []
    subject_ids_train, subject_ids_val = [], []

    downsample_factor = ORIG_SR // TARGET_SR  # 1000 // 125 = 8

    for subj in args.subjects:
        print(f"Processing subject {subj:02d}...")
        try:
            eeg_raw, song_sequence = load_nmedt_eeg(args.raw_dir, subj)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue
        eeg_ds = downsample_eeg(eeg_raw)
        eeg_norm = normalize_eeg(eeg_ds)

        # Decimate song_sequence labels to match downsampled EEG
        song_seq_ds = song_sequence[::downsample_factor][: eeg_ds.shape[1]]

        all_eeg, all_audio, all_labels = [], [], []
        for song_id in np.unique(song_seq_ds):
            mask = song_seq_ds == song_id
            song_eeg = eeg_norm[:, mask]
            song_eeg = truncate_to_max_duration(song_eeg)  # truncate per song
            if int(song_id) not in audio_dict:
                continue
            eeg_w, audio_w, lbl = extract_windows(
                song_eeg, audio_dict[int(song_id)], int(song_id),
                stride=TRAIN_STRIDE, delay_samples=delay_samples,
            )
            if len(eeg_w) > 0:
                all_eeg.append(eeg_w)
                all_audio.append(audio_w)
                all_labels.append(lbl)

        if not all_eeg:
            print(f"  No windows extracted for subject {subj}, skipping.")
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
        subject_ids_train.append(np.full(len(lbl_tr), subj, dtype=np.int32))

        all_val_eeg.append(eeg_v)
        all_val_audio.append(aud_v)
        all_val_labels.append(lbl_v)
        subject_ids_val.append(np.full(len(lbl_v), subj, dtype=np.int32))

    if not all_train_eeg:
        print("No data was processed. Check --raw_dir and subject IDs.")
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
        "subjects": args.subjects,
        "train_size": train_size,
        "val_size": val_size,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved preprocessed data to {out_dir}")
    print(f"  Train: {train_size} windows")
    print(f"  Val:   {val_size} windows")


if __name__ == "__main__":
    main()
