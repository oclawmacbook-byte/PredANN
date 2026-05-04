"""
Replace synthetic audio in nmedt_s01 with real MP3 audio.

Reconstructs the same windowing used in download_nmedt_subject.py:
  - Skip first 15s of each song
  - Extract 149 windows per song (stride=200 at 125Hz, window=66150 at 22050Hz)
  - Apply same train_test_split(seed=42, test_size=0.25, stratify=labels)

Song label → MP3 mapping (1-indexed file number):
  0→01, 1→02, 2→03, 3→04, 4→05, 5→06, [6→07 MISSING], 7→08, 8→09, 9→10
"""
from __future__ import annotations

import json
from pathlib import Path

import audioread
import numpy as np
import scipy.signal
from sklearn.model_selection import train_test_split

AUDIO_SR = 22050
TARGET_SR = 125
WINDOW_SIZE = 375
AUDIO_WINDOW_SIZE = int(WINDOW_SIZE * AUDIO_SR / TARGET_SR)  # 66150
TRAIN_STRIDE = 200
DELAY_SAMPLES = 25
MAX_DURATION_S = 240
SKIP_SECONDS = 15
SEED = 42

AUDIO_DIR = Path("data/nmedt_audio")
OUT_DIR = Path("data/nmedt_s01")

MP3_NAMES = {
    0: "01 - First Fires.mp3",
    1: "02 - Oino.mp3",
    2: "03 - Tiptoes.mp3",
    3: "04 - Careless Love.mp3",
    4: "05 - Lebanese Blonde.mp3",
    5: "06 - Canopée.mp3",
    # 6: missing (track 07 not available)
    7: "08 - Until the Sun Needs to Rise.mp3",
    8: "09 - Silent Shout.mp3",
    9: "10 - The Last Thing You Should Do.mp3",
}

N_SONGS = 10
SKIP_AUDIO = SKIP_SECONDS * AUDIO_SR  # 330750 samples to skip


def make_synthetic_audio(song_label: int, n_samples: int) -> np.ndarray:
    t = np.arange(n_samples) / AUDIO_SR
    base_freq = 110 * (2 ** (song_label / 12))
    audio = (
        np.sin(2 * np.pi * base_freq * t)
        + 0.5 * np.sin(2 * np.pi * base_freq * 2 * t)
        + 0.25 * np.sin(2 * np.pi * base_freq * 3 * t)
    ).astype(np.float32)
    audio /= np.abs(audio).max() + 1e-8
    return audio


def load_real_audio(song_label: int) -> np.ndarray | None:
    if song_label not in MP3_NAMES:
        return None
    path = AUDIO_DIR / MP3_NAMES[song_label]
    if not path.exists():
        print(f"  WARNING: {path} not found, using synthetic audio")
        return None
    print(f"  Loading {path.name} ...")
    with audioread.audio_open(str(path)) as f:
        src_sr = f.samplerate
        n_channels = f.channels
        raw_blocks = [np.frombuffer(b, dtype=np.int16) for b in f]
    raw = np.concatenate(raw_blocks).astype(np.float32) / 32768.0
    if n_channels > 1:
        raw = raw.reshape(-1, n_channels).mean(axis=1)
    if src_sr != AUDIO_SR:
        n_out = int(len(raw) * AUDIO_SR / src_sr)
        raw = scipy.signal.resample(raw, n_out)
    return raw.astype(np.float32)


def extract_audio_windows(song_audio: np.ndarray) -> np.ndarray:
    n_eeg_samples = MAX_DURATION_S * TARGET_SR  # 30000

    windows = []
    for start in range(0, n_eeg_samples - WINDOW_SIZE - DELAY_SAMPLES + 1, TRAIN_STRIDE):
        audio_start = SKIP_AUDIO + int(start * AUDIO_SR / TARGET_SR)
        audio_end = audio_start + AUDIO_WINDOW_SIZE
        if audio_end > len(song_audio):
            print(f"  WARNING: audio too short at start={start}, stopping at {len(windows)} windows")
            break
        windows.append(song_audio[audio_start:audio_end])

    return np.stack(windows).astype(np.float32)


def main() -> None:
    all_audio, all_labels = [], []

    for song_label in range(N_SONGS):
        print(f"Song {song_label} ({'MP3' if song_label in MP3_NAMES else 'synthetic'}):")
        audio = load_real_audio(song_label)

        if audio is None:
            # Synthetic fallback: generate enough samples
            n_needed = SKIP_AUDIO + int(MAX_DURATION_S * AUDIO_SR) + AUDIO_WINDOW_SIZE
            audio = np.zeros(n_needed, dtype=np.float32)
            synth = make_synthetic_audio(song_label, MAX_DURATION_S * AUDIO_SR + AUDIO_WINDOW_SIZE)
            audio[SKIP_AUDIO:SKIP_AUDIO + len(synth)] = synth

        windows = extract_audio_windows(audio)
        print(f"  {len(windows)} windows extracted")
        all_audio.append(windows)
        all_labels.extend([song_label] * len(windows))

    audio_all = np.concatenate(all_audio)
    labels_all = np.array(all_labels, dtype=np.int64)
    print(f"\nTotal: {len(labels_all)} windows")

    idx = np.arange(len(labels_all))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.25, random_state=SEED, stratify=labels_all
    )

    np.save(OUT_DIR / "train_audio.npy", audio_all[train_idx])
    np.save(OUT_DIR / "val_audio.npy", audio_all[val_idx])

    # Update meta.json
    meta_path = OUT_DIR / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["note"] = "Audio replaced with real MP3 data (song 6/track-07 is synthetic placeholder)."
    meta["audio_source"] = "real_mp3_except_song6"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: train_audio.npy ({len(train_idx)} windows), val_audio.npy ({len(val_idx)} windows)")
    print("meta.json updated.")


if __name__ == "__main__":
    main()
