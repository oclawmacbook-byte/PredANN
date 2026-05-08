from __future__ import annotations

import re as _re
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# NMED-T dataset constants
ORIG_SR = 1000       # Hz — raw recording sample rate
TARGET_SR = 125      # Hz — downsampled rate used for training
MAX_DURATION_S = 240  # 4 minutes per song segment
WINDOW_SIZE = 375    # samples at 125Hz = 3 seconds
AUDIO_WINDOW_SIZE = 66150  # 3s × 22050Hz
TRAIN_STRIDE = 200
VAL_STRIDE = 1
DELAY_SAMPLES = 25   # 200ms at 125Hz — empirically optimal (Table 2)

# NMED-T song trigger IDs (21-30) → song indices (0-9)
SONG_TRIGGERS = list(range(21, 31))

# Clean subject ID (1-20) → raw file number
CLEAN_TO_RAW = {
    1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11,
    11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 19, 18: 20,
    19: 21, 20: 23,
}

# Song durations in audio samples at 44100Hz (from NMED-T README)
SONG_AUDIO_LENGTHS_44100 = [
    278 * 44100, 271 * 44100, 276 * 44100, 294 * 44100,
    289 * 44100, 276 * 44100, 292 * 44100, 292 * 44100,
    293 * 44100, 298 * 44100,
]


def _extract_songs_from_mat(mat_path: Path, sfreq_default: int = ORIG_SR) -> dict[int, np.ndarray]:
    """Load one raw NMED-T .mat file and return {song_idx: eeg (128, T)} at ORIG_SR."""
    import h5py

    def _decode(val):
        if isinstance(val, bytes):
            return val.decode()
        return str(val)

    with h5py.File(str(mat_path), "r") as f:
        sfreq = int(np.array(f['fs']).squeeze())
        eeg = np.array(f['X'], dtype=np.float32)
        # h5py returns (time, channels) for MATLAB HDF5; transpose to (channels, time)
        if eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T
        eeg = np.delete(eeg, 128, axis=0)  # remove electrode 129 (vertex reference)

        din_group = f['DIN_1']
        # DIN_1 is a 2×N cell array stored as object references
        refs_row0 = din_group[0]  # trigger label refs
        refs_row1 = din_group[1]  # onset sample refs
        triggers = [_decode(np.array(f[r]).squeeze()) for r in refs_row0]
        onsets = [int(np.array(f[r]).squeeze()) for r in refs_row1]

    segments: dict[int, np.ndarray] = {}
    for song_idx, (trigger_id, audio_len_44100) in enumerate(
        zip(SONG_TRIGGERS, SONG_AUDIO_LENGTHS_44100)
    ):
        tid = str(trigger_id)
        if tid not in triggers:
            continue
        t_pos = triggers.index(tid)
        start_pos = t_pos + 1
        if start_pos >= len(triggers) or triggers[start_pos] != '128':
            continue
        # Onset is 1 second before the '128' trigger
        onset = onsets[start_pos] - sfreq
        if onset < 0:
            onset = 0
        eeg_len = int(audio_len_44100 / 44100 * sfreq)
        end = min(onset + eeg_len - 1, eeg.shape[1])
        if onset >= eeg.shape[1]:
            continue
        segments[song_idx] = eeg[:, onset:end]

    return segments


def load_nmedt_eeg(data_dir: str | Path, subject_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load raw NMED-T EEG for one subject (clean ID 1-20).

    Returns (eeg, song_sequence) where:
      eeg:           (128, total_T) float32 at ORIG_SR=1000Hz, all songs concatenated
      song_sequence: (total_T,) int64 with song index (0-9) for each sample
    """
    data_dir = Path(data_dir)
    raw_id = CLEAN_TO_RAW.get(subject_id)
    if raw_id is None:
        raise ValueError(f"Unknown clean subject ID: {subject_id}. Must be 1-20.")

    all_segs: dict[int, np.ndarray] = {}
    for trial in [1, 2]:
        mat_path = data_dir / f"{raw_id:02d}_{trial}_raw.mat"
        if not mat_path.exists():
            continue
        segs = _extract_songs_from_mat(mat_path)
        all_segs.update(segs)

    if not all_segs:
        raise FileNotFoundError(
            f"No EEG data found for subject {subject_id} (raw_id={raw_id:02d}) in {data_dir}"
        )

    eeg_parts, label_parts = [], []
    for idx in sorted(all_segs):
        seg = all_segs[idx]
        eeg_parts.append(seg)
        label_parts.append(np.full(seg.shape[1], idx, dtype=np.int64))

    return np.concatenate(eeg_parts, axis=1), np.concatenate(label_parts)


def _load_audio_from_mat(path: Path, target_sr: int = 22050) -> np.ndarray:
    """Extract audio waveform from a .mat file and resample to target_sr."""
    import librosa
    import scipy.io

    data = scipy.io.loadmat(str(path))
    audio_keys = [k for k in data if not k.startswith("_")]
    preferred = [k for k in audio_keys if any(
        kw in k.lower() for kw in ("audio", "wav", "stim", "data", "wave")
    )]
    key = preferred[0] if preferred else audio_keys[0]
    audio = np.array(data[key], dtype=np.float64).squeeze()

    sr_keys = [k for k in data if not k.startswith("_") and "sr" in k.lower()]
    src_sr = int(np.array(data[sr_keys[0]]).flat[0]) if sr_keys else 44100

    if src_sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=src_sr, target_sr=target_sr)
    return audio.astype(np.float32)


def load_nmedt_audio(data_dir: str | Path) -> dict[int, np.ndarray]:
    """
    Load the 10 song audio clips from NMED-T.

    Looks in data_dir/audio/ for files named song_01.*, 01 - *.mp3, etc.
    File numbering is 1-based (01 = first song = song index 0).
    Returns {song_idx (0-9): waveform at 22050Hz}.
    """
    import librosa

    data_dir = Path(data_dir)
    audio_dir = data_dir / "audio"
    songs: dict[int, np.ndarray] = {}
    for song_idx in range(10):
        file_num = song_idx + 1  # files are 1-indexed
        candidates = (
            list(audio_dir.glob(f"song_{file_num:02d}.mat")) +
            list(audio_dir.glob(f"{file_num:02d}.mat")) +
            list(audio_dir.glob(f"song_{file_num:02d}.*")) +
            list(audio_dir.glob(f"{file_num:02d}.*"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No audio file found for song {song_idx} (file {file_num:02d}) in {audio_dir}"
            )
        path = candidates[0]
        if path.suffix == ".mat":
            audio = _load_audio_from_mat(path)
        elif path.suffix == ".npy":
            audio = np.load(str(path)).astype(np.float32)
        else:
            audio, _ = librosa.load(str(path), sr=22050, mono=True)
        songs[song_idx] = audio
    return songs


def downsample_eeg(eeg: np.ndarray, orig_sr: int = ORIG_SR, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample EEG from orig_sr to target_sr along axis=1."""
    n_out = int(eeg.shape[1] * target_sr / orig_sr)
    return scipy.signal.resample(eeg, n_out, axis=1).astype(np.float32)


def normalize_eeg(eeg: np.ndarray, clamp: float = 20.0) -> np.ndarray:
    """RobustScaler per channel then clamp to ±clamp. Input: (n_channels, n_samples)."""
    scaler = RobustScaler()
    eeg_norm = scaler.fit_transform(eeg.T).T
    return np.clip(eeg_norm, -clamp, clamp).astype(np.float32)


def truncate_to_max_duration(
    eeg: np.ndarray,
    sr: int = TARGET_SR,
    max_duration_s: int = MAX_DURATION_S,
) -> np.ndarray:
    return eeg[:, : max_duration_s * sr]


def extract_windows(
    eeg: np.ndarray,
    audio_segment: np.ndarray,
    song_label: int,
    window_size: int = WINDOW_SIZE,
    stride: int = TRAIN_STRIDE,
    delay_samples: int = DELAY_SAMPLES,
    audio_window_size: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window over (eeg, audio) with optional EEG delay alignment.
    Returns (eeg_windows, audio_windows, labels).
    """
    n_channels, n_samples = eeg.shape
    if audio_window_size is None:
        audio_window_size = int(window_size * 22050 / TARGET_SR)

    audio_sr = 22050
    n_audio = len(audio_segment)
    eeg_windows, audio_windows, labels = [], [], []

    for start in range(0, n_samples - window_size - delay_samples + 1, stride):
        eeg_win = eeg[:, start + delay_samples: start + delay_samples + window_size]
        audio_start = int(start * audio_sr / TARGET_SR)
        audio_end = audio_start + audio_window_size
        if audio_end > n_audio:
            break
        eeg_windows.append(eeg_win)
        audio_windows.append(audio_segment[audio_start:audio_end])
        labels.append(song_label)

    if not eeg_windows:
        return (
            np.empty((0, n_channels, window_size), dtype=np.float32),
            np.empty((0, audio_window_size), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    return (
        np.stack(eeg_windows).astype(np.float32),
        np.stack(audio_windows).astype(np.float32),
        np.array(labels, dtype=np.int64),
    )


def split_dataset(
    eeg_windows: np.ndarray,
    audio_windows: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val split."""
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )
    return (
        eeg_windows[train_idx], eeg_windows[val_idx],
        audio_windows[train_idx], audio_windows[val_idx],
        labels[train_idx], labels[val_idx],
    )


def preprocess_subject(
    data_dir: str | Path,
    subject_id: int,
    audio_dict: dict[int, np.ndarray],
    stride: int = TRAIN_STRIDE,
    delay_samples: int = DELAY_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full preprocessing pipeline for a single subject."""
    eeg_raw, song_sequence = load_nmedt_eeg(data_dir, subject_id)
    eeg_ds = downsample_eeg(eeg_raw)
    eeg_norm = normalize_eeg(eeg_ds)

    factor = ORIG_SR // TARGET_SR
    song_seq_ds = song_sequence[::factor][: eeg_ds.shape[1]]

    all_eeg, all_audio, all_labels = [], [], []
    for song_id in np.unique(song_seq_ds):
        mask = song_seq_ds == song_id
        song_eeg = eeg_norm[:, mask]
        song_eeg = truncate_to_max_duration(song_eeg)
        if int(song_id) not in audio_dict:
            continue
        eeg_w, audio_w, lbl = extract_windows(
            song_eeg, audio_dict[int(song_id)], int(song_id),
            stride=stride, delay_samples=delay_samples,
        )
        if len(eeg_w) > 0:
            all_eeg.append(eeg_w)
            all_audio.append(audio_w)
            all_labels.append(lbl)

    return (
        np.concatenate(all_eeg),
        np.concatenate(all_audio),
        np.concatenate(all_labels),
    )
