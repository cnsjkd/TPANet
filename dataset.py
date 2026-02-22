from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import scipy.io as scio
except Exception:
    scio = None

try:
    import h5py
except Exception:
    h5py = None


NUM_SEED_SESSIONS = 3
NUM_SEED_TRIALS = 15
SEED_LABEL_MAP = {-1: 0, 0: 1, 1: 2}


@dataclass(frozen=True)
class TrialIndex:
    file_path: str
    session_id: int
    subject_id: int
    date: str
    trial_key: str
    trial_id: int
    label: int


def parse_subject_id(mat_path: Path) -> int:
    try:
        return int(mat_path.stem.split("_", 1)[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"invalid subject id in filename: {mat_path.name}") from exc


def parse_subject_date(mat_path: Path) -> str:
    parts = mat_path.stem.split("_", 1)
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"invalid SEED file naming (missing date): {mat_path.name}")
    return parts[1]


def _scan_seed_session_files(data_root: Path) -> Dict[int, Dict[int, Path]]:
    root = Path(data_root)
    mat_files = sorted(p for p in root.glob("*.mat") if p.name != "label.mat")
    if not mat_files:
        raise FileNotFoundError(f"no .mat files found in {root}")

    grouped: Dict[int, List[Tuple[str, Path]]] = defaultdict(list)
    for mat_path in mat_files:
        subject_id = parse_subject_id(mat_path)
        date = parse_subject_date(mat_path)
        grouped[subject_id].append((date, mat_path))

    session_to_subject_files: Dict[int, Dict[int, Path]] = {sid: {} for sid in range(1, NUM_SEED_SESSIONS + 1)}
    for subject_id, date_files in sorted(grouped.items()):
        if len(date_files) != NUM_SEED_SESSIONS:
            raise ValueError(
                f"subject {subject_id} should have exactly {NUM_SEED_SESSIONS} files, got {len(date_files)}"
            )
        date_files_sorted = sorted(date_files, key=lambda x: x[0])
        for session_id, (_, mat_path) in enumerate(date_files_sorted, start=1):
            session_to_subject_files[session_id][subject_id] = mat_path

    return session_to_subject_files


def list_subject_ids(data_root: Path, session_id: int) -> List[int]:
    session_id = int(session_id)
    if session_id < 1 or session_id > NUM_SEED_SESSIONS:
        raise ValueError(f"SEED session_id must be in [1, {NUM_SEED_SESSIONS}], got {session_id}")
    session_to_subject_files = _scan_seed_session_files(Path(data_root))
    subject_ids = sorted(session_to_subject_files[session_id].keys())
    if not subject_ids:
        raise FileNotFoundError(f"no subject files found for session {session_id} under {data_root}")
    return subject_ids


def list_common_subject_ids(data_root: Path, sessions: Sequence[int]) -> List[int]:
    subject_sets = [set(list_subject_ids(Path(data_root), int(s))) for s in sessions]
    if not subject_sets:
        return []
    return sorted(set.intersection(*subject_sets))


def _load_mat(file_path: str) -> Dict[str, np.ndarray]:
    if scio is not None:
        try:
            return scio.loadmat(file_path, verify_compressed_data_integrity=False)
        except NotImplementedError:
            pass
        except Exception:
            pass

    if h5py is None:
        raise RuntimeError("failed to read .mat with scipy; install h5py for MATLAB v7.3 files")

    out: Dict[str, np.ndarray] = {}
    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            try:
                out[key] = np.array(f[key])
            except Exception:
                continue
    return out


def _extract_trial_keys(samples: Dict[str, np.ndarray]) -> List[Tuple[str, int]]:
    trial_name_ids: List[Tuple[str, int]] = []
    for key in samples.keys():
        if not isinstance(key, str):
            continue
        if "eeg" not in key:
            continue
        matched = re.findall(r".*_eeg(\d+)$", key)
        if not matched:
            continue
        trial_name_ids.append((key, int(matched[0])))
    trial_name_ids.sort(key=lambda x: x[1])
    return trial_name_ids


def _align_trial_shape(trial: np.ndarray, num_channel: int) -> np.ndarray:
    arr = np.asarray(trial)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"unexpected trial shape: {arr.shape}")

    if arr.shape[0] == num_channel:
        out = arr
    elif arr.shape[1] == num_channel:
        out = arr.T
    else:
        raise ValueError(f"cannot align trial shape {arr.shape} to channels={num_channel}")

    return out.astype(np.float32, copy=False)


def _load_seed_labels(data_root: Path) -> List[int]:
    label_path = Path(data_root) / "label.mat"
    if not label_path.exists():
        raise FileNotFoundError(f"missing SEED label file: {label_path}")

    samples = _load_mat(str(label_path))
    if "label" not in samples:
        raise KeyError(f"'label' not found in {label_path}")

    labels_raw = np.asarray(samples["label"]).reshape(-1)
    if labels_raw.size != NUM_SEED_TRIALS:
        raise ValueError(f"{label_path} should contain {NUM_SEED_TRIALS} labels, got {labels_raw.size}")

    labels: List[int] = []
    for v in labels_raw.tolist():
        iv = int(v)
        if iv not in SEED_LABEL_MAP:
            raise ValueError(f"unexpected SEED label value {iv}, expected one of {sorted(SEED_LABEL_MAP.keys())}")
        labels.append(SEED_LABEL_MAP[iv])
    return labels


class SEEDIVRawTrialDataset(Dataset):
    """One item = one trial with variable number of non-overlap windows.

    Returned sample:
    - x: (W, C, T) where W is number of windows, C=num_channel, T=chunk_size
    - y: int label in [0,2]
    - meta: TrialIndex
    """

    def __init__(
        self,
        root: str | Path,
        sessions: Sequence[int] = (1, 2, 3),
        subject_ids: Optional[Sequence[int]] = None,
        trial_filter: Optional[Sequence[Tuple[int, int, int]]] = None,
        chunk_size: int = 800,
        num_channel: int = 62,
        cache_dir: Optional[str | Path] = None,
        per_channel_zscore: bool = True,
        mat_cache_size: int = 8,
        channel_mean: Optional[np.ndarray] = None,
        channel_std: Optional[np.ndarray] = None,
        normalization_tag: Optional[str] = None,
    ) -> None:
        self.root = Path(root)
        self.sessions = [int(s) for s in sessions]
        self.subject_ids = set(subject_ids) if subject_ids is not None else None
        self.trial_filter: Optional[Set[Tuple[int, int, int]]] = (
            set((int(s), int(sub), int(t)) for s, sub, t in trial_filter) if trial_filter is not None else None
        )
        self.chunk_size = int(chunk_size)
        self.num_channel = int(num_channel)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.per_channel_zscore = bool(per_channel_zscore)
        self.mat_cache_size = int(mat_cache_size)
        self.channel_mean: Optional[np.ndarray] = None
        self.channel_std: Optional[np.ndarray] = None

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.num_channel <= 0:
            raise ValueError("num_channel must be > 0")
        if self.per_channel_zscore and (channel_mean is not None or channel_std is not None):
            raise ValueError("per_channel_zscore and channel_mean/channel_std are mutually exclusive")
        if (channel_mean is None) != (channel_std is None):
            raise ValueError("channel_mean and channel_std must be both set or both None")
        if channel_mean is not None and channel_std is not None:
            mean = np.asarray(channel_mean, dtype=np.float32).reshape(-1)
            std = np.asarray(channel_std, dtype=np.float32).reshape(-1)
            if mean.shape[0] != self.num_channel or std.shape[0] != self.num_channel:
                raise ValueError(
                    f"channel_mean/std shape mismatch: got {mean.shape[0]}/{std.shape[0]}, "
                    f"expected {self.num_channel}"
                )
            self.channel_mean = mean
            self.channel_std = std
        if normalization_tag is None:
            if self.per_channel_zscore:
                normalization_tag = "trialz"
            elif self.channel_mean is not None:
                normalization_tag = "trainz"
            else:
                normalization_tag = "none"
        self.normalization_tag = re.sub(r"[^0-9A-Za-z_.-]+", "-", str(normalization_tag))[:40] or "none"

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.index: List[TrialIndex] = []
        self._mat_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._mat_cache_order: Deque[str] = deque()

        self._build_index()
        if not self.index:
            raise RuntimeError("no trials found; check --root and file naming")

    def _build_index(self) -> None:
        session_to_subject_files = _scan_seed_session_files(self.root)
        labels = _load_seed_labels(self.root)

        for session_id in self.sessions:
            sid = int(session_id)
            if sid < 1 or sid > NUM_SEED_SESSIONS:
                raise ValueError(f"SEED session id must be in [1, {NUM_SEED_SESSIONS}], got {sid}")

            subject_files = session_to_subject_files[sid]
            for subject_id, mat_path in sorted(subject_files.items()):
                if self.subject_ids is not None and subject_id not in self.subject_ids:
                    continue

                date = parse_subject_date(mat_path)
                samples = _load_mat(str(mat_path))
                trial_keys = _extract_trial_keys(samples)
                trial_map: Dict[int, str] = {}
                for trial_key, trial_id in trial_keys:
                    if 1 <= trial_id <= NUM_SEED_TRIALS:
                        if trial_id in trial_map:
                            raise ValueError(f"duplicate trial id={trial_id} in {mat_path}")
                        trial_map[trial_id] = trial_key

                expected_trials = set(range(1, NUM_SEED_TRIALS + 1))
                if set(trial_map.keys()) != expected_trials:
                    raise ValueError(
                        f"{mat_path} trial keys mismatch: got {sorted(trial_map.keys())}, "
                        f"expected 1..{NUM_SEED_TRIALS}"
                    )

                for trial_id in range(1, NUM_SEED_TRIALS + 1):
                    key = (sid, subject_id, trial_id)
                    if self.trial_filter is not None and key not in self.trial_filter:
                        continue
                    self.index.append(
                        TrialIndex(
                            file_path=str(mat_path),
                            session_id=sid,
                            subject_id=subject_id,
                            date=date,
                            trial_key=trial_map[trial_id],
                            trial_id=trial_id,
                            label=int(labels[trial_id - 1]),
                        )
                    )

    def __len__(self) -> int:
        return len(self.index)

    def _cache_path(self, ti: TrialIndex) -> Path:
        assert self.cache_dir is not None
        return (
            self.cache_dir
            / (
                f"s{ti.session_id}_sub{ti.subject_id}_{ti.date}_trial{ti.trial_id}_"
                f"w{self.chunk_size}_c{self.num_channel}_norm{self.normalization_tag}.npz"
            )
        )

    def _load_trial_mat(self, file_path: str) -> Dict[str, np.ndarray]:
        if file_path in self._mat_cache:
            return self._mat_cache[file_path]

        samples = _load_mat(file_path)
        self._mat_cache[file_path] = samples
        self._mat_cache_order.append(file_path)

        if len(self._mat_cache_order) > self.mat_cache_size:
            stale = self._mat_cache_order.popleft()
            if stale in self._mat_cache:
                del self._mat_cache[stale]

        return samples

    def __getitem__(self, index: int):
        ti = self.index[index]

        if self.cache_dir is not None:
            cache_path = self._cache_path(ti)
            if cache_path.exists():
                cached = np.load(cache_path, allow_pickle=False)
                x = cached["x"]
                y = int(cached["y"])
                return torch.from_numpy(x), y, ti

        samples = self._load_trial_mat(ti.file_path)
        trial = samples.get(ti.trial_key)
        if trial is None:
            trial_keys = _extract_trial_keys(samples)
            map_by_id = {tid: key for key, tid in trial_keys}
            fallback_key = map_by_id.get(ti.trial_id)
            if fallback_key is None:
                raise KeyError(f"trial {ti.trial_id} not found in {ti.file_path}")
            trial = samples[fallback_key]

        trial = _align_trial_shape(trial, self.num_channel)
        total_len = trial.shape[1]
        windows = total_len // self.chunk_size
        if windows <= 0:
            raise RuntimeError(f"trial too short: T={total_len} (<{self.chunk_size}) in {ti.file_path}")

        trial = trial[:, : windows * self.chunk_size]
        x = trial.reshape(self.num_channel, windows, self.chunk_size).transpose(1, 0, 2)

        if self.per_channel_zscore:
            mean = x.mean(axis=(0, 2), keepdims=True)
            std = x.std(axis=(0, 2), keepdims=True)
            x = (x - mean) / (std + 1e-6)
        elif self.channel_mean is not None and self.channel_std is not None:
            mean = self.channel_mean[None, :, None]
            std = self.channel_std[None, :, None]
            x = (x - mean) / (std + 1e-6)

        x = x.astype(np.float32, copy=False)
        y = int(ti.label)

        if self.cache_dir is not None:
            np.savez_compressed(self._cache_path(ti), x=x, y=y)

        return torch.from_numpy(x), y, ti


def collate_trials(batch):
    xs, ys, metas = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_windows = int(lengths.max().item())
    channels = int(xs[0].shape[1])
    chunk_size = int(xs[0].shape[2])

    x_pad = torch.zeros((len(xs), max_windows, channels, chunk_size), dtype=torch.float32)
    for i, x in enumerate(xs):
        x_pad[i, : x.shape[0]] = x

    y = torch.tensor(ys, dtype=torch.long)
    return x_pad, lengths, y, list(metas)
