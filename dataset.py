from __future__ import annotations

import re
from collections import deque
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


LABELS_SEED4 = [
    [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
]


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


def list_subject_ids(data_root: Path, session_id: int) -> List[int]:
    session_dir = Path(data_root) / str(session_id)
    subject_ids = sorted({parse_subject_id(p) for p in session_dir.glob("*.mat")})
    if not subject_ids:
        raise FileNotFoundError(f"no .mat files found in {session_dir}")
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
        matched = re.findall(r".*_eeg(\d+)", key)
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


class SEEDIVRawTrialDataset(Dataset):
    """One item = one trial with variable number of non-overlap windows.

    Returned sample:
    - x: (W, C, T) where W is number of windows, C=num_channel, T=chunk_size
    - y: int label in [0,3]
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

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.num_channel <= 0:
            raise ValueError("num_channel must be > 0")

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.index: List[TrialIndex] = []
        self._mat_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._mat_cache_order: Deque[str] = deque()

        self._build_index()
        if not self.index:
            raise RuntimeError("no trials found; check --root and file naming")

    def _build_index(self) -> None:
        for session_id in self.sessions:
            session_dir = self.root / str(session_id)
            if not session_dir.exists():
                raise FileNotFoundError(f"missing session folder: {session_dir}")

            for mat_path in sorted(session_dir.glob("*.mat")):
                try:
                    subject_id = parse_subject_id(mat_path)
                except ValueError:
                    continue

                if self.subject_ids is not None and subject_id not in self.subject_ids:
                    continue

                name_parts = mat_path.stem.split("_")
                date = name_parts[1] if len(name_parts) > 1 else "unknown"
                samples = _load_mat(str(mat_path))
                trial_keys = _extract_trial_keys(samples)
                session_labels = LABELS_SEED4[session_id - 1]
                trial_map: Dict[int, str] = {}
                for trial_key, trial_id in trial_keys:
                    if 1 <= trial_id <= 24:
                        if trial_id in trial_map:
                            raise ValueError(f"duplicate trial id={trial_id} in {mat_path}")
                        trial_map[trial_id] = trial_key

                expected_trials = set(range(1, 25))
                if set(trial_map.keys()) != expected_trials:
                    raise ValueError(
                        f"{mat_path} trial keys mismatch: got {sorted(trial_map.keys())}, expected 1..24"
                    )

                for trial_id in range(1, 25):
                    key = (session_id, subject_id, trial_id)
                    if self.trial_filter is not None and key not in self.trial_filter:
                        continue
                    self.index.append(
                        TrialIndex(
                            file_path=str(mat_path),
                            session_id=session_id,
                            subject_id=subject_id,
                            date=date,
                            trial_key=trial_map[trial_id],
                            trial_id=trial_id,
                            label=int(session_labels[trial_id - 1]),
                        )
                    )

    def __len__(self) -> int:
        return len(self.index)

    def _cache_path(self, ti: TrialIndex) -> Path:
        assert self.cache_dir is not None
        z_flag = 1 if self.per_channel_zscore else 0
        return (
            self.cache_dir
            / (
                f"s{ti.session_id}_sub{ti.subject_id}_{ti.date}_trial{ti.trial_id}_"
                f"w{self.chunk_size}_c{self.num_channel}_z{z_flag}.npz"
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
