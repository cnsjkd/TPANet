from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


class CSVLogger:
    def __init__(self, path: Path, headers: Sequence[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.headers = list(headers)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, row: Iterable[object]) -> None:
        with self.path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(row))
