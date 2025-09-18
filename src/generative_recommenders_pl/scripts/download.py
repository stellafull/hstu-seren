from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generative_recommenders_pl.data.download import DATASETS, DEFAULT_ROOT, download_dataset
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)


def main(argv: Iterable[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    dataset_names = args or list(DATASETS)
    root = DEFAULT_ROOT

    log.info("Ensuring datasets %s in %s", ", ".join(dataset_names), root)

    results = {name: download_dataset(name, root=root) for name in dataset_names}
    failed = [name for name, ok in results.items() if not ok]

    for name, ok in results.items():
        log.info("%s: %s", name, "ok" if ok else "failed")

    if failed:
        log.error("Failed to download datasets: %s", ", ".join(failed))
        return 1

    log.info("All requested datasets are available at %s", root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
