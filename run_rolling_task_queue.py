"""Command entrypoint used by the dashboard's two-slot rolling queue."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.rolling_task_queue import process_rolling_queue


def main() -> None:
    parser = argparse.ArgumentParser(description="顺序执行滚动定参任务队列")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--slot", type=int, default=0, choices=(0, 1), help="队列并行槽位")
    args = parser.parse_args()
    process_rolling_queue(args.root.resolve(), worker_slot=args.slot)


if __name__ == "__main__":
    main()
