"""Top-K checkpoint keeper. Saves at most K checkpoints, ranked by metric."""
from __future__ import annotations
import os
import heapq
from typing import Any, Dict, List, Tuple
import torch


class TopKCheckpointKeeper:
    def __init__(self, out_dir: str, k: int = 3, metric_lower_better: bool = True):
        self.out_dir = out_dir
        self.k = k
        self.lower = metric_lower_better
        # heap of (sort_key, path) — for lower-better we use -metric so heappush
        # pops worst first when len > k. heapq is a min-heap; we always want
        # to evict the WORST when full.
        self._heap: List[Tuple[float, str]] = []

    def _sort_key(self, metric: float) -> float:
        # Larger key = worse. heapq.heappop returns smallest key = best.
        # We want to evict worst → flip when popping with nlargest.
        return -metric if not self.lower else metric

    def maybe_save(self, metric: float, state: Dict[str, Any], tag: str) -> str:
        """Save state if metric is in current top-K; else no-op.
        Returns path saved (empty string if skipped)."""
        path = os.path.join(self.out_dir, f"best_{tag}_metric{metric:.4f}.pt")
        if len(self._heap) < self.k:
            torch.save(state, path)
            heapq.heappush(self._heap, (self._sort_key(metric), path))
            # also save "best.pt" pointer (full file copy) to current top-1
            self._refresh_best_alias()
            return path

        # Compare with worst-of-current
        worst_key, worst_path = max(self._heap, key=lambda kv: kv[0])
        new_key = self._sort_key(metric)
        if new_key < worst_key:  # better than worst
            torch.save(state, path)
            # remove worst
            self._heap.remove((worst_key, worst_path))
            try:
                os.remove(worst_path)
            except OSError:
                pass
            heapq.heappush(self._heap, (new_key, path))
            self._refresh_best_alias()
            return path
        return ""

    def _refresh_best_alias(self) -> None:
        if not self._heap:
            return
        # best = smallest sort_key
        best_key, best_path = min(self._heap, key=lambda kv: kv[0])
        try:
            alias = os.path.join(self.out_dir, "best.pt")
            if os.path.lexists(alias):
                os.remove(alias)
            # Use symlink if supported; otherwise plain copy
            try:
                os.symlink(os.path.basename(best_path), alias)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copyfile(best_path, alias)
        except OSError:
            pass
