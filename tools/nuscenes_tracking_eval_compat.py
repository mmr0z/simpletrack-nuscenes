#!/usr/bin/env python3
"""Launch the official nuScenes evaluator with Python 3.10 compatibility only."""

from __future__ import annotations

import collections
import collections.abc
import runpy


# motmetrics 1.1.3 is the API expected by nuscenes-devkit 1.1.11, but it still
# imports Iterable from collections. Python 3.10 moved it to collections.abc.
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable  # type: ignore[attr-defined]


if __name__ == "__main__":
    runpy.run_module("nuscenes.eval.tracking.evaluate", run_name="__main__")
