"""
length_filter.py
-----------------
Filters reads/sequences to the expected VHH amplicon length window.

Default: 397 bp +/- 10% -> [357, 437] bp (configurable).
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def compute_length_window(expected_length: int = 397, tolerance: float = 0.10) -> tuple[int, int]:
    """Return (min_len, max_len) given an expected length and a fractional tolerance."""
    delta = expected_length * tolerance
    min_len = int(round(expected_length - delta))
    max_len = int(round(expected_length + delta))
    return min_len, max_len


def filter_by_length(
    df: pd.DataFrame,
    expected_length: int = 397,
    tolerance: float = 0.10,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Filter a sequence DataFrame to the expected amplicon length window.

    If min_length/max_length are explicitly given they override the
    expected_length/tolerance computation (lets users hard-set a window,
    e.g. after inspecting the length histogram and spotting a secondary
    peak from a different construct).

    Returns (passed_df, failed_df, report_dict).
    """
    if min_length is None or max_length is None:
        min_length, max_length = compute_length_window(expected_length, tolerance)

    mask = (df["length"] >= min_length) & (df["length"] <= max_length)
    passed, failed = df[mask].copy(), df[~mask].copy()

    report = {
        "expected_length": expected_length,
        "tolerance": tolerance,
        "min_length": min_length,
        "max_length": max_length,
        "n_input": len(df),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "pct_passed": round(100 * len(passed) / len(df), 2) if len(df) else 0.0,
    }
    logger.info(
        "Length filter [%d-%d bp]: %d/%d passed (%.1f%%)",
        min_length, max_length, len(passed), len(df), report["pct_passed"],
    )
    return passed, failed, report
