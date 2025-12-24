#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data conversion utilities for TableWalker."""

from typing import Any, List, Dict
import pandas as pd
import numpy as np


def dataframe_to_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a pandas DataFrame to a list of rows for TableWalker.

    Args:
        df: The pandas DataFrame to convert.

    Returns:
        A list of dictionaries representing the rows.
    """
    rows = df.to_dict(orient="records")

    for row in rows:
        for key, value in row.items():
            if hasattr(value, "item"):
                row[key] = value.item()
            elif isinstance(value, pd.Timestamp):
                row[key] = value.isoformat()
            elif isinstance(value, np.datetime64):
                row[key] = pd.Timestamp(value).isoformat()
            elif isinstance(value, pd.Timedelta):
                row[key] = str(value)
            elif pd.isna(value):
                row[key] = None
            elif not isinstance(value, (int, float, str, bool, type(None))):
                row[key] = str(value)

    return rows


def sample_dataframe(df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
    """Sample a DataFrame to a maximum number of rows.

    Args:
        df: The pandas DataFrame to sample.
        max_rows: The maximum number of rows to return.

    Returns:
        A sampled DataFrame with at most max_rows rows.
    """
    if len(df) <= max_rows:
        return df
    else:
        return df.sample(n=max_rows, random_state=42)
