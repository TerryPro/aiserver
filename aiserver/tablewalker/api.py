#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""API functions for TableWalker integration."""

from typing import Optional, Dict, Any
import pandas as pd
from IPython.display import display
from .data_converter import dataframe_to_rows, sample_dataframe
from .field_mapper import generate_raw_fields

# MIME type must match the one defined in the frontend extension
MIME_TYPE = 'application/vnd.kanaries.tablewalker+json'


def show_df(df: pd.DataFrame, max_rows: int = 1000) -> None:
    """Display a DataFrame using TableWalker component.

    This implementation sends data via a custom MIME type, which is rendered
    by the frontend extension.

    Args:
        df: The pandas DataFrame to display.
        max_rows: Maximum number of rows to display. Default is 1000.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(df).__name__}")

    sampled_df = sample_dataframe(df, max_rows)
    rows = dataframe_to_rows(sampled_df)
    raw_fields = generate_raw_fields(df)

    payload = {
        'data': rows,
        'fields': raw_fields
    }

    # Display the data with the custom MIME type
    # We also provide a text/plain representation as a fallback
    display({
        MIME_TYPE: payload,
        'text/plain': str(sampled_df)
    }, raw=True)
