#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Field mapping utilities for TableWalker."""

from typing import List, Dict, Any
import pandas as pd


def map_field_types(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Map DataFrame column types to TableWalker field configurations.

    Args:
        df: The pandas DataFrame to analyze.

    Returns:
        A list of field configurations for TableWalker.
    """
    fields = []

    for col, dtype in df.dtypes.items():
        semantic_type = "quantitative"
        if pd.api.types.is_string_dtype(dtype):
            semantic_type = "nominal"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            semantic_type = "temporal"
        elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            semantic_type = "ordinal"

        analytic_type = "measure"
        if semantic_type in ["nominal", "ordinal", "temporal"]:
            analytic_type = "dimension"

        field = {
            "fid": col,
            "name": col,
            "semanticType": semantic_type,
            "analyticType": analytic_type
        }
        fields.append(field)

    return fields


def generate_raw_fields(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate rawFields configuration for TableWalker based on DataFrame.

    Args:
        df: The pandas DataFrame to analyze.

    Returns:
        A list of rawFields configurations.
    """
    return map_field_types(df)
