import re
from typing import Any
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

def detect_datatypes(
    df: pd.DataFrame,
    cat_max_unique: int = 10,
    cat_ratio_threshold: float = 0.05,
) -> pd.DataFrame:

    rows = []

    for col in df.columns:
        s = df[col]

        nn = int(s.notna().sum())
        un = int(s.dropna().nunique())
        pct_unique = float(un / nn) if nn else np.nan

        if is_bool_dtype(s):
            logical = "bool"
        elif is_datetime64_any_dtype(s):
            logical = "datetime"
        elif is_timedelta64_dtype(s):
            logical = "timedelta"
        elif is_integer_dtype(s):
            logical = "integer"
        elif is_float_dtype(s):
            logical = "float"
        elif is_numeric_dtype(s):
            logical = "numeric"
        elif is_string_dtype(s) or s.dtype == "object":
            sample = s.dropna().astype(str).head(100)
            numeric_pattern = r"^\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*$"
            ratio = sample.str.match(numeric_pattern).mean()
            logical = "float_like_string" if ratio > 0.9 else "string"
        else:
            logical = str(s.dtype)

        is_id = any(key in col.lower() for key in ["id", "uuid", "guid", "code", "key"])

        likely_cat = False
        if logical in {"bool", "string", "category"} or "string" in logical:
            likely_cat = True
        elif logical == "integer":
            if (un <= cat_max_unique) and (pct_unique <= cat_ratio_threshold):
                likely_cat = True

        if logical == "integer":
            value_nature = "binary_integer" if un == 2 else "integer_distinct"
        elif logical in {"float", "numeric", "float_like_string"}:
            value_nature = "binary_numeric" if un == 2 else "continuous"
        else:
            value_nature = None

        # High cardinality for nominal
        is_high_cardinality = likely_cat and (un > cat_max_unique)

        use_for_model = True
        if is_id:
            use_for_model = False
        if logical in {"datetime", "timedelta"}:
            use_for_model = False
        if is_high_cardinality:
            # Nominal with cardinality > cat_max_unique -> exclude from model
            use_for_model = False
        needs_minmax = False
        if use_for_model and not likely_cat:
            needs_minmax = True

        rows.append(
            {
                "column": col,
                "logical_type": logical,
                "n_unique": un,
                "pct_unique": pct_unique,
                "value_nature": value_nature,
                "likely_categorical": likely_cat,
                "is_high_cardinality": is_high_cardinality,
                "is_identifier": is_id,
                "use_for_model": use_for_model,
                "needs_minmax_scaling": needs_minmax,
            }
        )

    return pd.DataFrame(rows)

def build_column_summary(df: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:

    rows = []
    n_rows = len(df)
    schema_idx = schema.set_index("column")

    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)

        # Basic summary
        non_null = int(s.notna().sum())
        missing = int(s.isna().sum())
        missing_pct = float(missing / n_rows * 100) if n_rows else np.nan

        unique_count = int(s.nunique(dropna=True))
        uniques = s.dropna().astype(str).unique()
        example_values = ", ".join(uniques[:3])

        # Determine categorical/numeric
        if col in schema_idx.index:
            is_cat = bool(schema_idx.loc[col, "likely_categorical"])
            high_logical = "categorical" if is_cat else "numeric"
            is_high_card = bool(schema_idx.loc[col, "is_high_cardinality"])
        else:
            high_logical = "categorical" if s.dtype == "object" else "numeric"
            is_cat = (high_logical == "categorical")
            is_high_card = False

        if not is_cat:
            unique_val = ""
            cardinality_val = ""
        else:
            unique_val = unique_count
            cardinality_val = unique_count

        if not is_cat:
            num = pd.to_numeric(s, errors="coerce")

            mean_val = float(num.mean(skipna=True))
            median_val = float(num.median(skipna=True))
            mode_val = (
                float(num.mode(dropna=True).iloc[0])
                if not num.mode(dropna=True).empty else np.nan
            )
            min_val = float(num.min(skipna=True))
            max_val = float(num.max(skipna=True))
            pct95 = float(num.quantile(0.95))
            pct99 = float(num.quantile(0.99))

            q1 = num.quantile(0.25)
            q3 = num.quantile(0.75)
            iqr_val = float(q3 - q1)

        else:
            mean_val = ""
            median_val = ""
            mode_val = (
                s.mode(dropna=True).astype(str).iloc[0]
                if not s.mode().empty else ""
            )
            min_val = ""
            max_val = ""
            pct95 = ""
            pct99 = ""
            iqr_val = ""

        rows.append(
            {
                "column": col,
                "dtype": dtype_str,
                "logical_type": high_logical,
                "non_null": non_null,
                "missing": missing,
                "missing_%": missing_pct,
                "unique": unique_val,
                "cardinality": cardinality_val,
                "example_values": example_values,
                "is_high_cardinality": is_high_card,
                "Mean": mean_val,
                "Median": median_val,
                "Mode": mode_val,
                "Minimum": min_val,
                "Maximum": max_val,
                "95th Percentile": pct95,
                "99th Percentile": pct99,
                "Inter Quartile Range": iqr_val,
            }
        )

    return pd.DataFrame(rows)


def sanitize_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    num_re = re.compile(r"^\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*$")

    def convert(val: Any) -> Any:
        if isinstance(val, str):
            cleaned = (
                val.replace(",", "")
                .replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .strip()
            )
            if num_re.match(cleaned):
                try:
                    return float(cleaned)
                except ValueError:
                    return val
        return val
    return df.applymap(convert)

def parse_bracketed_numeric_series(s: pd.Series) -> pd.Series:

    if pd.api.types.is_numeric_dtype(s):
        return s

    cleaned = (
        s.astype(str)
        .str.replace(r"[()\[\]]", "", regex=True)
        .str.replace(",", "")
    )

    return pd.to_numeric(cleaned, errors="coerce")


def yes_like(value: Any) -> bool:
    return str(value).strip().lower() in {"y", "yes", "1", "true", "t"}
