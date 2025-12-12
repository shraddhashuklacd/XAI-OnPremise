from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import statsmodels.api as sm

LINEAR_MODEL_NAME = "LinearRegression"
LOGISTIC_MODEL_NAME = "LogisticRegression"

# Final output schemas
LINEAR_FINAL_COLS = [
    "model_name",
    "transformation",
    "target",
    "original_variable",
    "encoded_variable",
    "coef",
    "intercept",
    "p_value",
    "n_obs",
    "r_squared",     
]

LOGISTIC_FINAL_COLS = [
    "model_name",
    "transformation",
    "target",
    "original_variable",
    "encoded_variable",
    "coef",
    "intercept",
    "p_value",
    "r_squared",     
    "n_obs",
]


def _extract_original_name(col: str) -> str:
    return col.split("_")[0]


def _ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clean_xy(
    x: pd.Series | pd.DataFrame,
    y: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, pd.Series]:
    if isinstance(x, pd.Series):
        x = x.to_frame(name=x.name)

    target_col = y.name

    df = pd.concat([x, y], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    x_clean = df.drop(columns=[target_col])
    y_clean = df[target_col]

    return x_clean, y_clean


def run_bivariate_regression(
    X_raw: pd.DataFrame,
    X_minmax: pd.DataFrame,
    y: pd.Series | np.ndarray,
    output_dir: str | Path,
    target_name: str = "__TARGET__",
    transformation_label_raw: str = "original",
    transformation_label_mm: str = "min-max",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = _ensure_dir(output_dir)

    X_raw = X_raw.copy()
    X_minmax = X_minmax.copy()
    y = pd.Series(y).rename(target_name)

    raw_rows: List[Dict[str, Any]] = []

    for col in X_raw.columns:
        x_col = X_raw[col]

        if x_col.nunique(dropna=True) <= 1:
            raw_rows.append(
                {
                    "model_name": LINEAR_MODEL_NAME,
                    "transformation": transformation_label_raw,
                    "target": y.name,
                    "original_variable": _extract_original_name(col),
                    "encoded_variable": col,
                    "coef": np.nan,
                    "intercept": np.nan,
                    "std_err": np.nan,
                    "t_value": np.nan,
                    "p_value": np.nan,
                    "r_squared": np.nan,
                    #"adj_r_squared": np.nan,
                    "n_obs": int(x_col.notna().sum()),
                    "mean_x": float(x_col.mean(skipna=True)),
                    "mean_y": float(y.mean(skipna=True)),
                    "elasticity_at_mean": np.nan,
                    "note": "Skipped: zero variance",
                }
            )
            continue

        try:
            X_clean, y_clean = _clean_xy(x_col, y)

            if X_clean.empty or y_clean.empty:
                raise ValueError("Empty data after cleaning")

            X_design = sm.add_constant(X_clean, has_constant="add")
            model = sm.OLS(y_clean, X_design)
            result = model.fit()

            coef = result.params.get(col, np.nan)
            intercept = result.params.get("const", np.nan)
            std_err = result.bse.get(col, np.nan)
            t_value = result.tvalues.get(col, np.nan)
            p_value = result.pvalues.get(col, np.nan)

            r2 = float(result.rsquared) if hasattr(result, "rsquared") else np.nan
            adj_r2 = float(result.rsquared_adj) if hasattr(result, "rsquared_adj") else np.nan

            mean_x = float(X_clean[col].mean(skipna=True))
            mean_y = float(y_clean.mean(skipna=True))

            if mean_y == 0 or np.isnan(mean_y):
                elasticity = np.nan
            else:
                elasticity = float(coef) * (mean_x / mean_y)

            raw_rows.append(
                {
                    "model_name": LINEAR_MODEL_NAME,
                    "transformation": transformation_label_raw,
                    "target": y.name,
                    "original_variable": _extract_original_name(col),
                    "encoded_variable": col,
                    "coef": float(coef),
                    "intercept": float(intercept),
                    "std_err": float(std_err),
                    "t_value": float(t_value),
                    "p_value": float(p_value),
                    #"r_squared": r2,
                    "r_squared": adj_r2,
                    #"adj_r_squared": adj_r2,
                    "n_obs": int(len(y_clean)),
                    "mean_x": mean_x,
                    "mean_y": mean_y,
                    "elasticity_at_mean": elasticity if not np.isnan(elasticity) else np.nan,
                    "note": "",
                }
            )
        except Exception as e:
            raw_rows.append(
                {
                    "model_name": LINEAR_MODEL_NAME,
                    "transformation": transformation_label_raw,
                    "target": y.name,
                    "original_variable": _extract_original_name(col),
                    "encoded_variable": col,
                    "coef": np.nan,
                    "intercept": np.nan,
                    "std_err": np.nan,
                    "t_value": np.nan,
                    "p_value": np.nan,
                    "r_squared": np.nan,
                    #"adj_r_squared": np.nan,
                    "n_obs": int(x_col.notna().sum()),
                    "mean_x": float(x_col.mean(skipna=True)),
                    "mean_y": float(y.mean(skipna=True)),
                    "elasticity_at_mean": np.nan,
                    "note": f"Failed: {e}",
                }
            )

    raw_df = pd.DataFrame(raw_rows)

    if "p_value" in raw_df.columns:
        raw_df["__p_sort__"] = raw_df["p_value"].fillna(1e9)
        raw_df = (
            raw_df.sort_values("__p_sort__")
            .drop(columns="__p_sort__")
            .reset_index(drop=True)
        )

    raw_path = out_dir / "driver_analysis_coefficients.csv"
    raw_df_final = raw_df[LINEAR_FINAL_COLS]
    raw_df_final.to_csv(raw_path, index=False)
    print(f"Saved driver analysis coefficients (RAW) to: {raw_path}")

    mm_rows: List[Dict[str, Any]] = []

    for col in X_minmax.columns:
        x_col = X_minmax[col]

        if x_col.nunique(dropna=True) <= 1:
            mm_rows.append(
                {
                    "model_name": LINEAR_MODEL_NAME,
                    "transformation": transformation_label_mm,
                    "target": y.name,
                    "original_variable": _extract_original_name(col),
                    "encoded_variable": col,
                    "coef": np.nan,
                    "intercept": np.nan,
                    "std_err": np.nan,
                    "t_value": np.nan,
                    "p_value": np.nan,
                    "r_squared": np.nan,
                    #"adj_r_squared": np.nan,
                    "n_obs": int(x_col.notna().sum()),
                    "mean_x": float(x_col.mean(skipna=True)),
                    "mean_y": float(y.mean(skipna=True)),
                    "elasticity_at_mean": np.nan,
                    "note": "Skipped: zero variance",
                }
            )
            continue

        try:
            X_clean, y_clean = _clean_xy(x_col, y)

            if X_clean.empty or y_clean.empty:
                raise ValueError("Empty data after cleaning")

            X_design = sm.add_constant(X_clean, has_constant="add")
            model = sm.OLS(y_clean, X_design)
            result = model.fit()

            coef = result.params.get(col, np.nan)
            intercept = result.params.get("const", np.nan)
            std_err = result.bse.get(col, np.nan)
            t_value = result.tvalues.get(col, np.nan)
            p_value = result.pvalues.get(col, np.nan)

            r2 = float(result.rsquared) if hasattr(result, "rsquared") else np.nan
            adj_r2 = float(result.rsquared_adj) if hasattr(result, "rsquared_adj") else np.nan

            mean_x = float(X_clean[col].mean(skipna=True))
            mean_y = float(y_clean.mean(skipna=True))

            if mean_y == 0 or np.isnan(mean_y):
                elasticity = np.nan
            else:
                elasticity = float(coef) * (mean_x / mean_y)

            mm_rows.append(
                {
                    "model_name": LINEAR_MODEL_NAME,
                    "transformation": transformation_label_mm,
                    "target": y.name,
                    "original_variable": _extract_original_name(col),
                    "encoded_variable": col,
                    "coef": float(coef),
                    "intercept": float(intercept),
                    "std_err": float(std_err),
                    "t_value": float(t_value),
                    "p_value": float(p_value),
                    #"r_squared": r2,
                    "r_squared": adj_r2,
                    #"adj_r_squared": adj_r2,
                    "n_obs": int(len(y_clean)),
                    "mean_x": mean_x,
                    "mean_y": mean_y,
                    "elasticity_at_mean": elasticity if not np.isnan(elasticity) else np.nan,
                    "note": "",
                }
            )
        except Exception as e:
            mm_rows.append(
                {
                    "model_name": LINEAR_MODEL_NAME,
                    "transformation": transformation_label_mm,
                    "target": y.name,
                    "original_variable": _extract_original_name(col),
                    "encoded_variable": col,
                    "coef": np.nan,
                    "intercept": np.nan,
                    "std_err": np.nan,
                    "t_value": np.nan,
                    "p_value": np.nan,
                    "r_squared": np.nan,
                    #"adj_r_squared": np.nan,
                    "n_obs": int(x_col.notna().sum()),
                    "mean_x": float(x_col.mean(skipna=True)),
                    "mean_y": float(y.mean(skipna=True)),
                    "elasticity_at_mean": np.nan,
                    "note": f"Failed: {e}",
                }
            )

    mm_df = pd.DataFrame(mm_rows)

    if "p_value" in mm_df.columns:
        mm_df["__p_sort__"] = mm_df["p_value"].fillna(1e9)
        mm_df = (
            mm_df.sort_values("__p_sort__")
            .drop(columns="__p_sort__")
            .reset_index(drop=True)
        )

    mm_path = out_dir / "driver_analysis_coefficients_minmax.csv"
    mm_df_final = mm_df[LINEAR_FINAL_COLS]
    mm_df_final.to_csv(mm_path, index=False)
    print(f"Saved driver analysis coefficients (MIN-MAX) to: {mm_path}")
    return raw_df, mm_df


def run_bivariate_logistic(
    X_raw: pd.DataFrame,
    y: pd.Series | np.ndarray,
    output_dir: str | Path,
    target_name: str = "__TARGET__",
    file_suffix: str = "",
    transformation_label: str = "original",
) -> pd.DataFrame:
    out_dir = _ensure_dir(output_dir)

    X_raw = X_raw.copy()
    y = pd.Series(y).rename(target_name)

    uniq = sorted(y.dropna().unique().tolist())
    if len(uniq) != 2:
        print("Warning: Logistic regression expects binary target. Unique values:", uniq)

    rows: List[Dict[str, Any]] = []

    for col in X_raw.columns:
        x_col = X_raw[col]

        if x_col.nunique(dropna=True) <= 1:
            rows.append(
                {
                    "model_name": LOGISTIC_MODEL_NAME,
                    "transformation": transformation_label,
                    "target": y.name,
                    "original_variable": _extract_original_name(col),
                    "encoded_variable": col,
                    "coef": np.nan,
                    "intercept": np.nan,
                    "std_err": np.nan,
                    "z_value": np.nan,
                    "p_value": np.nan,
                    #"pseudo_r_squared": np.nan,
                    "r_squared": np.nan,
                    "n_obs": int(x_col.notna().sum()),
                    "mean_x": float(x_col.mean(skipna=True)),
                    "mean_y": float(y.mean(skipna=True)),
                    "note": "Skipped: zero variance",
                }
            )
            continue

        try:
            X_clean, y_clean = _clean_xy(x_col, y)

            if X_clean.empty or y_clean.empty:
                raise ValueError("Empty data after cleaning")

            if y_clean.nunique(dropna=True) < 2:
                raise ValueError("Only one class present after cleaning")

            X_design = sm.add_constant(X_clean, has_constant="add")
            model = sm.Logit(y_clean, X_design)
            result = model.fit(disp=False)

            coef = result.params.get(col, np.nan)
            intercept = result.params.get("const", np.nan)
            std_err = result.bse.get(col, np.nan)
            z_value = result.tvalues.get(col, np.nan)
            p_value = result.pvalues.get(col, np.nan)

            pseudo_r2 = float(result.prsquared) if hasattr(result, "prsquared") else np.nan

            rows.append(
                {
                    "model_name": LOGISTIC_MODEL_NAME,
                    "transformation": transformation_label,
                    "target": y.name,
                    "original_variable": _extract_original_name(col),
                    "encoded_variable": col,
                    "coef": float(coef),
                    "intercept": float(intercept),
                    "std_err": float(std_err),
                    "z_value": float(z_value),
                    "p_value": float(p_value),
                    #"pseudo_r_squared": pseudo_r2,
                    "r_squared": pseudo_r2,
                    "n_obs": int(len(y_clean)),
                    "mean_x": float(X_clean[col].mean(skipna=True)),
                    "mean_y": float(y_clean.mean(skipna=True)),
                    "note": "",
                }
            )
        except Exception as e:
            rows.append(
                {
                    "model_name": LOGISTIC_MODEL_NAME,
                    "transformation": transformation_label,
                    "target": y.name,
                    "original_variable": _extract_original_name(col),
                    "encoded_variable": col,
                    "coef": np.nan,
                    "intercept": np.nan,
                    "std_err": np.nan,
                    "z_value": np.nan,
                    "p_value": np.nan,
                    #"pseudo_r_squared": np.nan,
                    "r_squared": np.nan,
                    "n_obs": int(x_col.notna().sum()),
                    "mean_x": float(x_col.mean(skipna=True)),
                    "mean_y": float(y.mean(skipna=True)),
                    "note": f"Failed: {e}",
                }
            )

    df = pd.DataFrame(rows)

    if "p_value" in df.columns:
        df["__p_sort__"] = df["p_value"].fillna(1e9)
        df = (
            df.sort_values("__p_sort__")
            .drop(columns="__p_sort__")
            .reset_index(drop=True)
        )

    if transformation_label == "min-max":
        out_path = out_dir / "driver_analysis_coefficients_minmax.csv"
    else:
        out_path = out_dir / "driver_analysis_coefficients.csv"

    df_final = df[LOGISTIC_FINAL_COLS]
    df_final.to_csv(out_path, index=False)
    print(f"Saved driver analysis coefficients (Logistic, {transformation_label}) to: {out_path}")

    return df
