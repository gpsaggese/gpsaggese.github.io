"""
econml_utils.py

Reusable utilities + lightweight wrapper helpers for:

TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes

What this module is for
-----------------------
- Load cleaned NHANES 2021–2023 "meaningful" CSVs from ./data
- Build a merged, analysis-ready dataframe (one row per respondent)
- Construct:
    Outcomes: sbp_mean, dbp_mean, fasting_glucose_mg_dl
    Treatment: treatment_supplement (1 = any supplement use, 0 = none)
    Covariates: BMI, waist, lipids, hs-CRP, etc.
- Provide get_y_t_x() to safely return (Y, T, X, covariate_cols)

Project convention
------------------
Notebooks (econml.API.ipynb, econml.example.ipynb) should import and call these
utilities instead of embedding preprocessing logic inline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import warnings
import pandas as pd


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

ID_COL = "respondent_sequence_number"
ID_CANDIDATES = [ID_COL, "SEQN", "seqn"]

DEFAULT_FILES: Dict[str, str] = {
    "BPXO": "BPXO_L_meaningful.csv",
    "BMX": "BMX_L_meaningful.csv",
    "TCHOL": "TCHOL_L_meaningful.csv",
    "HDL": "HDL_L_meaningful.csv",
    "TRIGLY": "TRIGLY_L_meaningful.csv",
    "GLU": "GLU_L_meaningful.csv",
    "HSCRP": "HSCRP_L_meaningful.csv",
    "DSQTOT": "DSQTOT_L_meaningful.csv",
    "DEMO": "DEMO_L_meaningful.csv",
}

NA_STRINGS = ["", " ", ".", "NA", "NaN", "nan", "N/A", "missing", "Missing"]


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the participant identifier column is named `respondent_sequence_number`.

    Raises a clear error if no ID column is found.
    """
    found = None
    for candidate in ID_CANDIDATES:
        if candidate in df.columns:
            found = candidate
            break

    if found is None:
        raise KeyError(
            "No respondent ID column found. Expected one of: "
            f"{ID_CANDIDATES}. Found columns: {list(df.columns)[:30]}..."
        )

    if found != ID_COL:
        df = df.rename(columns={found: ID_COL})
    return df


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Expected it inside: {DATA_DIR}\n"
            f"Tip: confirm your cleaned CSVs are copied into ./data/"
        )


def _read_meaningful_csv(filename: str) -> pd.DataFrame:
    """Read a CSV from ./data with robust NA handling + ID normalization."""
    csv_path = DATA_DIR / filename
    _require_file(csv_path)

    df = pd.read_csv(
        csv_path,
        na_values=NA_STRINGS,
        keep_default_na=True,
        low_memory=False,
    )
    df = _normalize_id_column(df)
    return df


def _dedupe_on_id(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """If any component unexpectedly has duplicate respondent IDs, keep the first row."""
    if ID_COL not in df.columns:
        return df
    if df[ID_COL].duplicated().any():
        n_dup = int(df[ID_COL].duplicated().sum())
        warnings.warn(
            f"{name} has {n_dup} duplicate {ID_COL} rows. Keeping first occurrence per ID.",
            RuntimeWarning,
        )
        df = df.drop_duplicates(subset=[ID_COL], keep="first").copy()
    return df


def _to_numeric_safe(s: pd.Series) -> pd.Series:
    """Convert a Series to numeric where possible (errors -> NaN)."""
    return pd.to_numeric(s, errors="coerce")


def _encode_sex(series: pd.Series) -> pd.Series:
    """
    Encode sex to numeric:
      - NHANES common coding: 1=Male, 2=Female
      - or strings ("Male"/"Female")
    Output: 0=Male, 1=Female, NaN otherwise
    """
    if series is None:
        return series

    if pd.api.types.is_numeric_dtype(series):
        return series.map({1: 0, 2: 1})

    cleaned = series.astype(str).str.strip().str.lower()
    mapping = {
        "male": 0, "m": 0, "1": 0,
        "female": 1, "f": 1, "2": 1,
    }
    return cleaned.map(mapping)


def _encode_treatment(series: pd.Series) -> pd.Series:
    """
    Encode treatment to numeric:
      - expected: 1=yes, 2=no  -> 1/0
      - robust to strings "Yes"/"No"
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.map({1: 1, 2: 0})

    cleaned = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1, "yes": 1, "y": 1, "true": 1,
        "2": 0, "no": 0, "n": 0, "false": 0,
    }
    return cleaned.map(mapping)


def _standardize_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create consistent demographic columns WITHOUT breaking the original columns.

    - If `age_in_years_at_screening` exists, create `age_years`
    - If `gender` exists, create `sex` (0=Male, 1=Female)

    This makes downstream API logic more reliable (age bins, etc.).
    """
    df = df.copy()

    # Age
    if "age_years" not in df.columns:
        if "age_in_years_at_screening" in df.columns:
            df["age_years"] = _to_numeric_safe(df["age_in_years_at_screening"])
        elif "age" in df.columns:
            df["age_years"] = _to_numeric_safe(df["age"])

    # Sex
    if "sex" not in df.columns:
        if "gender" in df.columns:
            df["sex"] = _encode_sex(df["gender"])
        elif "riagendr" in df.columns:  # sometimes seen in raw NHANES
            df["sex"] = _encode_sex(df["riagendr"])

    return df


def _basic_range_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight cleaning steps:
    - Numeric coercion for key fields
    - Replace clearly impossible values with NaN
    """
    df = df.copy()

    # Outcomes
    for c in ["sbp_mean", "dbp_mean", "fasting_glucose_mg_dl"]:
        if c in df.columns:
            df[c] = _to_numeric_safe(df[c])

    # Numeric covariates (keep broad; missing ones are skipped)
    numeric_covs = [
        "age_years",
        "age_in_years_at_screening",
        "body_mass_index_kg_m2",
        "weight_kg",
        "waist_circumference_cm",
        "total_cholesterol_mg_dl",
        "direct_hdl_cholesterol_mg_dl",
        "LBXTLG",
        "triglycerides_mg_dl",
        "hs_c_reactive_protein_mg_l",
    ]
    for c in numeric_covs:
        if c in df.columns:
            df[c] = _to_numeric_safe(df[c])

    # Plausibility filters
    if "age_years" in df.columns:
        df.loc[(df["age_years"] < 0) | (df["age_years"] > 120), "age_years"] = pd.NA

    if "body_mass_index_kg_m2" in df.columns:
        df.loc[
            (df["body_mass_index_kg_m2"] <= 0) | (df["body_mass_index_kg_m2"] > 80),
            "body_mass_index_kg_m2",
        ] = pd.NA

    if "fasting_glucose_mg_dl" in df.columns:
        df.loc[
            (df["fasting_glucose_mg_dl"] <= 0) | (df["fasting_glucose_mg_dl"] > 600),
            "fasting_glucose_mg_dl",
        ] = pd.NA

    if "sbp_mean" in df.columns:
        df.loc[(df["sbp_mean"] < 50) | (df["sbp_mean"] > 300), "sbp_mean"] = pd.NA
    if "dbp_mean" in df.columns:
        df.loc[(df["dbp_mean"] < 30) | (df["dbp_mean"] > 200), "dbp_mean"] = pd.NA

    return df


# ---------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------

def load_bpxo_meaningful() -> pd.DataFrame:
    return _dedupe_on_id(_read_meaningful_csv(DEFAULT_FILES["BPXO"]), "BPXO")


def load_bmx_meaningful() -> pd.DataFrame:
    return _dedupe_on_id(_read_meaningful_csv(DEFAULT_FILES["BMX"]), "BMX")


def load_tchol_meaningful() -> pd.DataFrame:
    return _dedupe_on_id(_read_meaningful_csv(DEFAULT_FILES["TCHOL"]), "TCHOL")


def load_hdl_meaningful() -> pd.DataFrame:
    return _dedupe_on_id(_read_meaningful_csv(DEFAULT_FILES["HDL"]), "HDL")


def load_trigly_meaningful() -> pd.DataFrame:
    return _dedupe_on_id(_read_meaningful_csv(DEFAULT_FILES["TRIGLY"]), "TRIGLY")


def load_glu_meaningful() -> pd.DataFrame:
    return _dedupe_on_id(_read_meaningful_csv(DEFAULT_FILES["GLU"]), "GLU")


def load_hscrp_meaningful() -> pd.DataFrame:
    return _dedupe_on_id(_read_meaningful_csv(DEFAULT_FILES["HSCRP"]), "HSCRP")


def load_dsqtot_meaningful() -> pd.DataFrame:
    return _dedupe_on_id(_read_meaningful_csv(DEFAULT_FILES["DSQTOT"]), "DSQTOT")


def load_demo_meaningful() -> pd.DataFrame:
    return _dedupe_on_id(_read_meaningful_csv(DEFAULT_FILES["DEMO"]), "DEMO")


# ---------------------------------------------------------------------
# Blood pressure outcome construction
# ---------------------------------------------------------------------

def build_bp_outcomes(bpxo: pd.DataFrame) -> pd.DataFrame:
    """Compute mean systolic and diastolic BP from oscillometric readings."""
    bpxo = _normalize_id_column(bpxo).copy()

    sbp_cols = [
        "systolic_1st_oscillometric_reading",
        "systolic_2nd_oscillometric_reading",
        "systolic_3rd_oscillometric_reading",
    ]
    dbp_cols = [
        "diastolic_1st_oscillometric_reading",
        "diastolic_2nd_oscillometric_reading",
        "diastolic_3rd_oscillometric_reading",
    ]

    sbp_cols = [c for c in sbp_cols if c in bpxo.columns]
    dbp_cols = [c for c in dbp_cols if c in bpxo.columns]

    if not sbp_cols or not dbp_cols:
        raise ValueError(
            "BPXO dataframe is missing expected systolic/diastolic reading columns.\n"
            f"Found columns: {list(bpxo.columns)[:40]}..."
        )

    for c in sbp_cols + dbp_cols:
        bpxo[c] = _to_numeric_safe(bpxo[c])

    bpxo["sbp_mean"] = bpxo[sbp_cols].mean(axis=1)
    bpxo["dbp_mean"] = bpxo[dbp_cols].mean(axis=1)

    return bpxo[[ID_COL, "sbp_mean", "dbp_mean"]]


# ---------------------------------------------------------------------
# Build merged analysis dataframe
# ---------------------------------------------------------------------

def build_analysis_df(
    join: str = "inner",
    clean: bool = True,
) -> pd.DataFrame:
    """
    Build the merged NHANES analysis table for EconML.

    Parameters
    ----------
    join : {"inner","left"}
        - "inner" (default): complete-case merge across components
        - "left": keep BP sample and attach other components where available
    clean : bool
        Apply lightweight cleaning and plausibility checks.

    Returns
    -------
    pd.DataFrame
        One row per respondent.
    """
    if join not in {"inner", "left"}:
        raise ValueError("join must be one of {'inner','left'}")

    # Load components
    bpxo = load_bpxo_meaningful()
    bmx = load_bmx_meaningful()
    tchol = load_tchol_meaningful()
    hdl = load_hdl_meaningful()
    trigly = load_trigly_meaningful()
    glu = load_glu_meaningful()
    hscrp = load_hscrp_meaningful()
    dsqtot = load_dsqtot_meaningful()
    demo = load_demo_meaningful()

    # BP outcomes
    bp_outcomes = build_bp_outcomes(bpxo)

    # Merge to one row per respondent
    df = bp_outcomes.copy()
    components = [bmx, tchol, hdl, trigly, glu, hscrp, dsqtot, demo]
    for comp in components:
        df = df.merge(comp, on=ID_COL, how=join)

    # Treatment indicator
    if "any_dietary_supplements_taken" not in df.columns:
        raise KeyError(
            "Column `any_dietary_supplements_taken` not found after merging DSQTOT.\n"
            "Fix: confirm DSQTOT_L_meaningful.csv contains that column."
        )
    df["treatment_supplement"] = _encode_treatment(df["any_dietary_supplements_taken"])

    # Triglycerides normalization (support either name)
    if "LBXTLG" not in df.columns and "triglycerides_mg_dl" in df.columns:
        df["LBXTLG"] = df["triglycerides_mg_dl"]

    # Standardize demographics into age_years / sex (does not change existing cols)
    df = _standardize_demographics(df)

    # Encode sex if present (0/1). If already numeric, safe.
    if "sex" in df.columns:
        df["sex"] = _encode_sex(df["sex"])

    if clean:
        df = _basic_range_cleaning(df)

    return df


# ---------------------------------------------------------------------
# Helper to extract Y, T, X safely
# ---------------------------------------------------------------------

def get_y_t_x(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment_supplement",
    dropna: bool = True,
    include_demographics: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, List[str]]:
    """
    Extract (Y, T, X) for modeling from the merged analysis dataframe.

    Defaults are chosen to match your current notebook outputs:
    - include_demographics=False keeps X focused on the core biomarker covariates.

    Safety behavior:
    - Never includes the outcome itself in X (prevents leakage).
    - Coerces Y/T/X to numeric.
    - Optionally drops rows with any missing values in Y/T/X.
    """
    if outcome_col not in df.columns:
        raise KeyError(f"Outcome column `{outcome_col}` not found in dataframe.")
    if treatment_col not in df.columns:
        raise KeyError(f"Treatment column `{treatment_col}` not found in dataframe.")

    # Core covariates (matches the 8-column X you printed in your API notebook)
    core_covariates = [
        "body_mass_index_kg_m2",
        "weight_kg",
        "waist_circumference_cm",
        "total_cholesterol_mg_dl",
        "direct_hdl_cholesterol_mg_dl",
        "LBXTLG",
        "fasting_glucose_mg_dl",
        "hs_c_reactive_protein_mg_l",
    ]

    demographic_covariates = [
        "age_years",
        "sex",
    ]

    candidate_covariates = core_covariates + (demographic_covariates if include_demographics else [])

    # Prevent leakage if outcome appears in covariates
    candidate_covariates = [c for c in candidate_covariates if c != outcome_col]

    covariate_cols = [c for c in candidate_covariates if c in df.columns]
    if not covariate_cols:
        raise ValueError(
            "No covariate columns were found. Check build_analysis_df() merges and column names."
        )

    y = _to_numeric_safe(df[outcome_col])
    t = _to_numeric_safe(df[treatment_col])

    X = df[covariate_cols].copy()
    for c in X.columns:
        X[c] = _to_numeric_safe(X[c])

    if dropna:
        mask = y.notna() & t.notna()
        for c in X.columns:
            mask &= X[c].notna()

        y = y[mask]
        t = t[mask]
        X = X.loc[mask, :]

    return y, t, X, covariate_cols



def quick_summary(join: str = "inner") -> pd.DataFrame:
    df = build_analysis_df(join=join, clean=True)

    cols_of_interest = [
        "sbp_mean",
        "dbp_mean",
        "fasting_glucose_mg_dl",
        "treatment_supplement",
        "age_years",
        "sex",
        "body_mass_index_kg_m2",
        "total_cholesterol_mg_dl",
        "direct_hdl_cholesterol_mg_dl",
        "LBXTLG",
        "hs_c_reactive_protein_mg_l",
    ]
    cols_present = [c for c in cols_of_interest if c in df.columns]
    return df[cols_present].describe(include="all")
