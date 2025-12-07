"""
econml_utils.py

Helper functions for the MSML610 Project:

    TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes

This module is responsible for:

1. Loading the cleaned NHANES 2021–2023 CSV files from the local `data/` directory.
2. Building a merged, analysis-ready dataframe with:
   - Outcomes: sbp_mean, dbp_mean, fasting_glucose_mg_dl
   - Treatment: treatment_supplement (binary; 1 = any supplement use, 0 = none)
   - Baseline covariates: age, sex, BMI, lipids, glucose, hs-CRP, etc.
3. Providing a simple helper, `get_y_t_x`, that returns (Y, T, X, covariate_cols)
   for EconML and OLS experiments.

This module does NOT run any models directly. All modeling is done in
`econml.API.py`, which imports `build_analysis_df` and `get_y_t_x`.
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd

# ---------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

ID_COL = "respondent_sequence_number"
ID_CANDIDATES = [ID_COL, "SEQN", "seqn"]


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that the participant identifier column is named
    `respondent_sequence_number` in every dataframe.

    The upstream "meaningful" CSVs already rename SEQN, but this helper
    makes the merge logic robust in case any file still uses `SEQN` or
    a slightly different variant.
    """
    for candidate in ID_CANDIDATES:
        if candidate in df.columns:
            if candidate != ID_COL:
                df = df.rename(columns={candidate: ID_COL})
            break
    return df


def _read_meaningful_csv(name: str) -> pd.DataFrame:
    """
    Convenience wrapper to load a CSV from the `data/` folder and apply
    common NA handling + ID normalization.

    Parameters
    ----------
    name : str
        File name inside `data/`, e.g. "BPXO_L_meaningful.csv".

    Returns
    -------
    pd.DataFrame
    """
    csv_path = DATA_DIR / name
    df = pd.read_csv(csv_path, na_values=[".", " "])
    df = _normalize_id_column(df)
    return df


# ---------------------------------------------------------------------
# Loaders for each NHANES component
# ---------------------------------------------------------------------


def load_bpxo_meaningful() -> pd.DataFrame:
    """Oscillometric blood pressure readings (BPXO)."""
    return _read_meaningful_csv("BPXO_L_meaningful.csv")


def load_bmx_meaningful() -> pd.DataFrame:
    """Body measures (BMX) – BMI, weight, waist, etc."""
    return _read_meaningful_csv("BMX_L_meaningful.csv")


def load_tchol_meaningful() -> pd.DataFrame:
    """Total cholesterol (TCHOL)."""
    return _read_meaningful_csv("TCHOL_L_meaningful.csv")


def load_hdl_meaningful() -> pd.DataFrame:
    """Direct HDL cholesterol (HDL)."""
    return _read_meaningful_csv("HDL_L_meaningful.csv")


def load_trigly_meaningful() -> pd.DataFrame:
    """Triglycerides (TRIGLY) including LBXTLG."""
    return _read_meaningful_csv("TRIGLY_L_meaningful.csv")


def load_glu_meaningful() -> pd.DataFrame:
    """Fasting plasma glucose (GLU)."""
    return _read_meaningful_csv("GLU_L_meaningful.csv")


def load_hscrp_meaningful() -> pd.DataFrame:
    """High-sensitivity C-reactive protein (HSCRP)."""
    return _read_meaningful_csv("HSCRP_L_meaningful.csv")


def load_dsqtot_meaningful() -> pd.DataFrame:
    """
    Dietary supplements totals (DSQTOT).

    Expected to contain a column
        `any_dietary_supplements_taken`
    derived during data preparation.
    """
    return _read_meaningful_csv("DSQTOT_L_meaningful.csv")


def load_demo_meaningful() -> pd.DataFrame:
    """
    Demographics (DEMO).

    Expected to contain:
        - age_years
        - sex
        - survey weights (not used directly in this module)
    """
    return _read_meaningful_csv("DEMO_L_meaningful.csv")


# ---------------------------------------------------------------------
# Blood pressure outcome construction
# ---------------------------------------------------------------------


def build_bp_outcomes(bpxo: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean systolic and diastolic blood pressure from three
    oscillometric readings.

    Parameters
    ----------
    bpxo : pd.DataFrame
        Raw BPXO component with individual readings.

        Expected columns:
            systolic_1st_oscillometric_reading
            systolic_2nd_oscillometric_reading
            systolic_3rd_oscillometric_reading
            diastolic_1st_oscillometric_reading
            diastolic_2nd_oscillometric_reading
            diastolic_3rd_oscillometric_reading

    Returns
    -------
    pd.DataFrame
        Columns:
            respondent_sequence_number
            sbp_mean
            dbp_mean
    """
    bpxo = _normalize_id_column(bpxo)

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

    # Only average over columns that actually exist to be slightly robust.
    sbp_cols = [c for c in sbp_cols if c in bpxo.columns]
    dbp_cols = [c for c in dbp_cols if c in bpxo.columns]

    if not sbp_cols or not dbp_cols:
        raise ValueError(
            "BPXO dataframe is missing expected systolic/diastolic reading columns."
        )

    bpxo = bpxo.copy()
    bpxo["sbp_mean"] = bpxo[sbp_cols].mean(axis=1)
    bpxo["dbp_mean"] = bpxo[dbp_cols].mean(axis=1)

    return bpxo[[ID_COL, "sbp_mean", "dbp_mean"]]


# ---------------------------------------------------------------------
# Build merged analysis dataframe
# ---------------------------------------------------------------------


def build_analysis_df() -> pd.DataFrame:
    """
    Build the merged NHANES analysis table for EconML.

    This function:

    1. Loads all required NHANES components from `data/`.
    2. Computes sbp_mean and dbp_mean from BPXO readings.
    3. Merges anthropometrics, lipids, glucose, hs-CRP, supplements, and
       demographics on `respondent_sequence_number`.
    4. Constructs a binary treatment variable:

           treatment_supplement = 1  if any_dietary_supplements_taken == 1
                                   0  if any_dietary_supplements_taken == 2
                                   NaN otherwise

    Returns
    -------
    pd.DataFrame
        One row per respondent, with all outcomes, treatment, and
        covariates used in the project.
    """
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

    # Blood pressure outcomes
    bp_outcomes = build_bp_outcomes(bpxo)

    # Start from BP outcomes as the "core" sample and inner-join
    # everything else so that all variables are observed.
    df = bp_outcomes

    components = [bmx, tchol, hdl, trigly, glu, hscrp, dsqtot, demo]
    for comp in components:
        df = df.merge(comp, on=ID_COL, how="inner")

    # Construct treatment indicator from DSQTOT
    # The data-prep notebooks should have created a clean binary variable:
    #   any_dietary_supplements_taken == 1  -> "Yes"
    #   any_dietary_supplements_taken == 2  -> "No"
    if "any_dietary_supplements_taken" in df.columns:
        raw = df["any_dietary_supplements_taken"]
        df["treatment_supplement"] = raw.map({1: 1, 2: 0})
    else:
        # Fallback: if the column is missing, raise a clear error so that
        # it is obvious where to fix the data preparation.
        raise KeyError(
            "Column `any_dietary_supplements_taken` not found in merged dataframe. "
            "Please check the DSQTOT_L_meaningful.csv data-prep step."
        )

    return df


# ---------------------------------------------------------------------
# Helper to extract Y (outcome), T (treatment), and X (covariates)
# ---------------------------------------------------------------------


def get_y_t_x(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment_supplement",
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, List[str]]:
    """
    Extract (Y, T, X) for modeling from the merged analysis dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `build_analysis_df()`.
    outcome_col : str
        Name of the outcome column, e.g. "sbp_mean" or "fasting_glucose_mg_dl".
    treatment_col : str, default "treatment_supplement"
        Name of the binary treatment indicator column.

    Returns
    -------
    y : pd.Series
        Outcome values.
    t : pd.Series
        Treatment indicator (0/1).
    X : pd.DataFrame
        Covariate matrix.
    covariate_cols : list of str
        Names of covariate columns used in X.

    Notes
    -----
    We follow the project write-up and use a fixed set of baseline
    covariates. If a covariate is missing from the dataframe (for
    example if a lab component was not merged correctly), it will be
    silently dropped from `covariate_cols`. This makes the code a bit
    more robust while keeping the API stable.
    """
    # Candidate covariates (project standard)
    candidate_covariates = [
        "age_years",
        "sex",
        "body_mass_index_kg_m2",
        "weight_kg",
        "waist_circumference_cm",
        "total_cholesterol_mg_dl",
        "direct_hdl_cholesterol_mg_dl",
        "LBXTLG",  # triglycerides (mg/dL)
        "fasting_glucose_mg_dl",
        "hs_c_reactive_protein_mg_l",
    ]

    # Keep only covariates that are actually present
    covariate_cols = [c for c in candidate_covariates if c in df.columns]

    if not covariate_cols:
        raise ValueError(
            "No covariate columns were found in the dataframe. "
            "Check that build_analysis_df() merged all required components."
        )

    if outcome_col not in df.columns:
        raise KeyError(f"Outcome column `{outcome_col}` not found in dataframe.")

    if treatment_col not in df.columns:
        raise KeyError(f"Treatment column `{treatment_col}` not found in dataframe.")

    y = df[outcome_col]
    t = df[treatment_col]
    X = df[covariate_cols]

    return y, t, X, covariate_cols


# ---------------------------------------------------------------------
# (Optional) quick sanity-check helper
# ---------------------------------------------------------------------


def quick_summary() -> pd.DataFrame:
    """
    Small convenience function for notebooks: build the analysis
    dataframe and return basic descriptive statistics for the key
    variables (outcomes, treatment, and covariates).

    This is not used by the API, but is handy for exploratory checks
    in `econml.example.ipynb`.
    """
    df = build_analysis_df()

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
