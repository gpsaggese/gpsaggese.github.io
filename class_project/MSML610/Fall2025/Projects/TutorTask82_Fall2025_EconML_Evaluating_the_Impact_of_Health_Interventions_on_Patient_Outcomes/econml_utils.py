"""
Helper functions for MSML610 Project 3.
This module builds cleaned NHANES analysis tables and extracts
outcome (Y), treatment (T), and covariates (X) for EconML experiments.
"""


import pandas as pd
from pathlib import Path

# Base project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has 'respondent_sequence_number' as the ID column.
    If it has 'respondent_id', rename it. Otherwise raise an error.
    """
    if "respondent_sequence_number" in df.columns:
        return df
    if "respondent_id" in df.columns:
        return df.rename(columns={"respondent_id": "respondent_sequence_number"})
    raise KeyError(
        "Could not find respondent ID column in dataframe "
        "(expected 'respondent_sequence_number' or 'respondent_id')."
    )


# -------------------------------------------------------------------
# Loaders for individual NHANES components
# -------------------------------------------------------------------

def load_bpxo_meaningful() -> pd.DataFrame:
    """Load cleaned oscillometric blood pressure data (BPXO)."""
    csv_path = DATA_DIR / "BPXO_L_meaningful.csv"
    df = pd.read_csv(csv_path, na_values=[".", " "])
    df = _normalize_id_column(df)
    return df


def load_bmx_meaningful() -> pd.DataFrame:
    """Load cleaned body measures (BMX) data."""
    csv_path = DATA_DIR / "BMX_L_meaningful.csv"
    df = pd.read_csv(csv_path, na_values=[".", " "])
    df = _normalize_id_column(df)
    return df


def load_tchol_meaningful() -> pd.DataFrame:
    """Load total cholesterol (TCHOL)."""
    csv_path = DATA_DIR / "TCHOL_L_meaningful.csv"
    df = pd.read_csv(csv_path, na_values=[".", " "])
    df = _normalize_id_column(df)
    return df


def load_hdl_meaningful() -> pd.DataFrame:
    """Load HDL cholesterol (HDL)."""
    csv_path = DATA_DIR / "HDL_L_meaningful.csv"
    df = pd.read_csv(csv_path, na_values=[".", " "])
    df = _normalize_id_column(df)
    return df


def load_trigly_meaningful() -> pd.DataFrame:
    """Load triglycerides and LDL (TRIGLY)."""
    csv_path = DATA_DIR / "TRIGLY_L_meaningful.csv"
    df = pd.read_csv(csv_path, na_values=[".", " "])
    df = _normalize_id_column(df)
    return df


def load_glu_meaningful() -> pd.DataFrame:
    """Load fasting glucose (GLU)."""
    csv_path = DATA_DIR / "GLU_L_meaningful.csv"
    df = pd.read_csv(csv_path, na_values=[".", " "])
    df = _normalize_id_column(df)
    return df


def load_hscrp_meaningful() -> pd.DataFrame:
    """Load hs-CRP (HSCRP)."""
    csv_path = DATA_DIR / "HSCRP_L_meaningful.csv"
    df = pd.read_csv(csv_path, na_values=[".", " "])
    df = _normalize_id_column(df)
    return df


def load_dsqtot_meaningful() -> pd.DataFrame:
    """Load supplement totals (DSQTOT)."""
    csv_path = DATA_DIR / "DSQTOT_L_meaningful.csv"
    df = pd.read_csv(csv_path, na_values=[".", " "])
    df = _normalize_id_column(df)
    return df


def load_demo_meaningful() -> pd.DataFrame:
    """Load demographics (DEMO)."""
    csv_path = DATA_DIR / "DEMO_L_meaningful.csv"
    df = pd.read_csv(csv_path, na_values=[".", " "])
    df = _normalize_id_column(df)
    return df


# -------------------------------------------------------------------
# Builder functions: BP + anthropometrics + labs
# -------------------------------------------------------------------

def build_bp_outcomes() -> pd.DataFrame:
    """
    Compute sbp_mean and dbp_mean from three oscillometric readings.

    Returns:
      respondent_sequence_number, sbp_mean, dbp_mean
    """
    df = load_bpxo_meaningful()

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

    df["sbp_mean"] = df[sbp_cols].mean(axis=1)
    df["dbp_mean"] = df[dbp_cols].mean(axis=1)

    bp_outcomes = df[["respondent_sequence_number", "sbp_mean", "dbp_mean"]].copy()
    return bp_outcomes


def build_bp_bmi_df() -> pd.DataFrame:
    """
    Merge BP outcomes with BMI and basic anthropometrics.
    Uses:
      body_mass_index_kg_m2, weight_kg, waist_circumference_cm
    """
    bp = build_bp_outcomes()
    bmx = load_bmx_meaningful()

    anthropometric_cols = [
        "respondent_sequence_number",
        "body_mass_index_kg_m2",   # BMI
        "weight_kg",
        "waist_circumference_cm",
    ]
    cols_present = [c for c in anthropometric_cols if c in bmx.columns]
    bmx_small = bmx[cols_present].copy()

    df = bp.merge(
        bmx_small,
        on="respondent_sequence_number",
        how="left",
    )
    return df


def build_nhanes_core_df() -> pd.DataFrame:
    """
    Core analysis dataframe with:
      - BP outcomes
      - Anthropometrics
      - Lipids, triglycerides, fasting glucose, hs-CRP
    """
    df = build_bp_bmi_df()

    # Total cholesterol
    tch = load_tchol_meaningful()
    df = df.merge(
        tch[
            [
                "respondent_sequence_number",
                "total_cholesterol_mg_dl",
                "total_cholesterol_mmol_l",
            ]
        ],
        on="respondent_sequence_number",
        how="left",
    )

    # HDL (direct HDL)
    hdl = load_hdl_meaningful()
    df = df.merge(
        hdl[
            [
                "respondent_sequence_number",
                "direct_hdl_cholesterol_mg_dl",
                "direct_hdl_cholesterol_mmol_l",
            ]
        ],
        on="respondent_sequence_number",
        how="left",
    )

    # Triglycerides & LDL
    tri = load_trigly_meaningful()
    df = df.merge(
        tri[
            [
                "respondent_sequence_number",
                "LBXTLG",  # triglyceride mg/dL
                "triglyceride_mmol_l",
                "ldl_cholesterol_friedewald_mg_dl",
                "ldl_cholesterol_friedewald_mmol_l",
                "ldl_cholesterol_martin_hopkins_mg_dl",
                "ldl_cholesterol_martin_hopkins_mmol_l",
                "ldl_cholesterol_nih_equation_2_mg_dl",
                "ldl_cholesterol_nih_equation_2_mmol_l",
            ]
        ],
        on="respondent_sequence_number",
        how="left",
    )

    # Fasting glucose
    glu = load_glu_meaningful()
    df = df.merge(
        glu[
            [
                "respondent_sequence_number",
                "fasting_glucose_mg_dl",
                "fasting_glucose_mmol_l",
            ]
        ],
        on="respondent_sequence_number",
        how="left",
    )

    # hs-CRP
    crp = load_hscrp_meaningful()
    df = df.merge(
        crp[
            [
                "respondent_sequence_number",
                "hs_c_reactive_protein_mg_l",
                "hs_c_reactive_protein_comment_code",
            ]
        ],
        on="respondent_sequence_number",
        how="left",
    )

    return df


# -------------------------------------------------------------------
# Treatment + demographics + Y/T/X helper
# -------------------------------------------------------------------

def add_treatment_and_demo(core_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add treatment indicator from DSQTOT and demographics (age, gender) from DEMO.

    Treatment:
      treatment_supplement = 1 if any_dietary_supplements_taken == 1, else 0
    """
    df = core_df.copy()

    # ---- Supplements -> treatment ----
    dsq = load_dsqtot_meaningful()

    # exact column from your file
    supp_col = "any_dietary_supplements_taken"
    if supp_col not in dsq.columns:
        raise KeyError(
            f"Expected column '{supp_col}' in DSQTOT_L_meaningful.csv, "
            f"found: {list(dsq.columns)}"
        )

    df = df.merge(
        dsq[["respondent_sequence_number", supp_col]],
        on="respondent_sequence_number",
        how="left",
    )

    df["treatment_supplement"] = (df[supp_col] == 1).astype(int)

    # ---- Demographics: age, gender ----
    demo = load_demo_meaningful()

    age_col = "age_in_years_at_screening"
    sex_col = "gender"

    missing_demo = [c for c in [age_col, sex_col] if c not in demo.columns]
    if missing_demo:
        raise KeyError(
            f"Expected columns {missing_demo} in DEMO_L_meaningful.csv, "
            f"found: {list(demo.columns)}"
        )

    demo_small = demo[
        ["respondent_sequence_number", age_col, sex_col]
    ].copy()

    demo_small = demo_small.rename(
        columns={
            age_col: "age_years",
            sex_col: "sex",
        }
    )

    df = df.merge(demo_small, on="respondent_sequence_number", how="left")

    return df


def build_analysis_df() -> pd.DataFrame:
    """
    Full analysis dataframe with:
      - outcomes (sbp_mean, dbp_mean)
      - treatment_supplement
      - key covariates (labs, BMI, age, sex)
    """
    core = build_nhanes_core_df()
    df = add_treatment_and_demo(core)
    return df


def get_y_t_x(
    df: pd.DataFrame,
    outcome_col: str = "sbp_mean",
    treatment_col: str = "treatment_supplement",
):
    """
    Extract outcome Y, treatment T, and covariate matrix X from the analysis dataframe.

    Returns:
      y (Series), t (Series), X (DataFrame), covariate_columns (list)
    """
    y = df[outcome_col]
    t = df[treatment_col]

    # Covariates using your actual column names
    covariate_cols = [
        c
        for c in [
            "age_years",
            "sex",
            "body_mass_index_kg_m2",
            "weight_kg",
            "waist_circumference_cm",
            "total_cholesterol_mg_dl",
            "direct_hdl_cholesterol_mg_dl",
            "LBXTLG",  # triglyceride mg/dL
            "fasting_glucose_mg_dl",
            "hs_c_reactive_protein_mg_l",
        ]
        if c in df.columns
    ]

    X = df[covariate_cols]

    return y, t, X, covariate_cols
