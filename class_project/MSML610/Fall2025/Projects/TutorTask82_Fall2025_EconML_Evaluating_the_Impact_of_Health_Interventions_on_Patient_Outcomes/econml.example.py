"""
econml.example.py

Script-style walkthrough for the MSML610 project:

    TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes

This script mirrors the logic of `econml.example.ipynb` and is meant to be
a simple, reproducible entry point that another student (or the grader)
can run end-to-end.

What it does:
-------------
1. Builds the merged NHANES analysis dataframe using `build_analysis_df`.
2. Prints basic summaries of outcomes, treatment, and core covariates.
3. Runs:
    - DRLearner for mean systolic BP (sbp_mean)
    - DRLearner for fasting glucose (fasting_glucose_mg_dl)
4. Runs OLS baselines for both outcomes.
5. Saves a small set of figures into the `figs/` folder:
    - Histogram of SBP by treatment group
    - Histogram of SBP CATEs
    - Bar chart of SBP CATE by BMI quartile
    - Histogram of glucose CATEs

You can run this inside the .venv with:

    python econml.example.py
"""

import pathlib
import sys
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from econml_utils import build_analysis_df, get_y_t_x

# ---------------------------------------------------------------------
# Dynamically load econml.API.py as a module called "econml.API"
# ---------------------------------------------------------------------

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
FIGS_DIR = PROJECT_DIR / "figs"
FIGS_DIR.mkdir(exist_ok=True)

api_path = PROJECT_DIR / "econml.API.py"

print("Project directory:", PROJECT_DIR)
print("econml.API.py exists:", api_path.exists())

spec = importlib.util.spec_from_file_location("econml.API", api_path)
econml_api = importlib.util.module_from_spec(spec)
sys.modules["econml.API"] = econml_api
spec.loader.exec_module(econml_api)

print("econml.API loaded as module:", econml_api.__name__)
print("econml.API file path:", api_path)


# ---------------------------------------------------------------------
# Small helper: pretty section printing
# ---------------------------------------------------------------------


def print_section(title: str) -> None:
    bar = "-" * len(title)
    print("\n" + title)
    print(bar)


# ---------------------------------------------------------------------
# 1. Data loading and basic summaries
# ---------------------------------------------------------------------


def summarize_analysis_df(df: pd.DataFrame) -> None:
    """Print high-level information about the merged analysis dataset."""
    print_section("1) Analysis dataframe overview")

    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    # Show first few rows of core variables only (so the console isn't flooded)
    cols_core = [
        "respondent_sequence_number",
        "sbp_mean",
        "dbp_mean",
        "fasting_glucose_mg_dl",
        "body_mass_index_kg_m2",
        "weight_kg",
        "waist_circumference_cm",
        "treatment_supplement",
        "age_years",
        "sex",
    ]
    cols_present = [c for c in cols_core if c in df.columns]

    print("First 5 rows (core variables):")
    print(df[cols_present].head())
    print()

    print("Summary statistics (core variables):")
    print(df[cols_present].describe(include="all"))

    # Treatment distribution
    if "treatment_supplement" in df.columns:
        print("\nTreatment_supplement value counts:")
        print(df["treatment_supplement"].value_counts(dropna=False))
        print()
        # For binary 0/1, mean = proportion treated (ignoring NaNs)
        prop_treated = df["treatment_supplement"].mean()
        print(f"Proportion treated (any supplement use): {prop_treated:.3f}")


def plot_sbp_by_treatment(df: pd.DataFrame) -> None:
    """Histogram of SBP for treated vs control groups."""
    if "sbp_mean" not in df.columns or "treatment_supplement" not in df.columns:
        print("Skipping SBP-by-treatment plot (columns missing).")
        return

    print_section("Figure: SBP distribution by supplement use")

    temp = df[["sbp_mean", "treatment_supplement"]].dropna()
    treated = temp.loc[temp["treatment_supplement"] == 1, "sbp_mean"]
    control = temp.loc[temp["treatment_supplement"] == 0, "sbp_mean"]

    plt.figure(figsize=(8, 4))
    plt.hist(control, bins=30, alpha=0.5, label="No supplements (T=0)")
    plt.hist(treated, bins=30, alpha=0.5, label="Any supplements (T=1)")
    plt.xlabel("Mean systolic BP (mmHg)")
    plt.ylabel("Count")
    plt.title("SBP Distribution by Supplement Use")
    plt.legend()
    plt.tight_layout()

    out_path = FIGS_DIR / "sbp_by_treatment_hist.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------
# 2. DRLearner analysis for SBP
# ---------------------------------------------------------------------


def analyze_sbp_with_econml() -> dict:
    """Run DRLearner for SBP and create a few plots/summaries."""
    print_section("2) EconML DRLearner – SBP outcome")

    sbp_results = econml_api.run_sbp_supplement_experiment(random_state=42)

    ate_sbp = sbp_results["ate_sbp"]
    covariates = sbp_results["covariates"]
    cate_df = sbp_results["cate_df"]
    tau_col = sbp_results["tau_col"]
    age_effects = sbp_results["age_effects"]
    bmi_effects = sbp_results["bmi_effects"]

    print(f"ATE (DRLearner) for SBP: {ate_sbp:.4f}")
    print("\nCovariates used in X:")
    print(covariates)

    # Summary of CATE
    print("\nSummary of individual CATEs (tau_hat_sbp_mean):")
    print(cate_df[tau_col].describe())

    # Plot: histogram of CATEs
    plt.figure(figsize=(8, 4))
    cate_df[tau_col].hist(bins=40)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Estimated CATE on SBP (mmHg)")
    plt.ylabel("Count")
    plt.title("Distribution of Individual Treatment Effects on SBP (DRLearner)")
    plt.tight_layout()

    out_path = FIGS_DIR / "sbp_cate_hist.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved figure: {out_path}")

    # Plot: CATE by BMI quartile, if available
    if bmi_effects is not None:
        print("\nMean CATE on SBP by BMI quartile:")
        print(bmi_effects)

        plt.figure(figsize=(6, 4))
        bmi_effects.plot(kind="bar")
        plt.axhline(0, linestyle="--")
        plt.ylabel("Mean CATE on SBP (mmHg)")
        plt.title("Average Treatment Effect on SBP by BMI Quartile")
        plt.tight_layout()

        out_path = FIGS_DIR / "sbp_cate_by_bmi.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved figure: {out_path}")
    else:
        print("BMI effects not available (no BMI column). Skipping BMI plot.")

    # Optional: CATE vs BMI scatter
    if "body_mass_index_kg_m2" in cate_df.columns:
        plt.figure(figsize=(8, 4))
        plt.scatter(
            cate_df["body_mass_index_kg_m2"],
            cate_df[tau_col],
            s=10,
            alpha=0.3,
        )
        plt.axhline(0, linestyle="--")
        plt.xlabel("BMI (kg/m^2)")
        plt.ylabel("CATE on SBP (mmHg)")
        plt.title("CATE vs BMI (SBP outcome)")
        plt.tight_layout()

        out_path = FIGS_DIR / "sbp_cate_vs_bmi.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved figure: {out_path}")
    else:
        print("BMI column missing in cate_df; skipping CATE vs BMI scatter.")

    # Optional: age-based heterogeneity if available
    if age_effects is not None:
        print("\nMean CATE on SBP by age quartile:")
        print(age_effects)
    else:
        print("\nAge-based heterogeneity not available (no age_years column).")

    return sbp_results


# ---------------------------------------------------------------------
# 3. DRLearner analysis for fasting glucose
# ---------------------------------------------------------------------


def analyze_glucose_with_econml() -> dict:
    """Run DRLearner for fasting glucose and summarize results."""
    print_section("3) EconML DRLearner – fasting glucose outcome")

    glu_results = econml_api.run_glucose_supplement_experiment(random_state=42)

    ate_glu = glu_results["ate_glucose"]
    covariates = glu_results["covariates"]
    cate_df = glu_results["cate_df"]
    tau_col = glu_results["tau_col"]
    age_effects = glu_results["age_effects"]
    bmi_effects = glu_results["bmi_effects"]

    print(f"ATE (DRLearner) for fasting glucose: {ate_glu:.4f}")
    print("\nCovariates used in X:")
    print(covariates)

    print("\nSummary of individual CATEs (tau_hat_fasting_glucose_mg_dl):")
    print(cate_df[tau_col].describe())

    # Plot: histogram of glucose CATEs
    plt.figure(figsize=(8, 4))
    cate_df[tau_col].hist(bins=40)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Estimated CATE on fasting glucose (mg/dL)")
    plt.ylabel("Count")
    plt.title("Distribution of Individual Treatment Effects on Fasting Glucose")
    plt.tight_layout()

    out_path = FIGS_DIR / "glucose_cate_hist.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved figure: {out_path}")

    # BMI heterogeneity (if available)
    if bmi_effects is not None:
        print("\nMean CATE on fasting glucose by BMI quartile:")
        print(bmi_effects)
    else:
        print("BMI effects not available for glucose outcome.")

    # Age heterogeneity (if available)
    if age_effects is not None:
        print("\nMean CATE on fasting glucose by age quartile:")
        print(age_effects)
    else:
        print("Age-based heterogeneity not available for glucose outcome.")

    return glu_results


# ---------------------------------------------------------------------
# 4. OLS baseline comparison
# ---------------------------------------------------------------------


def compare_econml_vs_ols(
    sbp_results: dict,
    glu_results: dict,
) -> pd.DataFrame:
    """Run OLS baselines and build a small comparison table."""
    print_section("4) EconML vs OLS comparison")

    ols_sbp = econml_api.run_ols_for_outcome("sbp_mean")
    ols_glu = econml_api.run_ols_for_outcome("fasting_glucose_mg_dl")

    comp = pd.DataFrame(
        {
            "outcome": ["sbp_mean", "fasting_glucose_mg_dl"],
            "econml_ate": [sbp_results["ate_sbp"], glu_results["ate_glucose"]],
            "ols_treatment_coef": [
                ols_sbp["treatment_coef"],
                ols_glu["treatment_coef"],
            ],
            "n_obs_ols": [ols_sbp["n_obs"], ols_glu["n_obs"]],
        }
    )

    print("EconML ATE vs OLS treatment coefficient:")
    print(comp)

    out_path = PROJECT_DIR / "econml_vs_ols_comparison.csv"
    comp.to_csv(out_path, index=False)
    print(f"\nSaved comparison table to: {out_path}")

    return comp


# ---------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------


def main() -> None:
    # Load and summarize the merged NHANES dataset
    df = build_analysis_df()
    summarize_analysis_df(df)

    # Plot SBP distribution by treatment group
    plot_sbp_by_treatment(df)

    # DRLearner analyses
    sbp_results = analyze_sbp_with_econml()
    glu_results = analyze_glucose_with_econml()

    # OLS comparison
    compare_econml_vs_ols(sbp_results, glu_results)

    print_section("Done")
    print("econml.example.py finished successfully.")


if __name__ == "__main__":
    main()
