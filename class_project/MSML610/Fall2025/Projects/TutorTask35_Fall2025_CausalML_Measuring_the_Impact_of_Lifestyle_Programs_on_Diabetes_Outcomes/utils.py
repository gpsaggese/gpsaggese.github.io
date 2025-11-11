'''
Utility functions for the CausalML project
'''
from pathlib import Path
from typing import Union, Iterable, Optional
import pandas as pd

PathLike = Union[str, Path]

def load_unprocessed_data(
        unprocessed_dir: PathLike = "data/unprocessed",
        cast_categoricals: bool = True,
        categorical_cols: Optional[Iterable[str]] =(
                    "race","gender","age","admission_type_id","admission_source_id",
        "discharge_disposition_id","A1Cresult","max_glu_serum",
        "change","diabetesMed","readmitted"
        )) -> pd.DataFrame:
    """
    Load diabetic_data.csv with safe defaults (strings first, '?' -> NaN),
    cast known integer-like columns, and optionally cast common low-cardinality
    columns to 'category'. 
    Args:
        unprocessed_dir: Directory where diabetic_data.csv is located.
        cast_categoricals: Whether to cast specified columns to 'category' dtype.
        categorical_cols: Iterable of column names to cast as categorical if
            cast_categoricals is True.
    """
    unprocessed_dir = Path(unprocessed_dir)
    csv_path = unprocessed_dir / "diabetic_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")    
    df = pd.read_csv(csv_path, dtype=str, na_values=["?"], keep_default_na=True)
    required = ["encounter_id", "patient_nbr", "readmitted"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    int_cols = [
        "time_in_hospital","num_lab_procedures","num_procedures","num_medications",
        "number_outpatient","number_emergency","number_inpatient","number_diagnoses"
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    if cast_categoricals and categorical_cols:
        for c in categorical_cols:
            if c in df.columns:
                df[c] = df[c].astype('category')
    return df

