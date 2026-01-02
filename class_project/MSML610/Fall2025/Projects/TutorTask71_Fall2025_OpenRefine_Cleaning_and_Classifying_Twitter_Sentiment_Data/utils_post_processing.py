import re
import pandas as pd

URL_RE = re.compile(r"https?://\S+")
AT_RE  = re.compile(r"@\S+")
HASH_RE= re.compile(r"#")
NONALPHA_RE = re.compile(r"[^a-z\s]")
MULTISPACE_RE = re.compile(r"\s+")

def clean_text_grel_equiv(s: str) -> str:
    """Python equivalent of the GREL cleaning (for spot checks)."""
    s = (s or "").lower()
    s = URL_RE.sub("", s)
    s = AT_RE.sub("", s)
    s = HASH_RE.sub("", s)
    s = NONALPHA_RE.sub("", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

def add_counts(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    out = df.copy()
    out["word_count_py"] = out[text_col].fillna("").str.split().str.len()
    out["char_count_py"] = out[text_col].fillna("").str.len()
    return out

def quick_summary(df: pd.DataFrame, text_col: str = "text_clean", label_col: str = "target") -> dict:
    url_rate = df[text_col].astype(str).str.contains(r"https?://", regex=True).mean()
    at_rate  = df[text_col].astype(str).str.contains(r"@\w+", regex=True).mean()
    hash_rate= df[text_col].astype(str).str.contains(r"#", regex=True).mean()
    return {
        "rows": len(df),
        "url_rate": round(url_rate, 4),
        "mention_rate": round(at_rate, 4),
        "hashtag_symbol_rate": round(hash_rate, 4),
        "label_counts": df[label_col].value_counts(dropna=False).to_dict(),
    }
