import os, glob, zipfile, pandas as pd, numpy as np

UNSW_COLS = [
    'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl',
    'sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb',
    'smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt',
    'Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd',
    'is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm',
    'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','label'
]

def load_unsw_from_zip(zip_path, extract_dir="./data", columns=UNSW_COLS):
    """Extracts UNSW-NB15 CSVs, skips metadata rows, and loads clean numeric data."""
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    parts = sorted(glob.glob(f"{extract_dir}/**/*.csv", recursive=True))
    if not parts:
        raise FileNotFoundError("No CSVs found after extracting the archive.")

    dfs = []
    for p in parts:
        df_part = pd.read_csv(
            p, encoding="latin1", header=None, names=columns,
            low_memory=False, dtype=str
        )
        # Drop any metadata rows
        drop_mask = df_part['srcip'].str.contains(
            "No|Name|Type|Description|nominal|integer", case=False, na=False
        )
        df_part = df_part[~drop_mask]
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)

    # Convert numeric columns
    numeric_like = [c for c in df.columns if c not in ['srcip','dstip','proto','state','service','attack_cat']]
    for c in numeric_like:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Clean label
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

    # Drop any remaining rows where almost everything is NaN
    df = df.dropna(thresh=int(len(df.columns)*0.3))

    return df



def basic_eda(df):
    """Quick dataset summary and health check."""
    print("✅ Dataset Loaded Successfully")
    print(f"Shape: {df.shape}")
    print("\n--- Data Types ---")
    print(df.dtypes.value_counts())
    print("\n--- Missing Values (Top 10) ---")
    miss = df.isna().sum().sort_values(ascending=False)
    print(miss.head(10))
    print("\n--- Sample Rows ---")
    print(df.head(3))
