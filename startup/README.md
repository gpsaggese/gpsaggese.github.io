# GP's Startup Network Google Sheet

- This dir stores a local copy of the Google Sheet
  [GP's Startup Network](https://docs.google.com/spreadsheets/d/1Qa9g-j0NSdStV46W_dc85TMrwywIEPvJAhzV4Ymokb0/edit)
- Each tab of the Google Sheet is one CSV file
  `startup/gp_resources.<ID>.<tab_name>.csv`
  - `<ID>` is the position of the tab in the Google Sheet, starting from 0
  - E.g., `startup/gp_resources.0.FriendlyAngels.csv`
- The tab names have no spaces, so that `from_gsheet.py` and `to_gsheet.py` do
  a roundtrip
- The tabs are:

  | ID | Tab name              | CSV file                                         |
  | :- | :-------------------- | :----------------------------------------------- |
  | 0  | `FriendlyAngels`      | `startup/gp_resources.0.FriendlyAngels.csv`      |
  | 1  | `Accelerators`        | `startup/gp_resources.1.Accelerators.csv`        |
  | 2  | `FriendlyVCs`         | `startup/gp_resources.2.FriendlyVCs.csv`         |
  | 3  | `FriendlySalezBizDev` | `startup/gp_resources.3.FriendlySalezBizDev.csv` |
  | 4  | `AIGrants`            | `startup/gp_resources.4.AIGrants.csv`            |
  | 5  | `CryptoGrants`        | `startup/gp_resources.5.CryptoGrants.csv`        |
- `startup/ai_grants.md` is the notes on the grant programs, not a tab

# Summary of Commands

| Action       | Command          | Direction                      |
| :----------- | :--------------- | :----------------------------- |
| **Download** | `from_gsheet.py` | Google Sheet to the CSV files  |
| **Upload**   | `to_gsheet.py`   | CSV files to the Google Sheet  |

# Download

- Run `startup/download_gsheet.sh` to do the command below
  - Extra arguments go to `from_gsheet.py`, e.g., `--overwrite`
- Download all the tabs of the Google Sheet to the CSV files:
  ```bash
  > helpers_root/dev_scripts_helpers/google/from_gsheet.py \
    --url "https://docs.google.com/spreadsheets/d/1Qa9g-j0NSdStV46W_dc85TMrwywIEPvJAhzV4Ymokb0/edit" \
    --tabname _ALL_ \
    --output_file startup/gp_resources.csv
  ```
- `--tabname _ALL_` downloads all the tabs, one CSV file for each tab
- `--output_file` is the prefix of the CSV files, not the name of one file
- The command fails if a CSV file already exists
  - Add `--overwrite` to replace the existing CSV files
  - This also deletes the local changes that are not uploaded

# Upload

- Run `startup/upload_gsheet.sh` to do the command below
- Upload the CSV files to the tabs of the Google Sheet:
  ```bash
  > helpers_root/dev_scripts_helpers/google/to_gsheet.py \
    --input_file startup/gp_resources.csv \
    --url "https://docs.google.com/spreadsheets/d/1Qa9g-j0NSdStV46W_dc85TMrwywIEPvJAhzV4Ymokb0/edit" \
    --tabname _ALL_ \
    --overwrite
  ```
- `--tabname _ALL_` uploads all the CSV files, one tab for each file
- `--input_file` is the prefix of the CSV files, not the name of one file
- `--overwrite` replaces the content of the tabs that already exist
  - Without it, the command fails if a tab already exists
  - This cannot be undone, except with the version history of Google Sheets
- The CSV files store only the values, so the formulas and the formatting of the
  Google Sheet are lost
