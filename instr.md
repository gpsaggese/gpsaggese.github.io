Propose how to extend

./helpers_root/dev_scripts_helpers/google/from_gsheet.py and to_gsheet.py

to 

1) handle a gsheet with multiple tabs, saving each tab on different file like
file_name.<tab_name>.csv

2) add an option to to_gsheet.py to turn the first row bold and frozen (using existing functions
in the codebase) 

3) add an option to to_gsheet.py to control the "style" of the cells (default
   "clipped", wrapped or overflow)

4) add unit tests (disabled) to read / write gsheets in a tmp dir on an account
   and deleted after the unit tests

# Plan

## Findings

- `from_gsheet.py` reads one tab only
  - `hgodrapi.get_tabs_from_gsheet()` already lists the tab names
  - `hgodrapi.read_all_gsheets()` cannot be reused
    - It reads through a pickle cache, so the data can be stale
    - It returns a list with no tab names
    - `hgodrapi.from_gsheet()` asserts on an empty tab, so one empty tab aborts all
- `to_gsheet.py` already freezes and bolds the header, but with no option
  - It hardwires `freeze_rows=True` in the call to `hgodrapi.to_gsheet()`
  - `_freeze_rows_in_gsheet(bold=True)` couples "freeze" and "bold" in one call
    - It also asserts `num_rows_to_freeze > 0`, so "bold without freeze" is
      impossible today
  - `hgodrapi.to_gsheet()` sets every row to 20 px when `freeze_rows=True`
    - This clashes with `wrap` cells, so wrap mode needs auto-sized rows
- `hgodrapi.to_gsheet()` has only `set_text_wrapping_clip: bool` (CLIP or untouched)
  - The only caller in the repo is `save_df_to_tmp_gsheet()`
- There are no unit tests for `hgoogle_drive_api.py` or the two scripts
- No function deletes a Drive file or folder
  - `create_google_drive_folder()` and `create_or_overwrite_with_timestamp()` exist
  - `_TMP_GDRIVE_FOLDER_ID` is the existing temporary Drive folder

## Decisions

- Multi-tab download uses `--tabname _ALL_`, not `--tabs` or `--all_tabs`
- File name is `<output_file without .csv>.<ID>.<tab_name>.csv`
  - `<ID>` is the position of the tab in the spreadsheet
- Empty tabs are kept: one file per tab, a 1-1 relationship between files and tabs
- `to_gsheet.py` has one separate option for each of freeze, bold, cell style
  - Defaults: freeze on, bold on, cell style `clip`
  - The `--cell_style` values are `clip`, `wrap`, `overflow`
- In the lib, `set_text_wrapping_clip` is replaced by `cell_style`
  - There is no compatibility parameter: the only caller in the repo is updated

## Proposal

### 1) Multi-tab download (`from_gsheet.py`)

- Extend `--tabname` with the special value `_ALL_`, which downloads every tab to its
  own CSV file
  - Define the constant `_ALL_TABS = "_ALL_"`
  - Any other value is one tab name, as today
  - Without `--tabname`, the behavior is unchanged ("first tab" or the `gid` tab)
  - `--tabname` already overrides the `gid` in the URL, so `--tabname _ALL_` wins
    over it, and the script logs a warning
  - Limitation: a tab that is really named `_ALL_` cannot be selected by name
  - Update the help text of `--tabname` and the "Tab Selection" part of the module
    docstring
- Output file name is `<output_file without .csv>.<ID>.<tab_name>.csv`
  - `<ID>` is the position of the tab in the spreadsheet, from left to right
    - It starts from 0, like the Sheets API `index` (see Open Questions)
    - It is unique in the spreadsheet
  - E.g., `--output_file data.csv` with the tabs `Sheet1` and `My Data` gives
    `data.0.Sheet1.csv` and `data.1.My_Data.csv`
  - If `--output_file` has no `.csv` extension, the whole name is the prefix
  - The tab name is sanitized with `re.sub(r"[^\w.-]", "_", tab_name)`
    - `hio.purify_file_name()` is not usable: it keeps only the basename
    - Two tabs cannot give the same file name, because the ID is unique
  - If a tab is renamed or moved, the next run writes a new file and leaves the old
    one
- Compute all output paths first
  - If any file exists and `--overwrite` is not set, fail before any download
- Keep empty tabs: write one file for every tab
  - A tab with only a header row gives a CSV with only the header
  - A tab with no cells gives an empty file (0 bytes)
  - Log a warning for each empty tab
  - The single-tab download keeps its "sheet is empty" assertion, unchanged
- New lib code in `helpers/hgoogle_drive_api.py`
  - `_read_worksheet(worksheet) -> pd.DataFrame`
    - Use `get_all_records()`
    - With no records, return a DataFrame with the columns from
      `worksheet.row_values(1)`
    - `from_gsheet()` uses it and keeps its assertion
  - `from_gsheet_all_tabs(url, *, credentials)`
    - Returns a dict `{tab_name: df}` in tab order (tab names are unique)
    - The caller gets `<ID>` with `enumerate()` over the dict
    - Opens the spreadsheet once, does not use the cache, keeps empty tabs

### 2) Header freeze and bold (`to_gsheet.py`)

- Add two separate options with `hparser.add_bool_arg()`
  - `--freeze_header` / `--no_freeze_header`, default on
  - `--bold_header` / `--no_bold_header`, default on
- "Off" means "leave untouched": the script does not unfreeze or unbold a tab
  - Say this in the help text
- Pass `freeze_rows=args.freeze_header` and `bold_header=args.bold_header` to
  `hgodrapi.to_gsheet()`
- Lib change in `helpers/hgoogle_drive_api.py`
  - Move the bold request out of `_freeze_rows_in_gsheet()` into a new
    `_bold_rows_in_gsheet()` (same code, moved) and drop the `bold` parameter
  - `to_gsheet()` keeps `freeze_rows: bool = False` and adds `bold_header: bool =
    False`: freeze no longer implies bold
  - `save_df_to_tmp_gsheet()` also passes `bold_header=True`, so its look is the same
  - The 20 px row height stays tied to `freeze_rows` (no change)

### 3) Cell style (`to_gsheet.py`)

- Add `--cell_style {clip,wrap,overflow}`, default `clip`
  - Map to the Sheets API `wrapStrategy`: `CLIP`, `WRAP`, `OVERFLOW_CELL`
  - Note: today the script leaves the Sheets default, which is `overflow`, so the new
    default changes the look of existing uploads
- Lib change in `helpers/hgoogle_drive_api.py`
  - Replace `set_text_wrapping_clip: bool` with `cell_style: Optional[str] = None`
    (`None` leaves the style untouched)
  - Generalize `_set_text_wrapping_clip_in_gsheet()` to
    `_set_text_wrapping_in_gsheet(..., cell_style)` with a
    `_CELL_STYLE_TO_WRAP_STRATEGY` dict and a `hdbg.dassert_in()` on the value
  - Update `save_df_to_tmp_gsheet()` to pass `cell_style="clip"`
- For `wrap`, call a new `_auto_resize_rows_in_gsheet()` after the data is written
  - It uses the Sheets API `autoResizeDimensions` on the rows
  - It overrides the fixed 20 px height set when `freeze_rows=True`

### 4) Unit tests (disabled)

- New files, one test class per executable (`Test_from_gsheet_py`,
  `Test_to_gsheet_py`) plus one for the lib
  - `dev_scripts_helpers/google/test/test_from_gsheet.py` (+ `__init__.py`)
  - `dev_scripts_helpers/google/test/test_to_gsheet.py`
  - `helpers/test/test_hgoogle_drive_api.py`
  - Each file calls `pytest.importorskip("gspread")` before importing the module
  - Scripts run through `_run_main()` with a mocked `sys.argv`, files in
    `self.get_scratch_space()`
- Disabled with `@pytest.mark.skip(reason="Needs Google credentials and network")` on
  each class (same idiom as `helpers/lib_tasks/test/test_lib_tasks.py`)
- Account and cleanup
  - `setUpClass` creates a Drive folder `tmp_test_<class>_<timestamp>` inside
    `_TMP_GDRIVE_FOLDER_ID`, and creates the test sheets inside it
  - `tearDownClass` deletes that folder, which also deletes the sheets, even when a
    test fails
  - New lib code
    - `create_tmp_gdrive_folder(name_prefix, *, credentials) -> str`
    - `delete_gdrive_file(file_id, *, credentials)`: permanent delete through the
      Drive API `files().delete()`; deleting a folder deletes its content
- Test cases
  - `to_gsheet`: write a CSV and read it back, same data
  - `to_gsheet`: header frozen and bold by default
  - `to_gsheet`: `--no_freeze_header` gives bold and not frozen, `--no_bold_header`
    gives frozen and not bold (one helper method checks the Sheets API
    `frozenRowCount` and `textFormat.bold`)
  - `to_gsheet`: each `--cell_style` value sets the expected `wrapStrategy`
  - `to_gsheet`: existing tab without `--overwrite` fails
  - `from_gsheet`: one tab to CSV, same data as written
  - `from_gsheet --tabname _ALL_`: 4 tabs (data, a space in the name, header only, no
    cells) give 4 files named `data.<ID>.<tab_name>.csv`, with `<ID>` 0 to 3
    - The header-only file has only the header, the no-cells file is empty
  - `from_gsheet`: `--tabname _ALL_` with a `gid` in the URL still gives all tabs
  - `from_gsheet`: existing output file without `--overwrite` fails, nothing written
  - `delete_gdrive_file`: after the delete, the file lookup returns 404
- Also add small enabled tests with no network (pure logic)
  - Output file name builder: sanitizing, ID in the name, no `.csv` extension
  - `cell_style` to `wrapStrategy` mapping, invalid value fails
  - `_parse()` defaults: `freeze_header` and `bold_header` are `True`, `cell_style`
    is `clip`

### Not in scope

- No `lib_<name>.py` split: new reusable code goes in `hgoogle_drive_api.py`, as the
  scripts do today
- `--tabname` takes one tab name or `_ALL_`, not a list of tab names
- No reverse multi-tab upload (`data.<ID>.<tab_name>.csv` files to one sheet)
- The outdated `helpers/notebooks/hgoodle_drive_api.tutorial.py` is not touched

## TODO

- [ ] Get answers to the Open Questions below
- [ ] Lib: add `_read_worksheet()` and `from_gsheet_all_tabs()`
- [ ] Lib: split `_bold_rows_in_gsheet()` out of `_freeze_rows_in_gsheet()`; add
  `bold_header` to `to_gsheet()`
- [ ] Lib: add `cell_style`, `_set_text_wrapping_in_gsheet()`,
  `_auto_resize_rows_in_gsheet()`
- [ ] Lib: update `save_df_to_tmp_gsheet()` (`bold_header=True`, `cell_style="clip"`)
- [ ] Lib: add `create_tmp_gdrive_folder()` and `delete_gdrive_file()`
- [ ] `from_gsheet.py`: support `--tabname _ALL_`, output-name builder, update
  docstring
- [ ] `to_gsheet.py`: add `--freeze_header`, `--bold_header`, `--cell_style`, update
  docstring
- [ ] Add the enabled (no network) unit tests
- [ ] Add the disabled (Google account) unit tests with folder setup and teardown
- [ ] Update `dev_scripts_helpers/google/README.md` (what it does, examples)
- [ ] Run the enabled tests and the linter
- [ ] Run the disabled tests once by hand with real credentials, then confirm that
  the tmp Drive folder is gone (needs the user's go-ahead: it creates and deletes
  files on the account)
- [ ] `git add` the new files, do not commit

## Open Questions

- Q1: Does the tab position `<ID>` start from 0 or from 1?
  - **Recommended**: 0 (same as the Sheets API `index`): `data.0.Sheet1.csv`
  - Alternative: 1: `data.1.Sheet1.csv`
- Q2: How are the tests "disabled"?
  - **Recommended**: unconditional `@pytest.mark.skip`; remove the line to run them
  - Alternative: skip unless an env var (e.g., `RUN_GSHEET_TESTS=1`) is set
