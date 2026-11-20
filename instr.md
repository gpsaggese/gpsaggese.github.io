Propose how to extend

./helpers_root/dev_scripts_helpers/google/from_gsheet.py and to_gsheet.py

to

1) Handle a gsheet with multiple tabs. Save each tab in a different file. Use the
   file name file_name.<tab_name>.csv

2) Add an option to to_gsheet.py to make the first row bold and frozen. Use the
   existing functions in the codebase

3) Add an option to to_gsheet.py to control the "style" of the cells. The styles are
   "clipped", wrapped, and overflow. The default is "clipped"

4) Add unit tests that read and write gsheets in a tmp dir on an account. Disable
   the tests. Delete the tmp dir after the unit tests

# Plan

## Findings

- `from_gsheet.py` reads one tab only
  - `hgodrapi.get_tabs_from_gsheet()` already lists the tab names
  - `hgodrapi.read_all_gsheets()` cannot be reused
    - It reads through a pickle cache, so the data can be stale
    - It returns a list with no tab names
    - `hgodrapi.from_gsheet()` asserts on an empty tab. So one empty tab stops the
      read of all tabs
- `to_gsheet.py` already freezes and bolds the header, but with no option
  - It always sets `freeze_rows=True` in the call to `hgodrapi.to_gsheet()`
  - `_freeze_rows_in_gsheet(bold=True)` ties "freeze" and "bold" together in one call
    - It also asserts `num_rows_to_freeze > 0`. So today you cannot do
      "bold without freeze"
  - `hgodrapi.to_gsheet()` sets every row to 20 px when `freeze_rows=True`
    - This conflicts with `wrap` cells. So wrap mode needs auto-sized rows
- `hgodrapi.to_gsheet()` has only `set_text_wrapping_clip: bool` (CLIP or untouched)
  - The only caller in the repo is `save_df_to_tmp_gsheet()`
- There are no unit tests for `hgoogle_drive_api.py` or the two scripts
- No function deletes a Drive file or folder
  - `create_google_drive_folder()` and `create_or_overwrite_with_timestamp()` exist
  - `_TMP_GDRIVE_FOLDER_ID` is the existing temporary Drive folder

## Decisions

- Multi-tab download uses `--tabname _ALL_`, not `--tabs` or `--all_tabs`
- File name is `<output_file without .csv>.<ID>.<tab_name>.csv`
  - `<ID>` is the position of the tab in the spreadsheet. It starts from 0
- The disabled tests use `@pytest.mark.skipif` on the env var `RUN_GSHEET_TESTS`
  - The tests skip when the value is `0` or when the env var is not set
- The script keeps empty tabs. It writes one file per tab. Files and tabs have a 1-1
  relationship
- `to_gsheet.py` has one separate option for each of freeze, bold, cell style
  - Defaults: freeze on, bold on, cell style `clip`
  - The `--cell_style` values are `clip`, `wrap`, `overflow`
- In the lib, `cell_style` replaces `set_text_wrapping_clip`
  - There is no compatibility parameter. The change updates the only caller in the
    repo

## Proposal

### 1) Multi-tab download (`from_gsheet.py`)

- Extend `--tabname` with the special value `_ALL_`. This value downloads every tab
  to its own CSV file
  - Define the constant `_ALL_TABS = "_ALL_"`
  - Any other value is one tab name, as today
  - Without `--tabname`, the behavior is unchanged ("first tab" or the `gid` tab)
  - `--tabname` already overrides the `gid` in the URL. So `--tabname _ALL_`
    overrides it. The script logs a warning
  - Limitation: the script cannot select a tab named `_ALL_` by name
  - Update the help text of `--tabname`. Update the "Tab Selection" part of the
    module docstring
- Output file name is `<output_file without .csv>.<ID>.<tab_name>.csv`
  - `<ID>` is the position of the tab in the spreadsheet, from left to right
    - It starts from 0, like the Sheets API `index`
    - It is unique in the spreadsheet
  - E.g., `--output_file data.csv` with the tabs `Sheet1` and `My Data` gives
    `data.0.Sheet1.csv` and `data.1.My Data.csv`
  - If `--output_file` has no `.csv` extension, the whole name is the prefix
  - The tab name is kept. Only the characters that are not safe in a file name
    are encoded as `%XX` (see the roundtrip plan below, which replaced the first
    rule `re.sub(r"[^\w.-]", "_", tab_name)`)
    - Two tabs cannot give the same file name, because the ID is unique
  - If the user renames or moves a tab, the next run writes a new file. It leaves the
    old file
- Compute all output paths first
  - If any file exists and the user does not set `--overwrite`, fail before any
    download
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
    - It opens the spreadsheet once. It does not use the cache. It keeps empty tabs

### 2) Header freeze and bold (`to_gsheet.py`)

- Add two separate options with `hparser.add_bool_arg()`
  - `--freeze_header` / `--no_freeze_header`, default on
  - `--bold_header` / `--no_bold_header`, default on
- "Off" means "leave untouched". The script does not unfreeze or unbold a tab
  - Say this in the help text
- Pass `freeze_rows=args.freeze_header` and `bold_header=args.bold_header` to
  `hgodrapi.to_gsheet()`
- Lib change in `helpers/hgoogle_drive_api.py`
  - Move the bold request out of `_freeze_rows_in_gsheet()` into a new
    `_bold_rows_in_gsheet()`. Keep the code the same. Drop the `bold` parameter
  - `to_gsheet()` keeps `freeze_rows: bool = False`. It adds
    `bold_header: bool = False`. Freeze no longer implies bold
  - `save_df_to_tmp_gsheet()` also passes `bold_header=True`, so the sheet looks the
    same
  - The 20 px row height still depends on `freeze_rows` (no change)

### 3) Cell style (`to_gsheet.py`)

- Add `--cell_style {clip,wrap,overflow}`, default `clip`
  - Map to the Sheets API `wrapStrategy`: `CLIP`, `WRAP`, `OVERFLOW_CELL`
  - Note: today the script leaves the Sheets default, which is `overflow`. So the new
    default changes the look of existing uploads
- Lib change in `helpers/hgoogle_drive_api.py`
  - Replace `set_text_wrapping_clip: bool` with `cell_style: str = ""`
    (the empty string leaves the style untouched, as the coding rules prefer a
    `str` default to `Optional[str] = None`)
  - Generalize `_set_text_wrapping_clip_in_gsheet()` to
    `_set_text_wrapping_in_gsheet(..., cell_style)`. Use a
    `_CELL_STYLE_TO_WRAP_STRATEGY` dict. Use a `hdbg.dassert_in()` on the value
  - Update `save_df_to_tmp_gsheet()` to pass `cell_style="clip"`
- For `wrap`, call a new `_auto_resize_rows_in_gsheet()` after the script writes the
  data
  - It uses the Sheets API `autoResizeDimensions` on the rows
  - It overrides the fixed 20 px height that `freeze_rows=True` sets

### 4) Unit tests (disabled)

- New files, one test class per executable (`Test_from_gsheet_py`,
  `Test_to_gsheet_py`) plus one for the lib
  - `dev_scripts_helpers/google/test/test_from_gsheet.py` (+ `__init__.py`)
  - `dev_scripts_helpers/google/test/test_to_gsheet.py`
  - `helpers/test/test_hgoogle_drive_api.py`
  - Each file calls `pytest.importorskip("gspread")` before importing the module
  - Scripts run through `_run_main()` with a mocked `sys.argv`. Files go in
    `self.get_scratch_space()`
- Disable the tests with `@pytest.mark.skipif` on each class
  - The condition is `os.environ.get("RUN_GSHEET_TESTS", "0") == "0"`
  - The reason is "Needs Google credentials and network: set RUN_GSHEET_TESTS=1"
  - To run the tests, set `RUN_GSHEET_TESTS=1`
- Account and cleanup
  - `setUpClass` creates a Drive folder `tmp_test_<class>_<timestamp>` inside
    `_TMP_GDRIVE_FOLDER_ID`. It creates the test sheets inside this folder
  - `tearDownClass` deletes that folder. This also deletes the sheets, even when a
    test fails
  - New lib code
    - `create_tmp_gdrive_folder(name_prefix, *, credentials) -> str`
    - `delete_gdrive_file(file_id, *, credentials)`: permanent delete through the
      Drive API `files().delete()`. If the file is a folder, the delete also deletes
      its content
- Test cases
  - `to_gsheet`: write a CSV and read it back. Check for the same data
  - `to_gsheet`: header frozen and bold by default
  - `to_gsheet`: `--no_freeze_header` gives bold and not frozen. `--no_bold_header`
    gives frozen and not bold. One helper method checks the Sheets API
    `frozenRowCount` and `textFormat.bold`
  - `to_gsheet`: each `--cell_style` value sets the expected `wrapStrategy`
  - `to_gsheet`: existing tab without `--overwrite` fails
  - `from_gsheet`: one tab to CSV, same data as written
  - `from_gsheet --tabname _ALL_`: 4 tabs (data, a space in the name, header only, no
    cells) give 4 files named `data.<ID>.<tab_name>.csv`. `<ID>` goes from 0 to 3
    - The header-only file has only the header. The no-cells file is empty
  - `from_gsheet`: `--tabname _ALL_` with a `gid` in the URL still gives all tabs
  - `from_gsheet`: existing output file without `--overwrite` fails. The script
    writes nothing
  - `delete_gdrive_file`: after the delete, the file lookup returns 404
- Also add small enabled tests with no network (pure logic)
  - Output file name builder: sanitizing, ID in the name, no `.csv` extension
  - `cell_style` to `wrapStrategy` mapping, invalid value fails
  - `_parse()` defaults: `freeze_header` and `bold_header` are `True`, `cell_style`
    is `clip`

### Not in scope

- No `lib_<name>.py` split. New reusable code goes in `hgoogle_drive_api.py`, as the
  scripts do today
- `--tabname` takes one tab name or `_ALL_`, not a list of tab names
- No reverse multi-tab upload (`data.<ID>.<tab_name>.csv` files to one sheet)
- The plan does not touch the outdated
  `helpers/notebooks/hgoodle_drive_api.tutorial.py`

## TODO

- [x] Get answers to the open questions (all answered, see Decisions)
- [x] Lib: add `_read_worksheet()` and `from_gsheet_all_tabs()`
- [x] Lib: split `_bold_rows_in_gsheet()` out of `_freeze_rows_in_gsheet()`. Add
      `bold_header` to `to_gsheet()`
- [x] Lib: add `cell_style`, `_set_text_wrapping_in_gsheet()`,
      `_auto_resize_rows_in_gsheet()`
- [x] Lib: update `save_df_to_tmp_gsheet()` (`bold_header=True`, `cell_style="clip"`)
- [x] Lib: add `create_tmp_gdrive_folder()` and `delete_gdrive_file()`
- [x] `from_gsheet.py`: support `--tabname _ALL_`, output-name builder, update
      docstring
- [x] `to_gsheet.py`: add `--freeze_header`, `--bold_header`, `--cell_style`, update
      docstring
- [x] Add the enabled (no network) unit tests
- [x] Add the disabled (Google account) unit tests with folder setup and teardown
- [x] Update `dev_scripts_helpers/google/README.md` (what it does, examples)
- [x] Run the enabled tests and the linter
- [x] Run the disabled tests once by hand with real credentials. Then confirm that
      the tmp Drive folder is gone. This step needs approval from the user, because
      it creates and deletes files on the account
- [x] `git add` the new files. Do not commit

## Result

### Done
- Lib `helpers/hgoogle_drive_api.py`
  - Added `from_gsheet_all_tabs()` and `_read_worksheet()`. They keep the empty tabs
  - Split the bold request out of `_freeze_rows_in_gsheet()` into
    `_bold_rows_in_gsheet()`. `to_gsheet()` has the new `bold_header` parameter
  - Replaced `set_text_wrapping_clip` with `cell_style` (`clip`, `wrap`,
    `overflow`). The `wrap` style also fits the row height to the text
  - Added `create_tmp_gdrive_folder()` and `delete_gdrive_file()`
- `from_gsheet.py`: `--tabname _ALL_` writes one file per tab, named
  `<output_file without .csv>.<ID>.<tab_name>.csv`. The `<ID>` starts from 0
- `to_gsheet.py`: added `--freeze_header`, `--bold_header` (both on by default) and
  `--cell_style` (default `clip`)
- Tests: 9 enabled tests pass. 13 tests that use a Google account are disabled and
  run only with `RUN_GSHEET_TESTS=1`. They create a temporary Drive folder and
  delete it in `tearDownClass`
  - `dev_scripts_helpers/google/test/test_from_gsheet.py`
  - `dev_scripts_helpers/google/test/test_to_gsheet.py`
  - `helpers/test/test_hgoogle_drive_api.py`
- Updated `dev_scripts_helpers/google/README.md`
- Ran `ruff format` (clean) and `ruff check` on the changed files. The new files
  have the same import-order warning as the existing tests, which is a repo
  convention. `git add` is done for the new files. Nothing is committed

### Not done
- The 13 disabled tests did not run against a real Google account
  - They create and delete files on the Google Drive of the user. This step needs
    approval from the user
  - To run them: `RUN_GSHEET_TESTS=1 pytest dev_scripts_helpers/google/test
    helpers/test/test_hgoogle_drive_api.py`
  - Instead, an ad hoc script with fake Google APIs checked the files written for
    all tabs and the order of the Sheets API requests for each option
- Known limits
  - A tab with duplicate header cells makes `get_all_records()` fail, as it does
    today for one tab. With `--tabname _ALL_` this stops the whole download
  - The script cannot select a tab that is really named `_ALL_`

# Plan: Roundtrip of All Tabs

## How `to_gsheet.py` works today with many files

- It uploads one CSV file to one tab. It has no support for many files
  - `--input_file` must be one existing path. A glob such as `data.*.csv` fails
  - The tab name comes from `--tabname` (default `new_data`). The script does not
    read it from the file name
- To upload the files made by `from_gsheet.py --tabname _ALL_` today, you need a
  shell loop. The loop has these problems
  - You must pass each `--tabname` by hand. The `<ID>` and the clean tab name in
    the file name are not parsed
  - Without `--url`, each call creates a new temporary Google Sheet. The loop must
    pass the `--url` of the first call to the next calls
  - An empty file (a tab with no cells) fails, because `pd.read_csv()` raises
    `EmptyDataError`
  - The tab order is the loop order. A new Google Sheet keeps its empty `Sheet1`
    as an extra tab
  - The tab name is not exact. `My Data` goes to `data.1.My_Data.csv`, so it comes
    back as `My_Data`

## Proposal

- Add `--tabname _ALL_` to `to_gsheet.py`. It is the mirror of `from_gsheet.py`
  - `--input_file data.csv` finds the files `data.<ID>.<tab_name>.csv` in the same
    dir and sorts them by `<ID>`
  - Each file goes to the tab named by its `<tab_name>`
  - It is not valid with `--use_gid`
- Put the file name format in one place, the lib `helpers/hgoogle_drive_api.py`
  - `get_tab_csv_file_name(output_file, tab_id, tab_name) -> str` (moved from
    `from_gsheet.py`)
  - `get_tab_csv_files(input_file) -> List[Tuple[int, str, str]]`: the inverse. It
    returns `(tab_id, tab_name, file_path)` sorted by `tab_id`. It fails on a
    duplicate `<ID>`
  - `from_gsheet.py` and `to_gsheet.py` both call them, so they cannot drift
- Make the tab name in the file name reversible
  - Today `re.sub(r"[^\w.-]", "_", tab_name)` loses the original name
  - New rule: keep the name as it is. Replace only the characters `/ \ : * ? " < >
    | %` and the control characters with `%XX` (the hex code). The inverse
    decodes `%XX`
  - E.g., `My Data` gives `data.1.My Data.csv`. `a/b` gives `data.2.a%2Fb.csv`
- Empty tabs and empty files
  - An empty file (0 bytes) gives an empty tab with no cells
  - A header-only file gives a tab with only the header
- Where the tabs go
  - With `--url`: write each tab. Fail before writing anything if a tab exists and
    `--overwrite` is not set. Tabs that are not in the files stay as they are. New
    tabs go at the end, in `<ID>` order
  - Without `--url`: create a new temporary Google Sheet. Use its empty `Sheet1`
    for the first file. If that file is not named `Sheet1`, rename the tab. So the
    tabs and the files are 1-1, in the same order
- The formatting options (`--freeze_header`, `--bold_header`, `--cell_style`) apply
  to every tab
- Check all the files before the first write: the `<ID>` values are unique, and the
  file names parse

## Tests

- Enabled (no network)
  - `get_tab_csv_files()` is the inverse of `get_tab_csv_file_name()` for the names
    `Sheet1`, `My Data`, `a/b:c`, `100%`, `my-tab.v2`
  - The files of other prefixes in the same dir are ignored
  - Duplicate `<ID>` fails
- Disabled (Google account, `RUN_GSHEET_TESTS=1`)
  - Roundtrip: download all tabs of a test Google Sheet to dir A, upload them with
    `--tabname _ALL_` to a new temporary Google Sheet, download again to dir B.
    The two dirs have the same file names and content
  - The new Google Sheet has the same tab names and order as the first one
  - Upload to an existing Google Sheet fails before writing if one tab exists and
    `--overwrite` is not set

## Limits of the roundtrip

- The CSV files store values only
  - Formulas become their displayed text (e.g., `=HYPERLINK(...)` gives the link
    text, not the URL)
  - Formatting is lost: colors, column widths, data validation, merged cells
- `from_gsheet.py` converts text that looks like a number into a number. So `007`
  comes back as `7`
- A tab with duplicate header cells fails on download

## TODO

- [x] Get answers to the Questions below (yes: both recommended answers)
- [x] Lib: move the file name builder, add the reversible name and
      `get_tab_csv_files()`
- [x] `from_gsheet.py`: use the lib builder
- [x] `to_gsheet.py`: support `--tabname _ALL_`
- [x] Add the enabled and the disabled tests (including the roundtrip test)
- [x] Update the docstrings and `dev_scripts_helpers/google/README.md`

## Questions

- Q1: Do we make the tab name in the file name reversible?
  - **Recommended**: yes. `My Data` stays `My Data` in the file name. The roundtrip
    keeps the exact tab names
  - Alternative: keep `My_Data`. The roundtrip renames the tab to `My_Data`
- Q2: What happens to the tabs of an existing Google Sheet that are not in the
  files?
  - **Recommended**: they stay as they are. The script never deletes a tab
  - Alternative: add `--delete_other_tabs`, to make the Google Sheet match the
    files exactly

## Result of the Roundtrip

### Done
- `to_gsheet.py --tabname _ALL_` uploads the files
  `<input_file without .csv>.<ID>.<tab_name>.csv` to one tab each, in `<ID>` order.
  It is the inverse of `from_gsheet.py --tabname _ALL_`
  - The tab name in the file name is reversible. Only `/ \ : * ? " < > | %` and the
    control characters are encoded as `%XX`. `My Data` is `data.1.My Data.csv` and
    `a/b` is `data.4.a%2Fb.csv`
  - An empty file gives an empty tab. A header-only file gives a header-only tab
  - Without `--url`, the new temporary Google Sheet has exactly the tabs of the
    files, in order. The script deletes the empty first tab if no file uses its name
  - With `--url`, it fails before writing anything if a tab exists and
    `--overwrite` is not set. Tabs that are not in the files do not change
- Lib `helpers/hgoogle_drive_api.py`: `get_tab_csv_file_name()`,
  `get_tab_csv_files()`, `delete_tab_from_gsheet()`, `reorder_tabs_in_gsheet()`, and
  `create_tmp_gsheet(folder_id=...)`. `to_gsheet()` skips the write for a tab with no
  columns
- Live tests with `RUN_GSHEET_TESTS=1` passed on the real Google account: 36 tests
  - With `gspread` 5.12.4 (the version in `devops/docker_build/pip_list.txt`): 31
    passed in the full run. The 5 that failed with a `429` quota error passed in a rerun,
    after adding a 30 s wait before each test of `Test_to_gsheet_py`
  - With `gspread` 6.2.1: all the tests passed, except one wrong assertion that was
    fixed and one `429` that the wait removes
  - The roundtrip test passed: download, upload to a new temporary Google Sheet,
    download again. Same tab names, same order, same files
  - No `tmp_test_*` item was left in the temporary Google Drive folder

### Not done
- The Google Sheets API limits the read requests to 60 per minute and each tab needs
  about 9 of them. A Google Sheet with more than about 6 tabs can fail with a `429`
  error in the middle of an upload. Run it again with `--overwrite`
  - A fix is to pass the tab ID to the private helpers instead of looking it up by
    name each time. This would cut the reads to about 3 per tab. It is not done,
    because it changes many existing private functions
- The CSV files store only values. Formulas, formatting, and column widths of the
  original Google Sheet are not restored
- A tab with duplicate header cells fails on download, as it does for one tab today
