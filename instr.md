Add a script that creates or updates a gp_umd_ai_research@googlegroups.com
passing a 
--csv_file "/tmp/github_users.csv"

following the style

helpers_root/dev_scripts_helpers/github/invite_gh_contributors.py

## Plan

- [x] Ask clarifying questions
  - Backend: Google Workspace group via the Admin SDK Directory API
    (`@googlegroups.com` consumer groups have no API)
  - CSV: has an email column
  - Update mode: full sync (add missing, remove extra)
- [x] Create `helpers_root/dev_scripts_helpers/github/sync_google_group.py`
  - `uv run` shebang with inline dependencies, `chmod +x`
  - Docstring with `# Usage Example`, `_parse()` / `_main(parser)`
  - `--csv_file`, `--email_column`, `--group_email`, `--admin_email`,
    `--remove_extra` / `--no_remove_extra`, `--dry_run`, `-v`
- [x] Read emails from the CSV: strip, lowercase, drop empty, invalid, duplicates
- [x] Create the group if missing, add missing members, remove extra members
  - Members with the `OWNER` role are never removed
- [x] `--dry_run`: log `[DRY_RUN] Would ...` and do not change the group
- [x] Add unit tests and a `uv run --help` end-to-end test
- [x] Add the script to `dev_scripts_helpers/github/README.md`
- [x] `git add` the new files (no commit)

## Result

### Done
- Added `helpers_root/dev_scripts_helpers/github/sync_google_group.py`
  - Syncs a Workspace group to the emails in the CSV, idempotent
  - Same structure and conventions as `invite_gh_contributors.py`
- Added tests: `test/test_sync_google_group.py` (16 pass) and
  `test/test_sync_google_group_uv_run.py` (`uv run --help`, passes)
- Added the script to `dev_scripts_helpers/github/README.md`
- New files are staged with `git add`, not committed

### Not done
- Not run against the real Google API: no Workspace admin credentials here
  - Only the mocked Directory API is tested
- The group address is `gp_umd_ai_research@causify.ai` by default, not
  `@googlegroups.com`
  - The domain `causify.ai` is an assumption: override it with `--group_email`
- `/tmp/github_users.csv` does not exist, so the CSV was not tested
  - The email column name is assumed to be `Email`: override with `--email_column`
- Required one-time setup, outside the script
  - Enable the Admin SDK API in the Google Cloud project
  - Grant the scopes `admin.directory.group` and `admin.directory.group.member`
    to the service account with domain-wide delegation
  - Pass a Workspace admin with `--admin_email`
