# `create_project.py`

- Creates a new project/tutorial directory from `class_project/project_template`
  and manages the Docker infrastructure files shared between the template and
  existing projects
- Main inputs: `--src_dir` (template directory) and `--dst_dir` (project
  directory)
- Main outputs, depending on `--action`:
  - A new project directory
  - Copied/linked Docker files
  - A comparison report

## Examples

- Create a new project from the default template:
  ```bash
  > create_project.py --dst_dir tutorials/my_project
  ```

- Overwrite an existing project directory:
  ```bash
  > create_project.py --dst_dir tutorials/my_project --overwrite
  ```

- Copy only the Docker files into an existing project (no template renaming):
  ```bash
  > create_project.py --dst_dir tutorials/my_project --action copy_docker_files
  ```

- Replace unmodified Docker files with links back to the template, so future
  template updates propagate automatically:
  ```bash
  > create_project.py --dst_dir tutorials/my_project --action create_links
  ```

- Report which Docker files are in sync, linked, or customized:
  ```bash
  > create_project.py --dst_dir tutorials/my_project \
      --action compare_docker_files
  ```

- Preview `create_links` without changing anything on disk:
  ```bash
  > create_project.py --dst_dir tutorials/my_project --action create_links \
      --dry_run
  ```

## Configuration & Inputs

### Command-line Arguments

// TODO(ai_gp): Remove type column
| Argument | Type | Default | Description |
| :------- | :--- | :------ | :---------- |
| `--src_dir` | str | `$GIT_ROOT/class_project/project_template` | Source template directory |
| `--dst_dir` | str | required | Destination project directory |
| `--overwrite` | flag | false | Overwrite `--dst_dir` if it already exists (`create_project` only) |
| `--dry_run` | flag | false | Preview an action without changing anything on disk |
| `--action` | str, repeatable | `create_project` | Add an action to the list of actions to run |
| `--skip_action` | str, repeatable | - | Remove an action from the list of actions to run |
| `--all_actions` | flag | false | Start from the list of all valid actions |
| `--clear_actions` | flag | false | Start from an empty list of actions |
| `-v` | str | `INFO` | Logging verbosity |

- `$GIT_ROOT` is the root of the Git client, resolved via `helpers.hgit`

### Actions

// TODO(ai_gp): Make description shorter
| Action | Description |
| :----- | :----------- |
| `create_project` | Copy `--src_dir` to `--dst_dir`, rename the template files to the project name, and customize `docker_name.sh` |
| `copy_docker_files` | Copy all and only the Docker files (see below) from `--src_dir` to `--dst_dir` |
| `create_links` | Replace Docker files in `--dst_dir` with soft links to `--src_dir`, for files whose content is unmodified |
| `compare_docker_files` | Print a table showing, for each Docker file, whether `--dst_dir` has the same content, a link, or diverged content, and generate a `vimdiff` script for the files that are different or missing |

### Docker Files

- `copy_docker_files`, `create_links`, and `compare_docker_files` all operate
  on the same fixed file set (`_DOCKER_FILES` in the script):
  - `.dockerignore`, `bashrc`, `docker_*.sh`, `Dockerfile*`, `etc_sudoers`,
    `requirements.txt`, `run_jupyter.sh`, `utils.sh`, `version.sh`
- This excludes:
  - Project-specific template files (e.g., `template.example.py`), which
    `create_project` renames instead
  - Generated artifacts (e.g., `docker_build.version.log`)
  - The template's own documentation (`docker_scripts.README.md`)

## Output & Side Effects

### Files Created / Modified

- `create_project`:
  - Creates `--dst_dir` as a full copy of `--src_dir`
  - Renames `template*` files to `<project_name>*`
  - Rewrites `IMAGE_NAME` in `docker_name.sh` to `umd_project_<project_name>`
- `copy_docker_files`:
  - Creates `--dst_dir` if missing
  - Copies the Docker files into it
- `create_links`:
  - Removes and replaces unmodified Docker files in `--dst_dir` with soft links into
    `--src_dir`
  - Customized files are left untouched
- `compare_docker_files`
  - Prints a report table to stdout, showing what Docker file is `different`,
    `missing_in_src`, or `missing_in_dst`
  - Creates `tmp.create_project.vimdiff.sh` in the current dir, an executable
    script that runs `vimdiff` on each such file (`--dry_run` logs the plan
    instead of creating the script)

## Software Architecture

### Data Flow

1. **Parse**:
   - Read `--src_dir` (defaults to `$GIT_ROOT/class_project/project_template`)
   - Read `--dst_dir`
   - Read and parse the list of `--action`s to run
2. **Dispatch**: for each selected action, call the matching handler function
3. **Execute**: each handler either mutates the filesystem (copy, rename,
   symlink) or produces a read-only report

### Key Functions

- `_get_default_src_dir() -> str`
  - Resolves `--src_dir`'s default from the Git client root
- `_run_create_project(src_dir, dst_dir, ...) -> None`
  - Orchestrates `create_project`: copy, rename, customize
- `_copy_docker_files(src_dir, dst_dir, ...) -> None`
  - Implements `copy_docker_files`
- `_create_links(src_dir, dst_dir, ...) -> None`
  - Implements `create_links`
- `_compare_docker_files(src_dir, dst_dir) -> pd.DataFrame`
  - Implements `compare_docker_files`
- `_main(parser)`
  - Selects and runs actions using the `helpers.hselect_action` idiom

### Design Patterns

- **Action idiom**: actions are selected via `helpers.hselect_action` (`--action`,
  `--skip_action`, `--all_actions`, `--clear_actions`), so multiple actions can run
  in one invocation
- **Dry-run**: every mutating action accepts a `dry_run` flag and logs the intended
  change instead of performing it
- **Fixed file whitelist**: `_DOCKER_FILES` is the single source of truth for what
  counts as a "Docker file", shared by all three Docker-file actions
