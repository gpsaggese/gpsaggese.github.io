Step 1
create a script called compress_pdf.py using a dockerized version the code
in compress_pdf() in
./helpers_root/dev_scripts_helpers/documentation/lib_notes_to_pdf.py
if --backend "ghostscript_global"

Step 2
if --backend "ghostscript_dockerized"

docker pull minidocks/ghostscript
docker run --rm -v "$(pwd)":/data minidocks/ghostscript \
  gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
     -dNOPAUSE -dBATCH -dQUIET \
     -sOutputFile=/data/compressed.pdf /data/original.pdf

using the infrastructure already present in hdocker.py

## Plan
- [x] Step 1: create `compress_pdf.py`
  - [x] Add CLI script `helpers_root/dev_scripts_helpers/documentation/compress_pdf.py`
    following the `_parse()`/`_main()` script conventions
  - [x] Add `--input`/`-i`, `--output`/`-o` (default: overwrite `--input` in
    place), `--quality`, and `--backend` (choices: `ghostscript_global`,
    `ghostscript_dockerized`) CLI args
  - [x] Implement the `ghostscript_global` backend, adapted from
    `compress_pdf()` in `lib_notes_to_pdf.py` (host `gs` binary, not Docker)
  - [x] Have `ghostscript_dockerized` raise `NotImplementedError` (left for
    step 2)
  - [x] Add unit tests `helpers_root/dev_scripts_helpers/documentation/test/test_compress_pdf.py`
  - [x] `git add` the new files (no commit)
- [x] Step 2: implement the `ghostscript_dockerized` backend inside
  `compress_pdf.py` (`ARGUMENTS: step 2 ... inject the logic inside
  compress_pdf so that one can use either global ghostscript or a
  dockerized version`)
  - [x] Add `_compress_pdf_ghostscript_dockerized()`, using
    `helpers/hdocker.py`'s existing mount/path-conversion/run
    infrastructure (`get_docker_mount_context()`,
    `convert_caller_to_callee_docker_path()`,
    `build_and_run_docker_cmd()`) against the pre-built public
    `minidocks/ghostscript` image (no local Dockerfile build; image is
    auto-pulled if missing)
  - [x] Factor the shared `gs` options (`-sDEVICE=pdfwrite
    -dPDFSETTINGS=... -dCompatibilityLevel=1.4 -dNOPAUSE -dQUIET -dBATCH`)
    into `_build_gs_cmd_opts()`, used by both backends
  - [x] Wire `--backend ghostscript_dockerized` in `_main()` to call the
    new function instead of raising `NotImplementedError`
  - [x] Add `hdocker.add_dockerized_script_arg(parser)` for
    `--dockerized_use_sudo` (`--dockerized_force_rebuild` accepted but a
    no-op: no local image build)
  - [x] Update/add unit tests in `test_compress_pdf.py`
  - [x] Verify end-to-end with real Docker (image pull + `gs` run)
  - [x] `git add` the updated files (no commit)

## Result
- Done: created `compress_pdf.py` with a `--backend` flag; implemented the
  `ghostscript_global` path
  - Adapted `compress_pdf()` from `lib_notes_to_pdf.py`: same `gs` flags
    (`-sDEVICE=pdfwrite -dPDFSETTINGS=<quality> -dNOPAUSE -dQUIET -dBATCH`),
    generalized to support `--output` (not just in-place) via a
    `<output>.compressed.tmp` intermediate file
  - Found and fixed a real bug during testing: on this repo's thin client, a
    bare `gs` on `PATH` resolves to `dev_scripts_helpers/git/gs` (a `git
    status` alias), not Ghostscript, since that dir is ahead of
    `/opt/homebrew/bin` on `PATH`. Added `_find_gs_binary()`, which checks
    known Ghostscript install paths first, falling back to a plain `PATH`
    lookup
  - Verified end-to-end on a real PDF (`msml610/lectures_pdf/Lesson01.1-Intro.pdf`
    copy): compressed 1.5MB -> 616KB in place
  - Added `test_compress_pdf.py` (6 tests, all passing) covering
    `_find_gs_binary()`, `_compress_pdf_ghostscript_global()` (in-place and
    separate output file, via `capture_sys_calls`), and the CLI end-to-end
    (default backend, and `ghostscript_dockerized` raising
    `NotImplementedError`)
  - `git add`ed `compress_pdf.py` and its test file in `helpers_root` (not
    committed)
- Done: implemented the `ghostscript_dockerized` backend directly inside
  `compress_pdf.py`, so `--backend` now switches between the host `gs`
  binary and `gs` running in the `minidocks/ghostscript` Docker container
  - Reused `helpers/hdocker.py`'s existing infra rather than hand-rolling
    `docker run`: `get_docker_mount_context()` +
    `convert_caller_to_callee_docker_path()` to map host paths to the
    container's `/app` mount, and `build_and_run_docker_cmd()` (which
    already auto-pulls the image if missing, matching the `docker pull`
    step) to run it — same pattern used by `lib_svg.py`/`lib_pandoc.py`,
    but against a pre-built public image instead of a locally-built one
  - `minidocks/ghostscript` has no fixed entrypoint, so the full `gs ...`
    command is passed through as-is, matching the literal `docker run ...
    minidocks/ghostscript gs ...` example in this file
  - Extracted `_build_gs_cmd_opts()` (shared by both backends) and added
    `-dCompatibilityLevel=1.4`, matching the step-2 example; the global
    backend's test was updated for the new flag
  - Verified end-to-end with real Docker: pulled `minidocks/ghostscript`
    (~100MB) and compressed a real PDF in place inside
    `helpers_root` (output verified as a valid `%PDF-1.4`, 19-page file)
  - Added `Test__compress_pdf_ghostscript_dockerized` (2 tests: in-place
    and separate output file) and a CLI end-to-end test
    (`Test_compress_pdf_py::test2`, replacing the old
    `NotImplementedError` test), mocking only
    `hdocker.build_and_run_docker_cmd` (the Docker-execution boundary);
    the real (unmocked) `hdocker` path-conversion calls are used in the
    test itself to compute the expected `gs` command. 8/8 tests pass
  - `git add`ed the updated `compress_pdf.py` and `test_compress_pdf.py`
    in `helpers_root` (not committed)
- Not done: nothing outstanding from the instructions in this file; the
  `--dockerized_force_rebuild` flag is accepted for CLI consistency but is
  inert for this backend since no image is built locally
