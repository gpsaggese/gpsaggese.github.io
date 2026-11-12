# How to Sync the `helpers_root` Submodule

- `helpers_root` is the `helpers` repo, included in this repo as a submodule
- `helpers` moved from `causify-ai/helpers` to `gpsaggese/helpers`
  - `.gitmodules` points to the new repo since 2026-11-02
  - A clone or a worktree made before then still fetches `helpers_root` from the old
    repo, which does not have the new commits
- This guide fixes such a clone
  - To move the pointer of `helpers_root` forward, see
    `helpers_root/docs/tools/dev_system/all.update_helpers_submodule.how_to_guide.md`

## Symptom

- `git submodule update` fails with `not our ref`:
  ```text
  fatal: remote error: upload-pack: not our ref 3da59ef60cac55ac9ae896c8d63c84b55ca7fdff
  fatal: Fetched in submodule path 'helpers_root', but it did not contain 3da59ef60cac55ac9ae896c8d63c84b55ca7fdff. Direct fetching of that commit failed.
  ```
- Check the URLs that your clone uses:
  ```bash
  > git config --get submodule.helpers_root.url
  > git -C helpers_root remote get-url origin
  ```
  - If either one says `causify-ai/helpers`, the clone needs the fix

## Fix a Clone

- Run from the root of the repo:
  ```bash
  > git pull
  > git submodule sync --recursive
  > git submodule update --init --recursive
  ```
  - `git pull` gets the new `.gitmodules`
  - `git submodule sync` copies its URL into `.git/config` and into the `origin`
    remote of `helpers_root`
  - `git submodule update` checks out the commit of `helpers` that this repo points
    to
- Check that it worked:
  ```bash
  > git submodule status
  ```
  - The line of `helpers_root` starts with a space, not with `-` or `+`

## Fix a Worktree

- Each worktree has its own copy of `helpers_root`, so fix each one
- Check that the branch of the worktree has the new `.gitmodules`:
  ```bash
  > git config -f .gitmodules --get submodule.helpers_root.url
  ```
  - If it says `causify-ai/helpers`, the branch is older than the move: merge
    `master` into it first
- If the worktree sets its own URL, `git submodule sync` keeps it
  - Check it:
    ```bash
    > git config --worktree --get submodule.helpers_root.url
    ```
  - If it says `causify-ai/helpers`, set it to the new repo:
    ```bash
    > git config --worktree submodule.helpers_root.url https://github.com/gpsaggese/helpers.git
    ```
- Then sync and update:
  ```bash
  > git submodule sync --recursive
  > git submodule update --init --recursive
  ```
  - After the sync, `helpers_root` fetches over SSH, from the URL of `.gitmodules`:
    without an SSH key, see [Use HTTPS Instead of SSH](#use-https-instead-of-ssh)

## Use HTTPS Instead of SSH

- `.gitmodules` uses SSH: `git@github.com:gpsaggese/helpers.git`
- Without an SSH key for GitHub, use HTTPS for one command, as in step 4 of
  [`pr_process.how_to_guide.md`](pr_process.how_to_guide.md):
  ```bash
  > git -c url."https://github.com/".insteadOf="git@github.com:" \
      submodule update --init --recursive
  ```

## Do Not Move the Pointer by Accident

- A commit that stages `helpers_root` moves the pointer for everyone once it is
  merged
- Stage only the files you changed, never `git add .` or `git commit -a`, and check
  before each commit:
  ```bash
  > git diff --cached --name-only
  ```
  - `helpers_root` must not be in the list, unless you mean to move the pointer

## If the Commit Is Missing

- Some older commits of `master` point to `helpers` commits that are in neither repo,
  e.g., `347c111f` and `328259b3`
  - On such a commit, `git submodule update` fails with `not our ref` even with the
    new URL
  - Use a newer commit of `master`; the missing commits are tracked in
    [#874](https://github.com/gpsaggese/gpsaggese.github.io/issues/874)
