---
title: "TIL: How to Not Let Coding Agents Commit Code Automatically"
authors:
    - gpsaggese
date: 2026-06-23
description:
draft: false
categories:
    - Developer Tools
    - Software Engineering
---

TL;DR Coding agents shouldn't auto-commit. There are four ways to enforce
that, from a static permission deny to a dynamic per-session authorization
flag. The last one also needs a second, independent rule or the agent can
create its own authorization.

<!-- more -->

## What I Learned

Coding agents shouldn't auto-commit.

The review itself is the value, not just for catching bugs, but for keeping my
mental model synchronized with the codebase. When an agent changes code and
commits immediately, I lose the chance to understand what happened and why.

This breaks my workflow in three ways:

**1. Mental Model Drift** Each agent change should integrate into my
understanding of the codebase. If the agent commits before I review, I miss
why a particular approach was chosen, what edge cases were considered, or how
it affects surrounding code.

**2. Large, Opaque PRs** Auto-committed batches create large PRs that are hard
to review. I prefer targeted, self-contained prompts that produce reviewable
diffs, not stacks of committed changes.

**3. Loss of Agency** The point of agents with `--yolo` mode is independence.
But that independence needs bounds: agents execute tasks autonomously without
permission prompts, yet stay under _your_ control when it comes to git
history.

## The Problem: Current LLM Behavior

Current-generation LLMs still don't follow complex instructions reliably. I
tried:

- Adding rules to project-level `CLAUDE.md`
- Adding rules to user-level `~/.claude/CLAUDE.md`
- Writing explicit instructions in prompts

Agents committed anyway. They nailed the "make the change" part but buried
the "don't commit" instruction in the noise. Prose instructions are a
suggestion; the agent can misread or deprioritize them. A permission boundary
is enforced by Claude Code itself, not by the model, so the fix has to live
there.

I ended up with four variants of the same idea, in increasing order of
flexibility.

## Approach 1: A Static Deny in the Project's Local Settings

The simplest fix: explicitly deny git commits and pushes in
`.claude/settings.local.json`, scoped to one project:

```json
{
    "permissions": {
        "deny": ["Bash(*git commit:*)", "Bash(*git push:*)"]
    }
}
```

This works because it's a hard boundary at the permission level: Claude Code
evaluates deny rules against the literal text of every Bash command it is
about to run, regardless of which program produces that text. A `*` on both
sides means the rule matches the substring `git commit` (or `git push`)
appearing _anywhere_ in the command, including inside a compound command like
`cd repo && git commit -m "x"`, or even inside an unrelated string that
happens to contain those words.

The downside: it's static. To let the agent commit again, I have to hand-edit
the JSON, remove the lines, and remember to put them back afterward. In
practice I forgot to put them back more than once.

## Approach 2: A Script That Toggles the Deny List

`./helpers_root/dev_scripts_helpers/ai/control_cc_commit.py` automates
exactly that toggle, on the same `.claude/settings.local.json` deny list from
Approach 1:

```bash
# Remove git commit/push from the deny list (agent can commit/push).
> control_cc_commit.py --enable

# Restore git commit/push to the deny list (agent is blocked again).
> control_cc_commit.py --disable
```

`--enable` scans the deny list for any entry containing `"git commit"` or
`"git push"`, removes those entries, and writes what it removed to a
`.backup` file next to the settings file. `--disable` reads that backup,
re-adds the removed entries, and deletes the backup so state doesn't linger.

This is still the same mechanism as Approach 1 (a static deny list), just
with a safer, scriptable toggle instead of hand-editing JSON: no risk of
forgetting the exact denied patterns, and no risk of leaving stray backup
state around after re-enabling the block. It's still a project-local,
all-or-nothing switch: once `--enable` runs, every git commit/push is allowed
until `--disable` runs, with no per-command review in between.

## Approach 3: A Global Deny in `~/.claude/settings.json`

Approaches 1 and 2 are scoped to one project's `.claude/settings.local.json`.
The same deny-rule mechanism also works in the user-level
`~/.claude/settings.json`, which applies to every project on the machine, not
just one:

```json
{
    "permissions": {
        "deny": [
            "Bash(*git commit:*)",
            "Bash(*git commit *)",
            "Bash(*git commit -m *)",
            "Bash(*git commit -F *)",
            "Bash(*git push:*)"
        ]
    }
}
```

This is the broadest, simplest guarantee: Claude can never commit or push in
any repo, on this machine, until I edit this file. It's also the least
flexible: there's no way to say "yes, here, but not there" with this
mechanism alone, because a deny rule at any scope always wins.

That's not a detail, it's the load-bearing fact underneath all four
approaches: **Claude Code's own documentation states that deny rules from any
settings scope are evaluated before allow rules, so a user-level deny always
blocks a project-level allow for the same command.** A project can only ever
add restrictions on top of what the user settings already forbid, never lift
one. Practically, that means once git commit/push is denied in
`~/.claude/settings.json`, no `.claude/settings.local.json` in any project,
however permissive, can undo it. To get commits back, the deny has to be
removed at the scope that set it (or, for Approach 4 below, replaced by
something else entirely).

## Approach 4: A Flag File Plus a PreToolUse Hook

The first three approaches are all static: authorizing a commit means
editing a settings file (by hand or via a script) and remembering to revert
it. What I actually wanted was a session-scoped switch: "yes, for a while,
starting now," checked live, on every attempt, without touching any JSON.

That needs two pieces:

1. **No static deny rule for `git commit`/`git push` at all.** Instead, the
   default Bash permission behavior applies: an unmatched command prompts for
   approval. A `PreToolUse` hook can then skip that prompt (return `allow`) or
   replace it with a hard block (return `deny`) before the prompt would ever
   appear.
2. **A hook script, `.claude/hooks/check_git_auth.sh`, that decides which.**
   Wired into `~/.claude/settings.json`:

    ```json
    {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "/path/to/repo/.claude/hooks/check_git_auth.sh",
                            "if": "Bash(git commit:*)"
                        },
                        {
                            "type": "command",
                            "command": "/path/to/repo/.claude/hooks/check_git_auth.sh",
                            "if": "Bash(git push:*)"
                        }
                    ]
                }
            ]
        }
    }
    ```

   The `if` field means this hook only runs when the Bash command matches
   `git commit` or `git push`; every other command is untouched. The script
   itself just checks whether a flag file exists:

    ```bash
    AUTH_FILE=".../.claude/git_authorized"
    if [[ ! -e "$AUTH_FILE" ]]; then
        # emit {"hookSpecificOutput": {"permissionDecision": "deny", ...}}
    else
        # emit {"hookSpecificOutput": {"permissionDecision": "allow", ...}}
    fi
    ```

To authorize a session, I open a normal terminal (not through Claude) and
run:

```bash
touch .claude/git_authorized   # allow commits/pushes from now on
rm .claude/git_authorized      # revoke
```

The flag file lives at `.claude/git_authorized`, next to the script, and is
`.gitignore`d so it never gets committed as a side effect of the very thing
it's gating.

This is strictly more useful than Approaches 1-3: there's no settings file to
edit per toggle, the decision is made live on every attempt (so it composes
with anything else that also wants to deny or ask), and revoking is a single
`rm`.

## "Won't Claude Just Create the File Itself?"

This is the obvious hole: if authorization is just "does a file exist", why
can't Claude create that file and authorize itself?

It can't, but only because of a second, independent rule: not because of
anything about the hook or the flag file. Two facts matter here:

**Edit/Read deny rules don't stop an arbitrary subprocess.** Claude Code's
own documentation is explicit about this: _"Read and Edit deny rules apply to
Claude's built-in file tools and to file commands Claude Code recognizes in
Bash, such as `cat`, `head`, `tail`, and `sed`. They don't apply to arbitrary
subprocesses that read or write files indirectly, like a Python or Node
script that opens files itself."_ So denying the `Edit` tool on
`.claude/git_authorized` blocks Claude's Write/Edit tool and blocks a shell
redirect like `echo x > .claude/git_authorized`, but it does **not** block
`python3 -c "open('.claude/git_authorized', 'w').close()"`. And `python3 *`
happens to already be on this project's allow list for unrelated reasons.
That's a real gap, and it's exactly the kind of gap that looks closed until
someone (or something) actually tries it.

**What actually closes it is a deny rule matched on raw command text, not on
parsed file operations:**

```json
{
    "permissions": {
        "deny": [
            "Bash(*git_authorized*)",
            "Edit(/.claude/git_authorized)"
        ]
    }
}
```

`Bash(*git_authorized*)` matches the literal substring `git_authorized`
anywhere in a Bash command: `touch`, `echo`, `cat`, `cp`, a `python3 -c
"..."` one-liner, a `node -e "..."` one-liner, anything. It doesn't care what
program is being invoked, only what text Claude is about to hand to the
shell, so it closes exactly the gap the Edit/Read mechanism leaves open.
`Edit(/.claude/git_authorized)` is added alongside it, as a second, belt and
suspenders layer that also covers the Write/Edit tool path directly.

Two things make this a structural block rather than a behavioral one:

- **Deny always wins.** As in Approach 3, a deny rule at any scope is
  evaluated before any allow rule, and Claude Code's hooks documentation adds
  the other half: _"Hook decisions don't bypass permission rules... a
  matching deny rule blocks the call"_, even if a hook (including this exact
  `check_git_auth.sh`) returned `allow`. So even a hook that Claude fully
  understands and could describe back to me cannot talk its way past this
  deny; it isn't evaluating the hook's intent, it's matching text.
- **It isn't a request, it's a rule.** The block isn't "please don't touch
  this file"; it's enforced by Claude Code before the command ever runs, the
  same way Approaches 1-3 are. Nothing about the model choosing a different
  phrasing, a different language, or a different tool changes the outcome,
  because the check runs on the command Claude Code is about to execute, not
  on what Claude intended.

The one honest caveat: this is still pattern matching on command text, not an
OS-level guarantee. The documentation's own suggestion for a true guarantee
is OS-level sandboxing (`sandbox.filesystem.denyWrite` on the flag file's
path), which blocks every process regardless of tool or command text. I
haven't turned that on here; the Bash-text deny is enough for the threat
model I actually care about (an agent following its instructions and
occasionally getting creative), not for an adversarial one.

## Comparing the Four

| Approach | Scope | Toggle | Enforcement |
| --- | --- | --- | --- |
| 1. Static deny in `.claude/settings.local.json` | One project | Hand-edit JSON | Permission engine |
| 2. `control_cc_commit.py` | One project | CLI flag, with backup/restore | Same deny list as (1) |
| 3. Static deny in `~/.claude/settings.json` | Every project | Hand-edit JSON | Permission engine |
| 4. Flag file + `PreToolUse` hook | One project | `touch`/`rm` a file | Hook, backstopped by a second deny rule |

I use (4) day to day and keep (1)/(3) as the fallback if I ever want a hard
stop with no toggle at all.

## References and Further Reading

- [Claude Code Documentation](https://claude.ai/code): Permission and
  settings configuration
- [Claude Code Permission System](https://claude.ai/code): How to configure
  deny rules, hooks, and their precedence
