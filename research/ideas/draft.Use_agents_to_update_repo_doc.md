There are README files on a file basis and on a dir basis (e.g., README.md)

There are rules on how to maintain them (e.g., file.README.md)

We want to write a script that navigates the repo finds which dir / file has
been updated and update the corresponding README using an LLM agent

- Ingredients
  - Specs of README and exec.README
  - Find the last modification of the files and README (using git or an explicit tag)
    so that documentation is updated incrementally
  - Use llm_agent approach to update the doc
  - Have a way to run an update with --no_incremental

- Do a state of the art search to see what other systems exist (context7?) to handle
  this problem

### Convert update README into a script

- We want to have a command to find all the files modified after a date

E.g., research/ideas/prompt.update_README.md

- `find_newer_files.py` --file ... --dir ... --type ...
- `find_newer_files.py` --update_file ...

/Users/saggese/src/umd_classes3/helpers_root/.claude/skills/readme.create/SKILL.md
/Users/saggese/src/umd_classes3/helpers_root/.claude/skills/readme.write_architecture/SKILL.md
/Users/saggese/src/umd_classes3/helpers_root/.claude/skills/research_idea.update_readme/SKILL.md
