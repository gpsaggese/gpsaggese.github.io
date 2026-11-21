# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
this repository.

## Overview

This is the **UMD Classes Repository**: a comprehensive collection of educational
materials, tutorials, and projects for two graduate-level computer science courses
at the University of Maryland:

- **DATA605: Big Data Systems**: distributed systems, scalable data engineering,
  and big data technologies
- **MSML610: Advanced Machine Learning**: advanced ML techniques, research
  methods, and applied projects

The repository also contains shared development infrastructure and utilities via
the **helpers** subsystem.

## Repository Architecture

### Course Content Directories

| Directory | Purpose |
|-----------|---------|
| **data605/** | Lectures, tutorials, labs, and materials for DATA605: Big Data Systems |
| **msml610/** | Lectures, tutorials, labs, and materials for MSML610: Advanced Machine Learning |
| **class_project/** | Student project templates, examples, and guidelines |
| **tutorials/** | Standalone ML and data engineering tutorials (generic, not course-specific) |
| **class_scripts/** | Shell scripts and utilities for class operations |
| **dev_scripts_umd_classes/** | Development scripts specific to UMD classes repo management |

### Supporting Directories

| Directory | Purpose |
|-----------|---------|
| **helpers_root/** | Shared development infrastructure, utilities, and tools (see helpers_root/CLAUDE.md) |
| **assets/** | Logos, images, static content |
| **papers/** | Research papers, reading lists, academic references |
| **research/** | Research projects and experimental work |
| **books/** | Book references and educational materials |
| **website/** | Documentation and project website |

### Configuration and Infrastructure

- **`helpers_root/.claude/`**: Shared Claude Code configuration (symlinked as `.claude`)
- **`helpers_root/.pytest_cache/`**: Test cache for development
- **`helpers_root/config_root/`**: Configuration management system
- **`helpers_root/dev_scripts_helpers/`**: Development automation scripts

# Development Guidelines

### For Code Development

For all Python code, development patterns, testing conventions, and module
organization, refer to:

> See `helpers_root/CLAUDE.md` for comprehensive development guidance

This includes:
- Python code style and conventions
- Testing framework and patterns
- Module naming (`h<module>` convention)
- Task automation via `invoke`
- Linting and code quality checks

Key development commands are documented in `helpers_root/CLAUDE.md`.

## Working with Symlinks

The repository uses symlinks from root to `helpers_root/` for centralized
configuration:

```
.claude → helpers_root/.claude
.pytest_cache → helpers_root/.pytest_cache
.coveragerc → helpers_root/.coveragerc
conftest.py → helpers_root/conftest.py
# ... etc
```

This enables shared configuration while maintaining directory separation. When
working with symlinked files, the actual content lives in `helpers_root/`.

## Key Files and Configuration

- **`helpers_root/repo_config.yaml`**: Repository metadata, Docker config, S3
  buckets
- **`helpers_root/pytest.ini`**: Test markers and configuration
- **`helpers_root/pyproject.toml`**: Ruff linting rules (line length 81, Python
  3.11)
- **`how_to_contribute.md`**: Contribution guidelines for this repository
- **`.gitignore`**: Standard Python/Docker ignore patterns

# Important

- You MUST read and strictly follow `.claude/rules.md`

- You MUST never commit changes without user permission

- After writing code, you MUST run the commit gates
  (`helpers_root/dev_scripts_helpers/git/git_hooks/pre-commit.py`) and fix all
  errors before declaring the task done (see `helpers_root/CLAUDE.md`)
