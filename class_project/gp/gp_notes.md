# Difference in images

| Aspect | `python:3.12-slim` | `ubuntu:22.04` | `ubuntu:24.04` |
|---|---|---|---|
| What it is | Python-specific image | Generic Ubuntu OS | Generic Ubuntu OS |
| Python included? | ✓ Yes (3.12 pre-installed) | ✗ No (must install) | ✗ No (must install) |
| Pip included? | ✓ Yes | ✗ No | ✗ No |
| Base | Debian bookworm (minimal) | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |
| Image size | ~150–200 MB | ~70–80 MB (uncompressed ~250 MB) | ~80–90 MB (uncompressed ~280 MB) |
| Typical use case | Python applications | Any workload | Any workload |
| LTS support until | Until Python 3.12 EOL (Oct 2028) | April 2032 | April 2034 |
| Release date | Ongoing (updated with Python releases) | April 2022 | April 2024 |

python:3.12-slim

- Optimized for Python — everything already set up
- Minimal dependencies — only essentials for Python runtime
- Faster startup — no need to install Python/pip
- Smaller size — leanest option
- Trade-off — if you need system tools (curl, git, gcc), must install them separately

ubuntu:22.04 vs ubuntu:24.04 are both are full OS images, but:

ubuntu:22.04 (older, more stable)
- Longer time in field (proven, stable)
- Older packages (may have security patches backported)
- Some legacy software expects 22.04
- Support until 2032

ubuntu:24.04 (newer, modern)
- Newer system packages (better performance, newer features)
- Python 3.12 pre-installed in OS (vs 3.10 in 22.04)
- Newer GCC, better library versions (libgomp1, glibc, etc.)
- More suitable for modern development
- Support until 2034

## Install C/C++ toolchain

# uv vs pip

# 
- Run a Python script inside a container
  ```
  > cd tutorials/causalnex/
  > docker_cmd.sh "python /git_root/tutorials/causalnex/causalnex.API.py"
  ``` 

## In a notebook

jupyter lab --ContentsManager.allow_hidden=True


try:
  from IPython.display import display
except ImportError:
  display = print  # type: ignore
