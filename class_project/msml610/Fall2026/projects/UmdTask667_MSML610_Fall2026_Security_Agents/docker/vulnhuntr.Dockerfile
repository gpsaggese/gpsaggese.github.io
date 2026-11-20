# Vulnhuntr (Protect AI) in an isolated Python 3.10 image.
#
# Build context is a checkout of https://github.com/protectai/vulnhuntr pinned
# at commit ead88c5 (2025-02-06), cloned into the git-ignored `third_party/`:
#
# > git clone https://github.com/protectai/vulnhuntr.git third_party/vulnhuntr
# > git -C third_party/vulnhuntr checkout ead88c5
# > docker build -f docker/vulnhuntr.Dockerfile -t vulnhuntr:ead88c5 \
#     third_party/vulnhuntr
#
# Differences from the upstream Dockerfile:
# - Installs the hash-locked `requirements.txt`, so versions match what
#   upstream tested
# - Upstream `pyproject.toml` omits `requests`, which `LLMs.py` imports, so
#   `pip install .` alone gives a `vulnhuntr` command that cannot start
# - The last line of upstream `requirements.txt` (`python-dotenv==1.0.0`) has
#   no hash, which makes pip reject the whole file, so the PyPI hash of that
#   wheel is added to a copy of the file
FROM python:3.10-bookworm

ARG DOTENV_HASH=sha256:f5971a9226b701070a4bf2c38c89e5a3f0d64de8debda981d1db98583009122a

WORKDIR /usr/src/vulnhuntr
COPY . .
RUN sed "s/^python-dotenv==1.0.0\$/python-dotenv==1.0.0 --hash=${DOTENV_HASH}/" \
        requirements.txt > /tmp/requirements.locked.txt \
    && pip install --no-cache-dir -r /tmp/requirements.locked.txt \
    && pip install --no-cache-dir --no-deps .

# Vulnhuntr writes `vulnhuntr.log` to the working dir; runs mount `/out` here.
WORKDIR /out
ENTRYPOINT ["vulnhuntr"]
