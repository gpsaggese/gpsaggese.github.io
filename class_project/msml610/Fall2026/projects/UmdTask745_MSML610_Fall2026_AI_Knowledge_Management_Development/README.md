# AI Knowledge Management for Development Teams

Decision-aware question answering over Architecture Decision Records (ADRs): extract each
decision, link it to the code it governs, and answer "why was this designed this way?" with
citations, abstaining when no rationale is recorded.

- Master issue: gpsaggese/gpsaggese.github.io#745
- Team: Gowtham S, Surendra Kumar, Saurabh Gujar
- Spec: research/ideas/draft.AI_Knowledge_Management_Development.md

## Status

Project skeleton only: the course template, renamed for this project. The prototype
(parser, linker, decision graph, retrieval, answer verification, evaluation) is added in
follow-up PRs.

## Layout

- `*.API.py`, `*.example.py` and notebooks: template files, to be filled in
- `Dockerfile`, `docker_*.sh`: Docker setup from the template