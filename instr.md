### [ ] Standardize test/ dirs under msml610/tutorials to test_docker_all.py only

* Repo: umd_classes only (no helpers_root changes needed)
- [ ] umd_classes (https://github.com/gpsaggese/gpsaggese.github.io)

* Problem
- `msml610/tutorials/L*/test/` dirs are inconsistent:
  - 6 of 9 dirs (L05_statistical_learning, L07_prob_programming,
    L08_causal_inference, L09_kalman_filter, L09_multi_armed_bandits,
    L10_causal_discovery) carry a byte-identical `test/conftest.py` that
    duplicates what the repo-root `conftest.py` and `msml610/test/conftest.py`
    already provide (pytest path setup, `--dbg`/`--incremental` options,
    logger init). `L03_knowledge_representation` has no `conftest.py` and
    works fine, proving the per-dir copies are redundant.
  - 2 of 9 dirs have no `test/test_docker_all.py`, so their notebooks are not
    covered by an end-to-end Docker test:
    - `L06_bayesian_networks` (has an empty `test/` dir; notebooks:
      `L06_01_exact_inference.ipynb`, `L06_02_approximate_inference.ipynb`)
    - `L12_reinforcement_learning` (has no `test/` dir at all; notebooks:
      `L12_01_gridworld_4x3.ipynb`, `L12_02_gridworld_4x3_gymnasium.ipynb`)

* Solution

- [ ] PR1: Standardize `msml610/tutorials/L*/test/` dirs
  - Remove `test/conftest.py` from the 6 dirs listed above (rely on the
    existing root `conftest.py` and `msml610/test/conftest.py`)
  - Run the full `msml610/tutorials` unit test suite (not just the
    `@pytest.mark.slow` Docker tests) after removal to confirm nothing
    depended on the per-dir `sys.path` insertion or custom options
  - Add `test/test_docker_all.py` to `L06_bayesian_networks` and
    `L12_reinforcement_learning`, modeled on
    `L03_knowledge_representation/test/test_docker_all.py`
    (`Test_docker(hdoctest.DockerTestCase)`, one `@pytest.mark.slow` test
    method per notebook, calling `self.helper(notebook_name)`)
  - End state: every `msml610/tutorials/L*/test/` dir contains exactly one
    file, `test_docker_all.py`, and no `conftest.py`
