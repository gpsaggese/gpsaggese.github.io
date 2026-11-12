"""
Evaluation sets: disjoint seed ranges, task distributions and manifests.

Each set is a block of instance seeds inside its own range, so no training
seed can coincide with an evaluation seed. A manifest lists every instance's
generator settings and a content hash of its data and graph. Manifests are
written before any training; evaluation refuses an instance whose hash no
longer matches (the generator changed). Sealed sets (the confirmatory test
sets) can only be loaded by the confirmatory evaluation.

The stored Milestone 2 table (run `wp3_m2_table`) rebuilt its instances from
the seeds alone, before manifests were checked; a later re-run that checked
every manifest reproduced its PC and GES values exactly. `load_set` now also
checks a manifest's header, its ordered seeds and each instance's size and
family against the code.

Hashes are computed inside the project's Docker image; numbers are rounded to
10 decimals first, so last-bit differences rarely change them (a value within
about 1e-10 of a rounding boundary still can). The committed hashes were
computed on arm64 (Apple silicon); no x86-64 run has checked them yet.

Import as:

import tukey_sets as tsets
"""

import dataclasses
import functools
import hashlib
import json
import logging
import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

import helpers.hdbg as hdbg
import tukey_env as tenv

_LOG = logging.getLogger(__name__)

# Training draws instance seeds in [0, TRAIN_SEED_END).
TRAIN_SEED_END = 1_000_000_000
# First seed of each evaluation range; ranges are 1e9 wide and disjoint.
SEED_RANGES = {
    "validation": 1_000_000_000,
    "m2": 2_000_000_000,
    "test_main": 3_000_000_000,
    "test_proto": 4_000_000_000,
    "test_reserve": 5_000_000_000,
    "m4": 6_000_000_000,
    # Noise-level study v1 and the LLM-agent comparison v1 (sealed sets).
    "noise_v1": 7_000_000_000,
    "llm_v1": 8_000_000_000,
}
# Seed-stream tag of the size draw, separate from the instance's own streams.
_SIZE_STREAM = 17
# Seed-stream tag of a curriculum stage's draws (size, density, family).
_CURRICULUM_STREAM = 23
# Noise-level study v1 (its plan, generator check and results arrive with
# Milestone 4): the noise shares of the four levels, and the fallback grid that
# the generator check selects if the primary one fails its rule.
NOISE_GRIDS_V1 = {
    "primary": (0.2, 0.4, 0.6, 0.8),
    "fallback": (0.15, 0.35, 0.55, 0.75),
}
# The grid the generator check selected (the primary grid passed its rule).
NOISE_GRID_V1 = NOISE_GRIDS_V1["primary"]


# #############################################################################
# Task distributions
# #############################################################################


def draw_size(
    seed: int, *, n_nodes_range: Tuple[int, int] = (6, 8)
) -> Tuple[int, int]:
    """
    Draw a graph size: d uniform in a range, `round(rho * d)` edges.

    :param seed: instance seed
    :param n_nodes_range: smallest and largest d (inclusive)
    :return: number of nodes and of edges (rho ~ U[1, 1.5])
    """
    rng = np.random.default_rng([seed, _SIZE_STREAM])
    n_nodes = int(rng.integers(n_nodes_range[0], n_nodes_range[1] + 1))
    n_edges = int(round(rng.uniform(1.0, 1.5) * n_nodes))
    return n_nodes, n_edges


def main_task_config(seed: int, *, family: str = "mixed") -> tenv.GraphConfig:
    """
    Settings of one instance of the main task distribution.

    d in {6, 7, 8}, `round(rho * d)` edges with rho ~ U[1, 1.5], linear,
    noise family drawn per graph (`mixed`), standardized with train
    statistics. `family` selects one noise family instead.

    :param seed: instance seed
    :param family: generator family
    :return: generator settings
    """
    n_nodes, n_edges = draw_size(seed)
    if family == "linear_gauss_eqvar_raw":
        cfg = dataclasses.replace(
            tenv.EQVAR_RAW_CONFIG, n_nodes=n_nodes, n_edges=n_edges
        )
    else:
        cfg = tenv.GraphConfig(n_nodes=n_nodes, n_edges=n_edges, family=family)
    return cfg


def fixed_size_config(seed: int, *, n_nodes: int) -> tenv.GraphConfig:
    """
    Settings for graphs of one size (d fixed, rho ~ U[1, 1.5], `mixed`).

    :param seed: instance seed
    :param n_nodes: number of nodes
    :return: generator settings
    """
    _, n_edges = draw_size(seed, n_nodes_range=(n_nodes, n_nodes))
    cfg = tenv.GraphConfig(n_nodes=n_nodes, n_edges=n_edges, family="mixed")
    return cfg


def curriculum_config(seed: int, *, stage: int) -> tenv.GraphConfig:
    """
    Settings of one instance of a curriculum stage (plan 3.9).

    - Stage 1: d in {4, 5}, d edges, exponential noise
    - Stage 2: d in {5, 6, 7}, `round(rho * d)` edges with rho ~ U[1, 1.5],
      exponential or uniform noise
    - Stage 3: the main task (d in {6, 7, 8}, mixed noise)
    - Stage 4: d in {6, ..., 10}, rho ~ U[1, 2], mixed noise in 70% of the
      graphs and the nonlinear family in 30%

    :param seed: instance seed
    :param stage: 1 to 4
    :return: generator settings
    """
    hdbg.dassert_in(stage, (1, 2, 3, 4))
    if stage == 3:
        return main_task_config(seed)
    rng = np.random.default_rng([seed, _CURRICULUM_STREAM])
    if stage == 1:
        n_nodes = int(rng.integers(4, 6))
        n_edges = n_nodes
        family = "linear_exp"
    elif stage == 2:
        n_nodes = int(rng.integers(5, 8))
        n_edges = int(round(rng.uniform(1.0, 1.5) * n_nodes))
        family = str(rng.choice(["linear_exp", "linear_uniform"]))
    else:
        n_nodes = int(rng.integers(6, 11))
        n_edges = int(round(rng.uniform(1.0, 2.0) * n_nodes))
        family = "nonlinear" if rng.uniform() < 0.3 else "mixed"
    cfg = tenv.GraphConfig(n_nodes=n_nodes, n_edges=n_edges, family=family)
    return cfg


def noise_level_config(seed: int, *, level: int) -> tenv.GraphConfig:
    """
    Settings of one instance of a level of the noise-level study v1.

    Sizes as in the main task (d in {6, 7, 8}, `round(rho * d)` edges with
    rho ~ U[1, 1.5]); iSCM with one noise family drawn per graph and the
    level's exact noise share on every node with parents. A seed gives the
    same graph and draws at every level (`tenv._sample_iscm_noise_share()`).

    :param seed: instance seed
    :param level: 1 to 4, the position in `NOISE_GRID_V1`
    :return: generator settings
    """
    hdbg.dassert_in(level, (1, 2, 3, 4))
    n_nodes, n_edges = draw_size(seed)
    cfg = tenv.GraphConfig(
        n_nodes=n_nodes,
        n_edges=n_edges,
        family="iscm",
        noise_share=NOISE_GRID_V1[level - 1],
        noise_grid=NOISE_GRID_V1,
    )
    return cfg


def _constant_config(seed: int, *, cfg: tenv.GraphConfig) -> tenv.GraphConfig:
    """
    Return the same settings for every seed.

    :param seed: instance seed (unused)
    :param cfg: settings
    :return: `cfg`
    """
    _ = seed
    return cfg


TASKS: Dict[str, Callable[[int], tenv.GraphConfig]] = {
    "main": main_task_config,
    "main_linear_exp": functools.partial(main_task_config, family="linear_exp"),
    "main_linear_uniform": functools.partial(
        main_task_config, family="linear_uniform"
    ),
    "main_linear_gauss": functools.partial(
        main_task_config, family="linear_gauss"
    ),
    "main_eqvar_raw": functools.partial(
        main_task_config, family="linear_gauss_eqvar_raw"
    ),
    "proto": functools.partial(_constant_config, cfg=tenv.PROTO_CONFIG),
    "mixed_12_18": functools.partial(
        _constant_config,
        cfg=tenv.GraphConfig(n_nodes=12, n_edges=18, family="mixed"),
    ),
    "nonlinear_8_10": functools.partial(
        _constant_config,
        cfg=tenv.GraphConfig(n_nodes=8, n_edges=10, family="nonlinear"),
    ),
    # U, T1, T2 and three unrelated features.
    "confounder_observed_6": functools.partial(
        _constant_config,
        cfg=tenv.GraphConfig(
            n_nodes=6,
            n_edges=2,
            family="sklearn_confounder",
            confounder="observed",
        ),
    ),
    "main_pgmpy": functools.partial(main_task_config, family="pgmpy_lg"),
    "main_mixed_node": functools.partial(main_task_config, family="mixed_node"),
    "main_nonlinear": functools.partial(main_task_config, family="nonlinear"),
    "main_iscm": functools.partial(main_task_config, family="iscm"),
    # 7 features, 4 of them informative, and y.
    "sklearn_reg_8_4": functools.partial(
        _constant_config,
        cfg=tenv.GraphConfig(n_nodes=8, n_edges=4, family="sklearn_reg"),
    ),
    # The size of the Sachs data (11 variables, bnlearn's 17 arcs, 853 cells
    # in the evaluation file): the simulated side of the sim-to-real gap.
    "sachs_matched": functools.partial(
        _constant_config,
        cfg=tenv.GraphConfig(
            n_nodes=11, n_edges=17, family="mixed", n_samples=853
        ),
    ),
}
for _n_nodes in (4, 5, 6, 7, 8, 9, 10, 12, 15):
    TASKS[f"size_{_n_nodes}"] = functools.partial(
        fixed_size_config, n_nodes=_n_nodes
    )


def _topology_config(seed: int, *, topology: str) -> tenv.GraphConfig:
    """
    Settings for 8-node graphs of one shape (`mixed` noise).

    Chains and stars have 7 edges; scale-free and layered graphs have
    `round(rho * 8)` edges with rho ~ U[1, 1.5].

    :param seed: instance seed
    :param topology: one of `tenv.TOPOLOGIES` other than "er"
    :return: generator settings
    """
    cfg = dataclasses.replace(
        fixed_size_config(seed, n_nodes=8), topology=topology
    )
    if topology in ("chain", "star"):
        cfg = dataclasses.replace(cfg, n_edges=cfg.n_nodes - 1)
    return cfg


def _eqvar_config(seed: int, *, standardize: str) -> tenv.GraphConfig:
    """
    Settings of the equal-variance Gaussian task with another data scaling.

    The same seed gives the same graph, weights and rows as `main_eqvar_raw`,
    scaled differently, so the scalings can be compared on paired instances.

    :param seed: instance seed
    :param standardize: "post" or "rescale"
    :return: generator settings
    """
    cfg = dataclasses.replace(
        main_task_config(seed, family="linear_gauss_eqvar_raw"),
        standardize=standardize,
    )
    return cfg


def _raw_config(seed: int, *, task: str) -> tenv.GraphConfig:
    """
    Settings of a task with the data kept on its raw scale.

    :param seed: instance seed
    :param task: key of `TASKS`
    :return: generator settings with `standardize="none"`
    """
    cfg = dataclasses.replace(TASKS[task](seed), standardize="none")
    return cfg


for _topology in ("scale_free", "chain", "star", "layered"):
    TASKS[f"{_topology}_8"] = functools.partial(
        _topology_config, topology=_topology
    )
TASKS["main_eqvar_std"] = functools.partial(_eqvar_config, standardize="post")
TASKS["main_eqvar_rescaled"] = functools.partial(
    _eqvar_config, standardize="rescale"
)
for _stage in (1, 2, 3, 4):
    TASKS[f"curriculum_{_stage}"] = functools.partial(
        curriculum_config, stage=_stage
    )
# Hidden common cause U of T1 and T2 (`n_edges` counts U's edges).
for _variant, _n_edges in (("a", 2), ("b", 4), ("c", 2)):
    TASKS[f"confounder_hidden_{_variant}_6"] = functools.partial(
        _constant_config,
        cfg=tenv.GraphConfig(
            n_nodes=6,
            n_edges=_n_edges,
            family="hidden_confounder",
            confounder=f"hidden_{_variant}",
        ),
    )
# Generator families of the M1 sortability report, on the raw scale.
SORTABILITY_TASKS = {
    "proto": "proto",
    "linear_exp": "main_linear_exp",
    "linear_uniform": "main_linear_uniform",
    "linear_gauss": "main_linear_gauss",
    "eqvar_raw": "main_eqvar_raw",
    "mixed": "main",
    "mixed_node": "main_mixed_node",
    "nonlinear": "main_nonlinear",
    "iscm": "main_iscm",
    "pgmpy_lg": "main_pgmpy",
    "sklearn_reg": "sklearn_reg_8_4",
    "sklearn_confounder": "confounder_observed_6",
    "mixed_d10": "size_10",
    "mixed_d15": "size_15",
    "scale_free_d8": "scale_free_8",
}
for _task in SORTABILITY_TASKS.values():
    TASKS[f"raw_{_task}"] = functools.partial(_raw_config, task=_task)
# Noise-level study v1: one task per noise share.
for _level in (1, 2, 3, 4):
    TASKS[f"noise_v1_L{_level}"] = functools.partial(
        noise_level_config, level=_level
    )


# #############################################################################
# Evaluation sets
# #############################################################################


@dataclasses.dataclass(frozen=True)
class EvalSet:
    """
    A named block of instance seeds drawn from one task distribution.
    """

    name: str
    # Range in `SEED_RANGES` and offset of the block inside it.
    seed_range: str
    offset: int
    n_instances: int
    # Key of `TASKS`.
    task: str
    # Sealed sets are loaded only by the confirmatory evaluation.
    sealed: bool

    @property
    def seeds(self) -> List[int]:
        """
        Instance seeds of the set.
        """
        first = SEED_RANGES[self.seed_range] + self.offset
        return list(range(first, first + self.n_instances))


# M4 suites by name and task (plan 5.5); appending keeps every seed offset.
M4_SUITES = {
    # The M3 task and the curriculum's last stage.
    "main": "main",
    "s4": "curriculum_4",
    # Size transfer.
    "size_10": "size_10",
    "size_12": "size_12",
    "size_15": "size_15",
    # Generator families not seen in training.
    "pgmpy": "main_pgmpy",
    "iscm": "main_iscm",
    "mixed_node": "main_mixed_node",
    "nonlinear": "main_nonlinear",
    "sklearn_reg": "sklearn_reg_8_4",
    # Graph shapes not seen in training (8 nodes).
    "scale_free_8": "scale_free_8",
    "chain_8": "chain_8",
    "star_8": "star_8",
    "layered_8": "layered_8",
    # Variance shortcut: paired instances, three scalings.
    "eqvar_raw": "main_eqvar_raw",
    "eqvar_std": "main_eqvar_std",
    "eqvar_rescaled": "main_eqvar_rescaled",
    # Relatives shortcut: Gaussian, standardized.
    "gauss_std": "main_linear_gauss",
    # Confounding: observed, then hidden (PAG targets).
    "conf_observed": "confounder_observed_6",
    "conf_hidden_a": "confounder_hidden_a_6",
    "conf_hidden_b": "confounder_hidden_b_6",
    "conf_hidden_c": "confounder_hidden_c_6",
}


# M4 suites drawn on another suite's seeds: the same graphs and rows, scaled
# differently, so the variance-shortcut comparison is paired graph by graph.
M4_PAIRED = {"eqvar_std": "eqvar_raw", "eqvar_rescaled": "eqvar_raw"}
# Noise-level sets drawn on the first level's seeds (set name -> partner).
NOISE_PAIRED = {
    f"{prefix}_noise_v1_L{level}": f"{prefix}_noise_v1_L1"
    for prefix in ("V", "T")
    for level in (2, 3, 4)
}


def _make_sets() -> Dict[str, EvalSet]:
    """
    Define every evaluation set of milestones M1 to M4.

    :return: sets by name
    """
    sets = [
        EvalSet("V_main", "validation", 0, 600, "main", False),
        EvalSet("V_proto", "validation", 1_000_000, 200, "proto", False),
        EvalSet("T_main", "test_main", 0, 600, "main", True),
        EvalSet("T_proto", "test_proto", 0, 200, "proto", True),
        EvalSet("T_reserve", "test_reserve", 0, 600, "main", True),
    ]
    # M2 benchmark settings, each with a validation twin for tuning.
    m2_tasks = [
        ("A0", "proto"),
        ("A1", "main_linear_exp"),
        ("B", "main_linear_gauss"),
        ("Braw", "main_eqvar_raw"),
        ("C", "main_linear_uniform"),
        ("D", "main"),
        ("H", "mixed_12_18"),
        # Added with the M1 families (manifests written before the M2 table).
        ("E", "nonlinear_8_10"),
        ("F", "confounder_observed_6"),
        ("G", "main_pgmpy"),
    ]
    for k, (setting, task) in enumerate(m2_tasks):
        offset = k * 1_000_000
        sets.append(EvalSet(f"B_m2_{setting}", "m2", offset, 200, task, False))
        sets.append(
            EvalSet(
                f"V_m2_{setting}",
                "validation",
                10_000_000 + offset,
                200,
                task,
                False,
            )
        )
    # Validation graphs of one size, to measure the matched budget.
    for n_nodes in (4, 5, 6, 7, 8, 9, 10, 12, 15):
        sets.append(
            EvalSet(
                f"V_size_{n_nodes}",
                "validation",
                50_000_000 + n_nodes * 1_000_000,
                200,
                f"size_{n_nodes}",
                False,
            )
        )
    # M1 sortability report: 100 raw-scale graphs per family.
    for k, (name, task) in enumerate(SORTABILITY_TASKS.items()):
        sets.append(
            EvalSet(
                f"V_sort_{name}",
                "validation",
                90_000_000 + k * 100_000,
                100,
                f"raw_{task}",
                False,
            )
        )
    # M4 suites (plan 5.5): a validation set for development and a sealed set
    # for the single confirmatory run, both written before any M4 results.
    position = {suite: k for k, suite in enumerate(M4_SUITES)}
    for suite, task in M4_SUITES.items():
        k = position[M4_PAIRED.get(suite, suite)]
        sets.append(
            EvalSet(
                f"V_m4_{suite}",
                "validation",
                100_000_000 + k * 1_000_000,
                200,
                task,
                False,
            )
        )
        sets.append(
            EvalSet(f"T_m4_{suite}", "m4", k * 1_000_000, 200, task, True)
        )
    # Sim-to-real gap (plan 3.10): graphs of the Sachs data's size, played
    # like real data; the sealed set is read only by the final real-data run.
    sets.append(
        EvalSet(
            "V_sachs_matched",
            "validation",
            150_000_000,
            200,
            "sachs_matched",
            False,
        )
    )
    sets.append(
        EvalSet("T_sachs_matched", "m4", 100_000_000, 200, "sachs_matched", True)
    )
    # Noise-level study v1: the four levels share their seeds (`NOISE_PAIRED`),
    # so the levels of one seed are the same graph at four noise shares.
    for level in (1, 2, 3, 4):
        task = f"noise_v1_L{level}"
        sets.append(
            EvalSet(
                f"V_noise_v1_L{level}",
                "validation",
                200_000_000,
                200,
                task,
                False,
            )
        )
        sets.append(
            EvalSet(f"T_noise_v1_L{level}", "noise_v1", 0, 200, task, True)
        )
    # LLM-agent comparison v1 on the main task: a development pool for the
    # pilot and a sealed set for the approved run.
    sets.append(
        EvalSet("V_llm_v1_main", "validation", 210_000_000, 50, "main", False)
    )
    sets.append(EvalSet("T_llm_v1_main", "llm_v1", 0, 200, "main", True))
    sets_by_name = {eval_set.name: eval_set for eval_set in sets}
    hdbg.dassert_eq(len(sets_by_name), len(sets))
    return sets_by_name


EVAL_SETS = _make_sets()


# #############################################################################
# Manifests
# #############################################################################


def instance_hash(inst: tenv.Instance) -> str:
    """
    Hash an instance's data and graph.

    :param inst: instance
    :return: SHA-256 hex digest of the rounded data, graph and weights
    """
    digest = hashlib.sha256()
    arrays = [
        inst.d_train.to_numpy(),
        inst.d_test.to_numpy(),
        inst.truth.B.astype(float),
        inst.W,
    ]
    # A PAG target is part of the instance; other targets add nothing, so
    # their hashes are unchanged.
    if inst.truth.pag is not None:
        arrays.append(inst.truth.pag.astype(float))
    for array in arrays:
        rounded = np.round(np.ascontiguousarray(array, dtype=float), 10)
        # Map -0.0 to 0.0 so the sign of zero cannot change the bytes.
        digest.update((rounded + 0.0).tobytes())
    return digest.hexdigest()


def build_manifest(eval_set: EvalSet) -> Dict[str, Any]:
    """
    Sample every instance of a set and record its settings and hash.

    :param eval_set: set definition
    :return: manifest dictionary
    """
    instances = []
    for seed in eval_set.seeds:
        cfg = TASKS[eval_set.task](seed)
        inst = tenv.sample_instance(cfg, seed)
        instances.append(
            {
                "seed": seed,
                "n_nodes": cfg.n_nodes,
                "n_edges": cfg.n_edges,
                "family": cfg.family,
                "noise": inst.noise_families[0],
                "target_class": inst.truth.target_class,
                "sha256": instance_hash(inst),
            }
        )
        if cfg.noise_share is not None:
            # Only the noise-level sets: older manifests keep their bytes.
            instances[-1]["noise_share"] = cfg.noise_share
            instances[-1]["noise_grid"] = list(cfg.noise_grid)
    manifest = {
        "name": eval_set.name,
        "task": eval_set.task,
        "seed_range": eval_set.seed_range,
        "offset": eval_set.offset,
        "n_instances": eval_set.n_instances,
        "sealed": eval_set.sealed,
        "instances": instances,
    }
    return manifest


class ManifestHash(str):
    """
    An instance hash read from a manifest, with the instance's recorded meta.

    `meta` holds the scoring metadata the manifest recorded beside the hash
    (`target_class`, `noise`, and `noise_share` for the noise-level sets).
    The hash covers the data and the graph, not these fields, so
    `sample_checked()` compares them too (`check_semantics()`). Being a `str`,
    it travels through every caller (and worker processes) unchanged.
    """

    meta: Dict[str, Any]


def _manifest_hash(item: Dict[str, Any]) -> ManifestHash:
    """
    Build the expected hash of a manifest entry with its metadata.
    """
    expected = ManifestHash(item["sha256"])
    expected.meta = {
        key: item[key]
        for key in ("target_class", "noise", "noise_share")
        if key in item
    }
    return expected


def check_semantics(inst: tenv.Instance, expected_hash: str) -> None:
    """
    Check an instance's scoring metadata against its manifest entry.

    A plain `str` hash (no recorded metadata) checks nothing here.

    :param inst: sampled instance
    :param expected_hash: hash from `load_set()` (a `ManifestHash`)
    """
    meta = getattr(expected_hash, "meta", None) or {}
    actual = {
        "target_class": inst.truth.target_class,
        "noise": inst.noise_families[0],
        "noise_share": inst.cfg.noise_share,
    }
    for key, value in meta.items():
        hdbg.dassert_eq(
            actual[key],
            value,
            "Instance %s differs from its manifest on '%s'",
            inst.seed,
            key,
        )


def manifest_path(manifest_dir: str, name: str) -> str:
    """
    Return the path of a set's manifest.

    :param manifest_dir: directory of the manifests
    :param name: set name
    :return: path of `<name>.json`
    """
    return os.path.join(manifest_dir, f"{name}.json")


def load_set(
    name: str, manifest_dir: str, *, allow_sealed: bool = False
) -> List[Tuple[int, tenv.GraphConfig, str]]:
    """
    Load a set's seeds, settings and expected hashes from its manifest.

    The manifest must describe exactly the set the code defines: the same
    header (`name`, `task`, `seed_range`, `offset`, `n_instances`, `sealed`),
    the same seeds in the same order without duplicates, and each instance's
    recorded size and family equal to its settings.

    :param name: set name
    :param manifest_dir: directory of the manifests
    :param allow_sealed: True only in the confirmatory evaluation
    :return: `(seed, settings, sha256)` per instance; the hash is a
        `ManifestHash` carrying the recorded scoring metadata
    """
    eval_set = EVAL_SETS[name]
    if eval_set.sealed:
        hdbg.dassert(
            allow_sealed,
            "Set '%s' is sealed: only the confirmatory evaluation loads it",
            name,
        )
    with open(manifest_path(manifest_dir, name)) as file:
        manifest = json.load(file)
    for key in ("name", "task", "seed_range", "offset", "n_instances", "sealed"):
        hdbg.dassert_eq(
            manifest[key],
            getattr(eval_set, key),
            "Manifest of '%s' disagrees with the code on '%s'",
            name,
            key,
        )
    seeds = [item["seed"] for item in manifest["instances"]]
    hdbg.dassert_eq(len(seeds), eval_set.n_instances)
    hdbg.dassert_no_duplicates(seeds)
    hdbg.dassert_eq(
        seeds, eval_set.seeds, "Manifest of '%s' lists other seeds", name
    )
    items = []
    for item in manifest["instances"]:
        cfg = TASKS[eval_set.task](item["seed"])
        keys = ["n_nodes", "n_edges", "family"]
        if cfg.noise_share is not None:
            keys.append("noise_share")
            hdbg.dassert_eq(
                item["noise_grid"],
                list(cfg.noise_grid),
                "Instance %s of '%s' disagrees on 'noise_grid'",
                item["seed"],
                name,
            )
        for key in keys:
            hdbg.dassert_eq(
                item[key],
                getattr(cfg, key),
                "Instance %s of '%s' disagrees on '%s'",
                item["seed"],
                name,
                key,
            )
        items.append((item["seed"], cfg, _manifest_hash(item)))
    return items


def sample_checked(
    seed: int, cfg: tenv.GraphConfig, expected_hash: str
) -> tenv.Instance:
    """
    Sample a manifest instance and check that the generator did not change.

    :param seed: instance seed
    :param cfg: generator settings
    :param expected_hash: hash recorded in the manifest
    :return: the instance
    """
    inst = tenv.sample_instance(cfg, seed)
    check_instance(inst, expected_hash)
    return inst


def check_instance(inst: tenv.Instance, expected_hash: str) -> None:
    """
    Check that an instance is the one its set manifest recorded.

    :param inst: sampled instance
    :param expected_hash: hash recorded in the manifest
    """
    hdbg.dassert_eq(
        instance_hash(inst),
        expected_hash,
        "Instance %s differs from its manifest: the generator changed",
        inst.seed,
    )
    check_semantics(inst, expected_hash)
