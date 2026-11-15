"""
Test the evaluation sets: disjoint seeds, task distributions and manifests.

Import as:

import test.test_tukey_sets as ttetuset
"""

import collections
import dataclasses
import json
import logging
import os
import pickle
import unittest.mock as umock

import helpers.hunit_test as hunitest
import tukey_env as tenv
import tukey_sets as tsets

_LOG = logging.getLogger(__name__)

# The committed manifests, found from this file so the tests run from any
# directory.
_MANIFEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs/sets",
)

# The open sets whose manifests Milestones 1-2 commit: the main and prototype
# validation sets and the 20 sets of the Milestone 2 benchmark. Later
# milestones add the manifests of the sets they use.
_COMMITTED_SETS = sorted(
    ["V_main", "V_proto"]
    + [n for n in tsets.EVAL_SETS if n.startswith(("B_m2_", "V_m2_"))]
)


class Test_eval_sets(hunitest.TestCase):
    """
    Seeds of different sets never collide, nor with training seeds.
    """

    def test1(self) -> None:
        """
        Set seeds are distinct (except paired suites) and outside training.
        """
        # Prepare inputs.
        paired = {
            f"{prefix}{suite}"
            for suite in tsets.M4_PAIRED
            for prefix in ("V_m4_", "T_m4_")
        } | set(tsets.NOISE_PAIRED)
        # Run test and check outputs.
        seen = set()
        n_seeds = 0
        for name, eval_set in tsets.EVAL_SETS.items():
            seeds = eval_set.seeds
            self.assertGreaterEqual(min(seeds), tsets.TRAIN_SEED_END)
            if name in paired:
                continue
            seen.update(seeds)
            n_seeds += len(seeds)
        self.assertEqual(len(seen), n_seeds)

    def test4(self) -> None:
        """
        Paired suites reuse their partner's seeds, on their own task.
        """
        # Run test and check outputs.
        for suite, partner in tsets.M4_PAIRED.items():
            for prefix in ("V_m4_", "T_m4_"):
                eval_set = tsets.EVAL_SETS[prefix + suite]
                partner_set = tsets.EVAL_SETS[prefix + partner]
                self.assertEqual(eval_set.seeds, partner_set.seeds)
                self.assertNotEqual(eval_set.task, partner_set.task)
        for name, partner in tsets.NOISE_PAIRED.items():
            eval_set = tsets.EVAL_SETS[name]
            partner_set = tsets.EVAL_SETS[partner]
            self.assertEqual(eval_set.seeds, partner_set.seeds)
            self.assertEqual(eval_set.sealed, partner_set.sealed)
            self.assertNotEqual(eval_set.task, partner_set.task)

    def test2(self) -> None:
        """
        Exactly the confirmatory test sets of M3 and M4 are sealed.
        """
        # Run test.
        sealed = sorted(n for n, s in tsets.EVAL_SETS.items() if s.sealed)
        # Check outputs.
        expected = ["T_main", "T_proto", "T_reserve", "T_sachs_matched"] + [
            f"T_m4_{suite}" for suite in tsets.M4_SUITES
        ]
        # The noise-level study v1 and the LLM comparison v1.
        expected += [f"T_noise_v1_L{level}" for level in (1, 2, 3, 4)]
        expected += ["T_llm_v1_main"]
        self.assertEqual(sealed, sorted(expected))

    def test3(self) -> None:
        """
        Every M4 suite has a validation twin on the same task.
        """
        # Run test and check outputs.
        for suite, task in tsets.M4_SUITES.items():
            for prefix in ("V_m4_", "T_m4_"):
                eval_set = tsets.EVAL_SETS[prefix + suite]
                self.assertEqual(eval_set.task, task)
                self.assertEqual(eval_set.n_instances, 200)

    def test5(self) -> None:
        """
        The Sachs-matched graphs have the Sachs data's size and split.
        """
        # Run test and check outputs.
        for name in ("V_sachs_matched", "T_sachs_matched"):
            self.assertEqual(tsets.EVAL_SETS[name].task, "sachs_matched")
        cfg = tsets.TASKS["sachs_matched"](0)
        self.assertEqual((cfg.n_nodes, cfg.n_edges), (11, 17))
        inst = tenv.sample_instance(cfg, 1)
        # 853 cells, of which `tukey_real_data.split_rows()` trains on 571.
        self.assertEqual(inst.d_train.shape, (571, 11))
        self.assertEqual(inst.d_test.shape, (282, 11))
        self.assertEqual(int(inst.truth.B.sum()), 17)


class Test_main_task_config(hunitest.TestCase):
    """
    The main task distribution draws d in {6, 7, 8} and 1-1.5 edges per node.
    """

    def test1(self) -> None:
        """
        Sizes and edge counts over 3,000 seeds.
        """
        # Run test and check outputs.
        sizes: collections.Counter = collections.Counter()
        for seed in range(3000):
            cfg = tsets.main_task_config(seed)
            sizes[cfg.n_nodes] += 1
            self.assertGreaterEqual(cfg.n_edges, cfg.n_nodes)
            self.assertLessEqual(cfg.n_edges, round(1.5 * cfg.n_nodes))
            self.assertEqual(cfg.family, "mixed")
        self.assertEqual(sorted(sizes), [6, 7, 8])
        self.assertGreater(min(sizes.values()), 900)

    def test2(self) -> None:
        """
        A family override keeps the size and the eqvar settings are enforced.
        """
        # Run test.
        cfg = tsets.main_task_config(5)
        cfg_raw = tsets.main_task_config(5, family="linear_gauss_eqvar_raw")
        # Check outputs.
        self.assertEqual(
            (cfg.n_nodes, cfg.n_edges), (cfg_raw.n_nodes, cfg_raw.n_edges)
        )
        self.assertEqual(
            (cfg_raw.noise_scale, cfg_raw.standardize), ("equal", "none")
        )


class Test_manifests(hunitest.TestCase):
    """
    Manifests round-trip, detect generator changes and guard sealed sets.
    """

    def test1(self) -> None:
        """
        A written manifest loads back and its instances pass the hash check.
        """
        # Prepare inputs.
        manifest_dir = self.get_scratch_space()
        # Run test.
        manifest = tsets.build_manifest(tsets.EVAL_SETS["V_proto"])
        path = tsets.manifest_path(manifest_dir, "V_proto")
        with open(path, "w") as file:
            json.dump(manifest, file)
        items = tsets.load_set("V_proto", manifest_dir)
        # Check outputs.
        self.assertEqual(len(items), 200)
        for seed, cfg, sha256 in items[:3]:
            inst = tsets.sample_checked(seed, cfg, sha256)
            self.assertEqual(inst.cfg, tenv.PROTO_CONFIG)

    def test2(self) -> None:
        """
        The hash is stable across samples and changes with the data.
        """
        # Prepare inputs.
        cfg = tsets.main_task_config(12)
        # Run test.
        hash1 = tsets.instance_hash(tenv.sample_instance(cfg, 12))
        hash2 = tsets.instance_hash(tenv.sample_instance(cfg, 12))
        # Check outputs.
        self.assertEqual(hash1, hash2)
        # The check is strict on the machine type that wrote the manifests.
        with umock.patch("platform.machine", return_value="aarch64"):
            with self.assertRaises(AssertionError):
                tsets.sample_checked(13, tsets.main_task_config(13), hash1)

    def test3(self) -> None:
        """
        Sealed sets refuse to load unless the caller is the confirmatory run.

        A small stand-in set is sealed, so no real test graph is generated.
        """
        # Prepare inputs.
        manifest_dir = self.get_scratch_space()
        name = "T_stand_in"
        eval_set = dataclasses.replace(
            tsets.EVAL_SETS["V_proto"], name=name, sealed=True, n_instances=2
        )
        # Run test and check outputs.
        with umock.patch.dict(tsets.EVAL_SETS, {name: eval_set}):
            with open(tsets.manifest_path(manifest_dir, name), "w") as file:
                json.dump(tsets.build_manifest(eval_set), file)
            with self.assertRaises(AssertionError):
                tsets.load_set(name, manifest_dir)
            items = tsets.load_set(name, manifest_dir, allow_sealed=True)
        self.assertEqual([seed for seed, _, _ in items], eval_set.seeds)

    def test4(self) -> None:
        """
        On another machine type a different hash is one warning, not an error.
        """
        # Prepare inputs.
        cfg = tsets.main_task_config(12)
        hash1 = tsets.instance_hash(tenv.sample_instance(cfg, 12))
        # Start from fresh counts; `patch.dict` restores them afterwards.
        fresh = {"checked": 0, "differ": 0}
        # Run test and check outputs.
        with (
            umock.patch("platform.machine", return_value="x86_64"),
            umock.patch.dict(tsets._OTHER_MACHINE_COUNTS, fresh),
        ):
            with self.assertLogs(tsets._LOG, level="WARNING") as logs:
                # An instance with the right hash, then two with a wrong one:
                # all three are accepted.
                tsets.sample_checked(12, cfg, hash1)
                for seed in (13, 14):
                    inst = tsets.sample_checked(
                        seed, tsets.main_task_config(seed), hash1
                    )
                    self.assertEqual(inst.seed, seed)
            self.assertEqual(
                tsets._OTHER_MACHINE_COUNTS, {"checked": 3, "differ": 2}
            )
        # One warning, at the first difference, naming its seed and machine.
        self.assertEqual(len(logs.records), 1)
        self.assertIn("Instance 13 ", logs.output[0])
        self.assertIn("x86_64", logs.output[0])

    def test5(self) -> None:
        """
        On another machine type a majority of differences is reported.
        """
        # Prepare inputs.
        cfg = tsets.main_task_config(12)
        hash1 = tsets.instance_hash(tenv.sample_instance(cfg, 12))
        fresh = {"checked": 0, "differ": 0}
        # Run test.
        with (
            umock.patch("platform.machine", return_value="x86_64"),
            umock.patch.dict(tsets._OTHER_MACHINE_COUNTS, fresh),
            umock.patch.dict(tsets._OTHER_MACHINE_WARNED, {"majority": False}),
            umock.patch.object(tsets, "_MIN_CHECKED", 3),
        ):
            with self.assertLogs(tsets._LOG, level="WARNING") as logs:
                # Three wrong hashes out of the first three instances.
                for seed in (13, 14, 15):
                    tsets.sample_checked(
                        seed, tsets.main_task_config(seed), hash1
                    )
                # Later differences add no warning.
                tsets.sample_checked(16, tsets.main_task_config(16), hash1)
        # Check outputs.
        self.assertEqual(len(logs.records), 2)
        self.assertIn("generator may have changed", logs.output[1])

    def test6(self) -> None:
        """
        A majority of differences reached after the first 20 is reported once.
        """
        # Prepare inputs.
        same = [True] * 20 + [False] * 80
        fresh = {"checked": 0, "differ": 0}
        # Run test.
        with (
            umock.patch.dict(tsets._OTHER_MACHINE_COUNTS, fresh),
            umock.patch.dict(tsets._OTHER_MACHINE_WARNED, {"majority": False}),
        ):
            with self.assertLogs(tsets._LOG, level="WARNING") as logs:
                for seed, is_same in enumerate(same):
                    tsets._note_other_machine(seed, is_same)
            counts = dict(tsets._OTHER_MACHINE_COUNTS)
        # Check outputs.
        self.assertEqual(counts, {"checked": 100, "differ": 80})
        # The first difference (instance 20), then the first majority: 21 of
        # 41 comparisons.
        self.assertEqual(len(logs.records), 2)
        self.assertIn("Instance 20 ", logs.output[0])
        self.assertIn("21 of the 41 instances checked so far", logs.output[1])


def _tampered_manifest(manifest_dir: str, name: str, change: str) -> None:
    """
    Copy a committed manifest into a directory with one change.

    :param manifest_dir: destination directory
    :param name: unsealed set name
    :param change: `header` (wrong offset), `order` (two seeds swapped),
        `duplicate` (a seed listed twice), `count` (one instance missing) or
        `size` (an instance's recorded size)
    """
    with open(tsets.manifest_path(_MANIFEST_DIR, name)) as file:
        manifest = json.load(file)
    instances = manifest["instances"]
    if change == "header":
        manifest["offset"] += 1
    elif change == "order":
        instances[0], instances[1] = instances[1], instances[0]
    elif change == "duplicate":
        instances[1] = dict(instances[0])
    elif change == "count":
        del instances[-1]
        manifest["n_instances"] -= 1
    else:
        assert change == "size"
        instances[0]["n_nodes"] += 1
    with open(tsets.manifest_path(manifest_dir, name), "w") as file:
        json.dump(manifest, file)


class Test_load_set_checks(hunitest.TestCase):
    """
    A manifest must describe exactly the set the code defines.
    """

    def test1(self) -> None:
        """
        Every committed unsealed manifest passes the checks.
        """
        # Prepare inputs.
        names = [
            name
            for name, eval_set in tsets.EVAL_SETS.items()
            if not eval_set.sealed
            and os.path.exists(tsets.manifest_path(_MANIFEST_DIR, name))
        ]
        # Run test and check outputs.
        self.assertEqual(sorted(names), _COMMITTED_SETS)
        for name in names:
            items = tsets.load_set(name, _MANIFEST_DIR)
            self.assertEqual(
                [seed for seed, _, _ in items], tsets.EVAL_SETS[name].seeds
            )

    def test2(self) -> None:
        """
        Each kind of tampering is refused.
        """
        # Run test and check outputs.
        for change in ("header", "order", "duplicate", "count", "size"):
            manifest_dir = os.path.join(self.get_scratch_space(), change)
            os.makedirs(manifest_dir)
            _tampered_manifest(manifest_dir, "V_m2_A0", change)
            with self.assertRaises(AssertionError, msg=change):
                tsets.load_set("V_m2_A0", manifest_dir)


class Test_curriculum_config(hunitest.TestCase):
    """
    The curriculum stages draw the sizes, densities and families of plan 3.9.
    """

    def test1(self) -> None:
        """
        Sizes, edge counts and families over 2,000 seeds per stage.
        """
        # Prepare outputs.
        expected_nodes = {
            1: [4, 5],
            2: [5, 6, 7],
            3: [6, 7, 8],
            4: list(range(6, 11)),
        }
        expected_families = {
            1: {"linear_exp"},
            2: {"linear_exp", "linear_uniform"},
            3: {"mixed"},
            4: {"mixed", "nonlinear"},
        }
        max_rho = {1: 1.0, 2: 1.5, 3: 1.5, 4: 2.0}
        # Run test and check outputs.
        for stage in (1, 2, 3, 4):
            nodes = set()
            families: collections.Counter = collections.Counter()
            for seed in range(2000):
                cfg = tsets.curriculum_config(seed, stage=stage)
                nodes.add(cfg.n_nodes)
                families[cfg.family] += 1
                self.assertGreaterEqual(cfg.n_edges, cfg.n_nodes)
                self.assertLessEqual(
                    cfg.n_edges, round(max_rho[stage] * cfg.n_nodes)
                )
            self.assertEqual(sorted(nodes), expected_nodes[stage])
            self.assertEqual(set(families), expected_families[stage])
            if stage == 4:
                share = families["nonlinear"] / 2000
                self.assertGreater(share, 0.25)
                self.assertLess(share, 0.35)

    def test2(self) -> None:
        """
        Stage 3 is the main task.
        """
        # Run test and check outputs.
        for seed in range(50):
            self.assertEqual(
                tsets.curriculum_config(seed, stage=3),
                tsets.main_task_config(seed),
            )


# Hashes (first 16 hex digits) of instances drawn before the noise-share option
# existed (commit `fe4bc7d`): adding it must not change any historical draw.
_COMPAT_HASHES = {
    "proto": ["f33824518c9d503d", "261bacaaf7383f52", "0265eaf311561598"],
    "mixed": ["6e1bbbcc18fa0508", "b8d75adf5d49be61", "f311b180c72e0e1a"],
    "linear_exp": ["c65cad988eb247f2", "b8d75adf5d49be61", "6344199c7a2fd2de"],
    "linear_uniform": [
        "6e1bbbcc18fa0508",
        "9d7ae453fd3110b1",
        "d4f09ce5da8a4332",
    ],
    "linear_gauss": ["7e739bd3ff0f4e4b", "c8e0a79a408097ec", "f311b180c72e0e1a"],
    "linear_gauss_eqvar_raw": [
        "b89df81228ce16ad",
        "95512b2d1d7564e3",
        "e16fa02ce18f8a08",
    ],
    "mixed_node": ["dc88c3eb462e9c86", "f4ac322d139d8f4d", "051569613feee549"],
    "nonlinear": ["82d48a6e3409bff0", "e0acc4f6d6e16362", "b38ae8e3e11a299a"],
    "iscm": ["4487eda9dd9258bf", "fc4c669317ddf46c", "2dbb8afecec91eb7"],
    "iscm_rescale": ["d17014f21a33b8e6", "e34fda97bd1369a0", "ab52a27355c3b051"],
    "pgmpy_lg": ["9b3d09eac46948b3", "e0cec61b16a107b9", "6bcce7eb5f8714aa"],
    "sklearn_reg": ["3044a7ff5df26032", "a1c0bc731dc85095", "0821a553b54a2e5e"],
    "sklearn_confounder": [
        "6e449527365e8f71",
        "81c3cb82298c0a04",
        "66bf2502cdeec84c",
    ],
    "scale_free": ["f7c613b082754c58", "f82221bdb031117b", "769a4f580b59b38e"],
    "task_main": ["59c5c554c2e6796b", "f6708eecb48016ce"],
    "task_curriculum_1": ["e11e03650eec8874", "498a71b75df3fa7e"],
    "task_curriculum_4": ["04c69c30b4320d6e", "591aa59a29b0b28d"],
    "task_confounder_hidden_a_6": ["ebd0d610069261eb", "bd452ef8af63dc39"],
    "task_confounder_hidden_c_6": ["76e0162707383511", "59a2a06d6e9a931a"],
    "task_confounder_observed_6": ["0c3396fc38c23efa", "b536c9ecec80d5f5"],
    "task_main_iscm": ["88e8476ea4da862f", "e229fca0b1d4c772"],
}


class Test_generator_compatibility(hunitest.TestCase):
    """
    Every family still draws the instances it drew before the noise share.
    """

    def test1(self) -> None:
        """
        Families on seeds 0-2 and tasks on seeds 11-12 hash as recorded.
        """
        if not tsets.is_reference_machine():
            self.skipTest(
                "The recorded hashes are bit-level and were written on Apple "
                "silicon (README, 'Other CPUs')"
            )
        # Prepare inputs.
        G = tenv.GraphConfig
        configs = {
            "proto": tenv.PROTO_CONFIG,
            "mixed": G(),
            "linear_exp": G(family="linear_exp"),
            "linear_uniform": G(family="linear_uniform"),
            "linear_gauss": G(family="linear_gauss"),
            "linear_gauss_eqvar_raw": tenv.EQVAR_RAW_CONFIG,
            "mixed_node": G(family="mixed_node"),
            "nonlinear": G(family="nonlinear", n_nodes=8, n_edges=10),
            "iscm": G(family="iscm"),
            "iscm_rescale": G(family="iscm", standardize="rescale"),
            "pgmpy_lg": G(family="pgmpy_lg"),
            "sklearn_reg": G(family="sklearn_reg", n_nodes=7, n_edges=3),
            "sklearn_confounder": G(
                family="sklearn_confounder",
                n_nodes=6,
                n_edges=2,
                confounder="observed",
            ),
            "scale_free": G(n_nodes=8, n_edges=12, topology="scale_free"),
        }
        # Run test.
        actual = {}
        for name, cfg in configs.items():
            actual[name] = [
                tsets.instance_hash(tenv.sample_instance(cfg, seed))[:16]
                for seed in (0, 1, 2)
            ]
        for task in (
            "main",
            "curriculum_1",
            "curriculum_4",
            "confounder_hidden_a_6",
            "confounder_hidden_c_6",
            "confounder_observed_6",
            "main_iscm",
        ):
            actual["task_" + task] = [
                tsets.instance_hash(
                    tenv.sample_instance(tsets.TASKS[task](seed), seed)
                )[:16]
                for seed in (11, 12)
            ]
        # Check outputs.
        self.assertEqual(actual, _COMPAT_HASHES)


class Test_noise_sets(hunitest.TestCase):
    """
    The noise-level sets pair their levels and record the noise share.
    """

    def test1(self) -> None:
        """
        The levels of one seed are one graph at four noise shares.
        """
        # Prepare inputs.
        seed = tsets.EVAL_SETS["V_noise_v1_L1"].seeds[0]
        # Run test.
        configs = [tsets.TASKS[f"noise_v1_L{k}"](seed) for k in (1, 2, 3, 4)]
        shares = [cfg.noise_share for cfg in configs]
        # Check outputs.
        self.assertEqual(tuple(shares), tsets.NOISE_GRID_V1)
        self.assertEqual(len({(c.n_nodes, c.n_edges) for c in configs}), 1)

    def test2(self) -> None:
        """
        A manifest records the share, and a wrong share is refused.
        """
        # Prepare inputs.
        manifest_dir = self.get_scratch_space()
        eval_set = dataclasses.replace(
            tsets.EVAL_SETS["V_noise_v1_L2"], n_instances=2
        )
        # Run test and check outputs.
        manifest = tsets.build_manifest(eval_set)
        self.assertEqual(
            [item["noise_share"] for item in manifest["instances"]], [0.4, 0.4]
        )
        self.assertEqual(
            manifest["instances"][0]["noise_grid"], list(tsets.NOISE_GRID_V1)
        )
        # A legacy set's manifest gains no key.
        legacy = dataclasses.replace(tsets.EVAL_SETS["V_main"], n_instances=1)
        self.assertNotIn(
            "noise_share", tsets.build_manifest(legacy)["instances"][0]
        )
        manifest["instances"][1]["noise_share"] = 0.6
        name = "V_noise_v1_L2"
        with umock.patch.dict(tsets.EVAL_SETS, {name: eval_set}):
            with open(tsets.manifest_path(manifest_dir, name), "w") as file:
                json.dump(manifest, file)
            with self.assertRaises(AssertionError):
                tsets.load_set(name, manifest_dir)


class Test_manifest_semantics(hunitest.TestCase):
    """
    The recorded target class, noise and share are checked beside the hash.
    """

    def test1(self) -> None:
        """
        A changed target class is refused; the metadata survives pickling.
        """
        # Prepare inputs.
        manifest_dir = self.get_scratch_space()
        name = "V_noise_v1_L3"
        eval_set = dataclasses.replace(tsets.EVAL_SETS[name], n_instances=2)
        manifest = tsets.build_manifest(eval_set)
        # Run test and check outputs.
        with umock.patch.dict(tsets.EVAL_SETS, {name: eval_set}):
            with open(tsets.manifest_path(manifest_dir, name), "w") as file:
                json.dump(manifest, file)
            items = tsets.load_set(name, manifest_dir)
            seed, cfg, expected = items[0]
            self.assertEqual(expected.meta["noise_share"], 0.6)
            tsets.sample_checked(seed, cfg, expected)
            copied = pickle.loads(pickle.dumps(expected))
            self.assertEqual(copied.meta, expected.meta)
            other = {"dag": "cpdag", "cpdag": "dag"}
            manifest["instances"][0]["target_class"] = other[
                manifest["instances"][0]["target_class"]
            ]
            with open(tsets.manifest_path(manifest_dir, name), "w") as file:
                json.dump(manifest, file)
            seed, cfg, expected = tsets.load_set(name, manifest_dir)[0]
            with self.assertRaises(AssertionError):
                tsets.sample_checked(seed, cfg, expected)
        # A plain hash string checks the hash only, as before.
        inst = tenv.sample_instance(cfg, seed)
        tsets.check_semantics(inst, tsets.instance_hash(inst))

    def test2(self) -> None:
        """
        The first instance of every committed open set matches its metadata.
        """
        # Prepare inputs.
        manifest_dir = _MANIFEST_DIR
        # Run test.
        checked = 0
        for name, eval_set in sorted(tsets.EVAL_SETS.items()):
            path = tsets.manifest_path(manifest_dir, name)
            if eval_set.sealed or not os.path.exists(path):
                continue
            seed, cfg, expected = tsets.load_set(name, manifest_dir)[0]
            tsets.sample_checked(seed, cfg, expected)
            checked += 1
        # Check outputs.
        self.assertEqual(checked, len(_COMMITTED_SETS))
