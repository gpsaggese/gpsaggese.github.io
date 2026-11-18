"""
Test the frozen Phase 0 campaign manifest.
"""

import dataclasses
import json
import unittest

import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma


class TestPhaseZeroManifest(unittest.TestCase):
    """
    Test campaign validation and stable identity.
    """

    def test1(self) -> None:
        """
        Test the reviewed default is internally consistent and no-spend.
        """
        manifest = rnmonexma.DEFAULT_PHASE_ZERO_MANIFEST
        self.assertFalse(manifest.execution_enabled)
        self.assertEqual(manifest.capability_window_size, 60)
        self.assertEqual(
            sum(item.bank_items for item in manifest.benchmark_families),
            300,
        )
        self.assertTrue(all(not item.allow_fallbacks for item in manifest.routes))

    def test2(self) -> None:
        """
        Test serialization and hashing are deterministic and content-sensitive.
        """
        manifest = rnmonexma.DEFAULT_PHASE_ZERO_MANIFEST
        rendered = rnmonexma.render_phase_zero_manifest(manifest)
        self.assertEqual(json.loads(rendered)["campaign_id"], manifest.campaign_id)
        first_hash = rnmonexma.phase_zero_manifest_hash(manifest)
        self.assertEqual(first_hash, rnmonexma.phase_zero_manifest_hash(manifest))
        changed = dataclasses.replace(manifest, bank_version="five-family-pilot-v2")
        self.assertNotEqual(first_hash, rnmonexma.phase_zero_manifest_hash(changed))


class TestRouteSpec(unittest.TestCase):
    """
    Test route safety constraints.
    """

    def test1(self) -> None:
        """
        Test the manifest refuses to enable provider fallback.
        """
        with self.assertRaises(ValueError):
            dataclasses.replace(
                rnmonexma.DEFAULT_PHASE_ZERO_MANIFEST.routes[0],
                allow_fallbacks=True,
            )
