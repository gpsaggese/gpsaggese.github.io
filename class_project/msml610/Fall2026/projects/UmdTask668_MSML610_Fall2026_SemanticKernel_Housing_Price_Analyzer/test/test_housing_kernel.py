"""Tests for Semantic Kernel plugin registration and invocation."""

import json
import asyncio

from housing_kernel import create_kernel, PLUGIN_NAME


def test_kernel_registration():
    """Verify that the housing plugin is registered."""
    kernel = create_kernel()

    assert PLUGIN_NAME in kernel.plugins
    assert "dataset_summary" in kernel.plugins[PLUGIN_NAME]
    assert "compare_models" in kernel.plugins[PLUGIN_NAME]


def test_kernel_invocation():
    """Verify that Semantic Kernel can execute the plugin."""
    async def run():
        kernel = create_kernel()

        result = await kernel.invoke(
            plugin_name=PLUGIN_NAME,
            function_name="dataset_summary",
        )

        return json.loads(str(result))

    data = asyncio.run(run())

    assert data["rows"] == 414
    assert len(data["features"]) == 6
    assert data["missing_values"] == 0
