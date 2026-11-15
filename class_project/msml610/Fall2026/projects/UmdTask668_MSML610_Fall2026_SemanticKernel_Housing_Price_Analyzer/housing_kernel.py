"""Register and invoke the housing plugin using Semantic Kernel."""

import asyncio
import json

from semantic_kernel import Kernel
from housing_plugins.housing_plugin import HousingAnalysisPlugin

PLUGIN_NAME = "HousingAnalysis"


def create_kernel(plugin: HousingAnalysisPlugin | None = None) -> Kernel:
    """
    Create a kernel with the housing plugin registered.

    :param plugin: plugin instance to register; defaults to a plugin that
        loads the UCI dataset
    """
    kernel = Kernel()
    kernel.add_plugin(
        plugin if plugin is not None else HousingAnalysisPlugin(),
        plugin_name=PLUGIN_NAME,
    )
    return kernel


async def main():
    kernel = create_kernel()

    for function_name in ("dataset_summary", "compare_models"):
        result = await kernel.invoke(
            plugin_name=PLUGIN_NAME,
            function_name=function_name,
        )

        data = json.loads(str(result))
        print(f"\n{function_name}:")
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
