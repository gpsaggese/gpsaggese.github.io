"""
MCP (Model Context Protocol) Client for Fake News Detection.

This client provides a high-level interface to interact with the MCP server,
enabling predictions, model management, and context-aware deployment.

Usage:
    client = FakeNewsMCPClient()
    result = await client.predict("Article text here")
    models = await client.list_models()
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fake_news_client")


class FakeNewsMCPClient:
    """Client for interacting with Fake News Detection MCP server."""

    def __init__(self, server_script: str = "MCP.server.py"):
        """
        Initialize the MCP client.

        Args:
            server_script: Path to the MCP server script
        """
        self.server_script = server_script
        self.session: Optional[ClientSession] = None

    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect to the MCP server."""
        try:
            # Create stdio parameters for server communication
            params = StdioServerParameters(
                command="python",
                args=[self.server_script]
            )

            # Initialize client session
            self.session = ClientSession(params)
            await self.session.__aenter__()
            logger.info("Connected to MCP server")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
                logger.info("Disconnected from MCP server")
            except Exception as e:
                logger.error(f"Error disconnecting: {str(e)}")

    async def _call_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool arguments

        Returns:
            Tool result
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        try:
            result = await self.session.call_tool(tool_name, kwargs)
            return result.content[0].text if result.content else {}
        except Exception as e:
            logger.error(f"Tool call failed: {str(e)}")
            return {'error': str(e), 'status': 'error'}

    async def _read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource from the MCP server.

        Args:
            uri: Resource URI (e.g., "model://registry")

        Returns:
            Resource content
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        try:
            resource = await self.session.read_resource(uri)
            return json.loads(resource.contents[0].text) if resource.contents else {}
        except Exception as e:
            logger.error(f"Resource read failed: {str(e)}")
            return {'error': str(e), 'status': 'error'}

    # High-level API methods

    async def predict(
        self,
        text: str,
        model_id: Optional[str] = None,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Make a prediction on input text.

        Args:
            text: News article text
            model_id: Optional specific model to use
            return_confidence: Include confidence scores

        Returns:
            Prediction result with label and confidence
        """
        kwargs = {
            'text': text,
            'return_confidence': return_confidence
        }
        if model_id:
            kwargs['model_id'] = model_id

        return await self._call_tool('predict', **kwargs)

    async def batch_predict(
        self,
        texts: List[str],
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make predictions on multiple texts.

        Args:
            texts: List of article texts
            model_id: Optional specific model to use

        Returns:
            Batch prediction results with statistics
        """
        kwargs = {'texts': texts}
        if model_id:
            kwargs['model_id'] = model_id

        return await self._call_tool('batch_predict', **kwargs)

    async def register_model(
        self,
        model_name: str,
        architecture: str,
        training_config: Dict[str, Any],
        test_metrics: Dict[str, float],
        dataset: str,
        model_path: str
    ) -> Dict[str, Any]:
        """
        Register a new model version.

        Args:
            model_name: Descriptive model name
            architecture: Architecture description
            training_config: Training parameters
            test_metrics: Performance metrics
            dataset: Training dataset name
            model_path: Path to saved model

        Returns:
            Registered model metadata
        """
        return await self._call_tool(
            'register_model_version',
            model_name=model_name,
            architecture=architecture,
            training_config=training_config,
            test_metrics=test_metrics,
            dataset=dataset,
            model_path=model_path
        )

    async def list_models(self) -> Dict[str, Any]:
        """
        List all registered models.

        Returns:
            Registry with all models and statistics
        """
        return await self._call_tool('list_all_models')

    async def get_registry(self) -> Dict[str, Any]:
        """
        Get the model registry.

        Returns:
            Complete registry data
        """
        return await self._read_resource('model://registry')

    async def get_active_model(self) -> Dict[str, Any]:
        """
        Get information about the active model.

        Returns:
            Active model metadata
        """
        return await self._read_resource('model://active')

    async def get_model_metrics(self, model_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific model.

        Args:
            model_id: Model ID

        Returns:
            Model performance metrics
        """
        return await self._read_resource(f'model://metrics/{model_id}')

    async def get_model_architecture(self, model_id: str) -> Dict[str, Any]:
        """
        Get architecture details for a model.

        Args:
            model_id: Model ID

        Returns:
            Architecture and configuration
        """
        return await self._read_resource(f'model://architecture/{model_id}')

    async def set_active_model(self, model_id: str) -> Dict[str, Any]:
        """
        Set the active model for predictions.

        Args:
            model_id: Model ID to activate

        Returns:
            Status confirmation
        """
        return await self._call_tool('set_active_model', model_id=model_id)

    async def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Compare performance across models.

        Args:
            model_ids: List of model IDs to compare

        Returns:
            Comparison table with metrics
        """
        return await self._call_tool('compare_models', model_ids=model_ids)

    async def get_model_context(
        self,
        model_id: Optional[str] = None,
        include_performance: bool = True,
        include_config: bool = True
    ) -> Dict[str, Any]:
        """
        Get complete context for a model (MCP context protocol).

        Args:
            model_id: Model ID (uses active if not specified)
            include_performance: Include performance metrics
            include_config: Include training config

        Returns:
            Complete model context
        """
        kwargs = {
            'include_performance': include_performance,
            'include_config': include_config
        }
        if model_id:
            kwargs['model_id'] = model_id

        return await self._call_tool('get_model_context', **kwargs)


async def example_usage():
    """Example usage of the MCP client."""
    async with FakeNewsMCPClient() as client:
        # Get registry
        registry = await client.get_registry()
        print("Registry:", json.dumps(registry, indent=2))

        # Make a prediction
        text = "Breaking news: Scientists discover new planet in habitable zone"
        prediction = await client.predict(text)
        print("Prediction:", json.dumps(prediction, indent=2))

        # List all models
        models = await client.list_models()
        print("Models:", json.dumps(models, indent=2))

        # Get active model context
        context = await client.get_model_context()
        print("Model Context:", json.dumps(context, indent=2))


if __name__ == '__main__':
    # Run example
    asyncio.run(example_usage())
