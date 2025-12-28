"""
Explores the native API for Horovod distributed transformer language model training.

1. Citations:
   - Horovod: Sergeev, A., & Balso, M. D. (2018). Horovod: fast and easy
     distributed deep learning in TensorFlow. arXiv preprint arXiv:1802.05799.
   - Transformer Architecture: Vaswani, A., et al. (2017). Attention is all
     you need. Advances in neural information processing systems, 30.
   - GPT-style Language Modeling: Radford, A., et al. (2019). Language models
     are unsupervised multitask learners. OpenAI blog, 1(8), 9.

2. Make sure to run the linter on the script before committing changes.
   - Many changes would be pointed out by the linter to maintain consistency
     with coding style.

3. Reference documentation: horovod_transformer.API.md

The name of this script follows the format:
 - Since this project explores the Horovod distributed transformer API,
   it is named `horovod_transformer.API.py`

Follow the reference on coding style guide to write clean and readable code.
- https://github.com/causify-ai/helpers/blob/master/docs/coding/all.coding_style.how_to_guide.md
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.transformer_lm import TransformerLM
from src.data import get_dataloaders, load_preprocessed_data
from src.utils.config import load_config, Config
from src.utils.distributed import (
    setup_horovod,
    get_rank,
    get_world_size,
    metric_average,
    is_main_process,
)
from src.metrics import compute_perplexity, MetricsTracker
from src.generate import load_model_for_generation, generate_text

_LOG = logging.getLogger(__name__)


# #############################################################################
# API Exploration
# #############################################################################


class TransformerAPIExplorer:
    """
    Explores the API for the Horovod distributed transformer language model.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the API explorer with configuration.

        :param config_path: Path to the YAML configuration file.
        """
        self.config = load_config(config_path)
        _LOG.info("Loaded configuration from %s", config_path)

    def explore_model_creation(self) -> TransformerLM:
        """
        Explore creating a transformer language model instance.

        Demonstrates the TransformerLM API for model initialization with
        configurable architecture parameters.

        :return: Initialized TransformerLM model instance.
        """
        _LOG.info("Creating transformer model with config:")
        _LOG.info("  d_model: %d", self.config.model.d_model)
        _LOG.info("  n_layers: %d", self.config.model.n_layers)
        _LOG.info("  n_heads: %d", self.config.model.n_heads)

        model = TransformerLM(
            vocab_size=self.config.model.vocab_size,
            d_model=self.config.model.d_model,
            n_heads=self.config.model.n_heads,
            n_layers=self.config.model.n_layers,
            d_ff=self.config.model.d_ff,
            max_seq_len=self.config.model.max_seq_len,
            dropout=self.config.model.dropout,
        )

        param_count = sum(p.numel() for p in model.parameters())
        _LOG.info("Model created with %d parameters", param_count)

        return model

    def explore_data_loading(self) -> Tuple:
        """
        Explore the data loading API.

        Demonstrates loading preprocessed datasets and creating distributed
        data loaders for training and validation.

        :return: Tuple of (train_dataloader, val_dataloader).
        """
        _LOG.info("Loading preprocessed data from %s", self.config.data.data_dir)

        train_dataloader, val_dataloader = get_dataloaders(self.config)

        _LOG.info("Data loaders created:")
        _LOG.info("  Train batches: %d", len(train_dataloader))
        _LOG.info("  Val batches: %d", len(val_dataloader))

        return train_dataloader, val_dataloader

    def explore_model_forward(self, model: TransformerLM) -> Dict:
        """
        Explore the model forward pass API.

        Demonstrates how to perform forward passes with the transformer model,
        including input preparation and output interpretation.

        :param model: TransformerLM model instance.
        :return: Dictionary containing forward pass outputs.
        """
        _LOG.info("Exploring model forward pass")

        batch_size = 2
        seq_len = 128
        vocab_size = self.config.model.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = input_ids.clone()

        model.eval()
        with torch.no_grad():
            logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        _LOG.info("Forward pass completed:")
        _LOG.info("  Logits shape: %s", logits.shape)
        _LOG.info("  Loss value: %.4f", loss.item())
        _LOG.info("  Perplexity: %.2f", compute_perplexity(loss.item()))

        return {
            "logits": logits,
            "loss": loss.item(),
            "perplexity": compute_perplexity(loss.item()),
        }

    def explore_distributed_setup(self) -> Tuple[int, int, int]:
        """
        Explore the Horovod distributed training setup API.

        Demonstrates initializing Horovod and retrieving distributed training
        information such as world size, rank, and local rank.

        :return: Tuple of (world_size, rank, local_rank).
        """
        _LOG.info("Exploring Horovod distributed setup")

        world_size, rank, local_rank, use_cuda = setup_horovod(
            require_distributed=False
        )

        _LOG.info("Distributed setup:")
        _LOG.info("  World size: %d", world_size)
        _LOG.info("  Rank: %d", rank)
        _LOG.info("  Local rank: %d", local_rank)
        _LOG.info("  CUDA available: %s", use_cuda)

        return world_size, rank, local_rank

    def explore_metric_averaging(self, value: float, name: str = "test_metric") -> float:
        """
        Explore the distributed metric averaging API.

        Demonstrates how to average scalar metrics across all processes in
        distributed training using Horovod's allreduce operation.

        :param value: Metric value to average.
        :param name: Name for the metric operation.
        :return: Averaged metric value across all processes.
        """
        _LOG.info("Exploring metric averaging for metric: %s", name)
        _LOG.info("  Local value: %.4f", value)

        averaged_value = metric_average(value, name=name)

        _LOG.info("  Averaged value: %.4f", averaged_value)

        return averaged_value

    def explore_text_generation(
        self, checkpoint_path: str, prompt: str = "Once upon a time"
    ) -> str:
        """
        Explore the text generation API.

        Demonstrates loading a trained model checkpoint and generating text
        from a prompt using the autoregressive generation API.

        :param checkpoint_path: Path to model checkpoint file.
        :param prompt: Input text prompt for generation.
        :return: Generated text string.
        """
        _LOG.info("Exploring text generation API")
        _LOG.info("  Checkpoint: %s", checkpoint_path)
        _LOG.info("  Prompt: %s", prompt)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, config = load_model_for_generation(
            checkpoint_path=checkpoint_path,
            config_path="configs/config.yaml",
            device=device,
        )

        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            num_samples=1,
            device=device,
        )

        generated_text = generated_texts[0] if generated_texts else ""
        _LOG.info("  Generated: %s", generated_text[:100])

        return generated_text


def explore_config_api(config_path: str = "configs/config.yaml") -> Config:
    """
    Explore the configuration loading API.

    Demonstrates loading and accessing configuration values from YAML files
    using the Config class with nested attribute access.

    :param config_path: Path to configuration YAML file.
    :return: Config object with nested attribute access.
    """
    _LOG.info("Exploring configuration API")
    _LOG.info("  Config path: %s", config_path)

    config = load_config(config_path)

    _LOG.info("Configuration loaded:")
    _LOG.info("  Model d_model: %d", config.model.d_model)
    _LOG.info("  Training epochs: %d", config.training.epochs)
    _LOG.info("  Training learning_rate: %.2e", config.training.learning_rate)

    return config


def main():
    """
    Main function to explore the Horovod transformer API.
    """
    logging.basicConfig(level=logging.INFO)

    explorer = TransformerAPIExplorer()

    _LOG.info("=" * 60)
    _LOG.info("Exploring Horovod Transformer API")
    _LOG.info("=" * 60)

    config = explore_config_api()
    model = explorer.explore_model_creation()
    explorer.explore_model_forward(model)
    explorer.explore_distributed_setup()
    explorer.explore_metric_averaging(3.14159, "pi_approximation")

    _LOG.info("=" * 60)
    _LOG.info("API exploration complete")
    _LOG.info("=" * 60)


if __name__ == "__main__":
    main()

