"""
End-to-end example demonstrating how to use the Horovod distributed transformer
training API in a complete project workflow.

1. Citations:
   - Horovod: Sergeev, A., & Balso, M. D. (2018). Horovod: fast and easy
     distributed deep learning in TensorFlow. arXiv preprint arXiv:1802.05799.
   - Transformer Architecture: Vaswani, A., et al. (2017). Attention is all
     you need. Advances in neural information processing systems, 30.
   - BookCorpus Dataset: Zhu, Y., et al. (2015). Aligning books and movies:
     Towards story-like visual explanations by watching movies and reading books.
     Proceedings of the IEEE international conference on computer vision.

2. Make sure to run the linter on the script before committing changes.
   - Many changes would be pointed out by the linter to maintain consistency
     with coding style.

3. Reference documentation: horovod_transformer.example.md

The name of this script follows the format:
 - Since this project uses the Horovod distributed transformer API,
   it is named `horovod_transformer.example.py`

Follow the reference on coding style guide to write clean and readable code.
- https://github.com/causify-ai/helpers/blob/master/docs/coding/all.coding_style.how_to_guide.md
"""

import logging
import os
from typing import Optional

import torch

from src.models.transformer_lm import TransformerLM
from src.data import get_dataloaders
from src.utils.config import load_config
from src.utils.distributed import setup_horovod, get_rank, is_main_process
from src.generate import load_model_for_generation, generate_text
from src.train import run_distributed_training

_LOG = logging.getLogger(__name__)


# #############################################################################
# Example Workflow
# #############################################################################


class TransformerTrainingExample:
    """
    Demonstrates a complete workflow for distributed transformer training.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the example with configuration.

        :param config_path: Path to the YAML configuration file.
        """
        self.config = load_config(config_path)
        _LOG.info("Initialized training example with config: %s", config_path)

    def step1_preprocess_data(self) -> None:
        """
        Step 1: Preprocess the BookCorpus dataset.

        This step should be run once before training. It downloads and
        tokenizes the BookCorpus dataset, creating train/val splits and
        saving preprocessed data to disk.

        :return: None
        """
        _LOG.info("=" * 60)
        _LOG.info("Step 1: Data Preprocessing")
        _LOG.info("=" * 60)

        data_dir = self.config.data.data_dir
        if os.path.exists(data_dir):
            _LOG.info("Preprocessed data already exists at: %s", data_dir)
            _LOG.info("Skipping preprocessing step.")
            _LOG.info("To re-run preprocessing, delete the directory and run:")
            _LOG.info("  jupyter notebook notebooks/00_data_preprocessing.ipynb")
        else:
            _LOG.info("Preprocessed data not found at: %s", data_dir)
            _LOG.info("Please run preprocessing first:")
            _LOG.info("  jupyter notebook notebooks/00_data_preprocessing.ipynb")

    def step2_setup_distributed_training(self) -> tuple:
        """
        Step 2: Setup Horovod distributed training environment.

        Initializes Horovod, sets up CUDA devices, and returns distributed
        training information.

        :return: Tuple of (world_size, rank, local_rank, use_cuda).
        """
        _LOG.info("=" * 60)
        _LOG.info("Step 2: Distributed Training Setup")
        _LOG.info("=" * 60)

        world_size, rank, local_rank, use_cuda = setup_horovod(
            require_distributed=True
        )

        if is_main_process():
            _LOG.info("Distributed training initialized:")
            _LOG.info("  World size: %d", world_size)
            _LOG.info("  Using CUDA: %s", use_cuda)
            if use_cuda:
                _LOG.info("  GPU count: %d", torch.cuda.device_count())

        return world_size, rank, local_rank, use_cuda

    def step3_train_model(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        resume_checkpoint: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> None:
        """
        Step 3: Train the transformer model using distributed training.

        Runs the complete training loop with Horovod, including data loading,
        model initialization, optimizer setup, and training/validation loops.

        :param world_size: Total number of processes.
        :param rank: Process rank.
        :param local_rank: Local GPU rank.
        :param resume_checkpoint: Optional path to checkpoint to resume from.
        :param run_name: Optional name for this training run.
        :return: None
        """
        _LOG.info("=" * 60)
        _LOG.info("Step 3: Model Training")
        _LOG.info("=" * 60)

        if is_main_process():
            _LOG.info("Starting distributed training:")
            _LOG.info("  Config: configs/config.yaml")
            _LOG.info("  Epochs: %d", self.config.training.epochs)
            _LOG.info("  Per-GPU batch size: %d", self.config.training.per_gpu_batch_size)
            _LOG.info("  Global batch size: %d", 
                     self.config.training.per_gpu_batch_size * world_size)

        run_distributed_training(
            config=self.config,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            resume_checkpoint=resume_checkpoint,
            run_name=run_name,
            output_dir=None,
        )

        if is_main_process():
            _LOG.info("Training completed!")
            _LOG.info("Checkpoints saved in: %s", self.config.paths.checkpoint_dir)

    def step4_generate_text(
        self, checkpoint_path: str, prompt: str = "The future of AI"
    ) -> str:
        """
        Step 4: Generate text using a trained model checkpoint.

        Loads a trained model and generates text from a prompt using
        autoregressive generation with sampling.

        :param checkpoint_path: Path to model checkpoint.
        :param prompt: Input text prompt.
        :return: Generated text string.
        """
        _LOG.info("=" * 60)
        _LOG.info("Step 4: Text Generation")
        _LOG.info("=" * 60)

        if not os.path.exists(checkpoint_path):
            _LOG.error("Checkpoint not found: %s", checkpoint_path)
            _LOG.error("Please train a model first or provide a valid checkpoint path.")
            return ""

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _LOG.info("Loading model from: %s", checkpoint_path)
        _LOG.info("Using device: %s", device)

        model, tokenizer, config = load_model_for_generation(
            checkpoint_path=checkpoint_path,
            config_path="configs/config.yaml",
            device=device,
        )

        _LOG.info("Generating text from prompt: '%s'", prompt)
        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=self.config.generation.max_new_tokens,
            temperature=self.config.generation.temperature,
            top_k=self.config.generation.top_k,
            top_p=self.config.generation.top_p,
            num_samples=self.config.generation.num_samples,
            device=device,
        )

        generated_text = generated_texts[0] if generated_texts else ""
        _LOG.info("Generated text:")
        _LOG.info("  %s", generated_text)

        return generated_text

    def run_complete_workflow(
        self,
        train: bool = True,
        generate: bool = True,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Run the complete end-to-end workflow.

        Executes all steps: data preprocessing check, distributed setup,
        training, and text generation.

        :param train: Whether to run training step.
        :param generate: Whether to run generation step.
        :param checkpoint_path: Path to checkpoint for generation.
        :return: None
        """
        _LOG.info("=" * 80)
        _LOG.info("Complete Horovod Transformer Training Workflow")
        _LOG.info("=" * 80)

        self.step1_preprocess_data()

        if train:
            world_size, rank, local_rank, _ = self.step2_setup_distributed_training()
            self.step3_train_model(world_size, rank, local_rank)

        if generate:
            if checkpoint_path is None:
                checkpoint_path = os.path.join(
                    self.config.paths.checkpoint_dir, "best_model.pt"
                )
            self.step4_generate_text(checkpoint_path)


def example_minimal_training() -> None:
    """
    Minimal example showing the essential steps for training.

    This example demonstrates the core API usage without all the workflow
    management, suitable for quick testing or integration into other projects.

    :return: None
    """
    _LOG.info("Running minimal training example")

    config = load_config("configs/config.yaml")
    world_size, rank, local_rank, _ = setup_horovod(require_distributed=True)

    run_distributed_training(
        config=config,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )


def example_text_generation_only(checkpoint_path: str) -> None:
    """
    Example showing only text generation from a trained checkpoint.

    Useful for inference-only scenarios where training has already been completed.

    :param checkpoint_path: Path to trained model checkpoint.
    :return: None
    """
    _LOG.info("Running text generation example")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_model_for_generation(
        checkpoint_path=checkpoint_path,
        config_path="configs/config.yaml",
        device=device,
    )

    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a world where",
    ]

    for prompt in prompts:
        _LOG.info("Generating from prompt: '%s'", prompt)
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
        _LOG.info("Generated: %s", generated_texts[0][:100])


def main():
    """
    Main function demonstrating the complete workflow.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    example = TransformerTrainingExample()

    _LOG.info("=" * 80)
    _LOG.info("Horovod Transformer Training Example")
    _LOG.info("=" * 80)
    _LOG.info("")
    _LOG.info("This example demonstrates:")
    _LOG.info("  1. Data preprocessing workflow")
    _LOG.info("  2. Distributed training setup")
    _LOG.info("  3. Model training with Horovod")
    _LOG.info("  4. Text generation from trained model")
    _LOG.info("")
    _LOG.info("Note: For actual training, run with horovodrun:")
    _LOG.info("  horovodrun -np 4 python -m horovod_transformer.example")
    _LOG.info("")

    example.run_complete_workflow(train=False, generate=False)


if __name__ == "__main__":
    main()

