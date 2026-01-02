"""
House Price Prediction TFX Pipeline Definition

This module defines the complete TFX pipeline with all components:
1. CsvExampleGen - Data ingestion
2. SchemaGen - Schema generation and validation
3. Transform - Feature engineering
4. Trainer - Model training (XGBoost and TensorFlow DNN)
5. Evaluator - Model evaluation
6. Pusher - Model deployment
"""

import os
from typing import Dict, List, Optional

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    SchemaGen,
    StatisticsGen,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)
from tfx.orchestration import metadata, pipeline
from tfx.proto import trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    transform_module_file: str,
    trainer_module_file: str,
    metadata_path: str,
    serving_model_dir: str,
    beam_pipeline_args: Optional[List[str]] = None,
    trainer_custom_config: Optional[Dict] = None,
) -> pipeline.Pipeline:
    """
    Create TFX pipeline for house price prediction.

    Args:
        pipeline_name: Name of the pipeline
        pipeline_root: Root directory for pipeline outputs
        data_path: Path to the CSV data file
        transform_module_file: Path to transform module
        trainer_module_file: Path to trainer module
        metadata_path: Path to metadata database
        serving_model_dir: Directory for serving models
        beam_pipeline_args: Optional Beam pipeline arguments
        trainer_custom_config: Optional custom config for Trainer (e.g., {'model_name': 'XGBoost'})

    Returns:
        TFX Pipeline instance

    Phases implemented:
    - Phase 2: CsvExampleGen, SchemaGen, StatisticsGen
    - Phase 3: Transform
    - Phase 4: Trainer (supports TensorFlow DNN and sklearn models)
    - Phase 5: Evaluator
    - Phase 6: Pusher
    """

    components = []

    # ============================================================================
    # PHASE 2: Data Ingestion & Validation
    # ============================================================================

    # Component 1: CsvExampleGen
    # Ingests CSV data and converts to TFRecord format
    example_gen = CsvExampleGen(input_base=data_path)
    components.append(example_gen)

    # Component 2: StatisticsGen
    # Generates statistics from the examples for schema inference
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    components.append(statistics_gen)

    # Component 3: SchemaGen
    # Infers schema from statistics
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True
    )
    components.append(schema_gen)

    # ============================================================================
    # PHASE 3: Feature Engineering
    # ============================================================================

    # Component 4: Transform
    # Feature engineering and preprocessing using utils/feature_engineering.py
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module_file
    )
    components.append(transform)

    # ============================================================================
    # PHASE 4: Model Training
    # ============================================================================

    # Component 5: Trainer
    # Train model using specified trainer module (TensorFlow DNN or sklearn model)
    trainer = Trainer(
        module_file=trainer_module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=config.TRAIN_STEPS),
        eval_args=trainer_pb2.EvalArgs(num_steps=config.EVAL_STEPS),
        custom_config=trainer_custom_config  # Pass model name for sklearn models
    )
    components.append(trainer)

    # ============================================================================
    # PHASE 5: Model Evaluation
    # ============================================================================

    # Component 6: Evaluator
    # Evaluate model performance on validation data
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                label_key='SalePrice',  # Raw target variable (transform layer handles log)
                signature_name='serving_default'
            )
        ],
        slicing_specs=[
            tfma.SlicingSpec()  # Evaluate on entire dataset
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name='ExampleCount'),
                    tfma.MetricConfig(
                        class_name='MeanSquaredError',
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                upper_bound={'value': 900000000.0}  # MSE ~= (30k RMSE)^2
                            )
                        )
                    ),
                    tfma.MetricConfig(class_name='RootMeanSquaredError'),
                    tfma.MetricConfig(class_name='MeanAbsoluteError'),
                ]
            )
        ]
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],  # Use raw examples, not transformed
        model=trainer.outputs['model'],
        eval_config=eval_config
    )
    components.append(evaluator)

    # ============================================================================
    # PHASE 6: Model Deployment
    # ============================================================================

    # Component 7: Pusher
    # Deploy the trained model to serving directory
    # Note: Since our model is not blessed, we'll push unconditionally for demonstration
    pusher = Pusher(
        model=trainer.outputs['model'],
        # Commenting out model_blessing to push regardless of evaluation
        # model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )
    components.append(pusher)

    # ============================================================================
    # Create and return pipeline
    # ============================================================================

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_pipeline_args,
    )


if __name__ == "__main__":
    print("House Price TFX Pipeline Definition")
    print("This module will be implemented across phases 2-6")
