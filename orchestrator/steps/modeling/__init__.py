"""
Modeling module for the Modular Pipeline Orchestrator.

This module contains machine learning and statistical modeling steps for both
landslide susceptibility mapping and mineral targeting applications.

Available modeling steps:
- RandomForestStep: Random Forest classification/regression
- LogisticRegressionStep: Logistic regression for binary classification
- KMeansClusteringStep: K-means clustering for unsupervised analysis
- ModelValidationStep: Cross-validation and performance metrics

Each step follows the BaseStep interface and can be configured through
JSON process definitions with hyperparameters specific to each algorithm.
"""

from .random_forest_step import RandomForestStep
from .logistic_regression_step import LogisticRegressionStep
from .kmeans_clustering_step import KMeansClusteringStep
from .model_validation_step import ModelValidationStep

__all__ = [
    'RandomForestStep',
    'LogisticRegressionStep', 
    'KMeansClusteringStep',
    'ModelValidationStep'
]

# Register all modeling steps with the registry
try:
    from ..base import StepRegistry, register_step_safe, BaseStep
    logger.debug("✓ StepRegistry imported successfully")
    REGISTRY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"StepRegistry not available: {e}")
    StepRegistry = None
    register_step_safe = None
    BaseStep = None
    REGISTRY_AVAILABLE = False

def register_modeling_steps():
    """Register all modeling steps with the step registry."""
    StepRegistry.register('random_forest_training', RandomForestStep)
    StepRegistry.register('random_forest_classification', RandomForestStep)
    StepRegistry.register('random_forest_regression', RandomForestStep)
    StepRegistry.register('logistic_regression', LogisticRegressionStep)
    StepRegistry.register('logistic_regression_training', LogisticRegressionStep)
    StepRegistry.register('kmeans_clustering', KMeansClusteringStep)
    StepRegistry.register('unsupervised_clustering', KMeansClusteringStep)
    StepRegistry.register('model_validation', ModelValidationStep)
    StepRegistry.register('cross_validation', ModelValidationStep)
    StepRegistry.register('performance_evaluation', ModelValidationStep)

# Auto-register when module is imported
register_modeling_steps()
