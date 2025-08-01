"""
Prediction module for the Modular Pipeline Orchestrator.

This module contains prediction and mapping steps for both landslide susceptibility
and mineral targeting applications, focusing on generating actionable outputs from
trained models.

Available prediction steps:
- ClassificationPredictionStep: Multi-class prediction with confidence levels
- ProbabilityMappingStep: Continuous probability surface generation  
- UncertaintyQuantificationStep: Model confidence and uncertainty analysis

Each step integrates with trained models from the modeling module and produces
geospatially-aware outputs suitable for risk assessment and decision making.
"""

from .classification_prediction_step import ClassificationPredictionStep
from .probability_mapping_step import ProbabilityMappingStep
from .uncertainty_quantification_step import UncertaintyQuantificationStep

__all__ = [
    'ClassificationPredictionStep',
    'ProbabilityMappingStep', 
    'UncertaintyQuantificationStep'
]

# Register all prediction steps with the registry
from ..base.step_registry import StepRegistry

def register_prediction_steps():
    """Register all prediction steps with the step registry."""
    StepRegistry.register('classification_prediction', ClassificationPredictionStep)
    StepRegistry.register('multi_class_prediction', ClassificationPredictionStep)
    StepRegistry.register('susceptibility_prediction', ClassificationPredictionStep)
    StepRegistry.register('prospectivity_prediction', ClassificationPredictionStep)
    
    StepRegistry.register('probability_mapping', ProbabilityMappingStep)
    StepRegistry.register('risk_mapping', ProbabilityMappingStep)
    StepRegistry.register('susceptibility_mapping', ProbabilityMappingStep)
    StepRegistry.register('prospectivity_mapping', ProbabilityMappingStep)
    
    StepRegistry.register('uncertainty_quantification', UncertaintyQuantificationStep)
    StepRegistry.register('confidence_analysis', UncertaintyQuantificationStep)
    StepRegistry.register('prediction_uncertainty', UncertaintyQuantificationStep)

# Auto-register when module is imported
register_prediction_steps()
