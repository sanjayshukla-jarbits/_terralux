# orchestrator/steps/preprocessing/__init__.py
"""
Preprocessing steps for the modular pipeline orchestrator.

This module provides domain-agnostic preprocessing steps that can be configured
for different applications (landslide assessment, mineral targeting, etc.)
through hyperparameters rather than separate implementations.
"""

# Import all step classes to trigger registration
from .atmospheric_correction_step import AtmosphericCorrectionStep
from .geometric_correction_step import GeometricCorrectionStep
from .cloud_masking_step import CloudMaskingStep
from .spatial_resampling_step import SpatialResamplingStep
from .band_math_step import BandMathStep

__all__ = [
    'AtmosphericCorrectionStep',
    'GeometricCorrectionStep', 
    'CloudMaskingStep',
    'SpatialResamplingStep',
    'BandMathStep'
]

__version__ = "1.0.0"
__author__ = "TerraLux Development Team"
