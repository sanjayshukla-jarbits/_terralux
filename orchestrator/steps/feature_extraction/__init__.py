# orchestrator/steps/feature_extraction/__init__.py
"""
Feature extraction steps for the modular pipeline orchestrator.

This module provides domain-agnostic feature extraction steps that can be configured
for different applications (landslide assessment, mineral targeting, etc.)
through hyperparameters rather than separate implementations.

The feature extraction module supports:
- Spectral indices calculation (vegetation, water, urban, mineral indices)
- Topographic derivatives (slope, aspect, curvature, TPI, TRI)
- Texture analysis (GLCM, Gabor filters, LBP)
- Spectral absorption feature analysis
- Multi-source feature integration and stacking
- Statistical and morphological features
- Spatial context analysis
"""

# Import all step classes to trigger registration
from .spectral_indices_step import SpectralIndicesStep
from .topographic_derivatives_step import TopographicDerivativesStep
from .texture_analysis_step import TextureAnalysisStep
from .absorption_feature_step import AbsorptionFeatureStep
from .feature_integration_step import FeatureIntegrationStep

__all__ = [
    'SpectralIndicesStep',
    'TopographicDerivativesStep',
    'TextureAnalysisStep',
    'AbsorptionFeatureStep',
    'FeatureIntegrationStep'
]

__version__ = "1.0.0"
__author__ = "TerraLux Development Team"

# Feature extraction registry for easy access
FEATURE_EXTRACTION_REGISTRY = {
    'spectral_indices': SpectralIndicesStep,
    'topographic_derivatives': TopographicDerivativesStep,
    'texture_analysis': TextureAnalysisStep,
    'absorption_features': AbsorptionFeatureStep,
    'feature_integration': FeatureIntegrationStep
}

def get_available_feature_extractors():
    """Get list of available feature extraction types"""
    return list(FEATURE_EXTRACTION_REGISTRY.keys())

def create_feature_extraction_step(extractor_type: str, step_id: str, hyperparameters: dict):
    """Factory function to create feature extraction steps"""
    if extractor_type not in FEATURE_EXTRACTION_REGISTRY:
        raise ValueError(f"Unknown feature extractor type: {extractor_type}. Available: {list(FEATURE_EXTRACTION_REGISTRY.keys())}")
    
    return FEATURE_EXTRACTION_REGISTRY[extractor_type](step_id, hyperparameters)

# Predefined feature sets for common applications
LANDSLIDE_FEATURE_SET = {
    'spectral_indices': ['NDVI', 'NDWI', 'SAVI', 'EVI', 'BSI'],
    'topographic_derivatives': ['slope', 'aspect', 'curvature', 'tpi', 'tri', 'flow_accumulation'],
    'texture_features': ['contrast', 'dissimilarity', 'homogeneity', 'energy'],
    'focus': 'vegetation_terrain_stability'
}

MINERAL_FEATURE_SET = {
    'spectral_indices': ['clay_minerals', 'iron_oxides', 'carbonates', 'alteration_index'],
    'topographic_derivatives': ['slope', 'aspect', 'curvature'],
    'texture_features': ['contrast', 'correlation', 'energy', 'homogeneity'],
    'absorption_features': ['continuum_removed', 'absorption_depth', 'absorption_position'],
    'focus': 'mineral_signatures_alteration'
}

def get_predefined_feature_set(application_type: str) -> dict:
    """Get predefined feature set for specific application"""
    feature_sets = {
        'landslide_susceptibility': LANDSLIDE_FEATURE_SET,
        'mineral_targeting': MINERAL_FEATURE_SET
    }
    
    return feature_sets.get(application_type, {})
