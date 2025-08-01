"""
Feature Extraction Steps Module - CORRECTED VERSION
===================================================

This module provides feature extraction steps for spectral indices, topographic
derivatives, texture analysis, and other feature calculation tasks.

Available Steps:
- spectral_indices_extraction: Calculate vegetation, water, urban, and mineral indices
- topographic_derivatives: Generate slope, aspect, curvature, and terrain features
- texture_analysis: GLCM, Gabor filters, and local binary patterns
- absorption_features: Spectral absorption feature analysis for minerals
- feature_integration: Multi-source feature stacking and integration

FIXES APPLIED:
- Fixed __import__ calls to remove invalid 'package' parameter
- Added missing logger definition
- Enhanced error handling
- Fixed step registration
"""

import logging
import warnings
from typing import Dict, List, Optional, Any

# Configure logging - THIS WAS MISSING
logger = logging.getLogger(__name__)

# Module metadata
__version__ = "2.0.0-fixed"
__author__ = "TerraLux Development Team"

# Step registration tracking
_STEP_IMPORTS_SUCCESSFUL = {}
_AVAILABLE_STEPS = []
_STEP_ALIASES = {}
_IMPORT_ERRORS = {}

# Try to import the step registry for registration
StepRegistry = None
register_step_safe = None

# Add proper error handling for imports
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

def _safe_import_step(step_name: str, module_name: str, class_name: str, 
                     step_type: str, aliases: Optional[List[str]] = None,
                     category: str = 'feature_extraction',
                     name: Optional[str] = None) -> bool:
    """
    FIXED: Safely import and register a feature extraction step.
    
    Args:
        step_name: Human-readable step name
        module_name: Module name to import from
        class_name: Class name to import
        step_type: Step type identifier for registration
        aliases: Optional list of aliases
        category: Step category
        name: Optional name parameter (added for compatibility)
        
    Returns:
        True if import and registration successful
    """
    try:
        # FIXED: Correct import without invalid 'package' parameter
        # OLD (BROKEN): module = __import__(f'.{module_name}', package=__name__, fromlist=[class_name])
        # NEW (FIXED): Use proper absolute import path
        full_module_path = f'orchestrator.steps.{category}.{module_name}'
        module = __import__(full_module_path, fromlist=[class_name])
        step_class = getattr(module, class_name)
        
        # Register with StepRegistry if available
        if StepRegistry and register_step_safe:
            success = register_step_safe(
                step_type,
                step_class,
                category=category,
                aliases=aliases or [],
                metadata={
                    'description': f'{step_name} step',
                    'module': full_module_path,
                    'class': class_name,
                    'name': name or step_name
                }
            )
            if not success:
                logger.warning(f"Registration failed for {step_type}")
                _IMPORT_ERRORS[step_name] = "Registration failed"
                return False
        
        # Track successful import
        _STEP_IMPORTS_SUCCESSFUL[step_name] = True
        _AVAILABLE_STEPS.append(step_type)
        
        # Add aliases
        if aliases:
            for alias in aliases:
                _STEP_ALIASES[alias] = step_type
        
        # Make available in module namespace
        globals()[class_name] = step_class
        
        logger.debug(f"✓ Successfully imported and registered: {step_name}")
        return True
        
    except ImportError as e:
        logger.debug(f"✗ {step_name} not available: {e}")
        _STEP_IMPORTS_SUCCESSFUL[step_name] = False
        _IMPORT_ERRORS[step_name] = f"Import error: {e}"
        return False
    except Exception as e:
        logger.warning(f"✗ Error with {step_name}: {e}")
        _STEP_IMPORTS_SUCCESSFUL[step_name] = False
        _IMPORT_ERRORS[step_name] = f"Error: {e}"
        return False


def _register_feature_extraction_steps():
    """Register all feature extraction steps with FIXED imports."""
    logger.info("Registering feature extraction steps...")
    
    # Core feature extraction steps
    core_steps = [
        {
            'step_name': 'Spectral Indices Extraction',
            'module_name': 'spectral_indices_step',
            'class_name': 'SpectralIndicesStep',  # FIXED: Use actual class name
            'step_type': 'spectral_indices_extraction',
            'aliases': ['spectral_indices', 'indices', 'ndvi', 'spectral_features'],
            'name': 'spectral_indices_extraction'
        },
        {
            'step_name': 'Topographic Derivatives',
            'module_name': 'topographic_derivatives_step',
            'class_name': 'TopographicDerivativesStep',
            'step_type': 'topographic_derivatives',
            'aliases': ['slope_aspect', 'terrain_features', 'dem_derivatives'],
            'name': 'topographic_derivatives'
        },
        {
            'step_name': 'Texture Analysis',
            'module_name': 'texture_analysis_step',
            'class_name': 'TextureAnalysisStep',
            'step_type': 'texture_analysis',
            'aliases': ['glcm', 'texture_features', 'spatial_features'],
            'name': 'texture_analysis'
        }
    ]
    
    # Advanced feature extraction steps
    advanced_steps = [
        {
            'step_name': 'Absorption Feature Analysis',
            'module_name': 'absorption_features_step',
            'class_name': 'AbsorptionFeatureStep',
            'step_type': 'absorption_features',
            'aliases': ['spectral_absorption', 'mineral_features'],
            'name': 'absorption_features'
        },
        {
            'step_name': 'Feature Integration',
            'module_name': 'feature_integration_step',
            'class_name': 'FeatureIntegrationStep',
            'step_type': 'feature_integration',
            'aliases': ['feature_stacking', 'multi_source_features'],
            'name': 'feature_integration'
        }
    ]
    
    successful_core = 0
    successful_advanced = 0
    
    # Register core steps
    for step_config in core_steps:
        if _safe_import_step(**step_config):
            successful_core += 1
    
    # Register advanced steps
    for step_config in advanced_steps:
        if _safe_import_step(**step_config):
            successful_advanced += 1
    
    # Log results
    total_core = len(core_steps)
    total_advanced = len(advanced_steps)
    
    logger.info(f"✓ Core feature extraction steps: {successful_core}/{total_core} registered")
    if successful_advanced > 0:
        logger.info(f"✓ Advanced steps: {successful_advanced}/{total_advanced} registered")
    
    if successful_core < total_core:
        missing_core = [s['step_name'] for s in core_steps 
                       if not _STEP_IMPORTS_SUCCESSFUL.get(s['step_name'], False)]
        logger.warning(f"Missing core feature extraction steps: {missing_core}")
    
    return {
        'core_successful': successful_core,
        'core_total': total_core,
        'advanced_successful': successful_advanced,
        'advanced_total': total_advanced,
        'total_registered': successful_core + successful_advanced
    }


def get_import_status() -> Dict[str, bool]:
    """Get the import status of all step modules."""
    return _STEP_IMPORTS_SUCCESSFUL.copy()


def get_available_feature_extraction_steps() -> List[str]:
    """Get list of available feature extraction step types."""
    return _AVAILABLE_STEPS.copy()


def get_step_aliases() -> Dict[str, str]:
    """Get mapping of step aliases to canonical names."""
    return _STEP_ALIASES.copy()


def is_step_available(step_type: str) -> bool:
    """Check if a specific step type is available."""
    return step_type in _AVAILABLE_STEPS or step_type in _STEP_ALIASES


def get_missing_dependencies() -> List[str]:
    """Get list of missing dependencies based on import errors."""
    missing_deps = []
    for step_name, error in _IMPORT_ERRORS.items():
        if "No module named" in error:
            # Extract module name from error
            module_name = error.split("No module named ")[1].strip("'\"")
            if module_name not in missing_deps:
                missing_deps.append(module_name)
    return missing_deps


def validate_feature_extraction_setup() -> Dict[str, Any]:
    """Validate the feature extraction setup and return status."""
    return {
        'steps_available': len(_AVAILABLE_STEPS),
        'core_steps_missing': 3 - len([s for s in _AVAILABLE_STEPS if s in ['spectral_indices_extraction', 'topographic_derivatives', 'texture_analysis']]),
        'import_errors': len(_IMPORT_ERRORS),
        'registry_available': REGISTRY_AVAILABLE,
        'missing_dependencies': get_missing_dependencies()
    }


def print_module_status():
    """Print comprehensive module status."""
    print(f"Feature Extraction Module Status (v{__version__})")
    print("=" * 50)
    print(f"Available Steps: {len(_AVAILABLE_STEPS)}")
    print(f"Import Errors: {len(_IMPORT_ERRORS)}")
    print(f"Registry Available: {REGISTRY_AVAILABLE}")
    
    if _AVAILABLE_STEPS:
        print(f"Steps: {', '.join(_AVAILABLE_STEPS)}")
    
    if _IMPORT_ERRORS:
        print("Import Issues:")
        for step, error in _IMPORT_ERRORS.items():
            print(f"  - {step}: {error[:100]}...")


def get_help() -> str:
    """Get help information for the feature extraction module."""
    return f"""
Feature Extraction Module Help
=============================

This module provides feature extraction steps for spectral indices, topographic
derivatives, texture analysis, and other feature calculation tasks.

Available Step Types:
{chr(10).join(f'  • {step}' for step in _AVAILABLE_STEPS)}

Available Aliases:
{chr(10).join(f'  • {alias} → {canonical}' for alias, canonical in _STEP_ALIASES.items())}

Quick Start:
-----------
1. Check module status: print_module_status()
2. Validate setup: validate_feature_extraction_setup()
3. List available steps: get_available_feature_extraction_steps()

For detailed documentation, see individual step classes:
- SpectralIndicesStep: Calculate NDVI, NDWI, mineral indices, etc.
- TopographicDerivativesStep: Generate slope, aspect, curvature from DEM
- TextureAnalysisStep: GLCM, Gabor filters, LBP texture features
- AbsorptionFeatureStep: Spectral absorption analysis for minerals
- FeatureIntegrationStep: Multi-source feature stacking

Missing Dependencies:
{chr(10).join(f'  • {dep}' for dep in get_missing_dependencies())}
"""


# Initialize the feature extraction module
def _initialize_feature_extraction_module():
    """Initialize the feature extraction module."""
    logger.info(f"Initializing feature extraction module v{__version__}")
    
    # Register all available steps
    registration_results = _register_feature_extraction_steps()
    
    # Log initialization results
    total_registered = registration_results['total_registered']
    if total_registered > 0:
        logger.info(f"✓ Feature extraction module initialized with {total_registered} steps")
    else:
        logger.warning("⚠ Feature extraction module initialized with no steps")
    
    # Issue warnings if critical functionality is missing
    core_missing = registration_results['core_total'] - registration_results['core_successful']
    if core_missing > 0:
        warnings.warn(
            f"Core feature extraction steps not available ({core_missing} missing). "
            f"Install missing dependencies for full functionality: {get_missing_dependencies()}",
            RuntimeWarning
        )
    elif registration_results['total_registered'] == 0:
        warnings.warn(
            "No step implementations are available. Using mock implementations only. "
            "Install step dependencies for full functionality.",
            RuntimeWarning
        )
    
    return registration_results


# Export public API
__all__ = [
    # Step classes (dynamically added based on successful imports)
    'SpectralIndicesStep',
    'TopographicDerivativesStep',
    'TextureAnalysisStep',
    'AbsorptionFeatureStep',
    'FeatureIntegrationStep',
    
    # Utility functions
    'get_import_status',
    'get_available_feature_extraction_steps',
    'get_step_aliases',
    'is_step_available',
    'get_missing_dependencies',
    'validate_feature_extraction_setup',
    'print_module_status',
    'get_help',
    
    # Module metadata
    '__version__',
    '__author__'
]

# Initialize module on import
_initialization_results = _initialize_feature_extraction_module()

# Quick status check when run directly
if __name__ == "__main__":
    print("Running Feature Extraction Module Test...")
    print("=" * 50)
    
    # Print module status
    print_module_status()
    
    print(f"\nInitialization Results:")
    print(f"Core Steps: {_initialization_results['core_successful']}/{_initialization_results['core_total']}")
    print(f"Advanced Steps: {_initialization_results['advanced_successful']}/{_initialization_results['advanced_total']}")
    print(f"Total Registered: {_initialization_results['total_registered']}")
    
    # Show help if there are issues
    if _initialization_results['total_registered'] == 0:
        print(f"\nFor help resolving issues:")
        print("python -c \"from orchestrator.steps.feature_extraction import get_help; print(get_help())\"")
    
    # Exit with appropriate code
    exit_code = 0 if _initialization_results['total_registered'] > 0 else 1
    exit(exit_code)
