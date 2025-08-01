"""
Feature Extraction Steps Module
==============================

This module provides feature extraction steps for spectral indices, topographic
derivatives, texture analysis, and other feature calculation tasks.

Available Steps:
- spectral_indices_extraction: Calculate vegetation, water, urban, and mineral indices
- topographic_derivatives: Generate slope, aspect, curvature, and terrain features
- texture_analysis: GLCM, Gabor filters, and local binary patterns
- absorption_features: Spectral absorption feature analysis for minerals
- feature_integration: Multi-source feature stacking and integration

Features:
- Domain-agnostic feature extraction for multiple applications
- Configurable indices and parameters for different use cases
- Robust error handling and quality control
- Integration with existing feature extraction workflows
"""

import logging
import warnings
from typing import Dict, List, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

# Module metadata
__version__ = "2.0.0"
__author__ = "TerraLux Development Team"

# Step registration tracking
_STEP_IMPORTS_SUCCESSFUL = {}
_AVAILABLE_STEPS = []
_STEP_ALIASES = {}
_IMPORT_ERRORS = {}

# Try to import the step registry for registration
StepRegistry = None
register_step_safe = None

try:
    from ..base import StepRegistry, register_step_safe
    logger.debug("✓ StepRegistry imported successfully")
except ImportError as e:
    logger.warning(f"StepRegistry not available: {e}")


def _safe_import_step(step_name: str, module_name: str, class_name: str, 
                     step_type: str, aliases: Optional[List[str]] = None,
                     category: str = 'feature_extraction') -> bool:
    """
    Safely import and register a feature extraction step.
    
    Args:
        step_name: Human-readable step name
        module_name: Module name to import from
        class_name: Class name to import
        step_type: Step type identifier for registration
        aliases: Optional list of aliases
        category: Step category
        
    Returns:
        True if import and registration successful
    """
    try:
        # Import the module
        module = __import__(f'.{module_name}', package=__name__, fromlist=[class_name])
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
                    'module': f'orchestrator.steps.feature_extraction.{module_name}',
                    'class': class_name
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
    """Register all feature extraction steps."""
    logger.info("Registering feature extraction steps...")
    
    # Core feature extraction steps
    core_steps = [
        {
            'name': 'Spectral Indices Extraction',
            'module': 'spectral_indices_step',
            'class': 'SpectralIndicesStep',
            'type': 'spectral_indices_extraction',
            'aliases': ['spectral_indices', 'indices', 'ndvi', 'ndwi']
        },
        {
            'name': 'Topographic Derivatives',
            'module': 'topographic_derivatives_step',
            'class': 'TopographicDerivativesStep',
            'type': 'topographic_derivatives',
            'aliases': ['topographic_features', 'slope', 'aspect', 'curvature']
        },
        {
            'name': 'Texture Analysis',
            'module': 'texture_analysis_step',
            'class': 'TextureAnalysisStep',
            'type': 'texture_analysis',
            'aliases': ['texture_features', 'glcm', 'gabor']
        }
    ]
    
    # Advanced feature extraction steps
    advanced_steps = [
        {
            'name': 'Absorption Features',
            'module': 'absorption_feature_step',
            'class': 'AbsorptionFeatureStep',
            'type': 'absorption_features',
            'aliases': ['mineral_features', 'spectral_features']
        },
        {
            'name': 'Feature Integration',
            'module': 'feature_integration_step',
            'class': 'FeatureIntegrationStep', 
            'type': 'feature_integration',
            'aliases': ['feature_stacking', 'multi_feature']
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
        missing_core = [s['name'] for s in core_steps 
                       if not _STEP_IMPORTS_SUCCESSFUL.get(s['name'], False)]
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
    
    for step_name, error_msg in _IMPORT_ERRORS.items():
        if 'numpy' in error_msg.lower():
            missing_deps.append('numpy')
        elif 'rasterio' in error_msg.lower():
            missing_deps.append('rasterio')
        elif 'gdal' in error_msg.lower():
            missing_deps.append('gdal')
        elif 'skimage' in error_msg.lower():
            missing_deps.append('scikit-image')
        elif 'scipy' in error_msg.lower():
            missing_deps.append('scipy')
        elif 'cv2' in error_msg.lower():
            missing_deps.append('opencv-python')
    
    return list(set(missing_deps))


def validate_feature_extraction_setup() -> Dict[str, Any]:
    """Validate the feature extraction module setup."""
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'summary': {}
    }
    
    # Check core step availability
    core_steps = ['spectral_indices_extraction', 'topographic_derivatives', 'texture_analysis']
    missing_core = [step for step in core_steps if not is_step_available(step)]
    
    if missing_core:
        validation_results['warnings'].append(f"Missing core feature extraction steps: {missing_core}")
    
    # Check for import errors
    failed_imports = [name for name, success in _STEP_IMPORTS_SUCCESSFUL.items() if not success]
    if failed_imports:
        validation_results['warnings'].append(f"Failed imports: {failed_imports}")
    
    # Check dependencies
    missing_deps = get_missing_dependencies()
    if missing_deps:
        validation_results['warnings'].append(f"Missing dependencies: {missing_deps}")
    
    # Summary
    validation_results['summary'] = {
        'total_steps_available': len(_AVAILABLE_STEPS),
        'core_steps_available': len([s for s in core_steps if is_step_available(s)]),
        'failed_imports': len(failed_imports),
        'missing_dependencies': len(missing_deps),
        'overall_status': 'healthy' if len(_AVAILABLE_STEPS) > 0 else 'degraded'
    }
    
    return validation_results


def get_feature_categories() -> Dict[str, List[str]]:
    """Get feature extraction steps organized by category."""
    categories = {
        'spectral': [
            'spectral_indices_extraction',
            'absorption_features'
        ],
        'spatial': [
            'topographic_derivatives',
            'texture_analysis'
        ],
        'integration': [
            'feature_integration'
        ]
    }
    
    # Filter by availability
    available_categories = {}
    for category, steps in categories.items():
        available_steps = [step for step in steps if is_step_available(step)]
        if available_steps:
            available_categories[category] = available_steps
    
    return available_categories


def print_module_status():
    """Print comprehensive module status for debugging."""
    print("=" * 60)
    print("FEATURE EXTRACTION MODULE STATUS")
    print("=" * 60)
    
    # Basic info
    print(f"Version: {__version__}")
    print(f"Available Steps: {len(_AVAILABLE_STEPS)}")
    print(f"Registry Available: {StepRegistry is not None}")
    
    # Step import status
    print(f"\nStep Import Status:")
    for step_name, success in _STEP_IMPORTS_SUCCESSFUL.items():
        icon = "✓" if success else "✗"
        print(f"  {icon} {step_name}")
        if not success and step_name in _IMPORT_ERRORS:
            print(f"    Error: {_IMPORT_ERRORS[step_name]}")
    
    # Available steps and aliases
    if _AVAILABLE_STEPS:
        print(f"\nAvailable Step Types:")
        for step_type in _AVAILABLE_STEPS:
            print(f"  • {step_type}")
    
    if _STEP_ALIASES:
        print(f"\nStep Aliases:")
        for alias, canonical in _STEP_ALIASES.items():
            print(f"  • {alias} → {canonical}")
    
    # Feature categories
    categories = get_feature_categories()
    if categories:
        print(f"\nFeature Categories:")
        for category, steps in categories.items():
            print(f"  • {category}: {len(steps)} steps")
            for step in steps:
                print(f"    - {step}")
    
    # Validation summary
    validation = validate_feature_extraction_setup()
    print(f"\nValidation Summary:")
    print(f"  Status: {validation['summary']['overall_status']}")
    print(f"  Core Steps Available: {validation['summary']['core_steps_available']}/3")
    print(f"  Total Steps Available: {validation['summary']['total_steps_available']}")
    
    if validation['errors']:
        print(f"  Errors: {len(validation['errors'])}")
        for error in validation['errors']:
            print(f"    ✗ {error}")
    
    if validation['warnings']:
        print(f"  Warnings: {len(validation['warnings'])}")
        for warning in validation['warnings']:
            print(f"    ⚠ {warning}")
    
    print("=" * 60)


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

Feature Categories:
{chr(10).join(f'  • {category}: {", ".join(steps)}' for category, steps in get_feature_categories().items())}

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
    'get_feature_categories',
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
