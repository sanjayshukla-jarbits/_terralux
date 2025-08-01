"""
Preprocessing Steps Module
=========================

This module provides data preprocessing steps for atmospheric correction,
geometric correction, cloud masking, and other preprocessing tasks.

Available Steps:
- atmospheric_correction: Sen2Cor/FLAASH atmospheric correction
- geometric_correction: Orthorectification and georeferencing
- cloud_masking: Cloud and shadow detection and masking
- spatial_resampling: Resolution harmonization and resampling
- band_math: Generic band calculations and transformations
- data_validation: Comprehensive data quality validation
- inventory_generation: Data inventory and catalog generation

Features:
- Domain-agnostic preprocessing applicable to multiple use cases
- Robust error handling and quality control
- Integration with existing preprocessing workflows
- Configurable parameters for different applications
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
                     category: str = 'preprocessing',
                     name: Optional[str] = None) -> bool:
    """
    Safely import and register a preprocessing step.
    
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
                    'module': f'orchestrator.steps.preprocessing.{module_name}',
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


def _register_preprocessing_steps():
    """Register all preprocessing steps."""
    logger.info("Registering preprocessing steps...")
    
    # Core preprocessing steps
    core_steps = [
        {
            'step_name': 'Atmospheric Correction',
            'module_name': 'atmospheric_correction_step',
            'class_name': 'AtmosphericCorrectionStep',
            'step_type': 'atmospheric_correction',
            'aliases': ['sen2cor', 'flaash', 'atm_correction'],
            'name': 'atmospheric_correction'
        },
        {
            'step_name': 'Geometric Correction',
            'module_name': 'geometric_correction_step',
            'class_name': 'GeometricCorrectionStep',
            'step_type': 'geometric_correction',
            'aliases': ['orthorectification', 'georeferencing'],
            'name': 'geometric_correction'
        },
        {
            'step_name': 'Cloud Masking',
            'module_name': 'cloud_masking_step',
            'class_name': 'CloudMaskingStep',
            'step_type': 'cloud_masking',
            'aliases': ['cloud_detection', 'shadow_masking'],
            'name': 'cloud_masking'
        },
        {
            'step_name': 'Spatial Resampling',
            'module_name': 'spatial_resampling_step',
            'class_name': 'SpatialResamplingStep',
            'step_type': 'spatial_resampling',
            'aliases': ['resampling', 'resolution_harmonization'],
            'name': 'spatial_resampling'
        },
        {
            'step_name': 'Band Math',
            'module_name': 'band_math_step',
            'class_name': 'BandMathStep',
            'step_type': 'band_math',
            'aliases': ['band_calculation', 'band_operations'],
            'name': 'band_math'
        }
    ]
    
    # Additional processing steps
    additional_steps = [
        {
            'step_name': 'Data Validation',
            'module_name': 'data_validation_step',
            'class_name': 'DataValidationStep',
            'step_type': 'data_validation',
            'aliases': ['validation', 'quality_control'],
            'name': 'data_validation'
        },
        {
            'step_name': 'Inventory Generation',
            'module_name': 'inventory_generation_step',
            'class_name': 'InventoryGenerationStep',
            'step_type': 'inventory_generation',
            'aliases': ['inventory', 'catalog_generation'],
            'name': 'inventory_generation'
        }
    ]
    
    successful_core = 0
    successful_additional = 0
    
    # Register core steps
    for step_config in core_steps:
        if _safe_import_step(**step_config):
            successful_core += 1
    
    # Register additional steps
    for step_config in additional_steps:
        if _safe_import_step(**step_config):
            successful_additional += 1
    
    # Log results
    total_core = len(core_steps)
    total_additional = len(additional_steps)
    
    logger.info(f"✓ Core preprocessing steps: {successful_core}/{total_core} registered")
    if successful_additional > 0:
        logger.info(f"✓ Additional steps: {successful_additional}/{total_additional} registered")
    
    if successful_core < total_core:
        missing_core = [s['step_name'] for s in core_steps 
                       if not _STEP_IMPORTS_SUCCESSFUL.get(s['step_name'], False)]
        logger.warning(f"Missing core preprocessing steps: {missing_core}")
    
    return {
        'core_successful': successful_core,
        'core_total': total_core,
        'additional_successful': successful_additional,
        'additional_total': total_additional,
        'total_registered': successful_core + successful_additional
    }


def get_import_status() -> Dict[str, bool]:
    """Get the import status of all step modules."""
    return _STEP_IMPORTS_SUCCESSFUL.copy()


def get_available_preprocessing_steps() -> List[str]:
    """Get list of available preprocessing step types."""
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


def validate_preprocessing_setup() -> Dict[str, Any]:
    """Validate the preprocessing setup and return status."""
    status = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'available_steps': len(_AVAILABLE_STEPS),
        'missing_dependencies': get_missing_dependencies()
    }
    
    # Check core functionality
    core_steps = ['atmospheric_correction', 'geometric_correction', 'cloud_masking']
    available_core = [step for step in core_steps if is_step_available(step)]
    
    if len(available_core) == 0:
        status['valid'] = False
        status['errors'].append("No core preprocessing steps available")
    elif len(available_core) < len(core_steps):
        missing_core = [step for step in core_steps if not is_step_available(step)]
        status['warnings'].append(f"Missing core steps: {missing_core}")
    
    # Check for missing dependencies
    if status['missing_dependencies']:
        status['warnings'].append(f"Missing dependencies: {status['missing_dependencies']}")
    
    return status


def print_module_status():
    """Print detailed module status information."""
    print("Preprocessing Module Status")
    print("=" * 40)
    print(f"Module Version: {__version__}")
    print(f"Available Steps: {len(_AVAILABLE_STEPS)}")
    print(f"Step Aliases: {len(_STEP_ALIASES)}")
    print(f"Import Errors: {len(_IMPORT_ERRORS)}")
    
    if _AVAILABLE_STEPS:
        print(f"\nAvailable Step Types:")
        for step in sorted(_AVAILABLE_STEPS):
            print(f"  ✓ {step}")
    
    if _STEP_ALIASES:
        print(f"\nStep Aliases:")
        for alias, canonical in sorted(_STEP_ALIASES.items()):
            print(f"  • {alias} → {canonical}")
    
    if _IMPORT_ERRORS:
        print(f"\nImport Errors:")
        for step, error in _IMPORT_ERRORS.items():
            print(f"  ✗ {step}: {error}")
    
    # Validation status
    validation = validate_preprocessing_setup()
    print(f"\nValidation Status: {'✓ VALID' if validation['valid'] else '✗ INVALID'}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠ {warning}")
    
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  ✗ {error}")


def get_help() -> str:
    """Get help information for the preprocessing module."""
    return f"""
Preprocessing Module Help
========================

This module provides data preprocessing steps for atmospheric correction,
geometric correction, cloud masking, and other preprocessing tasks.

Available Step Types:
{chr(10).join(f'  • {step}' for step in _AVAILABLE_STEPS)}

Available Aliases:
{chr(10).join(f'  • {alias} → {canonical}' for alias, canonical in _STEP_ALIASES.items())}

Quick Start:
-----------
1. Check module status: print_module_status()
2. Validate setup: validate_preprocessing_setup()
3. List available steps: get_available_preprocessing_steps()

For detailed documentation, see individual step classes:
- AtmosphericCorrectionStep: Sen2Cor/FLAASH atmospheric correction
- GeometricCorrectionStep: Orthorectification and georeferencing
- CloudMaskingStep: Cloud and shadow detection and masking
- DataValidationStep: Comprehensive data quality validation

Missing Dependencies:
{chr(10).join(f'  • {dep}' for dep in get_missing_dependencies())}
"""


# Initialize the preprocessing module
def _initialize_preprocessing_module():
    """Initialize the preprocessing module."""
    logger.info(f"Initializing preprocessing module v{__version__}")
    
    # Register all available steps
    registration_results = _register_preprocessing_steps()
    
    # Log initialization results
    total_registered = registration_results['total_registered']
    if total_registered > 0:
        logger.info(f"✓ Preprocessing module initialized with {total_registered} steps")
    else:
        logger.warning("⚠ Preprocessing module initialized with no steps")
    
    # Issue warnings if critical functionality is missing
    core_missing = registration_results['core_total'] - registration_results['core_successful']
    if core_missing > 0:
        warnings.warn(
            f"Core preprocessing steps not available ({core_missing} missing). "
            f"Install missing dependencies for full functionality: {get_missing_dependencies()}",
            RuntimeWarning
        )
    
    return registration_results


# Export public API
__all__ = [
    # Step classes (dynamically added based on successful imports)
    'AtmosphericCorrectionStep',
    'GeometricCorrectionStep', 
    'CloudMaskingStep',
    'SpatialResamplingStep',
    'BandMathStep',
    'DataValidationStep',
    'InventoryGenerationStep',
    
    # Utility functions
    'get_import_status',
    'get_available_preprocessing_steps',
    'get_step_aliases',
    'is_step_available',
    'get_missing_dependencies',
    'validate_preprocessing_setup',
    'print_module_status',
    'get_help',
    
    # Module metadata
    '__version__',
    '__author__'
]

# Initialize module on import
_initialization_results = _initialize_preprocessing_module()

# Quick status check when run directly
if __name__ == "__main__":
    print("Running Preprocessing Module Test...")
    print("=" * 50)
    
    # Print module status
    print_module_status()
    
    print(f"\nInitialization Results:")
    print(f"Core Steps: {_initialization_results['core_successful']}/{_initialization_results['core_total']}")
    print(f"Additional Steps: {_initialization_results['additional_successful']}/{_initialization_results['additional_total']}")
    print(f"Total Registered: {_initialization_results['total_registered']}")
    
    # Show help if there are issues
    if _initialization_results['total_registered'] == 0:
        print(f"\nFor help resolving issues:")
        print("python -c \"from orchestrator.steps.preprocessing import get_help; print(get_help())\"")
    
    # Exit with appropriate code
    exit_code = 0 if _initialization_results['total_registered'] > 0 else 1
    exit(exit_code)
