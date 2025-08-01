"""
Visualization Steps Module - CORRECTED VERSION
==============================================

This module provides visualization and reporting steps for pipeline results,
including interactive maps, statistical plots, and comprehensive reports.

Available Steps:
- map_visualization: Interactive map creation with Folium/Plotly
- report_generation: Comprehensive PDF/HTML report generation
- statistical_plots: Statistical analysis and plotting
- comparison_analysis: Model and result comparison visualizations

FIXES APPLIED:
- Added missing logger definition (was causing NameError)  
- Fixed __import__ calls to remove invalid 'package' parameter
- Enhanced error handling
"""

import logging  # THIS WAS MISSING
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
                     category: str = 'visualization',
                     name: Optional[str] = None) -> bool:
    """
    FIXED: Safely import and register a visualization step.
    
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


def _register_visualization_steps():
    """Register all visualization steps with FIXED imports."""
    logger.info("Registering visualization steps...")
    
    # Core visualization steps
    core_steps = [
        {
            'step_name': 'Map Visualization',
            'module_name': 'map_visualization_step',
            'class_name': 'MapVisualizationStep',
            'step_type': 'map_visualization',
            'aliases': ['map', 'interactive_map', 'folium_map'],
            'name': 'map_visualization'
        },
        {
            'step_name': 'Report Generation',
            'module_name': 'report_generation_step',
            'class_name': 'ReportGenerationStep',
            'step_type': 'report_generation',
            'aliases': ['report', 'pdf_report', 'html_report'],
            'name': 'report_generation'
        },
        {
            'step_name': 'Statistical Plots',
            'module_name': 'statistical_plots_step',
            'class_name': 'StatisticalPlotsStep',
            'step_type': 'statistical_plots',
            'aliases': ['plots', 'charts', 'statistics'],
            'name': 'statistical_plots'
        }
    ]
    
    # Advanced visualization steps
    advanced_steps = [
        {
            'step_name': 'Comparison Analysis',
            'module_name': 'comparison_analysis_step',
            'class_name': 'ComparisonAnalysisStep',
            'step_type': 'comparison_analysis',
            'aliases': ['comparison', 'model_comparison', 'analysis'],
            'name': 'comparison_analysis'
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
    
    logger.info(f"✓ Core visualization steps: {successful_core}/{total_core} registered")
    if successful_advanced > 0:
        logger.info(f"✓ Advanced steps: {successful_advanced}/{total_advanced} registered")
    
    if successful_core < total_core:
        missing_core = [s['step_name'] for s in core_steps 
                       if not _STEP_IMPORTS_SUCCESSFUL.get(s['step_name'], False)]
        logger.warning(f"Missing core visualization steps: {missing_core}")
    
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


def get_available_visualization_steps() -> List[str]:
    """Get list of available visualization step types."""
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


def validate_visualization_setup() -> Dict[str, Any]:
    """Validate the visualization setup and return status."""
    return {
        'steps_available': len(_AVAILABLE_STEPS),
        'core_steps_missing': 3 - len([s for s in _AVAILABLE_STEPS if s in ['map_visualization', 'report_generation', 'statistical_plots']]),
        'import_errors': len(_IMPORT_ERRORS),
        'registry_available': REGISTRY_AVAILABLE,
        'missing_dependencies': get_missing_dependencies()
    }


def print_module_status():
    """Print comprehensive module status."""
    print(f"Visualization Module Status (v{__version__})")
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
    """Get help information for the visualization module."""
    return f"""
Visualization Module Help
=========================

This module provides visualization and reporting steps for pipeline results,
including interactive maps, statistical plots, and comprehensive reports.

Available Step Types:
{chr(10).join(f'  • {step}' for step in _AVAILABLE_STEPS)}

Available Aliases:
{chr(10).join(f'  • {alias} → {canonical}' for alias, canonical in _STEP_ALIASES.items())}

Quick Start:
-----------
1. Check module status: print_module_status()
2. Validate setup: validate_visualization_setup()
3. List available steps: get_available_visualization_steps()

Missing Dependencies:
{chr(10).join(f'  • {dep}' for dep in get_missing_dependencies())}
"""


# Initialize the visualization module
def _initialize_visualization_module():
    """Initialize the visualization module."""
    logger.info(f"Initializing visualization module v{__version__}")
    
    # Register all available steps
    registration_results = _register_visualization_steps()
    
    # Log initialization results
    total_registered = registration_results['total_registered']
    if total_registered > 0:
        logger.info(f"✓ Visualization module initialized with {total_registered} steps")
    else:
        logger.warning("⚠ Visualization module initialized with no steps")
    
    # Issue warnings if critical functionality is missing
    core_missing = registration_results['core_total'] - registration_results['core_successful']
    if core_missing > 0:
        warnings.warn(
            f"Core visualization steps not available ({core_missing} missing). "
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
    'MapVisualizationStep',
    'ReportGenerationStep',
    'StatisticalPlotsStep',
    'ComparisonAnalysisStep',
    
    # Utility functions
    'get_import_status',
    'get_available_visualization_steps',
    'get_step_aliases',
    'is_step_available',
    'get_missing_dependencies',
    'validate_visualization_setup',
    'print_module_status',
    'get_help',
    
    # Module metadata
    '__version__',
    '__author__'
]

# Initialize module on import
_initialization_results = _initialize_visualization_module()

# Quick status check when run directly
if __name__ == "__main__":
    print("Running Visualization Module Test...")
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
        print("python -c \"from orchestrator.steps.visualization import get_help; print(get_help())\"")
    
    # Exit with appropriate code
    exit_code = 0 if _initialization_results['total_registered'] > 0 else 1
    exit(exit_code)
