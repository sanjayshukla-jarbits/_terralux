#!/usr/bin/env python3
"""
Data Acquisition Steps Module - CORRECTED VERSION
================================================

This module provides data acquisition steps for satellite imagery, DEM data,
and local file discovery with FIXED import handling.

Available Steps:
- sentinel_hub_acquisition: Sentinel-2/1 satellite data from Sentinel Hub API
- dem_acquisition: Digital elevation models from various sources (SRTM, ASTER, etc.)
- local_files_discovery: Discovery and cataloging of local data files
- copernicus_hub_acquisition: Data from Copernicus Open Access Hub (optional)
- landsat_acquisition: Landsat satellite data (optional)

FIXES APPLIED:
- FIXED: Removed invalid 'package' parameter from __import__ calls
- ADDED: Missing logger definition that was causing NameError
- FIXED: Step registration to use corrected register_step_safe signature
- Enhanced error handling for missing modules
- Enhanced registration management
"""

import logging  # ADDED: This was missing and causing logger errors
import warnings
from typing import Dict, List, Optional, Any, Type

# ADDED: Configure logging that was missing
logger = logging.getLogger(__name__)

# Module metadata
__version__ = "2.0.0-fixed"
__author__ = "TerraLux Development Team"

# Step registration tracking
_STEP_IMPORTS_SUCCESSFUL = {}
_AVAILABLE_STEPS = []
_STEP_ALIASES = {}
_IMPORT_ERRORS = {}
_EXISTING_PIPELINE_AVAILABLE = False

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
                     category: str = 'data_acquisition',
                     name: Optional[str] = None) -> bool:
    """
    FIXED: Safely import and register a step with corrected import syntax.
    
    This function fixes the critical import syntax error that was causing
    the "unexpected keyword argument 'package'" errors in the execution log.
    
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
        
        # FIXED: Register with StepRegistry using corrected function signature
        if StepRegistry and register_step_safe:
            success = register_step_safe(
                step_type=step_type,           # Step type identifier
                step_class=step_class,         # Step class
                category=category,             # FIXED: Now properly supported
                aliases=aliases or [],         # FIXED: Now properly supported  
                description=f'{step_name} step',  # Additional metadata
                module=full_module_path,
                class_name=class_name,
                name=name or step_name
            )
            if not success:
                logger.warning(f"✗ Registration failed for {step_type}")
                _IMPORT_ERRORS[step_name] = "Registration failed"
                return False
        else:
            logger.warning(f"✗ Error with {step_name}: StepRegistry not available")
            return False
        
        # Track successful import and registration
        _STEP_IMPORTS_SUCCESSFUL[step_name] = True
        _AVAILABLE_STEPS.append(step_type)
        
        # Add aliases to tracking
        if aliases:
            for alias in aliases:
                _STEP_ALIASES[alias] = step_type
        
        # Make available in module namespace for direct access
        globals()[class_name] = step_class
        
        logger.info(f"✓ {step_name} registered successfully")
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


def _register_data_acquisition_steps():
    """Register all data acquisition steps with FIXED imports."""
    logger.info("Registering data acquisition steps...")
    
    # Core data acquisition steps
    core_steps_registered = 0
    total_core_steps = 3
    
    # 1. Sentinel Hub Acquisition
    if _safe_import_step(
        step_name="Sentinel Hub Acquisition",
        module_name="sentinel_hub_step",
        class_name="SentinelHubAcquisitionStep", 
        step_type="sentinel_hub_acquisition",
        aliases=["sentinel_hub", "sentinel2", "s2"],
        category="data_acquisition"
    ):
        core_steps_registered += 1
    
    # 2. DEM Acquisition  
    if _safe_import_step(
        step_name="DEM Acquisition",
        module_name="dem_acquisition_step",
        class_name="DEMAcquisitionStep",
        step_type="dem_acquisition", 
        aliases=["dem", "elevation", "srtm"],
        category="data_acquisition"
    ):
        core_steps_registered += 1
    
    # 3. Local Files Discovery
    if _safe_import_step(
        step_name="Local Files Discovery",
        module_name="local_files_step", 
        class_name="LocalFilesDiscoveryStep",
        step_type="local_files_discovery",
        aliases=["local_files", "file_discovery"],
        category="data_acquisition"
    ):
        core_steps_registered += 1
    
    # Optional steps (don't count toward core)
    optional_steps_registered = 0
    
    # 4. Copernicus Hub Acquisition (optional)
    if _safe_import_step(
        step_name="Copernicus Hub Acquisition",
        module_name="copernicus_hub_step",
        class_name="CopernicusHubAcquisitionStep",
        step_type="copernicus_hub_acquisition",
        aliases=["copernicus", "scihub"],
        category="data_acquisition"
    ):
        optional_steps_registered += 1
    
    # 5. Landsat Acquisition (optional)
    if _safe_import_step(
        step_name="Landsat Acquisition", 
        module_name="landsat_acquisition_step",
        class_name="LandsatAcquisitionStep",
        step_type="landsat_acquisition",
        aliases=["landsat", "landsat8", "l8"],
        category="data_acquisition"
    ):
        optional_steps_registered += 1
    
    # Log registration summary
    logger.info(f"✓ Core data acquisition steps: {core_steps_registered}/{total_core_steps} registered")
    if optional_steps_registered > 0:
        logger.info(f"✓ Optional steps: {optional_steps_registered} registered")
    
    # Check for missing core steps
    if core_steps_registered < total_core_steps:
        missing_steps = []
        for step_name, success in _STEP_IMPORTS_SUCCESSFUL.items():
            if not success and step_name in ["Sentinel Hub Acquisition", "DEM Acquisition", "Local Files Discovery"]:
                missing_steps.append(step_name)
        
        logger.warning(f"Missing core data acquisition steps: {missing_steps}")
    
    # Issue warning if no core steps available
    if core_steps_registered == 0:
        logger.warning("⚠ Data acquisition module initialized with no steps")
        
        # Show available alternatives
        missing_modules = []
        for step_name, error in _IMPORT_ERRORS.items():
            if "Import error" in error:
                missing_modules.append(error.split(": ", 1)[1])
        
        if missing_modules:
            warnings.warn(
                f"Core data acquisition steps not available ({total_core_steps} missing). "
                f"Install missing dependencies for full functionality: {list(set(missing_modules))}",
                RuntimeWarning
            )
    
    return {
        'core_registered': core_steps_registered,
        'total_core': total_core_steps,
        'optional_registered': optional_steps_registered,
        'available_steps': _AVAILABLE_STEPS.copy(),
        'errors': _IMPORT_ERRORS.copy()
    }


def get_available_steps() -> List[str]:
    """Get list of successfully registered step types."""
    return _AVAILABLE_STEPS.copy()


def get_step_aliases() -> Dict[str, str]:
    """Get mapping of aliases to step types."""
    return _STEP_ALIASES.copy()


def get_import_errors() -> Dict[str, str]:
    """Get import errors for debugging."""
    return _IMPORT_ERRORS.copy()


def get_registration_status() -> Dict[str, Any]:
    """Get comprehensive registration status."""
    return {
        'module_version': __version__,
        'registry_available': REGISTRY_AVAILABLE,
        'core_steps_successful': _STEP_IMPORTS_SUCCESSFUL,
        'available_steps': _AVAILABLE_STEPS,
        'step_aliases': _STEP_ALIASES,
        'import_errors': _IMPORT_ERRORS,
        'total_registered': len(_AVAILABLE_STEPS),
        'total_errors': len(_IMPORT_ERRORS)
    }


def print_module_status():
    """Print detailed module status for debugging."""
    status = get_registration_status()
    
    print(f"\n=== Data Acquisition Module Status ===")
    print(f"Version: {status['module_version']}")
    print(f"Registry Available: {status['registry_available']}")
    print(f"Steps Registered: {status['total_registered']}")
    print(f"Import Errors: {status['total_errors']}")
    
    if status['available_steps']:
        print(f"\nAvailable Steps:")
        for step in status['available_steps']:
            print(f"  ✓ {step}")
    
    if status['step_aliases']:
        print(f"\nAliases:")
        for alias, step_type in status['step_aliases'].items():
            print(f"  {alias} -> {step_type}")
    
    if status['import_errors']:
        print(f"\nImport Errors:")
        for step_name, error in status['import_errors'].items():
            print(f"  ✗ {step_name}: {error}")


# Module initialization
logger.info(f"Initializing data acquisition module v{__version__}")
logger.info("Registering data acquisition steps...")

# Register all steps
try:
    registration_results = _register_data_acquisition_steps()
    
    if registration_results['core_registered'] > 0:
        logger.info(f"✓ Core data acquisition steps: {registration_results['core_registered']}/{registration_results['total_core']} registered")
    else:
        logger.warning("⚠ Data acquisition module initialized with no steps")
        
except Exception as e:
    logger.error(f"Failed to register data acquisition steps: {e}")
    # Continue anyway to avoid breaking the module


# Export public API
__all__ = [
    # Step types (if successfully imported)
    'SentinelHubAcquisitionStep',
    'DEMAcquisitionStep', 
    'LocalFilesDiscoveryStep',
    'CopernicusHubAcquisitionStep',
    'LandsatAcquisitionStep',
    
    # Utility functions
    'get_available_steps',
    'get_step_aliases', 
    'get_import_errors',
    'get_registration_status',
    'print_module_status',
    
    # Module metadata
    '__version__'
]

# Only export successfully imported classes
for class_name in ['SentinelHubAcquisitionStep', 'DEMAcquisitionStep', 'LocalFilesDiscoveryStep', 
                   'CopernicusHubAcquisitionStep', 'LandsatAcquisitionStep']:
    if class_name not in globals():
        __all__.remove(class_name)


# Quick test functionality
if __name__ == "__main__":
    print("Testing Data Acquisition Module...")
    print_module_status()
    
    # Test step creation if registry available
    if REGISTRY_AVAILABLE and _AVAILABLE_STEPS:
        print(f"\nTesting step creation...")
        try:
            from ..base import create_step_from_config
            
            test_config = {
                'type': _AVAILABLE_STEPS[0],
                'hyperparameters': {'test': True}
            }
            
            step = create_step_from_config('test_step', test_config)
            if step:
                print(f"✓ Successfully created test step: {step.step_id}")
            else:
                print("✗ Failed to create test step")
                
        except Exception as e:
            print(f"✗ Step creation test failed: {e}")
    
    print(f"\nModule test completed. Registry available: {REGISTRY_AVAILABLE}")
    
    # Return appropriate exit code  
    exit(0 if REGISTRY_AVAILABLE and len(_AVAILABLE_STEPS) > 0 else 1)
