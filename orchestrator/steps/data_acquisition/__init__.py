"""
Data Acquisition Module for Modular Pipeline Orchestrator
=========================================================

This module provides comprehensive data acquisition capabilities for geospatial
analysis pipelines, designed with fail-fast principles for rapid development.
CORRECTED: Updated to use only real_sentinel_hub_step.py (removed sentinel_hub_step.py)

Key Features:
- Multi-source satellite data acquisition (Sentinel-2, Landsat, etc.)
- Digital Elevation Model (DEM) acquisition from various sources
- Local file discovery and validation
- Mock data generation for testing and development
- Comprehensive error handling and fallback mechanisms

Supported Data Sources:
- Sentinel Hub API for Sentinel-2 data
- SRTM, ASTER, ALOS DEM sources
- Local geospatial files (GeoTIFF, Shapefile, etc.)
- External data sources via plugins

Quick Start:
-----------
```python
# Import all data acquisition steps
from orchestrator.steps.data_acquisition import *

# Or import specific steps
from orchestrator.steps.data_acquisition import (
    RealSentinelHubAcquisitionStep, LocalFilesStep, DEMAcquisitionStep
)

# Check available step types
available_types = get_available_data_acquisition_steps()
print(f"Available: {available_types}")
```

Process Definition Example:
--------------------------
```json
{
  "steps": [
    {
      "id": "acquire_satellite_data",
      "type": "sentinel_hub_acquisition",
      "hyperparameters": {
        "bbox": [85.3, 27.6, 85.4, 27.7],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "data_collection": "SENTINEL-2-L2A"
      }
    },
    {
      "id": "discover_local_files",
      "type": "local_files_discovery",
      "hyperparameters": {
        "base_path": "/data/study_area",
        "file_patterns": ["*.tif", "*.shp"],
        "recursive": true
      }
    }
  ]
}
```
"""

import logging
import warnings
from typing import Dict, Any, List, Optional, Set, Union
from pathlib import Path

# Configure logging for the data acquisition module
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0-corrected"  # CORRECTED: Updated version
__author__ = "Data Acquisition Team"

# Core step imports with error handling for fail-fast approach
_STEP_IMPORTS_SUCCESSFUL = {}
_AVAILABLE_STEPS = []
_STEP_ALIASES = {}
_EXISTING_PIPELINE_AVAILABLE = False

# CORRECTED: Import RealSentinelHubAcquisitionStep only (removed old sentinel_hub_step)
try:
    from .real_sentinel_hub_step import RealSentinelHubAcquisitionStep
    _STEP_IMPORTS_SUCCESSFUL['real_sentinel_hub_step'] = True
    _AVAILABLE_STEPS.append('sentinel_hub_acquisition')
    _STEP_ALIASES.update({
        'sentinel2': 'sentinel_hub_acquisition',
        'sentinel_hub': 'sentinel_hub_acquisition',
        's2_acquisition': 'sentinel_hub_acquisition'
    })
    logger.debug("Successfully imported RealSentinelHubAcquisitionStep")
    
    # CORRECTED: Create alias for backward compatibility
    SentinelHubStep = RealSentinelHubAcquisitionStep
    SentinelHubAcquisitionStep = RealSentinelHubAcquisitionStep  # For compatibility
    
except ImportError as e:
    logger.warning(f"Failed to import RealSentinelHubAcquisitionStep: {e}")
    _STEP_IMPORTS_SUCCESSFUL['real_sentinel_hub_step'] = False
    RealSentinelHubAcquisitionStep = None
    SentinelHubStep = None
    SentinelHubAcquisitionStep = None

# CORRECTED: Import LocalFilesStep with better error handling
try:
    from .local_files_step import LocalFilesStep
    _STEP_IMPORTS_SUCCESSFUL['local_files_step'] = True
    _AVAILABLE_STEPS.append('local_files_discovery')
    _STEP_ALIASES.update({
        'local_files': 'local_files_discovery',
        'file_discovery': 'local_files_discovery',
        'local_data_loading': 'local_files_discovery'
    })
    logger.debug("Successfully imported LocalFilesStep")
except ImportError as e:
    logger.warning(f"Failed to import LocalFilesStep: {e}")
    _STEP_IMPORTS_SUCCESSFUL['local_files_step'] = False
    LocalFilesStep = None
except SyntaxError as e:
    logger.error(f"Syntax error in LocalFilesStep: {e}")
    _STEP_IMPORTS_SUCCESSFUL['local_files_step'] = False
    LocalFilesStep = None

# CORRECTED: Import DEM step with better error handling
try:
    from .dem_acquisition_step import DEMAcquisitionStep
    _STEP_IMPORTS_SUCCESSFUL['dem_acquisition_step'] = True
    _AVAILABLE_STEPS.append('dem_acquisition')
    _STEP_ALIASES.update({
        'elevation_data': 'dem_acquisition',
        'srtm_acquisition': 'dem_acquisition',
        'dem_loading': 'dem_acquisition'
    })
    logger.debug("Successfully imported DEMAcquisitionStep")
except ImportError as e:
    logger.warning(f"Failed to import DEMAcquisitionStep: {e}")
    _STEP_IMPORTS_SUCCESSFUL['dem_acquisition_step'] = False
    DEMAcquisitionStep = None

# Optional step imports (may not be available)
CopernicusHubStep = None
LandsatAcquisitionStep = None

try:
    from .copernicus_hub_step import CopernicusHubStep
    _STEP_IMPORTS_SUCCESSFUL['copernicus_hub_step'] = True
    _AVAILABLE_STEPS.append('copernicus_hub_acquisition')
    logger.debug("Successfully imported CopernicusHubStep")
except ImportError:
    logger.debug("CopernicusHubStep not available (optional)")
    _STEP_IMPORTS_SUCCESSFUL['copernicus_hub_step'] = False

try:
    from .landsat_acquisition_step import LandsatAcquisitionStep
    _STEP_IMPORTS_SUCCESSFUL['landsat_acquisition_step'] = True
    _AVAILABLE_STEPS.append('landsat_acquisition')
    logger.debug("Successfully imported LandsatAcquisitionStep")
except ImportError:
    logger.debug("LandsatAcquisitionStep not available (optional)")
    _STEP_IMPORTS_SUCCESSFUL['landsat_acquisition_step'] = False

# CORRECTED: Check for existing landslide_pipeline integration
DataAcquisitionFactory = None
FilePathManager = None

try:
    # Try to import existing landslide_pipeline components
    from landslide_pipeline.data_acquisition import DataAcquisitionFactory, FilePathManager
    _EXISTING_PIPELINE_AVAILABLE = True
    logger.debug("Existing landslide_pipeline integration available")
except ImportError:
    logger.debug("Existing landslide_pipeline not available")

# Initialize the module
logger.info(f"Initializing data acquisition module v{__version__}")
logger.info(f"Step imports: {sum(_STEP_IMPORTS_SUCCESSFUL.values())}/{len(_STEP_IMPORTS_SUCCESSFUL)} successful")
logger.info(f"Available steps: {_AVAILABLE_STEPS}")

if _EXISTING_PIPELINE_AVAILABLE:
    logger.info("✓ Existing landslide_pipeline integration available")
else:
    logger.debug("Existing landslide_pipeline integration not available")

# CORRECTED: Issue warning if no step implementations are available
if sum(_STEP_IMPORTS_SUCCESSFUL.values()) == 0:
    warnings.warn(
        "No step implementations are available. Using mock implementations only. "
        "Install step dependencies for full functionality.",
        RuntimeWarning
    )

# Utility functions
def get_import_status() -> Dict[str, bool]:
    """Get the import status of all step modules."""
    return _STEP_IMPORTS_SUCCESSFUL.copy()

def get_available_data_acquisition_steps() -> List[str]:
    """Get list of available data acquisition step types."""
    return _AVAILABLE_STEPS.copy()

def get_step_aliases() -> Dict[str, str]:
    """Get mapping of step aliases to canonical names."""
    return _STEP_ALIASES.copy()

def is_step_available(step_type: str) -> bool:
    """Check if a specific step type is available."""
    # Check both canonical names and aliases
    return step_type in _AVAILABLE_STEPS or step_type in _STEP_ALIASES

def get_missing_dependencies() -> List[str]:
    """Get list of missing dependencies."""
    missing = []
    
    # Check for missing required libraries
    if not RealSentinelHubAcquisitionStep:
        missing.append("sentinelhub (pip install sentinelhub)")
    
    if not DEMAcquisitionStep:
        missing.append("elevation/rasterio libraries")
    
    if not LocalFilesStep:
        missing.append("geopandas/rasterio libraries")
    
    return missing

def print_module_status():
    """Print comprehensive module status information."""
    print(f"\n📡 Data Acquisition Module Status (v{__version__})")
    print("=" * 50)
    
    # Import status
    print("\n🔧 Import Status:")
    for module, status in _STEP_IMPORTS_SUCCESSFUL.items():
        status_icon = "✓" if status else "❌"
        print(f"  {status_icon} {module}")
    
    # Available steps
    print(f"\n🚀 Available Steps ({len(_AVAILABLE_STEPS)}):")
    for step in _AVAILABLE_STEPS:
        print(f"  • {step}")
    
    # Aliases
    if _STEP_ALIASES:
        print(f"\n🔗 Step Aliases ({len(_STEP_ALIASES)}):")
        for alias, canonical in _STEP_ALIASES.items():
            print(f"  • {alias} → {canonical}")
    
    # Missing dependencies
    missing = get_missing_dependencies()
    if missing:
        print(f"\n⚠️  Missing Dependencies ({len(missing)}):")
        for dep in missing:
            print(f"  • {dep}")
    
    # Integration status
    print(f"\n🔌 Integration Status:")
    integration_icon = "✓" if _EXISTING_PIPELINE_AVAILABLE else "❌"
    print(f"  {integration_icon} Existing landslide_pipeline integration")
    
    print("\n" + "=" * 50)

def validate_data_acquisition_setup() -> Dict[str, Any]:
    """
    Validate the data acquisition setup.
    
    Returns:
        Comprehensive validation results
    """
    validation_results = {
        'overall_status': 'unknown',
        'step_imports': _STEP_IMPORTS_SUCCESSFUL.copy(),
        'available_steps': _AVAILABLE_STEPS.copy(),
        'step_aliases': _STEP_ALIASES.copy(),
        'existing_pipeline_integration': _EXISTING_PIPELINE_AVAILABLE,
        'step_validations': {},
        'missing_dependencies': [],
        'warnings': [],
        'capabilities': {}
    }
    
    # Validate individual steps
    if RealSentinelHubAcquisitionStep:
        validation_results['step_validations']['sentinel_hub'] = {
            'status': 'available', 
            'type': 'real',
            'class': 'RealSentinelHubAcquisitionStep'
        }
    else:
        validation_results['step_validations']['sentinel_hub'] = {
            'status': 'unavailable', 
            'reason': 'import_failed'
        }
        validation_results['warnings'].append("Sentinel Hub step unavailable - install sentinelhub library")
    
    if DEMAcquisitionStep:
        validation_results['step_validations']['dem'] = {
            'status': 'available',
            'class': 'DEMAcquisitionStep'
        }
    else:
        validation_results['step_validations']['dem'] = {
            'status': 'unavailable', 
            'reason': 'import_failed'
        }
        validation_results['warnings'].append("DEM acquisition step unavailable")
    
    if LocalFilesStep:
        validation_results['step_validations']['local_files'] = {
            'status': 'available',
            'class': 'LocalFilesStep'
        }
    else:
        validation_results['step_validations']['local_files'] = {
            'status': 'unavailable', 
            'reason': 'import_failed'
        }
        validation_results['warnings'].append("Local files step unavailable")
    
    # Collect missing dependencies
    validation_results['missing_dependencies'] = get_missing_dependencies()
    
    # Determine overall status
    successful_imports = sum(_STEP_IMPORTS_SUCCESSFUL.values())
    total_core_steps = len([k for k in _STEP_IMPORTS_SUCCESSFUL.keys() 
                           if k in ['real_sentinel_hub_step', 'dem_acquisition_step', 'local_files_step']])
    
    if successful_imports == 0:
        validation_results['overall_status'] = 'failed'
        validation_results['warnings'].append("No data acquisition steps available")
    elif successful_imports >= total_core_steps:
        validation_results['overall_status'] = 'success'
    else:
        validation_results['overall_status'] = 'partial'
        validation_results['warnings'].append(f"Only {successful_imports}/{total_core_steps} core steps available")
    
    # Determine capabilities
    validation_results['capabilities'] = {
        'satellite_data_acquisition': is_step_available('sentinel_hub_acquisition'),
        'dem_data_acquisition': is_step_available('dem_acquisition'),
        'local_files_discovery': is_step_available('local_files_discovery'),
        'existing_pipeline_integration': _EXISTING_PIPELINE_AVAILABLE,
        'mock_data_generation': True,  # Always available
        'fail_fast_development': True  # Always available
    }
    
    return validation_results

def create_data_acquisition_workflow(workflow_type: str = 'standard', **config) -> Dict[str, Any]:
    """
    Create a standard data acquisition workflow configuration.
    
    Args:
        workflow_type: Type of workflow ('standard', 'satellite_only', 'dem_only', 'local_only', 'multi_source')
        **config: Additional configuration parameters
        
    Returns:
        Workflow configuration dictionary
    """
    workflows = {
        'standard': {
            'name': 'Standard Data Acquisition',
            'steps': [
                {
                    'id': 'satellite_data',
                    'type': 'sentinel_hub_acquisition',
                    'hyperparameters': {
                        'data_collection': 'SENTINEL-2-L2A',
                        'resolution': 10,
                        'max_cloud_coverage': 20
                    }
                },
                {
                    'id': 'elevation_data',
                    'type': 'dem_acquisition',
                    'hyperparameters': {
                        'source': 'SRTM',
                        'resolution': 30
                    }
                },
                {
                    'id': 'local_files',
                    'type': 'local_files_discovery',
                    'hyperparameters': {
                        'file_patterns': ['*.tif', '*.shp'],
                        'recursive': True
                    }
                }
            ]
        },
        'satellite_only': {
            'name': 'Satellite Data Only',
            'steps': [
                {
                    'id': 'satellite_data',
                    'type': 'sentinel_hub_acquisition',
                    'hyperparameters': {
                        'data_collection': 'SENTINEL-2-L2A',
                        'resolution': 10,
                        'bands': ['B02', 'B03', 'B04', 'B08']
                    }
                }
            ]
        },
        'dem_only': {
            'name': 'DEM Data Only',
            'steps': [
                {
                    'id': 'elevation_data',
                    'type': 'dem_acquisition',
                    'hyperparameters': {
                        'source': 'SRTM',
                        'resolution': 30,
                        'generate_derivatives': True
                    }
                }
            ]
        },
        'local_only': {
            'name': 'Local Files Only',
            'steps': [
                {
                    'id': 'local_files',
                    'type': 'local_files_discovery',
                    'hyperparameters': {
                        'recursive': True,
                        'validate_files': True,
                        'load_metadata': True
                    }
                }
            ]
        }
    }
    
    if workflow_type not in workflows:
        available_types = list(workflows.keys())
        raise ValueError(f"Unknown workflow type '{workflow_type}'. Available: {available_types}")
    
    workflow = workflows[workflow_type].copy()
    
    # Apply configuration overrides
    for key, value in config.items():
        if key in workflow:
            workflow[key] = value
    
    logger.info(f"Created {workflow_type} workflow: {workflow['name']}")
    logger.debug(f"Workflow steps: {[step['type'] for step in workflow['steps']]}")
    
    return workflow

def run_quick_test() -> Dict[str, Any]:
    """
    Run a quick test of all available data acquisition steps.
    
    Returns:
        Test results dictionary
    """
    test_results = {
        'module_info': {
            'version': __version__,
            'available_steps': _AVAILABLE_STEPS.copy(),
            'step_aliases': _STEP_ALIASES.copy()
        },
        'import_status': _STEP_IMPORTS_SUCCESSFUL.copy(),
        'step_tests': {},
        'overall_status': 'unknown'
    }
    
    failed_tests = 0
    
    # Test RealSentinelHubAcquisitionStep
    if RealSentinelHubAcquisitionStep:
        try:
            test_config = {
                'type': 'sentinel_hub_acquisition',
                'hyperparameters': {
                    'bbox': [85.3, 27.6, 85.4, 27.7],
                    'resolution': 60,
                    'fallback_to_mock': True
                }
            }
            test_step = RealSentinelHubAcquisitionStep('test_sentinel', test_config)
            test_results['step_tests']['sentinel_hub'] = {
                'status': 'success',
                'step_created': True,
                'step_type': test_step.step_type
            }
        except Exception as e:
            test_results['step_tests']['sentinel_hub'] = {
                'status': 'failed',
                'error': str(e)
            }
            failed_tests += 1
    else:
        test_results['step_tests']['sentinel_hub'] = {
            'status': 'skipped',
            'reason': 'Step not available'
        }
    
    # Test DEMAcquisitionStep
    if DEMAcquisitionStep:
        try:
            test_config = {
                'type': 'dem_acquisition',
                'hyperparameters': {
                    'bbox': [85.3, 27.6, 85.4, 27.7],
                    'resolution': 90,
                    'source': 'SRTM'
                }
            }
            test_step = DEMAcquisitionStep('test_dem', test_config)
            test_results['step_tests']['dem_acquisition'] = {
                'status': 'success',
                'step_created': True,
                'step_type': test_step.step_type
            }
        except Exception as e:
            test_results['step_tests']['dem_acquisition'] = {
                'status': 'failed',
                'error': str(e)
            }
            failed_tests += 1
    else:
        test_results['step_tests']['dem_acquisition'] = {
            'status': 'skipped',
            'reason': 'Step not available'
        }
    
    # Test LocalFilesStep
    if LocalFilesStep:
        try:
            test_config = {
                'type': 'local_files_discovery',
                'hyperparameters': {
                    'base_path': '.',
                    'file_patterns': ['*.tif'],
                    'recursive': False
                }
            }
            test_step = LocalFilesStep('test_local_files', test_config)
            test_results['step_tests']['local_files'] = {
                'status': 'success',
                'step_created': True,
                'step_type': test_step.step_type
            }
        except Exception as e:
            test_results['step_tests']['local_files'] = {
                'status': 'failed',
                'error': str(e)
            }
            failed_tests += 1
    else:
        test_results['step_tests']['local_files'] = {
            'status': 'skipped',
            'reason': 'Step not available'
        }
    
    # Determine overall test status
    total_available_steps = len([step for step in [RealSentinelHubAcquisitionStep, DEMAcquisitionStep, LocalFilesStep] if step is not None])
    
    if total_available_steps == 0:
        test_results['overall_status'] = 'no_steps_available'
    elif failed_tests == 0:
        test_results['overall_status'] = 'success'
    elif failed_tests < total_available_steps:
        test_results['overall_status'] = 'partial_success'
    else:
        test_results['overall_status'] = 'failed'
    
    return test_results

def get_help() -> str:
    """Get comprehensive help information for the data acquisition module."""
    return f"""
Data Acquisition Module Help (v{__version__})
{'=' * 50}

This module provides satellite data acquisition, DEM acquisition, and local file
discovery capabilities for geospatial analysis pipelines.

Available Step Types:
{chr(10).join(f'  • {step}' for step in _AVAILABLE_STEPS)}

Available Aliases:
{chr(10).join(f'  • {alias} → {canonical}' for alias, canonical in _STEP_ALIASES.items())}

Quick Start:
-----------
1. Import the module: from orchestrator.steps.data_acquisition import *
2. Check status: print_module_status()
3. Create workflow: workflow = create_data_acquisition_workflow('standard')
4. Use in JSON processes or programmatically

Example Usage:
-------------
# Check what's available
available = get_available_data_acquisition_steps()
print(f"Available steps: {{available}}")

# Validate setup
validation = validate_data_acquisition_setup()
print(f"Status: {{validation['overall_status']}}")

For detailed documentation, see individual step classes:
- RealSentinelHubAcquisitionStep: Sentinel-2 satellite data acquisition
- DEMAcquisitionStep: Digital elevation model acquisition  
- LocalFilesStep: Local file discovery and validation
"""

# CORRECTED: Export public API with proper class names (removed old imports)
__all__ = [
    # Core step classes (may be None if not available)
    'RealSentinelHubAcquisitionStep',
    'SentinelHubStep',  # Alias for backward compatibility
    'SentinelHubAcquisitionStep',  # Another alias for compatibility
    'DEMAcquisitionStep', 
    'LocalFilesStep',
    
    # Optional step classes
    'CopernicusHubStep',
    'LandsatAcquisitionStep',
    
    # Integration with existing pipeline
    'DataAcquisitionFactory',
    'FilePathManager',
    
    # Utility functions
    'get_import_status',
    'get_available_data_acquisition_steps',
    'get_step_aliases',
    'is_step_available',
    'get_missing_dependencies',
    'validate_data_acquisition_setup',
    'create_data_acquisition_workflow',
    'run_quick_test',
    'print_module_status',
    'get_help',
    
    # Module metadata
    '__version__',
    '__author__'
]

# Module initialization
def _initialize_module():
    """Initialize the data acquisition module."""
    logger.info(f"Initializing data acquisition module v{__version__}")
    
    # Log import summary
    successful_imports = sum(_STEP_IMPORTS_SUCCESSFUL.values())
    total_imports = len(_STEP_IMPORTS_SUCCESSFUL)
    logger.info(f"Step imports: {successful_imports}/{total_imports} successful")
    
    if successful_imports > 0:
        logger.info(f"Available steps: {_AVAILABLE_STEPS}")
    else:
        logger.warning("No data acquisition steps available")
        
    logger.info("✓ _terralux orchestrator loaded (v1.0.0-terralux)")

# Initialize on import
_initialize_module()
