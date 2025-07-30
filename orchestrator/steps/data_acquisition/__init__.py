"""
Data Acquisition Module for Modular Pipeline Orchestrator
=========================================================

This module provides comprehensive data acquisition capabilities for geospatial
analysis pipelines, designed with fail-fast principles for rapid development.

Key Features:
- Multi-source satellite data acquisition (Sentinel-2, Landsat, etc.)
- Digital Elevation Model (DEM) acquisition from various sources
- Local file discovery and validation
- Integration with existing landslide_pipeline infrastructure
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
    SentinelHubStep, DEMAcquisitionStep, LocalFilesStep
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
      "id": "acquire_elevation_data", 
      "type": "dem_acquisition",
      "hyperparameters": {
        "bbox": [85.3, 27.6, 85.4, 27.7],
        "source": "SRTM",
        "resolution": 30
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
from typing import Dict, Any, List, Optional, Set, Union
from pathlib import Path

# Configure logging for the data acquisition module
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0-dev"
__author__ = "Data Acquisition Team"

# Core step imports with error handling for fail-fast approach
_STEP_IMPORTS_SUCCESSFUL = {}
_AVAILABLE_STEPS = []
_STEP_ALIASES = {}

# Import SentinelHubStep
try:
    from .sentinel_hub_step import SentinelHubStep, validate_sentinel_hub_config, create_test_sentinel_step
    _STEP_IMPORTS_SUCCESSFUL['sentinel_hub_step'] = True
    _AVAILABLE_STEPS.append('sentinel_hub_acquisition')
    _STEP_ALIASES.update({
        'sentinel2': 'sentinel_hub_acquisition',
        'sentinel_hub': 'sentinel_hub_acquisition',
        's2_acquisition': 'sentinel_hub_acquisition'
    })
    logger.debug("Successfully imported SentinelHubStep")
except ImportError as e:
    logger.warning(f"Failed to import SentinelHubStep: {e}")
    _STEP_IMPORTS_SUCCESSFUL['sentinel_hub_step'] = False
    SentinelHubStep = None
    validate_sentinel_hub_config = None
    create_test_sentinel_step = None

# Import DEMAcquisitionStep
try:
    from .dem_acquisition_step import DEMAcquisitionStep, validate_dem_config, create_test_dem_step, get_supported_sources
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
    validate_dem_config = None
    create_test_dem_step = None
    get_supported_sources = None

# Import LocalFilesStep
try:
    from .local_files_step import LocalFilesStep, validate_local_files_config, create_test_local_files_step, get_supported_formats, get_common_patterns
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
    validate_local_files_config = None
    create_test_local_files_step = None
    get_supported_formats = None
    get_common_patterns = None

# Optional/experimental steps (import without failing)
try:
    from .copernicus_hub_step import CopernicusHubStep
    _STEP_IMPORTS_SUCCESSFUL['copernicus_hub_step'] = True
    _AVAILABLE_STEPS.append('copernicus_hub_acquisition')
    logger.debug("Successfully imported CopernicusHubStep")
except ImportError:
    _STEP_IMPORTS_SUCCESSFUL['copernicus_hub_step'] = False
    CopernicusHubStep = None
    logger.debug("CopernicusHubStep not available (optional)")

try:
    from .landsat_acquisition_step import LandsatAcquisitionStep
    _STEP_IMPORTS_SUCCESSFUL['landsat_acquisition_step'] = True
    _AVAILABLE_STEPS.append('landsat_acquisition')
    logger.debug("Successfully imported LandsatAcquisitionStep")
except ImportError:
    _STEP_IMPORTS_SUCCESSFUL['landsat_acquisition_step'] = False
    LandsatAcquisitionStep = None
    logger.debug("LandsatAcquisitionStep not available (optional)")

# Integration with existing landslide_pipeline
try:
    from landslide_pipeline.data.data_acquisition import DataAcquisitionFactory
    from landslide_pipeline.utils.file_path_manager import FilePathManager
    _EXISTING_PIPELINE_AVAILABLE = True
    logger.debug("Existing landslide_pipeline integration available")
except ImportError:
    DataAcquisitionFactory = None
    FilePathManager = None
    _EXISTING_PIPELINE_AVAILABLE = False
    logger.debug("Existing landslide_pipeline not available")


def get_import_status() -> Dict[str, bool]:
    """
    Get the import status of all data acquisition steps.
    
    Returns:
        Dictionary mapping step names to import success status
    """
    return _STEP_IMPORTS_SUCCESSFUL.copy()


def get_available_data_acquisition_steps() -> List[str]:
    """
    Get list of available data acquisition step types.
    
    Returns:
        List of available step type identifiers
    """
    return _AVAILABLE_STEPS.copy()


def get_step_aliases() -> Dict[str, str]:
    """
    Get mapping of step aliases to canonical step types.
    
    Returns:
        Dictionary mapping aliases to step types
    """
    return _STEP_ALIASES.copy()


def is_step_available(step_type: str) -> bool:
    """
    Check if a specific step type is available.
    
    Args:
        step_type: Step type identifier or alias
        
    Returns:
        True if step is available, False otherwise
    """
    # Check direct step type
    if step_type in _AVAILABLE_STEPS:
        return True
    
    # Check aliases
    if step_type in _STEP_ALIASES:
        canonical_type = _STEP_ALIASES[step_type]
        return canonical_type in _AVAILABLE_STEPS
    
    return False


def get_missing_dependencies() -> List[str]:
    """
    Get list of missing dependencies for data acquisition steps.
    
    Returns:
        List of missing dependency information
    """
    missing = []
    
    # Check for failed imports
    for step_name, success in _STEP_IMPORTS_SUCCESSFUL.items():
        if not success:
            missing.append(f"Step module: {step_name}")
    
    # Check for missing external dependencies
    if SentinelHubStep:
        try:
            sentinel_config = validate_sentinel_hub_config()
            for issue in sentinel_config.get('issues', []):
                missing.append(f"Sentinel Hub: {issue}")
        except:
            pass
    
    if DEMAcquisitionStep:
        try:
            dem_config = validate_dem_config()
            for issue in dem_config.get('issues', []):
                missing.append(f"DEM acquisition: {issue}")
        except:
            pass
    
    if LocalFilesStep:
        try:
            files_config = validate_local_files_config()
            for issue in files_config.get('issues', []):
                missing.append(f"Local files: {issue}")
        except:
            pass
    
    return missing


def validate_data_acquisition_setup() -> Dict[str, Any]:
    """
    Validate the complete data acquisition setup.
    
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
    if SentinelHubStep:
        try:
            validation_results['step_validations']['sentinel_hub'] = validate_sentinel_hub_config()
        except Exception as e:
            validation_results['step_validations']['sentinel_hub'] = {'error': str(e)}
    
    if DEMAcquisitionStep:
        try:
            validation_results['step_validations']['dem'] = validate_dem_config()
        except Exception as e:
            validation_results['step_validations']['dem'] = {'error': str(e)}
    
    if LocalFilesStep:
        try:
            validation_results['step_validations']['local_files'] = validate_local_files_config()
        except Exception as e:
            validation_results['step_validations']['local_files'] = {'error': str(e)}
    
    # Collect missing dependencies
    validation_results['missing_dependencies'] = get_missing_dependencies()
    
    # Determine overall status
    successful_imports = sum(_STEP_IMPORTS_SUCCESSFUL.values())
    total_core_steps = len([k for k in _STEP_IMPORTS_SUCCESSFUL.keys() 
                           if not k.endswith('_step') or k in ['sentinel_hub_step', 'dem_acquisition_step', 'local_files_step']])
    
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
        workflow_type: Type of workflow ('standard', 'satellite_only', 'local_only', 'comprehensive')
        **config: Workflow configuration parameters
        
    Returns:
        Workflow configuration dictionary
    """
    bbox = config.get('bbox', [85.3, 27.6, 85.4, 27.7])
    start_date = config.get('start_date', '2023-01-01')
    end_date = config.get('end_date', '2023-12-31')
    area_name = config.get('area_name', 'study_area')
    
    workflows = {
        'standard': {
            'process_info': {
                'name': f'Standard Data Acquisition - {area_name}',
                'version': '1.0.0',
                'description': 'Standard workflow with satellite and DEM data'
            },
            'steps': [
                {
                    'id': 'acquire_satellite_data',
                    'type': 'sentinel_hub_acquisition',
                    'hyperparameters': {
                        'bbox': bbox,
                        'start_date': start_date,
                        'end_date': end_date,
                        'data_collection': 'SENTINEL-2-L2A',
                        'resolution': 20
                    }
                },
                {
                    'id': 'acquire_elevation_data',
                    'type': 'dem_acquisition',
                    'hyperparameters': {
                        'bbox': bbox,
                        'source': 'SRTM',
                        'resolution': 30,
                        'generate_derivatives': True
                    }
                }
            ]
        },
        
        'satellite_only': {
            'process_info': {
                'name': f'Satellite Data Acquisition - {area_name}',
                'version': '1.0.0',
                'description': 'Satellite data acquisition only'
            },
            'steps': [
                {
                    'id': 'acquire_satellite_data',
                    'type': 'sentinel_hub_acquisition',
                    'hyperparameters': {
                        'bbox': bbox,
                        'start_date': start_date,
                        'end_date': end_date,
                        'data_collection': 'SENTINEL-2-L2A',
                        'resolution': 10,
                        'bands': ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
                    }
                }
            ]
        },
        
        'local_only': {
            'process_info': {
                'name': f'Local Files Discovery - {area_name}',
                'version': '1.0.0',
                'description': 'Local files discovery and validation'
            },
            'steps': [
                {
                    'id': 'discover_local_files',
                    'type': 'local_files_discovery',
                    'hyperparameters': {
                        'base_path': config.get('base_path', '/data/study_area'),
                        'file_patterns': config.get('file_patterns', ['*.tif', '*.shp']),
                        'recursive': True,
                        'validate_files': True,
                        'load_metadata': True
                    }
                }
            ]
        },
        
        'comprehensive': {
            'process_info': {
                'name': f'Comprehensive Data Acquisition - {area_name}',
                'version': '1.0.0',
                'description': 'Multi-source comprehensive data acquisition'
            },
            'steps': [
                {
                    'id': 'acquire_satellite_data',
                    'type': 'sentinel_hub_acquisition',
                    'hyperparameters': {
                        'bbox': bbox,
                        'start_date': start_date,
                        'end_date': end_date,
                        'data_collection': 'SENTINEL-2-L2A',
                        'resolution': 20
                    }
                },
                {
                    'id': 'acquire_elevation_data',
                    'type': 'dem_acquisition',
                    'hyperparameters': {
                        'bbox': bbox,
                        'source': 'SRTM',
                        'resolution': 30,
                        'generate_derivatives': True
                    }
                },
                {
                    'id': 'discover_local_files',
                    'type': 'local_files_discovery',
                    'hyperparameters': {
                        'base_path': config.get('base_path', '/data/auxiliary'),
                        'file_patterns': ['*.tif', '*.shp', '*.gpkg'],
                        'recursive': True
                    }
                }
            ]
        }
    }
    
    if workflow_type not in workflows:
        raise ValueError(f"Unknown workflow type: {workflow_type}. Available: {list(workflows.keys())}")
    
    return workflows[workflow_type]


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
    
    # Test each available step
    failed_tests = 0
    
    # Test SentinelHubStep
    if SentinelHubStep and create_test_sentinel_step:
        try:
            test_step = create_test_sentinel_step('test_sentinel', resolution=60)
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
    if DEMAcquisitionStep and create_test_dem_step:
        try:
            test_step = create_test_dem_step('test_dem', resolution=90)
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
    if LocalFilesStep and create_test_local_files_step:
        try:
            test_step = create_test_local_files_step('test_local_files', generate_mock_if_empty=True)
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
    
    # Determine overall status
    total_tests = len([t for t in test_results['step_tests'].values() if t['status'] != 'skipped'])
    if total_tests == 0:
        test_results['overall_status'] = 'no_tests'
    elif failed_tests == 0:
        test_results['overall_status'] = 'success'
    elif failed_tests < total_tests:
        test_results['overall_status'] = 'partial'
    else:
        test_results['overall_status'] = 'failed'
    
    return test_results


def print_module_status():
    """Print comprehensive status of the data acquisition module."""
    print(f"\n=== Data Acquisition Module v{__version__} ===")
    
    # Import status
    print("\nStep Import Status:")
    for step_name, success in _STEP_IMPORTS_SUCCESSFUL.items():
        status = "✓" if success else "✗"
        print(f"  {status} {step_name}")
    
    # Available steps
    if _AVAILABLE_STEPS:
        print(f"\nAvailable Step Types ({len(_AVAILABLE_STEPS)}):")
        for step_type in _AVAILABLE_STEPS:
            print(f"  • {step_type}")
    else:
        print("\n⚠ No step types available")
    
    # Aliases
    if _STEP_ALIASES:
        print(f"\nStep Aliases ({len(_STEP_ALIASES)}):")
        for alias, canonical in _STEP_ALIASES.items():
            print(f"  • {alias} → {canonical}")
    
    # Validation summary
    try:
        validation = validate_data_acquisition_setup()
        print(f"\nOverall Status: {validation['overall_status'].upper()}")
        
        if validation['capabilities']:
            print("\nCapabilities:")
            for capability, available in validation['capabilities'].items():
                status = "✓" if available else "✗"
                print(f"  {status} {capability}")
        
        if validation['missing_dependencies']:
            print(f"\nMissing Dependencies ({len(validation['missing_dependencies'])}):")
            for dep in validation['missing_dependencies'][:5]:  # Show first 5
                print(f"  ⚠ {dep}")
            if len(validation['missing_dependencies']) > 5:
                print(f"  ... and {len(validation['missing_dependencies'])-5} more")
    
    except Exception as e:
        print(f"\n⚠ Validation error: {e}")
    
    # Integration status
    if _EXISTING_PIPELINE_AVAILABLE:
        print("\n✓ Existing landslide_pipeline integration available")
    else:
        print("\n✗ Existing landslide_pipeline integration not available")


def get_help() -> str:
    """Get help text for the data acquisition module."""
    return f"""
Data Acquisition Module Help
===========================

This module provides comprehensive data acquisition capabilities for geospatial
analysis pipelines with fail-fast development support.

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

# Create a test step
if is_step_available('sentinel_hub_acquisition'):
    step = create_test_sentinel_step('test')

# Validate setup
validation = validate_data_acquisition_setup()
print(f"Status: {{validation['overall_status']}}")

For detailed documentation, see individual step classes:
- SentinelHubStep: Sentinel-2 satellite data acquisition
- DEMAcquisitionStep: Digital elevation model acquisition  
- LocalFilesStep: Local file discovery and validation
"""


# Export public API
__all__ = [
    # Core step classes (may be None if not available)
    'SentinelHubStep',
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
    
    # Step-specific utilities (may be None)
    'validate_sentinel_hub_config',
    'create_test_sentinel_step',
    'validate_dem_config',
    'create_test_dem_step',
    'get_supported_sources',
    'validate_local_files_config',
    'create_test_local_files_step',
    'get_supported_formats',
    'get_common_patterns',
    
    # Module metadata
    '__version__',
    '__author__'
]


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
    
    # Log integration status
    if _EXISTING_PIPELINE_AVAILABLE:
        logger.info("Existing landslide_pipeline integration available")
    else:
        logger.debug("Existing landslide_pipeline integration not available")


# Initialize module on import
_initialize_module()

# Quick test when run directly
if __name__ == "__main__":
    print("Running data acquisition module test...")
    
    # Print status
    print_module_status()
    
    # Run quick test
    print("\n" + "="*60)
    test_results = run_quick_test()
    
    print(f"\nModule Test Results:")
    print(f"Available Steps: {len(test_results['module_info']['available_steps'])}")
    print(f"Overall Status: {test_results['overall_status'].upper()}")
    
    # Show test results
    for step_name, result in test_results['step_tests'].items():
        status_symbol = {"success": "✓", "failed": "✗", "skipped": "⚠"}[result['status']]
        print(f"  {status_symbol} {step_name}: {result['status']}")
        if result['status'] == 'failed':
            print(f"    Error: {result.get('error', 'Unknown error')}")
    
    # Show help if not fully functional
    if test_results['overall_status'] in ['failed', 'partial', 'no_tests']:
        print(f"\nFor help:")
        print("python -c \"from orchestrator.steps.data_acquisition import get_help; print(get_help())\"")
    
    print(f"\nData acquisition module test completed!")
