#!/usr/bin/env python3
"""
Complete Integration Example: Real Sentinel Hub with JSON Process
================================================================

This example demonstrates how to integrate the real Sentinel Hub acquisition step
with your existing JSON process execution framework.

Features:
- Real API integration with fallback to mock
- JSON process definition support
- Template variable substitution
- Comprehensive error handling and logging
- Cache management and performance optimization

Author: Pipeline Development Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SentinelHub.Integration")

class EnhancedProcessExecutor:
    """Enhanced process executor with real Sentinel Hub integration"""
    
    def __init__(self, use_real_api: bool = True, cache_dir: Optional[Path] = None):
        self.use_real_api = use_real_api
        self.cache_dir = cache_dir or Path.home() / ".sentinel_hub_cache"
        
        # Register step types (real + fallback)
        self.step_types = {}
        self._register_step_types()
        
        logger.info(f"Process executor initialized (real_api: {use_real_api})")
    
    def _register_step_types(self):
        """Register both real and mock step implementations"""
        
        # Try to import real implementation
        try:
            from orchestrator.steps.data_acquisition.real_sentinel_hub_step import RealSentinelHubAcquisitionStep
            self.step_types['sentinel_hub_acquisition'] = RealSentinelHubAcquisitionStep
            logger.info("✓ Real Sentinel Hub step registered")
        except ImportError as e:
            logger.warning(f"Real Sentinel Hub step not available: {e}")
            self.use_real_api = False
        
        # Always include mock implementations as fallback
        from test_data_acquisition_core import (
            DEMAcquisitionStep,
            LocalFilesDiscoveryStep
        )
        
        # Mock Sentinel Hub step as fallback
        if 'sentinel_hub_acquisition' not in self.step_types:
            from test_data_acquisition_core import SentinelHubAcquisitionStep
            self.step_types['sentinel_hub_acquisition'] = SentinelHubAcquisitionStep
            logger.info("✓ Mock Sentinel Hub step registered as fallback")
        
        self.step_types.update({
            'dem_acquisition': DEMAcquisitionStep,
            'local_files_discovery': LocalFilesDiscoveryStep
        })
    
    def load_and_execute_process(self, process_file: str, template_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Load and execute a process with real/mock step integration"""
        
        # Load process definition
        with open(process_file, 'r') as f:
            process_def = json.load(f)
        
        logger.info(f"🚀 Executing: {process_def['process_info']['name']}")
        
        # Enhance template variables with API configuration
        enhanced_vars = self._enhance_template_variables(template_vars)
        
        # Substitute variables
        process_def = self._substitute_variables(process_def, enhanced_vars)
        
        # Execute process
        return self._execute_process_steps(process_def, enhanced_vars)
    
    def _enhance_template_variables(self, template_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance template variables with API configuration"""
        enhanced = template_vars.copy()
        
        if self.use_real_api:
            # Add Sentinel Hub credentials from environment
            enhanced.update({
                'sentinel_hub_client_id': os.getenv('SENTINEL_HUB_CLIENT_ID', ''),
                'sentinel_hub_client_secret': os.getenv('SENTINEL_HUB_CLIENT_SECRET', ''),
                'use_real_api': True,
                'cache_directory': str(self.cache_dir),
                'fallback_to_mock': True  # Always allow fallback
            })
        else:
            enhanced.update({
                'use_real_api': False,
                'fallback_to_mock': True
            })
        
        return enhanced
    
    def _substitute_variables(self, obj: Any, variables: Dict[str, Any]) -> Any:
        """Recursively substitute template variables"""
        if isinstance(obj, dict):
            return {k: self._substitute_variables(v, variables) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_variables(item, variables) for item in obj]
        elif isinstance(obj, str):
            for var_name, var_value in variables.items():
                obj = obj.replace(f"{{{var_name}}}", str(var_value))
            return obj
        return obj
    
    def _execute_process_steps(self, process_def: Dict[str, Any], template_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all process steps"""
        
        results = {
            'process_name': process_def['process_info']['name'],
            'status': 'success',
            'execution_time': datetime.now().isoformat(),
            'step_results': {},
            'artifacts': [],
            'performance': {
                'total_execution_time': 0,
                'steps_executed': 0,
                'real_api_calls': 0,
                'mock_fallbacks': 0,
                'cache_hits': 0
            }
        }
        
        start_time = datetime.now()
        
        # Execute each step
        for step_config in process_def['steps']:
            step_id = step_config['id']
            step_type = step_config['type']
            
            logger.info(f"📋 Executing step: {step_id} ({step_type})")
            
            try:
                # Enhance step configuration
                enhanced_config = self._enhance_step_config(step_config, template_vars)
                
                # Create and execute step
                step_result = self._execute_single_step(enhanced_config)
                
                # Track performance metrics
                self._update_performance_metrics(results['performance'], step_result)
                
                # Store results
                results['step_results'][step_id] = step_result
                
                # Extract artifacts
                self._extract_artifacts(step_result, step_id, results['artifacts'])
                
                logger.info(f"   ✓ {step_id}: {step_result.get('status', 'completed')}")
                
            except Exception as e:
                logger.error(f"   ❌ {step_id}: {str(e)}")
                results['step_results'][step_id] = {
                    'status': 'failed',
                    'error': str(e)
                }
                results['status'] = 'partial_failure'
        
        # Calculate total execution time
        end_time = datetime.now()
        results['performance']['total_execution_time'] = (end_time - start_time).total_seconds()
        results['performance']['steps_executed'] = len(process_def['steps'])
        
        return results
    
    def _enhance_step_config(self, step_config: Dict[str, Any], template_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance step configuration with API settings"""
        enhanced = step_config.copy()
        
        # Add API configuration to hyperparameters
        if 'hyperparameters' not in enhanced:
            enhanced['hyperparameters'] = {}
        
        # For Sentinel Hub steps, add API configuration
        if step_config['type'] == 'sentinel_hub_acquisition':
            enhanced['hyperparameters'].update({
                'client_id': template_vars.get('sentinel_hub_client_id'),
                'client_secret': template_vars.get('sentinel_hub_client_secret'),
                'cache_directory': template_vars.get('cache_directory'),
                'fallback_to_mock': template_vars.get('fallback_to_mock', True),
                'use_real_api': template_vars.get('use_real_api', False)
            })
        
        return enhanced
    
    def _execute_single_step(self, step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step"""
        step_type = step_config['type']
        
        if step_type not in self.step_types:
            raise ValueError(f"Unknown step type: {step_type}")
        
        step_class = self.step_types[step_type]
        step = step_class(step_config)
        
        return step.execute()
    
    def _update_performance_metrics(self, performance: Dict[str, Any], step_result: Dict[str, Any]):
        """Update performance metrics based on step result"""
        if step_result.get('mock', False):
            performance['mock_fallbacks'] += 1
        elif step_result.get('cache_used', False):
            performance['cache_hits'] += 1
        else:
            performance['real_api_calls'] += 1
    
    def _extract_artifacts(self, step_result: Dict[str, Any], step_id: str, artifacts: List[Dict[str, Any]]):
        """Extract artifacts from step result"""
        artifact_keys = [
            'imagery_data', 'elevation_data', 'discovered_files',
            'harmonized_stack', 'spectral_indices'
        ]
        
        for key in artifact_keys:
            if key in step_result:
                artifacts.append({
                    'step_id': step_id,
                    'type': key,
                    'path': step_result[key],
                    'mock': step_result.get('mock', False),
                    'cached': step_result.get('cache_used', False)
                })


def create_enhanced_process_definition() -> str:
    """Create an enhanced process definition with real API configuration"""
    
    process_def = {
        "process_info": {
            "name": "Enhanced Data Acquisition with Real Sentinel Hub",
            "version": "1.0.0",
            "description": "Real satellite data acquisition with fallback to mock",
            "application_type": "data_acquisition"
        },
        "global_config": {
            "template_variables": {
                "bbox": "{bbox}",
                "start_date": "{start_date}",
                "end_date": "{end_date}",
                "area_name": "{area_name}"
            },
            "output_directory": "outputs/{area_name}/enhanced_acquisition",
            "cache_enabled": True,
            "performance_monitoring": True
        },
        "steps": [
            {
                "id": "acquire_sentinel_real",
                "type": "sentinel_hub_acquisition",
                "description": "Real Sentinel-2 data acquisition with API",
                "hyperparameters": {
                    "bbox": "{bbox}",
                    "start_date": "{start_date}",
                    "end_date": "{end_date}",
                    "data_collection": "SENTINEL-2-L2A",
                    "resolution": 20,
                    "bands": ["B02", "B03", "B04", "B08"],
                    "max_cloud_coverage": 30,
                    "output_format": "GTiff",
                    "crs": "EPSG:4326",
                    "use_cache": True,
                    "force_download": False,
                    "timeout_seconds": 300
                }
            },
            {
                "id": "acquire_dem_srtm",
                "type": "dem_acquisition",
                "description": "SRTM DEM with topographic derivatives",
                "hyperparameters": {
                    "bbox": "{bbox}",
                    "source": "SRTM",
                    "resolution": 90,
                    "generate_derivatives": True,
                    "derivatives": ["slope", "aspect", "curvature"],
                    "void_fill": True,
                    "use_mock_data": True
                }
            },
            {
                "id": "discover_local_data",
                "type": "local_files_discovery",
                "description": "Discover existing local geospatial data",
                "hyperparameters": {
                    "base_path": "data/local/{area_name}",
                    "file_patterns": ["*.tif", "*.shp", "*.geojson"],
                    "recursive": True,
                    "validate_files": True,
                    "generate_mock_if_empty": True,
                    "mock_file_count": 2
                }
            }
        ]
    }
    
    # Save to temporary file
    temp_dir = Path(tempfile.mkdtemp(prefix="enhanced_process_"))
    process_file = temp_dir / "enhanced_acquisition_process.json"
    
    with open(process_file, 'w') as f:
        json.dump(process_def, f, indent=2)
    
    return str(process_file)


def demo_real_api_integration():
    """Demonstrate real API integration with comprehensive testing"""
    
    print("🛰️ ENHANCED SENTINEL HUB INTEGRATION DEMO")
    print("=" * 50)
    
    # Check if real API is available
    has_credentials = bool(
        os.getenv('SENTINEL_HUB_CLIENT_ID') and 
        os.getenv('SENTINEL_HUB_CLIENT_SECRET')
    )
    
    print(f"📡 Real API credentials: {'✓ Available' if has_credentials else '❌ Not found'}")
    print(f"🔄 Will use: {'Real API with mock fallback' if has_credentials else 'Mock implementation'}")
    
    # Test parameters for small, fast acquisition
    test_params = {
        'bbox': [85.30, 27.60, 85.32, 27.62],  # Small area in Nepal
        'start_date': '2023-06-01',
        'end_date': '2023-06-03',  # Short time range
        'area_name': 'nepal_enhanced_test'
    }
    
    print(f"\n📍 Test Parameters:")
    print(f"   Area: {test_params['area_name']}")
    print(f"   Bbox: {test_params['bbox']}")
    print(f"   Dates: {test_params['start_date']} to {test_params['end_date']}")
    
    try:
        # Create enhanced process executor
        executor = EnhancedProcessExecutor(use_real_api=has_credentials)
        
        # Create process definition
        process_file = create_enhanced_process_definition()
        print(f"\n📄 Process definition: {process_file}")
        
        # Execute process
        print(f"\n🚀 Executing enhanced process...")
        start_time = datetime.now()
        
        results = executor.load_and_execute_process(process_file, test_params)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Display results
        print(f"\n📊 EXECUTION RESULTS:")
        print(f"=" * 30)
        print(f"Process: {results['process_name']}")
        print(f"Status: {results['status']}")
        print(f"Execution time: {execution_time:.2f}s")
        print(f"Steps executed: {results['performance']['steps_executed']}")
        
        # Performance breakdown
        perf = results['performance']
        print(f"\n📈 Performance Breakdown:")
        print(f"   Real API calls: {perf['real_api_calls']}")
        print(f"   Mock fallbacks: {perf['mock_fallbacks']}")
        print(f"   Cache hits: {perf['cache_hits']}")
        
        # Step results
        print(f"\n📋 Step Results:")
        for step_id, step_result in results['step_results'].items():
            status = step_result.get('status', 'unknown')
            mock_used = step_result.get('mock', False)
            cached = step_result.get('cache_used', False)
            
            status_icon = "✓" if status == 'completed' else "❌"
            api_info = "mock" if mock_used else ("cached" if cached else "real")
            
            print(f"   {status_icon} {step_id}: {status} ({api_info})")
            
            # Show additional details for Sentinel Hub step
            if step_id == 'acquire_sentinel_real' and 'metadata' in step_result:
                metadata = step_result['metadata']
                if 'file_size_mb' in metadata:
                    print(f"      File size: {metadata['file_size_mb']:.2f} MB")
                if 'data_shape' in metadata:
                    print(f"      Data shape: {metadata['data_shape']}")
        
        # Artifacts
        if results['artifacts']:
            print(f"\n📁 Generated Artifacts:")
            for artifact in results['artifacts']:
                api_type = "mock" if artifact['mock'] else ("cached" if artifact['cached'] else "real")
                print(f"   - {artifact['step_id']}: {artifact['type']} ({api_type})")
                print(f"     Path: {artifact['path']}")
        
        # Success/failure summary
        failed_steps = [
            step_id for step_id, result in results['step_results'].items()
            if result.get('status') == 'failed'
        ]
        
        if not failed_steps:
            print(f"\n✅ All steps completed successfully!")
        else:
            print(f"\n⚠️  Some steps failed: {failed_steps}")
        
        # Real API usage summary
        if has_credentials and perf['real_api_calls'] > 0:
            print(f"\n🎉 Successfully used real Sentinel Hub API!")
            print(f"   Consider checking your quota usage at:")
            print(f"   https://apps.sentinel-hub.com/")
        elif has_credentials and perf['real_api_calls'] == 0:
            print(f"\n💡 Real API was available but used cached/mock data")
            print(f"   This is expected for repeated runs with caching enabled")
        else:
            print(f"\n💡 Demo completed using mock data")
            print(f"   Set SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET to test real API")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_production_process_template():
    """Create a production-ready process template"""
    
    production_process = {
        "process_info": {
            "name": "Production Landslide Susceptibility with Real Data",
            "version": "1.0.0",
            "description": "Complete landslide susceptibility workflow with real satellite data",
            "application_type": "landslide_susceptibility",
            "author": "Pipeline Development Team",
            "created": datetime.now().isoformat()
        },
        "global_config": {
            "template_variables": {
                "bbox": "{bbox}",
                "start_date": "{start_date}",
                "end_date": "{end_date}",
                "area_name": "{area_name}",
                "landslide_inventory_path": "{landslide_inventory_path}"
            },
            "output_directory": "outputs/{area_name}/landslide_susceptibility",
            "cache_enabled": True,
            "parallel_execution": False,
            "error_handling": "continue_on_failure",
            "performance_monitoring": True
        },
        "steps": [
            {
                "id": "acquire_sentinel2_data",
                "type": "sentinel_hub_acquisition",
                "description": "Acquire Sentinel-2 multispectral imagery",
                "hyperparameters": {
                    "bbox": "{bbox}",
                    "start_date": "{start_date}",
                    "end_date": "{end_date}",
                    "data_collection": "SENTINEL-2-L2A",
                    "resolution": 10,
                    "bands": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
                    "max_cloud_coverage": 15,
                    "atmospheric_correction": True,
                    "cloud_mask": True,
                    "use_cache": True
                }
            },
            {
                "id": "acquire_sentinel1_data",
                "type": "sentinel_hub_acquisition",
                "description": "Acquire Sentinel-1 SAR data",
                "hyperparameters": {
                    "bbox": "{bbox}",
                    "start_date": "{start_date}",
                    "end_date": "{end_date}",
                    "data_collection": "SENTINEL-1-GRD",
                    "resolution": 10,
                    "bands": ["VV", "VH"],
                    "orbit_direction": "ASCENDING",
                    "use_cache": True,
                    "optional": True
                }
            },
            {
                "id": "acquire_high_res_dem",
                "type": "dem_acquisition",
                "description": "High-resolution DEM with derivatives",
                "hyperparameters": {
                    "bbox": "{bbox}",
                    "source": "ASTER",
                    "resolution": 30,
                    "generate_derivatives": True,
                    "derivatives": ["slope", "aspect", "curvature", "roughness"],
                    "void_fill": True,
                    "fallback_source": "SRTM"
                }
            },
            {
                "id": "discover_auxiliary_data",
                "type": "local_files_discovery",
                "description": "Discover local geological and infrastructure data",
                "hyperparameters": {
                    "base_path": "data/auxiliary/{area_name}",
                    "file_patterns": ["*.shp", "*.geojson", "*.gpkg"],
                    "data_types": ["geology", "infrastructure", "hydrology"],
                    "validate_files": True,
                    "load_metadata": True
                }
            }
        ]
    }
    
    # Save production template
    template_file = Path("production_landslide_process_template.json")
    with open(template_file, 'w') as f:
        json.dump(production_process, f, indent=2)
    
    print(f"✓ Production template created: {template_file}")
    return str(template_file)


def show_next_steps():
    """Show next steps for full implementation"""
    
    print(f"\n🚀 NEXT STEPS FOR FULL IMPLEMENTATION")
    print("=" * 40)
    
    print(f"\n1. 📡 **Set Up Real API Access**")
    print(f"   - Get Sentinel Hub account: https://apps.sentinel-hub.com/")
    print(f"   - Set environment variables:")
    print(f"     export SENTINEL_HUB_CLIENT_ID='your-client-id'")
    print(f"     export SENTINEL_HUB_CLIENT_SECRET='your-client-secret'")
    
    print(f"\n2. 🧪 **Test Real Data Acquisition**")
    print(f"   - Run: python test_real_acquisition.py")
    print(f"   - Verify downloaded data quality")
    print(f"   - Check quota usage on dashboard")
    
    print(f"\n3. 🔧 **Implement Additional Steps**")
    print(f"   - Preprocessing: atmospheric correction, cloud masking")
    print(f"   - Feature extraction: spectral indices, texture analysis")
    print(f"   - Data harmonization: CRS transformation, resampling")
    
    print(f"\n4. 🏭 **Production Deployment**")
    print(f"   - Set up automated workflows")
    print(f"   - Configure monitoring and alerting")
    print(f"   - Implement data quality checks")
    
    print(f"\n5. 📊 **Performance Optimization**")
    print(f"   - Enable parallel processing")
    print(f"   - Optimize cache settings")
    print(f"   - Monitor resource usage")
    
    print(f"\n📚 **Resources**")
    print(f"   - Sentinel Hub Documentation: https://docs.sentinel-hub.com/")
    print(f"   - API Reference: https://docs.sentinel-hub.com/api/latest/")
    print(f"   - Community Forum: https://forum.sentinel-hub.com/")


def main():
    """Main demonstration function"""
    
    # Run the comprehensive demo
    print("Starting enhanced Sentinel Hub integration demo...")
    
    success = demo_real_api_integration()
    
    if success:
        # Create production template
        print(f"\n" + "="*50)
        create_production_process_template()
        
        # Show next steps
        show_next_steps()
        
        print(f"\n✅ Enhanced integration demo completed successfully!")
    else:
        print(f"\n❌ Demo failed. Check the error messages above.")
    
    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
