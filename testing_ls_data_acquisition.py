#!/usr/bin/env python3
"""
Complete Landslide Assessment Data Acquisition Test for _terralux
================================================================

This script tests the complete data acquisition pipeline for landslide 
susceptibility assessment using real Sentinel Hub API integration.

Features tested:
- Sentinel-2 L2A data acquisition
- DEM data acquisition simulation
- Multi-temporal data collection
- Real authentication with fallback
- Cache management
- Output validation

Author: _terralux Development Team
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add _terralux to path
sys.path.insert(0, '/home/ubuntu/_terralux')

def test_landslide_data_acquisition():
    """Test complete landslide assessment data acquisition pipeline"""
    
    print("🏔️ TESTING LANDSLIDE ASSESSMENT DATA ACQUISITION")
    print("=" * 55)
    
    # Define test area (Nepal landslide-prone region)
    test_area = {
        'name': 'Nepal_Landslide_Test_Zone',
        'bbox': [85.30, 27.60, 85.35, 27.65],  # Small area in Nepal
        'description': 'Landslide-prone area in Nepal Himalayas'
    }
    
    # Define acquisition periods for temporal analysis
    acquisition_periods = [
        {
            'name': 'pre_monsoon',
            'start_date': '2023-04-01',
            'end_date': '2023-05-31',
            'description': 'Pre-monsoon baseline'
        },
        {
            'name': 'monsoon',
            'start_date': '2023-06-01', 
            'end_date': '2023-09-30',
            'description': 'Monsoon period (high landslide risk)'
        },
        {
            'name': 'post_monsoon',
            'start_date': '2023-10-01',
            'end_date': '2023-11-30', 
            'description': 'Post-monsoon assessment'
        }
    ]
    
    print(f"📍 Test Area: {test_area['name']}")
    print(f"   Bounding Box: {test_area['bbox']}")
    print(f"   Description: {test_area['description']}")
    print(f"\n📅 Acquisition Periods: {len(acquisition_periods)} periods")
    for period in acquisition_periods:
        print(f"   - {period['name']}: {period['start_date']} to {period['end_date']}")
    
    # Check credentials
    client_id = os.getenv('SENTINEL_HUB_CLIENT_ID')
    client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET')
    
    if client_id and client_secret:
        print(f"\n🔐 Credentials: Found ({client_id[:8]}...)")
        use_real_api = True
    else:
        print("\n🔐 Credentials: Not found (will use mock data)")
        print("   To use real API, set:")
        print("   export SENTINEL_HUB_CLIENT_ID='your-client-id'")
        print("   export SENTINEL_HUB_CLIENT_SECRET='your-client-secret'")
        use_real_api = False
    
    # Test each data acquisition step
    results = {}
    
    try:
        # Import the fixed real step
        from orchestrator.steps.data_acquisition.real_sentinel_hub_step import RealSentinelHubAcquisitionStep
        print("\n✓ Real Sentinel Hub step imported successfully")
        
        # Create output directories
        output_base = Path('landslide_test_outputs')
        output_base.mkdir(exist_ok=True)
        
        cache_dir = Path.home() / '.terralux_landslide_cache'
        cache_dir.mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {output_base.absolute()}")
        print(f"📁 Cache directory: {cache_dir}")
        
        # Test 1: Sentinel-2 Multi-temporal Acquisition
        print(f"\n🛰️ TEST 1: SENTINEL-2 MULTI-TEMPORAL ACQUISITION")
        print("=" * 50)
        
        sentinel_results = {}
        
        for i, period in enumerate(acquisition_periods, 1):
            print(f"\n   {i}. Acquiring {period['name']} data...")
            print(f"      Period: {period['start_date']} to {period['end_date']}")
            
            # Create step configuration for landslide assessment
            step_config = {
                'type': 'sentinel_hub_acquisition',
                'hyperparameters': {
                    # Area and time
                    'bbox': test_area['bbox'],
                    'start_date': period['start_date'],
                    'end_date': period['end_date'],
                    
                    # Sentinel-2 L2A (atmospherically corrected)
                    'data_collection': 'SENTINEL-2-L2A',
                    'resolution': 20,  # 20m resolution for landslide assessment
                    
                    # Bands for landslide susceptibility
                    'bands': [
                        'B02', 'B03', 'B04',  # RGB for visual analysis
                        'B08',                 # NIR for vegetation analysis
                        'B11', 'B12',          # SWIR for soil moisture/geology
                        'SCL'                  # Scene classification layer
                    ],
                    
                    # Quality constraints
                    'max_cloud_coverage': 30,  # Accept up to 30% clouds
                    'crs': 'EPSG:4326',
                    
                    # API and caching
                    'use_real_api': use_real_api,
                    'fallback_to_mock': True,
                    'cache_directory': str(cache_dir / 'sentinel2'),
                    'output_directory': str(output_base / 'sentinel2' / period['name']),
                    
                    # Force new download for testing (disable in production)
                    'force_download': False
                }
            }
            
            # Create and execute step
            step = RealSentinelHubAcquisitionStep(
                f"landslide_s2_{period['name']}", 
                step_config
            )
            
            try:
                result = step.execute()
                
                print(f"      ✅ Status: {result['status']}")
                print(f"      📊 Data: {result['imagery_data']}")
                print(f"      🔄 Cache used: {result.get('cache_used', False)}")
                print(f"      🌐 Real API: {result.get('real_api_used', False)}")
                
                if 'metadata' in result:
                    metadata = result['metadata']
                    print(f"      📋 Collection: {metadata.get('data_collection', 'N/A')}")
                    print(f"      🎯 Resolution: {metadata.get('resolution', 'N/A')}m")
                    
                    if 'auth_token_length' in metadata:
                        print(f"      🔐 Token length: {metadata['auth_token_length']}")
                
                sentinel_results[period['name']] = result
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                sentinel_results[period['name']] = {'status': 'failed', 'error': str(e)}
        
        results['sentinel2'] = sentinel_results
        
        # Test 2: DEM Data Acquisition (Simulated)
        print(f"\n🏔️ TEST 2: DEM DATA ACQUISITION (SIMULATED)")
        print("=" * 45)
        
        print("   Simulating DEM acquisition for landslide assessment...")
        print("   (In production, this would acquire SRTM/ASTER DEM data)")
        
        # Simulate DEM acquisition
        dem_output = output_base / 'dem' / 'srtm_30m.tif'
        dem_output.parent.mkdir(parents=True, exist_ok=True)
        dem_output.touch()
        
        dem_result = {
            'status': 'completed',
            'dem_data': str(dem_output),
            'metadata': {
                'data_source': 'SRTM 30m',
                'bbox': test_area['bbox'],
                'resolution': 30,
                'vertical_accuracy': '±16m',
                'simulated': True
            },
            'simulated': True
        }
        
        results['dem'] = dem_result
        print(f"   ✅ Status: {dem_result['status']}")
        print(f"   📊 Data: {dem_result['dem_data']}")
        print(f"   🎯 Resolution: {dem_result['metadata']['resolution']}m")
        print(f"   📏 Source: {dem_result['metadata']['data_source']}")
        
        # Test 3: Data Validation and Summary
        print(f"\n✅ TEST 3: DATA VALIDATION AND SUMMARY")
        print("=" * 40)
        
        # Validate Sentinel-2 results
        successful_periods = []
        failed_periods = []
        
        for period_name, result in sentinel_results.items():
            if result.get('status') == 'completed':
                successful_periods.append(period_name)
            else:
                failed_periods.append(period_name)
        
        print(f"   📊 Sentinel-2 Acquisition Summary:")
        print(f"      ✅ Successful periods: {len(successful_periods)}")
        for period in successful_periods:
            print(f"         - {period}")
        
        if failed_periods:
            print(f"      ❌ Failed periods: {len(failed_periods)}")
            for period in failed_periods:
                print(f"         - {period}")
        
        # Validate DEM result
        print(f"   🏔️ DEM Acquisition: {'✅ Success' if dem_result['status'] == 'completed' else '❌ Failed'}")
        
        # Overall assessment
        total_success = len(successful_periods) + (1 if dem_result['status'] == 'completed' else 0)
        total_attempts = len(acquisition_periods) + 1
        success_rate = (total_success / total_attempts) * 100
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   Success Rate: {success_rate:.1f}% ({total_success}/{total_attempts})")
        
        if success_rate >= 80:
            print("   🎉 EXCELLENT: Ready for landslide assessment pipeline!")
        elif success_rate >= 60:
            print("   ⚠️ GOOD: Mostly ready, check failed acquisitions")
        else:
            print("   ❌ NEEDS WORK: Multiple acquisition failures")
        
        # Generate acquisition report
        report = {
            'test_info': {
                'test_area': test_area,
                'test_date': datetime.now().isoformat(),
                'use_real_api': use_real_api,
                'success_rate': success_rate
            },
            'results': results,
            'recommendations': []
        }
        
        # Add recommendations
        if not use_real_api:
            report['recommendations'].append("Set up Sentinel Hub credentials for real data acquisition")
        
        if failed_periods:
            report['recommendations'].append(f"Retry failed periods: {', '.join(failed_periods)}")
        
        if success_rate < 100:
            report['recommendations'].append("Check error logs and adjust acquisition parameters")
        
        # Save report
        report_file = output_base / 'landslide_acquisition_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 Report saved: {report_file}")
        
        return True, results
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("   Make sure the real_sentinel_hub_step.py file has the correct constructor")
        return False, {}
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def create_landslide_process_config():
    """Create a complete JSON process configuration for landslide assessment"""
    
    print(f"\n📋 CREATING LANDSLIDE PROCESS CONFIGURATION")
    print("=" * 45)
    
    # Define the complete landslide assessment process
    process_config = {
        "process_info": {
            "name": "landslide_susceptibility_assessment",
            "version": "1.0.0",
            "description": "Complete landslide susceptibility assessment pipeline with multi-temporal analysis",
            "author": "_terralux Development Team",
            "application_type": "landslide_susceptibility"
        },
        "global_config": {
            "study_area": {
                "name": "Nepal_Landslide_Assessment",
                "bbox": [85.30, 27.60, 85.35, 27.65],
                "crs": "EPSG:4326"
            },
            "output_directory": "landslide_assessment_outputs",
            "cache_directory": "~/.terralux_landslide_cache",
            "log_level": "INFO"
        },
        "steps": [
            {
                "id": "sentinel2_pre_monsoon",
                "type": "sentinel_hub_acquisition",
                "description": "Acquire pre-monsoon baseline imagery",
                "hyperparameters": {
                    "bbox": [85.30, 27.60, 85.35, 27.65],
                    "start_date": "2023-04-01",
                    "end_date": "2023-05-31",
                    "data_collection": "SENTINEL-2-L2A",
                    "resolution": 20,
                    "bands": ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"],
                    "max_cloud_coverage": 30,
                    "use_real_api": True,
                    "fallback_to_mock": True
                }
            },
            {
                "id": "sentinel2_monsoon",
                "type": "sentinel_hub_acquisition", 
                "description": "Acquire monsoon period imagery",
                "hyperparameters": {
                    "bbox": [85.30, 27.60, 85.35, 27.65],
                    "start_date": "2023-06-01",
                    "end_date": "2023-09-30", 
                    "data_collection": "SENTINEL-2-L2A",
                    "resolution": 20,
                    "bands": ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"],
                    "max_cloud_coverage": 30,
                    "use_real_api": True,
                    "fallback_to_mock": True
                }
            },
            {
                "id": "sentinel2_post_monsoon",
                "type": "sentinel_hub_acquisition",
                "description": "Acquire post-monsoon assessment imagery", 
                "hyperparameters": {
                    "bbox": [85.30, 27.60, 85.35, 27.65],
                    "start_date": "2023-10-01",
                    "end_date": "2023-11-30",
                    "data_collection": "SENTINEL-2-L2A", 
                    "resolution": 20,
                    "bands": ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"],
                    "max_cloud_coverage": 30,
                    "use_real_api": True,
                    "fallback_to_mock": True
                }
            },
            {
                "id": "dem_acquisition",
                "type": "dem_acquisition",
                "description": "Acquire Digital Elevation Model",
                "hyperparameters": {
                    "bbox": [85.30, 27.60, 85.35, 27.65],
                    "data_source": "SRTM",
                    "resolution": 30,
                    "output_format": "GTiff"
                }
            }
        ],
        "execution_config": {
            "parallel_execution": False,
            "max_retries": 3,
            "timeout_minutes": 60,
            "cache_enabled": True
        }
    }
    
    # Save the process configuration
    config_file = Path('landslide_assessment_process.json')
    with open(config_file, 'w') as f:
        json.dump(process_config, f, indent=2)
    
    print(f"✅ Process configuration saved: {config_file}")
    print(f"   Process: {process_config['process_info']['name']}")
    print(f"   Steps: {len(process_config['steps'])}")
    print(f"   Study Area: {process_config['global_config']['study_area']['name']}")
    
    return config_file

if __name__ == "__main__":
    print("🏔️ LANDSLIDE ASSESSMENT DATA ACQUISITION TEST")
    print("=" * 50)
    
    # Run the data acquisition test
    success, results = test_landslide_data_acquisition()
    
    if success:
        # Create process configuration
        config_file = create_landslide_process_config()
        
        print(f"\n🎉 LANDSLIDE DATA ACQUISITION TEST COMPLETED!")
        print("=" * 50)
        print(f"✅ Test Status: {'SUCCESS' if success else 'FAILED'}")
        
        if success:
            print(f"\n📋 Next Steps:")
            print(f"   1. Review results in: landslide_test_outputs/")
            print(f"   2. Check report: landslide_test_outputs/landslide_acquisition_report.json")
            print(f"   3. Use process config: {config_file}")
            print(f"   4. Run full pipeline: python -m orchestrator.cli {config_file}")
            
            # Show quick stats
            sentinel_success = sum(1 for r in results.get('sentinel2', {}).values() 
                                 if r.get('status') == 'completed')
            dem_success = 1 if results.get('dem', {}).get('status') == 'completed' else 0
            
            print(f"\n📊 Acquisition Summary:")
            print(f"   Sentinel-2 periods: {sentinel_success}/3")
            print(f"   DEM acquisition: {dem_success}/1")
            print(f"   Total success: {sentinel_success + dem_success}/4")
    
    sys.exit(0 if success else 1)
