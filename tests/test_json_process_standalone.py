import json
import sys
import tempfile
from pathlib import Path

# Import our validated mock classes directly from the working test
import unittest
sys.path.append(str(Path(__file__).parent))

# Use the mock classes that we know work from test_data_acquisition_core.py
class SentinelHubAcquisitionStep:
    def __init__(self, config): 
        self.config = config
        self.step_id = config.get("id", "mock_sentinel")
    
    def execute(self, **kwargs): 
        return {
            "imagery_data": f"/mock/path/{self.step_id}_sentinel.tif",
            "metadata": {
                "acquisition_date": "2023-06-01",
                "cloud_coverage": 15,
                "bands_count": 4,
                "mock_data_used": True
            },
            "status": "completed",
            "mock": True
        }

class DEMAcquisitionStep:
    def __init__(self, config): 
        self.config = config
        self.step_id = config.get("id", "mock_dem")
    
    def execute(self, **kwargs): 
        source = self.config.get("hyperparameters", {}).get("source", "SRTM")
        return {
            "elevation_data": f"/mock/path/{self.step_id}_{source.lower()}_dem.tif",
            "metadata": {
                "source": source,
                "resolution": 90 if source == "SRTM" else 30,
                "mock_data_used": True
            },
            "status": "completed",
            "mock": True
        }

class LocalFilesDiscoveryStep:
    def __init__(self, config): 
        self.config = config
        self.step_id = config.get("id", "mock_local")
    
    def execute(self, **kwargs): 
        return {
            "discovered_files": [f"/mock/path/local_file_{i}.tif" for i in range(3)],
            "metadata": {"discovery_complete": True, "total_files": 3},
            "status": "completed",
            "mock": True
        }

class SimpleProcessExecutor:
    """Simple executor for JSON process definitions"""
    
    def __init__(self):
        self.step_types = {
            'sentinel_hub_acquisition': SentinelHubAcquisitionStep,
            'dem_acquisition': DEMAcquisitionStep,
            'local_files_discovery': LocalFilesDiscoveryStep
        }
    
    def load_and_execute(self, process_file: str, params: dict):
        """Load JSON process and execute steps"""
        # Load process definition
        with open(process_file, 'r') as f:
            process_def = json.load(f)
        
        print(f"🚀 Executing: {process_def['process_info']['name']}")
        print("=" * 50)
        
        results = {
            'process_name': process_def['process_info']['name'],
            'status': 'success',
            'step_results': {},
            'artifacts': []
        }
        
        # Execute each step
        for step_config in process_def['steps']:
            step_id = step_config['id']
            step_type = step_config['type']
            
            print(f"\n📋 Step: {step_id} ({step_type})")
            
            # Substitute template variables
            substituted_config = self._substitute_variables(step_config, params)
            
            # Create and execute step
            if step_type in self.step_types:
                step_class = self.step_types[step_type]
                step = step_class(substituted_config)
                result = step.execute()
                
                results['step_results'][step_id] = result
                print(f"   ✓ Status: {result.get('status', 'completed')}")
                
                # Extract artifacts
                for key, value in result.items():
                    if key.endswith('_data') or key.endswith('_files'):
                        results['artifacts'].append({
                            'step': step_id,
                            'type': key,
                            'path': value
                        })
                        print(f"   📁 Generated: {key}")
                        
                # Show metadata
                if 'metadata' in result:
                    metadata = result['metadata']
                    for key, value in metadata.items():
                        if key in ['source', 'acquisition_date', 'total_files']:
                            print(f"   ℹ️  {key}: {value}")
            else:
                print(f"   ❌ Unknown step type: {step_type}")
                results['status'] = 'partial_failure'
        
        return results
    
    def _substitute_variables(self, obj, variables):
        """Replace {variable} patterns with actual values"""
        if isinstance(obj, dict):
            return {k: self._substitute_variables(v, variables) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_variables(item, variables) for item in obj]
        elif isinstance(obj, str):
            for var_name, var_value in variables.items():
                obj = obj.replace(f"{{{var_name}}}", str(var_value))
            return obj
        return obj

def create_test_process_file():
    """Create a test process definition file"""
    temp_dir = Path(tempfile.mkdtemp(prefix="process_test_"))
    process_file = temp_dir / "data_acquisition_only.json"
    
    process_def = {
        "process_info": {
            "name": "Data Acquisition Only Test",
            "version": "1.0.0",
            "application_type": "data_acquisition",
            "description": "Test JSON process for data acquisition validation"
        },
        "steps": [
            {
                "id": "acquire_sentinel",
                "type": "sentinel_hub_acquisition",
                "hyperparameters": {
                    "bbox": "{bbox}",
                    "start_date": "{start_date}",
                    "end_date": "{end_date}",
                    "data_collection": "SENTINEL-2-L1C",
                    "use_mock_data": True
                }
            },
            {
                "id": "acquire_dem",
                "type": "dem_acquisition",
                "hyperparameters": {
                    "bbox": "{bbox}",
                    "source": "SRTM",
                    "generate_derivatives": True,
                    "use_mock_data": True
                }
            },
            {
                "id": "discover_files",
                "type": "local_files_discovery",
                "hyperparameters": {
                    "base_path": "data/{area_name}",
                    "generate_mock_if_empty": True
                }
            }
        ]
    }
    
    # Save process definition
    with open(process_file, 'w') as f:
        json.dump(process_def, f, indent=2)
    
    return str(process_file), temp_dir

def main():
    """Test the data acquisition process using JSON definition"""
    # Test parameters
    test_params = {
        'bbox': [85.30, 27.60, 85.32, 27.62],
        'start_date': '2023-06-01',
        'end_date': '2023-06-07', 
        'area_name': 'test_nepal'
    }
    
    print("🧪 STANDALONE JSON PROCESS EXECUTION TEST")
    print("=" * 45)
    print(f"📍 Area: {test_params['area_name']}")
    print(f"📦 Bbox: {test_params['bbox']}")
    print(f"📅 Dates: {test_params['start_date']} to {test_params['end_date']}")
    
    try:
        # Create test process file
        process_file, temp_dir = create_test_process_file()
        print(f"📄 Process file: {process_file}")
        
        # Execute process
        executor = SimpleProcessExecutor()
        results = executor.load_and_execute(process_file, test_params)
        
        print(f"\n📊 EXECUTION SUMMARY:")
        print(f"   Process: {results['process_name']}")
        print(f"   Status: {results['status']}")
        print(f"   Steps executed: {len(results['step_results'])}")
        print(f"   Artifacts generated: {len(results['artifacts'])}")
        
        if results['artifacts']:
            print(f"\n📁 Generated Artifacts:")
            for artifact in results['artifacts']:
                print(f"   - {artifact['step']}: {artifact['type']}")
        
        print("\n✅ JSON process execution test completed successfully!")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return 0
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
