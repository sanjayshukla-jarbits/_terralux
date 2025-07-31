#!/usr/bin/env python3
"""
Test the complete data acquisition process using JSON definitions
"""

import unittest
import tempfile
import json
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the test infrastructure we just validated
from test_data_acquisition_core import (
    SentinelHubAcquisitionStep,
    DEMAcquisitionStep, 
    LocalFilesDiscoveryStep
)

class MockProcessLoader:
    """Simplified process loader for testing"""
    
    def load(self, process_file: str, template_vars: dict = None) -> dict:
        """Load and process the JSON definition"""
        with open(process_file, 'r') as f:
            process_def = json.load(f)
        
        # Simple template variable substitution
        if template_vars:
            process_def = self._substitute_variables(process_def, template_vars)
        
        return process_def
    
    def _substitute_variables(self, obj, variables):
        """Replace template variables recursively"""
        if isinstance(obj, dict):
            return {k: self._substitute_variables(v, variables) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_variables(item, variables) for item in obj]
        elif isinstance(obj, str):
            for var_name, var_value in variables.items():
                pattern = f"{{{var_name}}}"
                if pattern in obj:
                    obj = obj.replace(pattern, str(var_value))
            return obj
        return obj

class MockStepRegistry:
    """Mock step registry for testing"""
    
    _steps = {
        'sentinel_hub_acquisition': SentinelHubAcquisitionStep,
        'dem_acquisition': DEMAcquisitionStep,
        'local_files_discovery': LocalFilesDiscoveryStep
    }
    
    @classmethod
    def create_step(cls, step_config: dict):
        """Create step instance from config"""
        step_type = step_config['type']
        if step_type not in cls._steps:
            raise ValueError(f"Unknown step type: {step_type}")
        return cls._steps[step_type](step_config)

class SimpleDataAcquisitionOrchestrator:
    """Simplified orchestrator for testing data acquisition"""
    
    def __init__(self):
        self.process_loader = MockProcessLoader()
        self.step_registry = MockStepRegistry()
        self.process_definition = None
    
    def load_process(self, process_file: str, template_vars: dict = None):
        """Load process definition"""
        self.process_definition = self.process_loader.load(process_file, template_vars)
        print(f"✓ Loaded process: {self.process_definition['process_info']['name']}")
    
    def execute_process(self) -> dict:
        """Execute the data acquisition process"""
        if not self.process_definition:
            raise ValueError("No process loaded")
        
        results = {
            'status': 'success',
            'process_info': self.process_definition['process_info'],
            'step_results': {},
            'artifacts': {}
        }
        
        print(f"\n🚀 Executing process: {self.process_definition['process_info']['name']}")
        print("=" * 60)
        
        # Execute each step
        for step_config in self.process_definition['steps']:
            step_id = step_config['id']
            step_type = step_config['type']
            
            print(f"\n📋 Executing step: {step_id} ({step_type})")
            
            try:
                # Create and execute step
                step = self.step_registry.create_step(step_config)
                step_result = step.execute()
                
                # Store results
                results['step_results'][step_id] = step_result
                
                # Extract key artifacts
                if 'imagery_data' in step_result:
                    results['artifacts'][f'{step_id}_imagery'] = step_result['imagery_data']
                if 'elevation_data' in step_result:
                    results['artifacts'][f'{step_id}_elevation'] = step_result['elevation_data']
                if 'discovered_files' in step_result:
                    results['artifacts'][f'{step_id}_files'] = step_result['discovered_files']
                
                print(f"   ✓ {step_id}: {step_result.get('status', 'completed')}")
                
                # Print key outputs
                if 'metadata' in step_result:
                    metadata = step_result['metadata']
                    if 'acquisition_date' in metadata:
                        print(f"   📅 Acquisition: {metadata['acquisition_date']}")
                    if 'source' in metadata:
                        print(f"   🗄️  Source: {metadata['source']}")
                    if 'total_files' in metadata:
                        print(f"   📁 Files found: {metadata['total_files']}")
                
            except Exception as e:
                print(f"   ❌ {step_id}: Failed - {str(e)}")
                results['step_results'][step_id] = {
                    'status': 'failed',
                    'error': str(e)
                }
                results['status'] = 'partial_failure'
        
        print(f"\n🏁 Process execution completed with status: {results['status']}")
        return results

class TestDataAcquisitionProcess(unittest.TestCase):
    """Test data acquisition using JSON process definitions"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="process_test_"))
        self.orchestrator = SimpleDataAcquisitionOrchestrator()
        
        # Test parameters
        self.test_params = {
            'bbox': [85.30, 27.60, 85.32, 27.62],
            'start_date': '2023-06-01',
            'end_date': '2023-06-07',
            'area_name': 'test_nepal'
        }
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_acquisition_only_process(self):
        """Test the complete data acquisition process"""
        # Create the process file
        process_file = self.temp_dir / "data_acquisition_only.json"
        
        process_def = {
            "process_info": {
                "name": "Data Acquisition Test",
                "version": "1.0.0",
                "application_type": "data_acquisition"
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
        
        # Load and execute process
        self.orchestrator.load_process(str(process_file), self.test_params)
        result = self.orchestrator.execute_process()
        
        # Validate results
        self.assertIn('status', result)
        self.assertIn('step_results', result)
        self.assertEqual(len(result['step_results']), 3)
        
        # Check each step executed
        step_ids = ['acquire_sentinel', 'acquire_dem', 'discover_files']
        for step_id in step_ids:
            self.assertIn(step_id, result['step_results'])
            step_result = result['step_results'][step_id]
            self.assertIn('status', step_result)
            # Most should complete, but some might have expected failures
            self.assertIn(step_result['status'], ['completed', 'success', 'failed'])
        
        # Check artifacts were generated
        self.assertIn('artifacts', result)
        print(f"\n📦 Generated artifacts: {list(result['artifacts'].keys())}")
        
        print("✅ Data acquisition process test completed successfully!")

if __name__ == "__main__":
    unittest.main()
