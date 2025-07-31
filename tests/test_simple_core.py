#!/usr/bin/env python3
"""
Simple Core Data Acquisition Test
=================================

A simplified version to test basic functionality and identify issues.
"""

import unittest
import tempfile
import time
from pathlib import Path

# Mock classes for development
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
                "mock_data_used": True,
                "fallback_used": self.config.get("hyperparameters", {}).get("fallback_to_mock", False)
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
            "derivatives": {
                "slope": f"/mock/path/{self.step_id}_slope.tif",
                "aspect": f"/mock/path/{self.step_id}_aspect.tif"
            } if self.config.get("hyperparameters", {}).get("generate_derivatives", False) else {},
            "metadata": {
                "source": source,
                "resolution": 90 if source == "SRTM" else 30,
                "mock_data_used": True,
                "derivatives_generated": ["slope", "aspect"] if self.config.get("hyperparameters", {}).get("generate_derivatives", False) else []
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
            "discovered_files": [
                f"/mock/path/local_raster_{i}.tif" for i in range(3)
            ],
            "file_inventory": {
                "total_files": 3,
                "file_types": {"raster": 2, "vector": 1},
                "total_size_mb": 25,
                "mock_files_generated": True
            },
            "metadata": {"discovery_complete": True},
            "status": "completed",
            "mock": True
        }

class SimpleDataAcquisitionTest(unittest.TestCase):
    """Simple test class for basic functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="simple_test_")
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_sentinel_acquisition(self):
        """Test Sentinel data acquisition."""
        step_config = {
            "id": "test_sentinel",
            "type": "sentinel_hub_acquisition",
            "hyperparameters": {
                "bbox": [85.30, 27.60, 85.32, 27.62],
                "use_mock_data": True
            }
        }
        
        step = SentinelHubAcquisitionStep(step_config)
        result = step.execute()
        
        self.assertIn("imagery_data", result)
        self.assertIn("metadata", result)
        self.assertEqual(result.get("status"), "completed")
        print("✓ Sentinel acquisition test passed")
    
    def test_dem_acquisition(self):
        """Test DEM data acquisition."""
        step_config = {
            "id": "test_dem",
            "type": "dem_acquisition",
            "hyperparameters": {
                "bbox": [85.30, 27.60, 85.32, 27.62],
                "source": "SRTM",
                "generate_derivatives": True,
                "use_mock_data": True
            }
        }
        
        step = DEMAcquisitionStep(step_config)
        result = step.execute()
        
        self.assertIn("elevation_data", result)
        self.assertIn("derivatives", result)
        self.assertIn("metadata", result)
        self.assertEqual(len(result["derivatives"]), 2)
        self.assertEqual(result.get("status"), "completed")
        print("✓ DEM acquisition test passed")
    
    def test_local_files_discovery(self):
        """Test local files discovery."""
        step_config = {
            "id": "test_local",
            "type": "local_files_discovery",
            "hyperparameters": {
                "base_path": str(self.temp_dir),
                "generate_mock_if_empty": True,
                "mock_file_count": 3
            }
        }
        
        step = LocalFilesDiscoveryStep(step_config)
        result = step.execute()
        
        self.assertIn("discovered_files", result)
        self.assertIn("file_inventory", result)
        self.assertEqual(len(result["discovered_files"]), 3)
        self.assertEqual(result.get("status"), "completed")
        print("✓ Local files discovery test passed")

def run_simple_tests():
    """Run simple core tests."""
    print("🧪 SIMPLE CORE DATA ACQUISITION TESTS")
    print("=" * 45)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(SimpleDataAcquisitionTest)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ All simple tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures + result.errors)} tests failed!")
        return False

if __name__ == "__main__":
    import sys
    success = run_simple_tests()
    sys.exit(0 if success else 1)
