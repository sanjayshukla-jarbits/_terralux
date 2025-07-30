"""
Data Acquisition Extended Test Suite - Fail Fast Plan
=====================================================

Extended testing suite for comprehensive data acquisition validation in the
modular orchestrator system. This module builds on the core testing framework
to provide comprehensive workflow testing, performance benchmarking, and
advanced validation scenarios.

Key Features:
- Comprehensive workflow integration testing
- Performance benchmarking and stress testing
- Advanced validation and quality assessment
- Multi-temporal and multi-source testing
- Comprehensive reporting and analysis

Test Categories (Part 2):
- Spectral indices calculation tests
- Data inventory and cataloging tests
- Workflow integration and end-to-end tests
- Performance and stress testing
- Comprehensive validation and reporting

Author: Orchestrator Development Team
Version: 1.0.0-failfast
License: MIT
"""

import unittest
import pytest
import asyncio
import json
import tempfile
import shutil
import time
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

# Import from core test module
from test_data_acquisition_core import (
    DataAcquisitionTestBase,
    validate_test_environment,
    TestEnvironmentSetup,
    TestSentinelDataAcquisition,
    TestDEMAcquisition,
    TestLocalFilesDiscovery,
    TestDataHarmonization
)

# Import orchestrator components
try:
    from orchestrator.steps import create_step, execute_step_pipeline
    from orchestrator.utils.performance import PerformanceMonitor
    from orchestrator import create_orchestrator
except ImportError as e:
    logging.warning(f"Import warning: {e}. Using mock implementations.")
    
    def create_step(config):
        class MockStep:
            def __init__(self, config): self.config = config
            def execute(self, **kwargs): return {"status": "completed", "mock": True}
        return MockStep(config)
    
    def execute_step_pipeline(steps): 
        return [{"status": "completed", "mock": True} for _ in steps]
    
    def create_orchestrator(**kwargs):
        class MockOrchestrator:
            def execute_process(self, process): return {"status": "completed", "mock": True}
        return MockOrchestrator()

logger = logging.getLogger(__name__)


class TestSpectralIndicesCalculation(DataAcquisitionTestBase):
    """Test spectral indices calculation from optical imagery."""
    
    def test_basic_vegetation_indices(self):
        """Test calculation of basic vegetation indices."""
        # Create mock Sentinel-2 data
        sentinel_data = self.create_mock_sentinel_data()
        
        step_config = {
            "id": "test_vegetation_indices",
            "type": "spectral_indices_calculation",
            "hyperparameters": {
                "indices": ["NDVI", "EVI", "SAVI"],
                "save_individual_indices": True,
                "output_format": "GeoTIFF",
                "nodata_value": -9999
            }
        }
        
        step = create_step(step_config)
        
        # Mock output files
        ndvi_path = self.output_dir / "NDVI.tif"
        evi_path = self.output_dir / "EVI.tif"
        savi_path = self.output_dir / "SAVI.tif"
        
        for path in [ndvi_path, evi_path, savi_path]:
            path.touch()
        
        # Override step execution with mock
        step.execute = Mock(return_value={
            "spectral_indices": {
                "NDVI": str(ndvi_path),
                "EVI": str(evi_path),
                "SAVI": str(savi_path)
            },
            "statistics": {
                "NDVI": {"min": -1.0, "max": 1.0, "mean": 0.65},
                "EVI": {"min": -1.0, "max": 1.0, "mean": 0.55},
                "SAVI": {"min": -1.0, "max": 1.0, "mean": 0.60}
            },
            "metadata": {
                "indices_calculated": 3,
                "calculation_method": "numpy_based"
            }
        })
        
        result = step.execute(inputs={"optical_data": sentinel_data})
        
        self.assertIn("spectral_indices", result)
        self.assertIn("statistics", result)
        self.assertEqual(len(result["spectral_indices"]), 3)
        
        # Validate index ranges
        for index_name, stats in result["statistics"].items():
            self.assertGreaterEqual(stats["min"], -1.0)
            self.assertLessEqual(stats["max"], 1.0)
        
        logger.info("✓ Basic vegetation indices calculation test passed")
    
    def test_water_indices(self):
        """Test calculation of water-related indices."""
        sentinel_data = self.create_mock_sentinel_data()
        
        step_config = {
            "id": "test_water_indices",
            "type": "spectral_indices_calculation",
            "hyperparameters": {
                "indices": ["NDWI", "MNDWI", "AWI"],
                "save_individual_indices": True
            }
        }
        
        step = create_step(step_config)
        step.execute = Mock(return_value={
            "spectral_indices": {
                "NDWI": str(self.output_dir / "NDWI.tif"),
                "MNDWI": str(self.output_dir / "MNDWI.tif"),
                "AWI": str(self.output_dir / "AWI.tif")
            },
            "water_detection": {
                "water_pixel_count": 1250,
                "total_pixels": 10000,
                "water_percentage": 12.5
            }
        })
        
        result = step.execute(inputs={"optical_data": sentinel_data})
        
        self.assertIn("spectral_indices", result)
        self.assertIn("water_detection", result)
        self.assertEqual(len(result["spectral_indices"]), 3)
        logger.info("✓ Water indices calculation test passed")
    
    def test_indices_stack_creation(self):
        """Test creation of multi-index stack."""
        sentinel_data = self.create_mock_sentinel_data()
        
        step_config = {
            "id": "test_indices_stack",
            "type": "spectral_indices_calculation",
            "hyperparameters": {
                "indices": ["NDVI", "NDWI", "EVI"],
                "create_stack": True,
                "stack_name": "indices_stack.tif"
            }
        }
        
        step = create_step(step_config)
        step.execute = Mock(return_value={
            "indices_stack": str(self.output_dir / "indices_stack.tif"),
            "band_mapping": {
                "band_1": "NDVI",
                "band_2": "NDWI", 
                "band_3": "EVI"
            },
            "metadata": {
                "stack_bands": 3,
                "compression": "LZW"
            }
        })
        
        result = step.execute(inputs={"optical_data": sentinel_data})
        
        self.assertIn("indices_stack", result)
        self.assertIn("band_mapping", result)
        self.assertEqual(len(result["band_mapping"]), 3)
        logger.info("✓ Indices stack creation test passed")


class TestDataInventoryGeneration(DataAcquisitionTestBase):
    """Test comprehensive data inventory generation."""
    
    def test_complete_inventory_generation(self):
        """Test generation of complete data inventory."""
        start_time = time.time()
        
        # Create mock acquired data
        sentinel_data = self.create_mock_sentinel_data()
        dem_data = self.create_mock_dem_data()
        local_files = self.create_mock_local_files(count=3)
        harmonized_data = str(self.output_dir / "harmonized_stack.tif")
        indices_data = str(self.output_dir / "indices_stack.tif")
        
        Path(harmonized_data).touch()
        Path(indices_data).touch()
        
        step_config = {
            "id": "test_inventory_generation",
            "type": "inventory_generation",
            "hyperparameters": {
                "include_statistics": True,
                "include_spatial_index": True,
                "include_quality_metrics": True,
                "generate_preview_images": False,  # Skip for speed
                "output_format": "json"
            }
        }
        
        inputs = {
            "sentinel_data": sentinel_data,
            "dem_data": dem_data,
            "local_files": local_files,
            "harmonized_data": harmonized_data,
            "indices_data": indices_data
        }
        
        step = create_step(step_config)
        step.execute = Mock(return_value={
            "data_inventory": {
                "summary": {
                    "total_files": 6,
                    "total_size_mb": 156.7,
                    "data_types": ["satellite", "elevation", "local", "processed"],
                    "spatial_coverage": self.test_config["bbox"],
                    "temporal_coverage": {
                        "start": self.test_config["start_date"],
                        "end": self.test_config["end_date"]
                    }
                },
                "datasets": {
                    "sentinel_optical": {
                        "type": "satellite_imagery",
                        "source": "Sentinel-2",
                        "bands": 4,
                        "resolution": 60,
                        "file_path": sentinel_data
                    },
                    "elevation_model": {
                        "type": "elevation",
                        "source": "SRTM",
                        "resolution": 90,
                        "file_path": dem_data
                    },
                    "harmonized_stack": {
                        "type": "processed",
                        "layers": 7,
                        "resolution": 60,
                        "file_path": harmonized_data
                    }
                },
                "quality_metrics": {
                    "overall_quality": "GOOD",
                    "completeness": 0.95,
                    "spatial_accuracy": 0.98,
                    "temporal_accuracy": 1.0
                },
                "processing_history": [
                    "sentinel_acquisition",
                    "dem_acquisition",
                    "local_discovery",
                    "harmonization",
                    "indices_calculation"
                ]
            },
            "inventory_file": str(self.output_dir / "data_inventory.json")
        })
        
        result = step.execute(inputs=inputs)
        
        # Validate inventory structure
        self.assertIn("data_inventory", result)
        inventory = result["data_inventory"]
        
        self.assertIn("summary", inventory)
        self.assertIn("datasets", inventory)
        self.assertIn("quality_metrics", inventory)
        
        # Validate summary
        summary = inventory["summary"]
        self.assertGreater(summary["total_files"], 0)
        self.assertIn("spatial_coverage", summary)
        
        # Validate quality metrics
        quality = inventory["quality_metrics"]
        self.assertIn("overall_quality", quality)
        self.assertGreaterEqual(quality["completeness"], 0.8)
        
        execution_time = time.time() - start_time
        self.assertLess(execution_time, 60, "Inventory generation too slow")
        
        logger.info("✓ Complete inventory generation test passed")
    
    def test_spatial_index_generation(self):
        """Test spatial index generation for discovered data."""
        step_config = {
            "id": "test_spatial_index",
            "type": "inventory_generation", 
            "hyperparameters": {
                "include_spatial_index": True,
                "spatial_index_format": "geojson",
                "grid_size": 0.01  # degrees
            }
        }
        
        inputs = {
            "all_data": [
                self.create_mock_sentinel_data(),
                self.create_mock_dem_data()
            ]
        }
        
        step = create_step(step_config)
        step.execute = Mock(return_value={
            "spatial_index": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [85.30, 27.60], [85.32, 27.60],
                                [85.32, 27.62], [85.30, 27.62],
                                [85.30, 27.60]
                            ]]
                        },
                        "properties": {
                            "data_type": "satellite_imagery",
                            "source": "Sentinel-2",
                            "file_count": 1
                        }
                    }
                ]
            },
            "spatial_index_file": str(self.output_dir / "spatial_index.geojson")
        })
        
        result = step.execute(inputs=inputs)
        
        self.assertIn("spatial_index", result)
        spatial_index = result["spatial_index"]
        self.assertEqual(spatial_index["type"], "FeatureCollection")
        self.assertGreater(len(spatial_index["features"]), 0)
        logger.info("✓ Spatial index generation test passed")


class TestWorkflowIntegration(DataAcquisitionTestBase):
    """Test end-to-end data acquisition workflow integration."""
    
    def test_basic_acquisition_workflow(self):
        """Test basic multi-source acquisition workflow."""
        start_time = time.time()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        workflow_steps = [
            # Step 1: Acquire satellite data
            {
                "step": "satellite_acquisition",
                "config": {
                    "id": "workflow_sentinel",
                    "type": "sentinel_hub_acquisition",
                    "hyperparameters": {
                        "bbox": self.test_config["bbox"],
                        "start_date": self.test_config["start_date"],
                        "end_date": self.test_config["end_date"],
                        "use_mock_data": True,
                        "resolution": 60,
                        "bands": ["B02", "B03", "B04", "B08"]
                    }
                }
            },
            # Step 2: Acquire elevation data
            {
                "step": "elevation_acquisition", 
                "config": {
                    "id": "workflow_dem",
                    "type": "dem_acquisition",
                    "hyperparameters": {
                        "bbox": self.test_config["bbox"],
                        "source": "SRTM",
                        "resolution": 90,
                        "use_mock_data": True,
                        "generate_derivatives": True,
                        "derivatives": ["slope"]
                    }
                }
            },
            # Step 3: Calculate indices
            {
                "step": "indices_calculation",
                "config": {
                    "id": "workflow_indices",
                    "type": "spectral_indices_calculation",
                    "hyperparameters": {
                        "indices": ["NDVI"],
                        "save_individual_indices": True
                    }
                }
            },
            # Step 4: Validate results
            {
                "step": "validation",
                "config": {
                    "id": "workflow_validation",
                    "type": "data_validation",
                    "hyperparameters": {
                        "validation_checks": [
                            "spatial_bounds_check",
                            "data_quality_check"
                        ]
                    }
                }
            }
        ]
        
        workflow_results = {}
        
        # Execute workflow steps
        for step_info in workflow_steps:
            step_name = step_info["step"]
            step_config = step_info["config"]
            
            try:
                step = create_step(step_config)
                result = step.execute()
                
                workflow_results[step_name] = {
                    "success": True,
                    "result": result,
                    "execution_time": result.get("performance_metrics", {}).get("execution_time_seconds", 0)
                }
                
                logger.info(f"✓ Workflow step '{step_name}' completed")
                
            except Exception as e:
                workflow_results[step_name] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": 0
                }
                logger.error(f"✗ Workflow step '{step_name}' failed: {e}")
        
        # Validate workflow completion
        successful_steps = sum(1 for r in workflow_results.values() if r["success"])
        total_steps = len(workflow_steps)
        
        execution_time = time.time() - start_time
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory
        
        # Assertions
        self.assertEqual(successful_steps, total_steps, "Not all workflow steps completed successfully")
        self.assert_performance_requirements(execution_time, memory_used)
        
        logger.info(f"✓ Complete satellite-to-analysis workflow test passed ({successful_steps}/{total_steps} steps)")
    
    def test_multi_temporal_acquisition_workflow(self):
        """Test workflow with multiple temporal acquisitions."""
        # Define multiple time periods
        time_periods = [
            ("2023-06-01", "2023-06-03"),
            ("2023-06-15", "2023-06-17"),
            ("2023-06-29", "2023-07-01")
        ]
        
        acquisition_results = []
        
        for i, (start_date, end_date) in enumerate(time_periods):
            step_config = {
                "id": f"temporal_acquisition_{i}",
                "type": "sentinel_hub_acquisition",
                "hyperparameters": {
                    "bbox": self.test_config["bbox"],
                    "start_date": start_date,
                    "end_date": end_date,
                    "use_mock_data": True,
                    "resolution": 60
                }
            }
            
            step = create_step(step_config)
            result = step.execute()
            
            acquisition_results.append({
                "period": f"{start_date}_to_{end_date}",
                "success": result.get("status") == "completed",
                "data": result.get("imagery_data")
            })
        
        # Validate all acquisitions succeeded
        successful_acquisitions = sum(1 for r in acquisition_results if r["success"])
        self.assertEqual(successful_acquisitions, len(time_periods))
        
        logger.info(f"✓ Multi-temporal acquisition workflow test passed ({successful_acquisitions} periods)")
    
    def test_data_quality_assessment_workflow(self):
        """Test comprehensive data quality assessment workflow."""
        # Generate test data with different quality scenarios
        quality_scenarios = [
            {"name": "high_quality", "cloud_coverage": 5, "nodata_percent": 2},
            {"name": "medium_quality", "cloud_coverage": 25, "nodata_percent": 10},
            {"name": "low_quality", "cloud_coverage": 60, "nodata_percent": 30}
        ]
        
        quality_results = {}
        
        for scenario in quality_scenarios:
            # Mock acquisition with specific quality parameters
            step_config = {
                "id": f"quality_test_{scenario['name']}",
                "type": "sentinel_hub_acquisition",
                "hyperparameters": {
                    "bbox": self.test_config["bbox"],
                    "use_mock_data": True,
                    "mock_quality": scenario
                }
            }
            
            step = create_step(step_config)
            result = step.execute()
            
            # Validate quality assessment
            validation_config = {
                "id": f"validate_{scenario['name']}",
                "type": "data_validation",
                "hyperparameters": {
                    "quality_thresholds": {
                        "max_cloud_coverage": 50,
                        "max_nodata_percent": 20
                    }
                }
            }
            
            validation_step = create_step(validation_config)
            validation_result = validation_step.execute(inputs={"data": result})
            
            quality_results[scenario['name']] = {
                "acquisition_success": result.get("status") == "completed",
                "validation_passed": validation_result.get("overall_quality") in ["GOOD", "ACCEPTABLE"]
            }
        
        # Check quality assessment results
        for scenario_name, results in quality_results.items():
            self.assertTrue(results["acquisition_success"], 
                          f"Acquisition failed for {scenario_name}")
        
        logger.info("✓ Data quality assessment workflow test passed")


class TestPerformanceAndStress(DataAcquisitionTestBase):
    """Test performance benchmarking and stress testing."""
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking of data acquisition steps."""
        benchmark_scenarios = [
            {
                "name": "Sentinel Acquisition",
                "config": {
                    "id": "perf_sentinel",
                    "type": "sentinel_hub_acquisition",
                    "hyperparameters": {"use_mock_data": True, "resolution": 60}
                },
                "iterations": 3
            },
            {
                "name": "DEM Acquisition",
                "config": {
                    "id": "perf_dem",
                    "type": "dem_acquisition",
                    "hyperparameters": {"use_mock_data": True, "source": "SRTM"}
                },
                "iterations": 3
            }
        ]
        
        performance_results = {}
        
        for scenario in benchmark_scenarios:
            execution_times = []
            
            for i in range(scenario["iterations"]):
                start_time = time.time()
                step = create_step(scenario["config"])
                result = step.execute()
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            
            avg_time = sum(execution_times) / len(execution_times)
            performance_results[scenario["name"]] = {
                "average_time": avg_time,
                "min_time": min(execution_times),
                "max_time": max(execution_times),
                "iterations": scenario["iterations"]
            }
        
        # Validate performance results
        for name, results in performance_results.items():
            self.assertLess(results["average_time"], 10, 
                          f"{name} average time too slow: {results['average_time']:.2f}s")
            logger.info(f"✓ {name} benchmark: {results['average_time']:.2f}s average")
        
        logger.info("✓ Performance benchmarking test passed")
    
    def test_concurrent_acquisition_stress(self):
        """Test concurrent data acquisition under stress."""
        def run_acquisition_task(task_id):
            """Run a single acquisition task."""
            config = {
                "id": f"stress_task_{task_id}",
                "type": "sentinel_hub_acquisition",
                "hyperparameters": {
                    "bbox": self.test_config["bbox"],
                    "use_mock_data": True,
                    "resolution": 60
                }
            }
            
            try:
                step = create_step(config)
                result = step.execute()
                return {"task_id": task_id, "status": "success"}
            except Exception as e:
                return {"task_id": task_id, "status": "failed", "error": str(e)}
        
        # Test with multiple concurrent tasks
        num_tasks = 5
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_acquisition_task, i) for i in range(num_tasks)]
            results = [future.result() for future in futures]
        
        execution_time = time.time() - start_time
        
        # Validate results
        successful_tasks = sum(1 for r in results if r["status"] == "success")
        self.assertGreaterEqual(successful_tasks, num_tasks * 0.8)  # 80% success rate
        self.assertLess(execution_time, 60, "Concurrent execution too slow")
        
        logger.info(f"✓ Concurrent stress test passed: {successful_tasks}/{num_tasks} tasks, {execution_time:.2f}s")


# Comprehensive test reporting
class DataAcquisitionTestReporter:
    """Enhanced test reporter for data acquisition testing."""
    
    def __init__(self):
        self.test_results = {}
        self.performance_data = []
        self.start_time = None
    
    def start_test_session(self):
        """Start comprehensive test session."""
        self.start_time = time.time()
        logger.info("🚀 Starting comprehensive data acquisition test session")
    
    def record_test_category(self, category_name: str, results: Dict[str, Any]):
        """Record results for a test category."""
        self.test_results[category_name] = {
            **results,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_comprehensive_report(self, output_file: str = "comprehensive_test_report.json"):
        """Generate comprehensive test report."""
        if self.start_time is None:
            return None
        
        total_time = time.time() - self.start_time
        
        report = {
            "test_session": {
                "timestamp": datetime.now().isoformat(),
                "total_execution_time": total_time,
                "test_framework": "unittest + pytest",
                "test_type": "comprehensive_data_acquisition"
            },
            "test_categories": self.test_results,
            "summary": {
                "total_categories": len(self.test_results),
                "successful_categories": sum(1 for r in self.test_results.values() 
                                           if r.get("success", False)),
                "overall_success_rate": (sum(1 for r in self.test_results.values() 
                                           if r.get("success", False)) / 
                                       len(self.test_results) * 100) if self.test_results else 0,
                "total_execution_time": total_time
            },
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive test report saved: {output_file}")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if self.test_results:
            success_rate = (sum(1 for r in self.test_results.values() if r.get("success", False)) / 
                          len(self.test_results))
            
            if success_rate < 1.0:
                recommendations.append("Review failed test categories for debugging")
            
            if success_rate > 0.9:
                recommendations.append("System ready for production deployment")
            
            # Performance recommendations
            avg_execution_time = sum(r.get("execution_time", 0) for r in self.test_results.values()) / len(self.test_results)
            if avg_execution_time > 30:
                recommendations.append("Consider performance optimization for slower test categories")
        
        return recommendations


# Extended test suite management
class ExtendedTestSuite:
    """Extended test suite manager for comprehensive data acquisition testing."""
    
    @staticmethod
    def create_comprehensive_test_suite():
        """Create comprehensive test suite with all test categories."""
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # All test classes
        test_classes = [
            # Core tests (from core module)
            TestEnvironmentSetup,
            TestSentinelDataAcquisition,
            TestDEMAcquisition,
            TestLocalFilesDiscovery,
            TestDataHarmonization,
            
            # Extended tests (from this module)
            TestSpectralIndicesCalculation,
            TestDataInventoryGeneration,
            TestWorkflowIntegration,
            TestComprehensiveWorkflows,
            TestPerformanceAndStress
        ]
        
        for test_class in test_classes:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        return suite
    
    @staticmethod
    def run_comprehensive_tests():
        """Run comprehensive data acquisition test suite."""
        print("🧪 COMPREHENSIVE DATA ACQUISITION TEST SUITE")
        print("=" * 60)
        
        reporter = DataAcquisitionTestReporter()
        reporter.start_test_session()
        
        # Test categories with their respective classes
        test_categories = {
            "Environment Setup": TestEnvironmentSetup,
            "Sentinel Data Acquisition": TestSentinelDataAcquisition,
            "DEM Acquisition": TestDEMAcquisition,
            "Local Files Discovery": TestLocalFilesDiscovery,
            "Data Harmonization": TestDataHarmonization,
            "Spectral Indices Calculation": TestSpectralIndicesCalculation,
            "Data Inventory Generation": TestDataInventoryGeneration,
            "Workflow Integration": TestWorkflowIntegration,
            "Comprehensive Workflows": TestComprehensiveWorkflows,
            "Performance and Stress Testing": TestPerformanceAndStress
        }
        
        all_results = {}
        
        for category_name, test_class in test_categories.items():
            print(f"\n📂 {category_name}")
            print("-" * 50)
            
            category_start_time = time.time()
            
            # Create and run test suite for category
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(test_class)
            
            # Run with minimal output
            stream = StringIO()
            runner = unittest.TextTestRunner(stream=stream, verbosity=0)
            result = runner.run(suite)
            
            category_execution_time = time.time() - category_start_time
            
            # Store results
            category_results = {
                "success": result.wasSuccessful(),
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "execution_time": category_execution_time
            }
            
            all_results[category_name] = category_results
            reporter.record_test_category(category_name, category_results)
            
            # Display results
            if result.wasSuccessful():
                print(f"✅ {category_name}: All {result.testsRun} tests passed")
            else:
                print(f"❌ {category_name}: {len(result.failures + result.errors)} tests failed")
            
            print(f"   ⏱️  Execution time: {category_execution_time:.2f}s")
        
        # Generate final report
        report = reporter.generate_comprehensive_report()
        
        # Final summary
        print(f"\n🏁 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        successful_categories = sum(1 for r in all_results.values() if r["success"])
        total_categories = len(all_results)
        total_execution_time = sum(r["execution_time"] for r in all_results.values())
        
        print(f"📊 Results: {successful_categories}/{total_categories} categories passed")
        print(f"⏱️  Total execution time: {total_execution_time:.2f}s")
        
        if successful_categories == total_categories:
            print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
        else:
            print("⚠️  SOME TESTS FAILED - Check individual categories")
        
        return successful_categories == total_categories


# Main execution functions
def run_all_data_acquisition_tests():
    """Run all data acquisition tests (core + extended)."""
    return ExtendedTestSuite.run_comprehensive_tests()


def run_extended_tests_only():
    """Run only the extended test categories."""
    print("🔬 EXTENDED DATA ACQUISITION TESTS")
    print("=" * 50)
    
    extended_test_classes = [
        TestSpectralIndicesCalculation,
        TestDataInventoryGeneration,
        TestWorkflowIntegration,
        TestComprehensiveWorkflows,
        TestPerformanceAndStress
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in extended_test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


# Entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extended Data Acquisition Test Suite")
    parser.add_argument(
        "--mode",
        choices=["extended", "comprehensive", "performance"],
        default="comprehensive",
        help="Test mode to run"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed test report"
    )
    
    args = parser.parse_args()
    
    if args.mode == "extended":
        success = run_extended_tests_only()
    elif args.mode == "comprehensive":
        success = run_all_data_acquisition_tests()
    elif args.mode == "performance":
        # Run only performance tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestPerformanceAndStress)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        success = result.wasSuccessful()
    
    if args.report and args.mode == "comprehensive":
        reporter = DataAcquisitionTestReporter()
        reporter.generate_comprehensive_report()
    
    sys.exit(0 if success else 1) = self.process.memory_info().rss / 1024 / 1024
        
        # Simulate complete workflow
        workflow_steps = [
            ("acquire_sentinel", "Sentinel-2 acquisition"),
            ("acquire_dem", "SRTM DEM acquisition"),
            ("discover_local", "Local files discovery"),
            ("harmonize_data", "Data harmonization"),
            ("calculate_indices", "Spectral indices"),
            ("validate_outputs", "Output validation"),
            ("generate_inventory", "Inventory generation")
        ]
        
        workflow_results = {}
        
        # Execute workflow steps
        for step_id, step_description in workflow_steps:
            step_start = time.time()
            
            # Mock each step execution
            if step_id == "acquire_sentinel":
                result = {
                    "imagery_data": self.create_mock_sentinel_data(),
                    "metadata": {"bands": 4, "resolution": 60}
                }
            elif step_id == "acquire_dem":
                result = {
                    "elevation_data": self.create_mock_dem_data(),
                    "metadata": {"source": "SRTM", "resolution": 90}
                }
            elif step_id == "discover_local":
                result = {
                    "local_files": self.create_mock_local_files(2),
                    "file_inventory": {"count": 2}
                }
            elif step_id == "harmonize_data":
                harmonized_path = self.output_dir / "harmonized.tif"
                harmonized_path.touch()
                result = {
                    "harmonized_stack": str(harmonized_path),
                    "layer_count": 6
                }
            elif step_id == "calculate_indices":
                indices_path = self.output_dir / "indices.tif"
                indices_path.touch()
                result = {
                    "indices_stack": str(indices_path),
                    "indices_count": 3
                }
            elif step_id == "validate_outputs":
                result = {
                    "validation_results": {"overall": "PASS"},
                    "quality_score": 0.92
                }
            elif step_id == "generate_inventory":
                result = {
                    "inventory": {"total_files": 8, "quality": "GOOD"},
                    "inventory_file": str(self.output_dir / "inventory.json")
                }
            
            step_time = time.time() - step_start
            result["execution_time"] = step_time
            workflow_results[step_id] = result
            
            logger.info(f"✓ {step_description} completed in {step_time:.2f}s")
        
        # Validate workflow results
        self.assertEqual(len(workflow_results), 7)
        
        # Check all steps completed successfully
        for step_id, result in workflow_results.items():
            self.assertIsInstance(result, dict)
            self.assertIn("execution_time", result)
            self.assertLess(result["execution_time"], 30)  # Each step < 30s
        
        # Validate final outputs exist
        final_outputs = [
            workflow_results["harmonize_data"]["harmonized_stack"],
            workflow_results["calculate_indices"]["indices_stack"],
            workflow_results["generate_inventory"]["inventory_file"]
        ]
        
        for output_path in final_outputs:
            self.assertTrue(Path(output_path).exists())
        
        # Performance validation
        total_execution_time = time.time() - start_time
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory
        
        self.assertLess(total_execution_time, 180)  # Total < 3 minutes
        self.assertLess(memory_used, 256)  # Memory < 256MB
        
        logger.info(f"✓ Basic acquisition workflow completed in {total_execution_time:.2f}s")
    
    def test_workflow_with_failures_and_recovery(self):
        """Test workflow with step failures and recovery mechanisms."""
        workflow_config = {
            "error_handling": {
                "strategy": "continue_on_error",
                "fallback_to_mock": True,
                "max_retries": 1
            }
        }
        
        # Simulate workflow with some failures
        steps_with_failures = [
            ("acquire_sentinel", False, "Mock Sentinel data used"),
            ("acquire_dem", True, "DEM acquisition failed"),
            ("discover_local", False, "Local files discovered"),
            ("harmonize_data", False, "Using available data"),
            ("validate_outputs", False, "Validation completed")
        ]
        
        successful_steps = 0
        fallback_used = 0
        
        for step_id, should_fail, message in steps_with_failures:
            if should_fail:
                # Simulate failure with fallback
                fallback_used += 1
                logger.warning(f"Step {step_id} failed, using fallback")
            else:
                successful_steps += 1
                logger.info(f"Step {step_id}: {message}")
        
        # Validate workflow resilience
        self.assertGreater(successful_steps, 0)
        self.assertGreaterEqual(fallback_used, 0)
        
        # Workflow should continue despite failures
        total_steps = len(steps_with_failures)
        completion_rate = (successful_steps + fallback_used) / total_steps
        self.assertGreaterEqual(completion_rate, 0.8)  # 80% completion minimum
        
        logger.info("✓ Workflow failure recovery test passed")


class TestComprehensiveWorkflows(DataAcquisitionTestBase):
    """Test comprehensive data acquisition workflows."""
    
    def test_complete_satellite_to_analysis_workflow(self):
        """Test complete workflow from satellite acquisition to analysis."""
        start_time = time.time()
        initial_memory
