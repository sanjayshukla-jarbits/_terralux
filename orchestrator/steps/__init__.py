"""
Orchestrator Steps Module - Fail Fast Plan
==========================================

This module contains all step implementations for the modular orchestrator
system, designed for rapid development, testing, and validation of geospatial
data workflows.

Key Features:
- Modular step-based workflow execution
- Fail-fast execution with performance monitoring
- Comprehensive mock data support for testing
- Integration with existing landslide_pipeline structure
- Extensive error handling and recovery mechanisms
- Performance optimization for rapid development

Step Categories:
- data_acquisition: Satellite imagery, DEM, and local files acquisition
- data_processing: Harmonization, transformation, and analysis
- data_validation: Quality assessment and validation
- utilities: Mock data generation and helper functions

Author: Orchestrator Development Team
Version: 1.0.0-failfast
License: MIT
"""

import sys
import logging
import warnings
from typing import Dict, Any, List, Optional, Union, Type
from abc import ABC, abstractmethod

# Module metadata
__version__ = "1.0.0-failfast"
__author__ = "Orchestrator Development Team"
__license__ = "MIT"
__description__ = "Modular workflow steps for geospatial data processing - Fail Fast Plan"

# Configure module-level logging
logger = logging.getLogger(__name__)

# Step execution configuration
STEP_CONFIG = {
    "fail_fast_mode": True,
    "use_mock_data": True,
    "performance_monitoring": True,
    "default_timeout": 300,
    "max_retry_attempts": 2,
    "enable_caching": True,
    "auto_fallback": True
}

# Base Step Class
class BaseStep(ABC):
    """
    Abstract base class for all orchestrator steps.
    
    Provides common functionality for step execution, error handling,
    performance monitoring, and mock data support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize step with configuration.
        
        Args:
            config: Step configuration dictionary
        """
        self.config = config
        self.step_id = config.get("id", "unknown_step")
        self.step_type = config.get("type", "unknown_type")
        self.step_name = config.get("name", self.step_id)
        self.description = config.get("description", "")
        self.hyperparameters = config.get("hyperparameters", {})
        self.timeout = config.get("timeout", STEP_CONFIG["default_timeout"])
        self.retry_attempts = config.get("retry_attempts", STEP_CONFIG["max_retry_attempts"])
        
        # Performance tracking
        self.execution_start_time = None
        self.execution_end_time = None
        self.execution_metrics = {}
        
        # Mock data configuration
        self.use_mock_data = self.hyperparameters.get(
            "use_mock_data", 
            STEP_CONFIG["use_mock_data"]
        )
        
        logger.debug(f"Initialized step: {self.step_id} ({self.step_type})")
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any] = None, context: Any = None) -> Dict[str, Any]:
        """
        Execute the step with given inputs and context.
        
        Args:
            inputs: Input data from previous steps
            context: Execution context
            
        Returns:
            Dictionary containing step outputs
        """
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any] = None) -> bool:
        """
        Validate input data before execution.
        
        Args:
            inputs: Input data to validate
            
        Returns:
            True if inputs are valid
        """
        # Default implementation - can be overridden by specific steps
        required_inputs = self.config.get("inputs", {})
        
        if not required_inputs:
            return True
        
        if not inputs:
            inputs = {}
        
        for input_key, input_config in required_inputs.items():
            if input_config.get("required", False) and input_key not in inputs:
                logger.error(f"Required input '{input_key}' not provided for step {self.step_id}")
                return False
        
        return True
    
    def start_performance_monitoring(self):
        """Start performance monitoring for the step."""
        import time
        import psutil
        
        self.execution_start_time = time.time()
        
        try:
            process = psutil.Process()
            self.execution_metrics["start_memory_mb"] = process.memory_info().rss / 1024 / 1024
            self.execution_metrics["start_cpu_percent"] = process.cpu_percent()
        except Exception as e:
            logger.warning(f"Could not start performance monitoring: {e}")
    
    def stop_performance_monitoring(self):
        """Stop performance monitoring and calculate metrics."""
        import time
        import psutil
        
        if self.execution_start_time is None:
            return
        
        self.execution_end_time = time.time()
        execution_time = self.execution_end_time - self.execution_start_time
        
        self.execution_metrics["execution_time_seconds"] = execution_time
        
        try:
            process = psutil.Process()
            end_memory = process.memory_info().rss / 1024 / 1024
            start_memory = self.execution_metrics.get("start_memory_mb", 0)
            
            self.execution_metrics["end_memory_mb"] = end_memory
            self.execution_metrics["memory_used_mb"] = end_memory - start_memory
            self.execution_metrics["end_cpu_percent"] = process.cpu_percent()
        except Exception as e:
            logger.warning(f"Could not complete performance monitoring: {e}")
        
        logger.info(f"Step {self.step_id} completed in {execution_time:.2f}s")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the step."""
        return self.execution_metrics.copy()


# Import data acquisition steps with error handling
try:
    from .data_acquisition import (
        SentinelHubAcquisitionStep,
        DEMAcquisitionStep,
        LocalFilesDiscoveryStep,
        ASTERDEMAcquisitionStep,
        LandsatAcquisitionStep,
        MODISAcquisitionStep
    )
    
    DATA_ACQUISITION_AVAILABLE = True
    logger.debug("Data acquisition steps loaded successfully")
    
    # Register step types
    DATA_ACQUISITION_STEPS = {
        "sentinel_hub_acquisition": SentinelHubAcquisitionStep,
        "dem_acquisition": DEMAcquisitionStep,
        "local_files_discovery": LocalFilesDiscoveryStep,
        "aster_dem_acquisition": ASTERDEMAcquisitionStep,
        "landsat_acquisition": LandsatAcquisitionStep,
        "modis_acquisition": MODISAcquisitionStep
    }
    
except ImportError as e:
    logger.warning(f"Data acquisition steps not available: {e}")
    DATA_ACQUISITION_AVAILABLE = False
    
    # Mock implementations for development
    class MockAcquisitionStep(BaseStep):
        """Mock acquisition step for development."""
        def execute(self, inputs=None, context=None):
            self.start_performance_monitoring()
            
            # Simulate processing time
            import time
            time.sleep(0.1)
            
            result = {
                "status": "completed",
                "mock": True,
                "step_type": self.step_type,
                "output_data": f"/mock/path/{self.step_id}_output.tif",
                "metadata": {
                    "processing_time": 0.1,
                    "mock_data_used": True
                }
            }
            
            self.stop_performance_monitoring()
            result["performance_metrics"] = self.get_performance_metrics()
            
            return result
    
    # Create mock step classes
    SentinelHubAcquisitionStep = type("SentinelHubAcquisitionStep", (MockAcquisitionStep,), {})
    DEMAcquisitionStep = type("DEMAcquisitionStep", (MockAcquisitionStep,), {})
    LocalFilesDiscoveryStep = type("LocalFilesDiscoveryStep", (MockAcquisitionStep,), {})
    ASTERDEMAcquisitionStep = type("ASTERDEMAcquisitionStep", (MockAcquisitionStep,), {})
    LandsatAcquisitionStep = type("LandsatAcquisitionStep", (MockAcquisitionStep,), {})
    MODISAcquisitionStep = type("MODISAcquisitionStep", (MockAcquisitionStep,), {})
    
    DATA_ACQUISITION_STEPS = {
        "sentinel_hub_acquisition": SentinelHubAcquisitionStep,
        "dem_acquisition": DEMAcquisitionStep,
        "local_files_discovery": LocalFilesDiscoveryStep,
        "aster_dem_acquisition": ASTERDEMAcquisitionStep,
        "landsat_acquisition": LandsatAcquisitionStep,
        "modis_acquisition": MODISAcquisitionStep
    }

# Import data processing steps with error handling
try:
    from .data_processing import (
        DataHarmonizationStep,
        SpectralIndicesCalculationStep,
        TemporalAggregationStep,
        SpatialFilteringStep,
        DataTransformationStep,
        FeatureExtractionStep
    )
    
    DATA_PROCESSING_AVAILABLE = True
    logger.debug("Data processing steps loaded successfully")
    
    # Register step types
    DATA_PROCESSING_STEPS = {
        "data_harmonization": DataHarmonizationStep,
        "spectral_indices_calculation": SpectralIndicesCalculationStep,
        "temporal_aggregation": TemporalAggregationStep,
        "spatial_filtering": SpatialFilteringStep,
        "data_transformation": DataTransformationStep,
        "feature_extraction": FeatureExtractionStep
    }
    
except ImportError as e:
    logger.warning(f"Data processing steps not available: {e}")
    DATA_PROCESSING_AVAILABLE = False
    
    # Mock implementations
    class MockProcessingStep(BaseStep):
        """Mock processing step for development."""
        def execute(self, inputs=None, context=None):
            self.start_performance_monitoring()
            
            import time
            time.sleep(0.05)
            
            result = {
                "status": "completed",
                "mock": True,
                "step_type": self.step_type,
                "processed_data": f"/mock/path/{self.step_id}_processed.tif",
                "metadata": {
                    "processing_time": 0.05,
                    "mock_processing": True,
                    "input_count": len(inputs) if inputs else 0
                }
            }
            
            self.stop_performance_monitoring()
            result["performance_metrics"] = self.get_performance_metrics()
            
            return result
    
    # Create mock step classes
    DataHarmonizationStep = type("DataHarmonizationStep", (MockProcessingStep,), {})
    SpectralIndicesCalculationStep = type("SpectralIndicesCalculationStep", (MockProcessingStep,), {})
    TemporalAggregationStep = type("TemporalAggregationStep", (MockProcessingStep,), {})
    SpatialFilteringStep = type("SpatialFilteringStep", (MockProcessingStep,), {})
    DataTransformationStep = type("DataTransformationStep", (MockProcessingStep,), {})
    FeatureExtractionStep = type("FeatureExtractionStep", (MockProcessingStep,), {})
    
    DATA_PROCESSING_STEPS = {
        "data_harmonization": DataHarmonizationStep,
        "spectral_indices_calculation": SpectralIndicesCalculationStep,
        "temporal_aggregation": TemporalAggregationStep,
        "spatial_filtering": SpatialFilteringStep,
        "data_transformation": DataTransformationStep,
        "feature_extraction": FeatureExtractionStep
    }

# Import validation steps with error handling
try:
    from .data_validation import (
        DataValidationStep,
        QualityAssessmentStep,
        SpatialValidationStep,
        TemporalValidationStep,
        MetadataValidationStep,
        InventoryGenerationStep
    )
    
    DATA_VALIDATION_AVAILABLE = True
    logger.debug("Data validation steps loaded successfully")
    
    # Register step types
    DATA_VALIDATION_STEPS = {
        "data_validation": DataValidationStep,
        "quality_assessment": QualityAssessmentStep,
        "spatial_validation": SpatialValidationStep,
        "temporal_validation": TemporalValidationStep,
        "metadata_validation": MetadataValidationStep,
        "inventory_generation": InventoryGenerationStep
    }
    
except ImportError as e:
    logger.warning(f"Data validation steps not available: {e}")
    DATA_VALIDATION_AVAILABLE = False
    
    # Mock implementations
    class MockValidationStep(BaseStep):
        """Mock validation step for development."""
        def execute(self, inputs=None, context=None):
            self.start_performance_monitoring()
            
            import time
            time.sleep(0.02)
            
            result = {
                "status": "completed",
                "mock": True,
                "step_type": self.step_type,
                "validation_results": {
                    "overall_quality": "GOOD",
                    "validation_checks": ["spatial_bounds", "data_integrity"],
                    "passed_checks": 2,
                    "failed_checks": 0,
                    "warnings": [],
                    "mock_validation": True
                },
                "validation_report": f"/mock/path/{self.step_id}_validation_report.json"
            }
            
            self.stop_performance_monitoring()
            result["performance_metrics"] = self.get_performance_metrics()
            
            return result
    
    # Create mock step classes
    DataValidationStep = type("DataValidationStep", (MockValidationStep,), {})
    QualityAssessmentStep = type("QualityAssessmentStep", (MockValidationStep,), {})
    SpatialValidationStep = type("SpatialValidationStep", (MockValidationStep,), {})
    TemporalValidationStep = type("TemporalValidationStep", (MockValidationStep,), {})
    MetadataValidationStep = type("MetadataValidationStep", (MockValidationStep,), {})
    InventoryGenerationStep = type("InventoryGenerationStep", (MockValidationStep,), {})
    
    DATA_VALIDATION_STEPS = {
        "data_validation": DataValidationStep,
        "quality_assessment": QualityAssessmentStep,
        "spatial_validation": SpatialValidationStep,
        "temporal_validation": TemporalValidationStep,
        "metadata_validation": MetadataValidationStep,
        "inventory_generation": InventoryGenerationStep
    }

# Import utility steps with error handling
try:
    from .utilities import (
        MockDataGenerationStep,
        FileOperationsStep,
        DataConversionStep,
        MetadataExtractionStep,
        PerformanceMonitoringStep
    )
    
    UTILITIES_AVAILABLE = True
    logger.debug("Utility steps loaded successfully")
    
    # Register step types
    UTILITY_STEPS = {
        "mock_data_generation": MockDataGenerationStep,
        "file_operations": FileOperationsStep,
        "data_conversion": DataConversionStep,
        "metadata_extraction": MetadataExtractionStep,
        "performance_monitoring": PerformanceMonitoringStep
    }
    
except ImportError as e:
    logger.warning(f"Utility steps not available: {e}")
    UTILITIES_AVAILABLE = False
    
    # Mock implementations
    class MockUtilityStep(BaseStep):
        """Mock utility step for development."""
        def execute(self, inputs=None, context=None):
            self.start_performance_monitoring()
            
            import time
            time.sleep(0.01)
            
            result = {
                "status": "completed",
                "mock": True,
                "step_type": self.step_type,
                "utility_output": f"/mock/path/{self.step_id}_output",
                "metadata": {
                    "processing_time": 0.01,
                    "mock_utility": True
                }
            }
            
            self.stop_performance_monitoring()
            result["performance_metrics"] = self.get_performance_metrics()
            
            return result
    
    # Create mock step classes
    MockDataGenerationStep = type("MockDataGenerationStep", (MockUtilityStep,), {})
    FileOperationsStep = type("FileOperationsStep", (MockUtilityStep,), {})
    DataConversionStep = type("DataConversionStep", (MockUtilityStep,), {})
    MetadataExtractionStep = type("MetadataExtractionStep", (MockUtilityStep,), {})
    PerformanceMonitoringStep = type("PerformanceMonitoringStep", (MockUtilityStep,), {})
    
    UTILITY_STEPS = {
        "mock_data_generation": MockDataGenerationStep,
        "file_operations": FileOperationsStep,
        "data_conversion": DataConversionStep,
        "metadata_extraction": MetadataExtractionStep,
        "performance_monitoring": PerformanceMonitoringStep
    }

# Complete step registry
ALL_STEPS = {
    **DATA_ACQUISITION_STEPS,
    **DATA_PROCESSING_STEPS,
    **DATA_VALIDATION_STEPS,
    **UTILITY_STEPS
}

# Module-level functions
def get_available_step_types() -> List[str]:
    """Get list of all available step types."""
    return list(ALL_STEPS.keys())

def get_step_class(step_type: str) -> Type[BaseStep]:
    """
    Get step class for a given step type.
    
    Args:
        step_type: Type of step to get
        
    Returns:
        Step class
        
    Raises:
        ValueError: If step type is not found
    """
    if step_type not in ALL_STEPS:
        available_types = list(ALL_STEPS.keys())
        raise ValueError(f"Unknown step type: {step_type}. Available: {available_types}")
    
    return ALL_STEPS[step_type]

def create_step(step_config: Dict[str, Any]) -> BaseStep:
    """
    Factory function to create a step instance from configuration.
    
    Args:
        step_config: Step configuration dictionary
        
    Returns:
        Initialized step instance
        
    Raises:
        ValueError: If step type is not recognized
    """
    step_type = step_config.get("type")
    if not step_type:
        raise ValueError("Step configuration must include 'type' field")
    
    step_class = get_step_class(step_type)
    return step_class(step_config)

def validate_step_config(step_config: Dict[str, Any]) -> List[str]:
    """
    Validate step configuration structure.
    
    Args:
        step_config: Step configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ["id", "type", "description"]
    for field in required_fields:
        if field not in step_config:
            errors.append(f"Missing required field: {field}")
    
    # Check step type exists
    step_type = step_config.get("type")
    if step_type and step_type not in ALL_STEPS:
        errors.append(f"Unknown step type: {step_type}")
    
    # Validate outputs structure
    outputs = step_config.get("outputs", {})
    if outputs:
        for output_key, output_config in outputs.items():
            if not isinstance(output_config, dict):
                errors.append(f"Output '{output_key}' must be a dictionary")
            else:
                required_output_fields = ["key", "type"]
                for field in required_output_fields:
                    if field not in output_config:
                        errors.append(f"Output '{output_key}' missing required field: {field}")
    
    return errors

def get_step_categories() -> Dict[str, List[str]]:
    """Get step types organized by category."""
    return {
        "data_acquisition": list(DATA_ACQUISITION_STEPS.keys()),
        "data_processing": list(DATA_PROCESSING_STEPS.keys()),
        "data_validation": list(DATA_VALIDATION_STEPS.keys()),
        "utilities": list(UTILITY_STEPS.keys())
    }

def get_step_info(step_type: str) -> Dict[str, Any]:
    """
    Get information about a specific step type.
    
    Args:
        step_type: Type of step to get info for
        
    Returns:
        Dictionary with step information
    """
    if step_type not in ALL_STEPS:
        raise ValueError(f"Unknown step type: {step_type}")
    
    step_class = ALL_STEPS[step_type]
    
    # Determine category
    category = "unknown"
    for cat, steps in get_step_categories().items():
        if step_type in steps:
            category = cat
            break
    
    return {
        "step_type": step_type,
        "step_class": step_class.__name__,
        "category": category,
        "description": getattr(step_class, "__doc__", "No description available"),
        "mock_implementation": not any([
            DATA_ACQUISITION_AVAILABLE and step_type in DATA_ACQUISITION_STEPS,
            DATA_PROCESSING_AVAILABLE and step_type in DATA_PROCESSING_STEPS,
            DATA_VALIDATION_AVAILABLE and step_type in DATA_VALIDATION_STEPS,
            UTILITIES_AVAILABLE and step_type in UTILITY_STEPS
        ])
    }

def get_module_status() -> Dict[str, Any]:
    """Get status of the steps module."""
    return {
        "version": __version__,
        "components": {
            "data_acquisition": DATA_ACQUISITION_AVAILABLE,
            "data_processing": DATA_PROCESSING_AVAILABLE,
            "data_validation": DATA_VALIDATION_AVAILABLE,
            "utilities": UTILITIES_AVAILABLE
        },
        "total_steps": len(ALL_STEPS),
        "step_categories": get_step_categories(),
        "config": STEP_CONFIG
    }

def configure_steps(**kwargs) -> None:
    """
    Configure module-wide step settings.
    
    Args:
        **kwargs: Configuration parameters to update
    """
    STEP_CONFIG.update(kwargs)
    logger.info(f"Steps configuration updated: {kwargs}")

def execute_step_pipeline(steps_configs: List[Dict[str, Any]], 
                         context: Any = None) -> List[Dict[str, Any]]:
    """
    Execute a pipeline of steps in sequence.
    
    Args:
        steps_configs: List of step configurations
        context: Execution context
        
    Returns:
        List of step execution results
    """
    results = []
    step_outputs = {}
    
    for step_config in steps_configs:
        try:
            # Create step instance
            step = create_step(step_config)
            
            # Prepare inputs from previous steps
            inputs = {}
            step_inputs = step_config.get("inputs", {})
            
            for input_key, input_config in step_inputs.items():
                source_step = input_config.get("source")
                source_key = input_config.get("key")
                
                if source_step and source_step in step_outputs:
                    if source_key in step_outputs[source_step]:
                        inputs[input_key] = step_outputs[source_step][source_key]
            
            # Execute step
            logger.info(f"Executing step: {step.step_id}")
            result = step.execute(inputs=inputs, context=context)
            
            # Store outputs for next steps
            step_outputs[step.step_id] = result
            results.append(result)
            
            logger.info(f"Step {step.step_id} completed successfully")
            
        except Exception as e:
            error_result = {
                "step_id": step_config.get("id", "unknown"),
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
            results.append(error_result)
            logger.error(f"Step {step_config.get('id')} failed: {e}")
            
            # Stop on error in fail-fast mode
            if STEP_CONFIG["fail_fast_mode"]:
                break
    
    return results

# Package exports
__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__license__",
    "__description__",
    
    # Base class
    "BaseStep",
    
    # Data acquisition steps
    "SentinelHubAcquisitionStep",
    "DEMAcquisitionStep",
    "LocalFilesDiscoveryStep",
    "ASTERDEMAcquisitionStep", 
    "LandsatAcquisitionStep",
    "MODISAcquisitionStep",
    
    # Data processing steps
    "DataHarmonizationStep",
    "SpectralIndicesCalculationStep",
    "TemporalAggregationStep",
    "SpatialFilteringStep",
    "DataTransformationStep",
    "FeatureExtractionStep",
    
    # Data validation steps
    "DataValidationStep",
    "QualityAssessmentStep",
    "SpatialValidationStep",
    "TemporalValidationStep",
    "MetadataValidationStep",
    "InventoryGenerationStep",
    
    # Utility steps
    "MockDataGenerationStep",
    "FileOperationsStep",
    "DataConversionStep",
    "MetadataExtractionStep",
    "PerformanceMonitoringStep",
    
    # Module functions
    "get_available_step_types",
    "get_step_class",
    "create_step",
    "validate_step_config",
    "get_step_categories",
    "get_step_info",
    "get_module_status",
    "configure_steps",
    "execute_step_pipeline",
    
    # Step registries
    "ALL_STEPS",
    "DATA_ACQUISITION_STEPS",
    "DATA_PROCESSING_STEPS",
    "DATA_VALIDATION_STEPS",
    "UTILITY_STEPS",
    
    # Configuration
    "STEP_CONFIG",
    
    # Availability flags
    "DATA_ACQUISITION_AVAILABLE",
    "DATA_PROCESSING_AVAILABLE",
    "DATA_VALIDATION_AVAILABLE",
    "UTILITIES_AVAILABLE"
]

# Module initialization logging
logger.info(f"Orchestrator Steps module v{__version__} initialized")
logger.info(f"Available components: Acquisition={DATA_ACQUISITION_AVAILABLE}, "
           f"Processing={DATA_PROCESSING_AVAILABLE}, Validation={DATA_VALIDATION_AVAILABLE}, "
           f"Utilities={UTILITIES_AVAILABLE}")
logger.info(f"Total registered steps: {len(ALL_STEPS)}")

if STEP_CONFIG["fail_fast_mode"]:
    logger.info("Fail-fast mode enabled for step execution")

# Environment check on import
if not any([DATA_ACQUISITION_AVAILABLE, DATA_PROCESSING_AVAILABLE, 
           DATA_VALIDATION_AVAILABLE, UTILITIES_AVAILABLE]):
    warnings.warn(
        "No step implementations are available. Using mock implementations only. "
        "Install step dependencies for full functionality.",
        RuntimeWarning
    )
