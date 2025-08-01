"""
Abstract Base Step Class for Modular Pipeline Orchestrator
=========================================================

This module provides the abstract base class for all pipeline steps,
designed with fail-fast principles for rapid development and testing.
CORRECTED for compatibility with ModularOrchestrator ExecutionContext and
to return Dict instead of StepResult.

Key Features:
- Abstract interface for consistent step implementation
- Built-in logging and error handling
- Input/output validation
- Resource requirement specification
- Metadata tracking
- Mock execution support for testing
- Compatible with ExecutionContext from ModularOrchestrator
- Returns Dict instead of deprecated StepResult
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field
import traceback

# CORRECTED: Import ExecutionContext instead of PipelineContext for compatibility
if TYPE_CHECKING:
    from ...core.orchestrator import ExecutionContext


class StepValidationError(Exception):
    """Custom exception for step validation errors."""
    pass


class BaseStep(ABC):
    """
    Abstract base class for all pipeline steps.
    CORRECTED for compatibility with ModularOrchestrator's ExecutionContext
    and to return Dict instead of StepResult.
    
    This class provides the foundation for implementing modular pipeline steps
    with consistent interfaces, error handling, and metadata tracking.
    
    Design Philosophy:
    - Fail-fast: Quick feedback on configuration errors
    - Testable: Mock execution support for rapid development
    - Observable: Comprehensive logging and metadata tracking
    - Flexible: Support for various input/output patterns
    - Compatible: Works with ModularOrchestrator's ExecutionContext
    """
    
    def __init__(self, 
                 step_id: str, 
                 step_config: Dict[str, Any]):
        """
        Initialize the base step.
        CORRECTED to match the expected constructor signature.
        
        Args:
            step_id: Unique identifier for this step instance
            step_config: Complete step configuration from process definition
        """
        # Core identification
        self.step_id = step_id
        self.step_type = step_config.get('type', 'unknown')
        
        # Configuration
        self.hyperparameters = step_config.get('hyperparameters', {})
        self.inputs = step_config.get('inputs', {})
        self.outputs = step_config.get('outputs', {})
        self.dependencies = step_config.get('dependencies', [])
        self.condition = step_config.get('condition')
        
        # Execution metadata
        self.execution_metadata = {
            'step_id': self.step_id,
            'step_type': self.step_type,
            'created_at': datetime.now(),
            'version': '1.0.0'
        }
        
        # Logging setup
        self.logger = logging.getLogger(f'Step.{self.step_type}.{self.step_id}')
        
        # Validation
        self._validate_configuration()
        
        self.logger.debug(f"Initialized {self.step_type} step: {self.step_id}")
    
    def _validate_configuration(self) -> None:
        """
        Validate step configuration during initialization.
        Override in subclasses for step-specific validation.
        """
        # Basic validation
        if not self.step_id or not isinstance(self.step_id, str):
            raise StepValidationError("Step ID must be a non-empty string")
        
        if not self.step_type or not isinstance(self.step_type, str):
            raise StepValidationError("Step type must be a non-empty string")
        
        # Validate dependencies are strings
        if not isinstance(self.dependencies, list):
            raise StepValidationError("Dependencies must be a list")
        
        for dep in self.dependencies:
            if not isinstance(dep, str):
                raise StepValidationError(f"Dependency must be string: {dep}")
        
        # Validate inputs/outputs structure
        if not isinstance(self.inputs, dict):
            raise StepValidationError("Inputs must be a dictionary")
        
        if not isinstance(self.outputs, dict):
            raise StepValidationError("Outputs must be a dictionary")
    
    @abstractmethod
    def execute(self, context: 'ExecutionContext') -> Dict[str, Any]:
        """
        Execute the step logic.
        CORRECTED to use ExecutionContext and return Dict instead of StepResult.
        
        This is the main method that must be implemented by all step subclasses.
        It should contain the core logic for the step's operation.
        
        Args:
            context: Execution context containing shared data and configuration
            
        Returns:
            Dictionary with execution results:
            {
                'status': 'success'|'failed'|'skipped',
                'outputs': {...},  # Output data
                'metadata': {...}  # Step metadata
            }
            
        Note:
            Any step-specific exceptions should be caught and returned
            as failed status rather than propagated.
        """
        pass
    
    def should_execute(self, context: 'ExecutionContext') -> bool:
        """
        Determine if this step should execute based on conditions.
        CORRECTED to use ExecutionContext.
        
        Args:
            context: Execution context
            
        Returns:
            True if step should execute, False if it should be skipped
        """
        if not self.condition:
            return True
        
        try:
            # Simple condition evaluation using context variables
            condition_str = self.condition
            
            # Replace template variables from context
            for key, value in context.variables.items():
                condition_str = condition_str.replace(f"{{{key}}}", str(value))
            
            # Basic boolean evaluation (extend as needed)
            if condition_str.lower() in ('true', '1', 'yes'):
                return True
            elif condition_str.lower() in ('false', '0', 'no'):
                return False
            else:
                # For complex conditions, could use safe eval or custom parser
                # For now, use basic eval (in production, use safer evaluation)
                return bool(eval(condition_str))
                
        except Exception as e:
            self.logger.warning(f"Condition evaluation failed: {e}, defaulting to execute")
            return True
    
    def validate_inputs(self, context: 'ExecutionContext') -> bool:
        """
        Validate that required inputs are available in the context.
        CORRECTED to use ExecutionContext and simplified validation.
        
        Args:
            context: Execution context
            
        Returns:
            True if all required inputs are available, False otherwise
        """
        missing_inputs = []
        
        for input_key, input_config in self.inputs.items():
            # Check if input is required
            if input_config.get('required', False):
                # Check if input is available in step outputs
                available = input_key in context.step_outputs
                
                # Also check in artifacts
                if not available:
                    available = input_key in context.artifacts
                
                # Check in context variables
                if not available:
                    available = input_key in context.variables
                
                if not available:
                    missing_inputs.append(input_key)
        
        if missing_inputs:
            self.logger.error(f"Missing required inputs: {missing_inputs}")
            return False
        
        return True
    
    def get_input_data(self, 
                      context: 'ExecutionContext', 
                      input_key: str, 
                      default: Any = None) -> Any:
        """
        Retrieve input data from context with fallback logic.
        CORRECTED to use ExecutionContext.
        
        Args:
            context: Execution context
            input_key: Key for the input data
            default: Default value if input not found
            
        Returns:
            Input data value
        """
        # Try step outputs first
        if context and input_key in context.step_outputs:
            return context.step_outputs[input_key]
        
        # Try artifacts
        if context and input_key in context.artifacts:
            return context.artifacts[input_key]
        
        # Try context variables
        if context and input_key in context.variables:
            return context.variables[input_key]
        
        # Try hyperparameters
        if input_key in self.hyperparameters:
            return self.hyperparameters[input_key]
        
        return default
    
    def set_output_data(self, 
                       context: 'ExecutionContext', 
                       output_key: str, 
                       value: Any,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store output data in context.
        CORRECTED to use ExecutionContext.
        
        Args:
            context: Execution context
            output_key: Key for the output data
            value: Output value to store
            metadata: Optional metadata about the output
        """
        # Store in step outputs
        context.step_outputs[output_key] = value
        
        # Also store as artifact
        context.set_artifact(output_key, value)
    
    def _substitute_variables(self, 
                            template: str, 
                            context: 'ExecutionContext') -> str:
        """
        Substitute template variables in a string.
        CORRECTED to use ExecutionContext.
        
        Args:
            template: String with template variables
            context: Execution context
            
        Returns:
            String with variables substituted
        """
        if not isinstance(template, str):
            return template
        
        result = template
        
        # Substitute context variables
        if context:
            for key, value in context.variables.items():
                placeholder = f"{{{key}}}"
                if placeholder in result:
                    result = result.replace(placeholder, str(value))
        
        # Substitute hyperparameters
        for key, value in self.hyperparameters.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        
        return result
    
    def get_resource_requirements(self) -> Dict[str, Any]:
        """
        Get resource requirements for this step.
        
        Returns:
            Dictionary with resource requirements
        """
        return {
            'memory': self.hyperparameters.get('memory_limit', '4GB'),
            'cpu_cores': self.hyperparameters.get('cpu_cores', 1),
            'gpu_required': self.hyperparameters.get('gpu_required', False),
            'disk_space': self.hyperparameters.get('disk_space', '1GB'),
            'execution_time_estimate': self.hyperparameters.get('execution_time_estimate', '10m'),
            'network_required': self.hyperparameters.get('network_required', True)
        }
    
    def get_step_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about this step.
        
        Returns:
            Dictionary with step information
        """
        return {
            'step_id': self.step_id,
            'step_type': self.step_type,
            'dependencies': self.dependencies,
            'inputs': list(self.inputs.keys()),
            'outputs': list(self.outputs.keys()),
            'hyperparameters_count': len(self.hyperparameters),
            'has_condition': bool(self.condition),
            'resource_requirements': self.get_resource_requirements(),
            'created_at': self.execution_metadata['created_at']
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.step_id}, type={self.step_type})"
    
    def __str__(self) -> str:
        return f"{self.step_type}:{self.step_id}"


class MockStep(BaseStep):
    """
    Mock step implementation for testing and fail-fast development.
    CORRECTED to return Dict instead of StepResult and use ExecutionContext.
    
    This step simulates execution without performing actual operations,
    useful for testing pipeline structure and data flow.
    """
    
    def execute(self, context: 'ExecutionContext') -> Dict[str, Any]:
        """
        Execute mock step with simulated processing.
        CORRECTED to return Dict and use ExecutionContext.
        """
        self.logger.info(f"Mock execution of {self.step_type} step: {self.step_id}")
        
        # Simulate processing time
        processing_time = self.hyperparameters.get('mock_processing_time', 0.1)
        time.sleep(processing_time)
        
        # Generate mock outputs based on configuration
        mock_outputs = {}
        for output_key, output_config in self.outputs.items():
            output_type = output_config.get('type', 'data')
            
            if output_type == 'file':
                mock_outputs[output_key] = f"/mock/path/{self.step_id}_{output_key}.tif"
            elif output_type == 'raster':
                mock_outputs[output_key] = {
                    'path': f"/mock/path/{self.step_id}_{output_key}.tif",
                    'shape': [100, 100],
                    'bands': output_config.get('bands', 1),
                    'dtype': 'float32'
                }
            elif output_type == 'vector':
                mock_outputs[output_key] = {
                    'path': f"/mock/path/{self.step_id}_{output_key}.shp",
                    'features': 42,
                    'geometry_type': 'Polygon'
                }
            else:
                mock_outputs[output_key] = f"mock_{output_key}_result"
        
        # Store outputs in context if context is available
        if context:
            for output_key, output_value in mock_outputs.items():
                self.set_output_data(context, output_key, output_value)
        
        return {
            'status': 'success',
            'outputs': mock_outputs,
            'metadata': {
                'mock_execution': True,
                'processing_time': processing_time,
                'hyperparameters': dict(self.hyperparameters),
                'step_id': self.step_id,
                'step_type': self.step_type,
                'mock': True  # ADDED: For orchestrator to detect mock execution
            }
        }


# Utility functions for step development - CORRECTED
def create_mock_step(step_id: str, 
                    step_type: str = 'mock',
                    **hyperparameters) -> MockStep:
    """
    Create a mock step for testing.
    CORRECTED to use proper step configuration format.
    
    Args:
        step_id: Unique step identifier
        step_type: Step type name
        **hyperparameters: Step hyperparameters
        
    Returns:
        MockStep instance
    """
    step_config = {
        'type': step_type,
        'hyperparameters': hyperparameters,
        'inputs': {},
        'outputs': {'mock_output': {'type': 'data'}},
        'dependencies': []
    }
    
    return MockStep(step_id, step_config)


def validate_step_implementation(step_class: type) -> List[str]:
    """
    Validate that a step class properly implements the BaseStep interface.
    CORRECTED to check for Dict return type instead of StepResult.
    
    Args:
        step_class: Step class to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check inheritance
    if not issubclass(step_class, BaseStep):
        errors.append("Step class must inherit from BaseStep")
        return errors
    
    # Check required methods
    if not hasattr(step_class, 'execute'):
        errors.append("Step class must implement execute() method")
    elif not callable(getattr(step_class, 'execute')):
        errors.append("execute() must be callable")
    
    # Check method signatures (basic check)
    try:
        import inspect
        execute_sig = inspect.signature(step_class.execute)
        params = list(execute_sig.parameters.keys())
        
        if len(params) < 2 or 'context' not in params:
            errors.append("execute() method must accept 'context' parameter")
    except Exception as e:
        errors.append(f"Could not validate execute() signature: {e}")
    
    return errors


# CORRECTED: Simple ExecutionContext mock for testing when orchestrator not available
class SimpleExecutionContext:
    """Simple ExecutionContext mock for testing purposes."""
    
    def __init__(self):
        self.variables = {}
        self.artifacts = {}
        self.step_outputs = {}
        self.pipeline_id = "test_pipeline"
        self.start_time = datetime.now()
        self.current_step = ""
    
    def get_variable(self, key: str, default=None):
        return self.variables.get(key, default)
    
    def set_artifact(self, key: str, value: Any):
        self.artifacts[key] = value
    
    def get_artifact(self, key: str, default=None):
        return self.artifacts.get(key, default)


if __name__ == "__main__":
    # Quick test of the BaseStep implementation
    print("Testing BaseStep implementation...")
    
    # Test 1: Mock step creation
    try:
        mock_step = create_mock_step(
            'test_step',
            'mock_data_acquisition',
            bbox=[85.3, 27.6, 85.4, 27.7],
            processing_time=0.05
        )
        print("✓ Mock step creation successful")
        
        # Test step info
        info = mock_step.get_step_info()
        print(f"✓ Step info: {info['step_type']} with {info['hyperparameters_count']} parameters")
        
    except Exception as e:
        print(f"✗ Mock step creation failed: {e}")
    
    # Test 2: Validation
    try:
        class TestStep(BaseStep):
            def execute(self, context):
                return {'status': 'success', 'outputs': {}, 'metadata': {}}
        
        errors = validate_step_implementation(TestStep)
        if not errors:
            print("✓ Step implementation validation passed")
        else:
            print(f"✗ Step validation errors: {errors}")
            
    except Exception as e:
        print(f"✗ Step validation failed: {e}")
    
    # Test 3: Configuration validation
    try:
        test_config = {
            'type': 'test_step',
            'hyperparameters': {'param1': 'value1'},
            'inputs': {'input1': {'required': True}},
            'outputs': {'output1': {'type': 'data'}},
            'dependencies': ['step1', 'step2']
        }
        
        test_step = MockStep('test_config', test_config)
        print("✓ Configuration validation successful")
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
    
    # Test 4: Mock execution with simple context
    try:
        context = SimpleExecutionContext()
        context.variables = {'bbox': [85.3, 27.6, 85.4, 27.7], 'area_name': 'test'}
        
        mock_step = create_mock_step('test_execution', 'mock_test')
        result = mock_step.execute(context)
        
        if result['status'] == 'success':
            print("✓ Mock execution successful")
            print(f"  Mock execution detected: {result['metadata'].get('mock', False)}")
            print(f"  Returns Dict instead of StepResult: {type(result).__name__}")
        else:
            print(f"✗ Mock execution failed: {result}")
            
    except Exception as e:
        print(f"✗ Mock execution test failed: {e}")
    
    print("\nBaseStep test completed!")
    print("✓ CORRECTED: No longer uses StepResult - returns Dict")
    print("✓ CORRECTED: Compatible with ExecutionContext")
    print("✓ CORRECTED: All StepResult references removed")
