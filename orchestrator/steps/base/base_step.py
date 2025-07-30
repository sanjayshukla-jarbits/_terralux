"""
Abstract Base Step Class for Modular Pipeline Orchestrator
=========================================================

This module provides the abstract base class for all pipeline steps,
designed with fail-fast principles for rapid development and testing.

Key Features:
- Abstract interface for consistent step implementation
- Built-in logging and error handling
- Input/output validation
- Resource requirement specification
- Metadata tracking
- Mock execution support for testing
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field
import traceback

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from ...core.context_manager import PipelineContext

# Step execution results dataclass
@dataclass
class StepResult:
    """Standard result structure for step execution."""
    status: str  # 'success', 'failed', 'skipped'
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    artifacts_created: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class StepValidationError(Exception):
    """Custom exception for step validation errors."""
    pass


class BaseStep(ABC):
    """
    Abstract base class for all pipeline steps.
    
    This class provides the foundation for implementing modular pipeline steps
    with consistent interfaces, error handling, and metadata tracking.
    
    Design Philosophy:
    - Fail-fast: Quick feedback on configuration errors
    - Testable: Mock execution support for rapid development
    - Observable: Comprehensive logging and metadata tracking
    - Flexible: Support for various input/output patterns
    """
    
    def __init__(self, 
                 step_id: str, 
                 step_config: Dict[str, Any]):
        """
        Initialize the base step.
        
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
    def execute(self, context: 'PipelineContext') -> StepResult:
        """
        Execute the step logic.
        
        This is the main method that must be implemented by all step subclasses.
        It should contain the core logic for the step's operation.
        
        Args:
            context: Pipeline execution context containing shared data and configuration
            
        Returns:
            StepResult with execution status, outputs, and metadata
            
        Raises:
            Exception: Any step-specific exceptions should be caught and returned
                      as failed StepResult rather than propagated
        """
        pass
    
    def should_execute(self, context: 'PipelineContext') -> bool:
        """
        Determine if this step should execute based on conditions.
        
        Args:
            context: Pipeline execution context
            
        Returns:
            True if step should execute, False if it should be skipped
        """
        if not self.condition:
            return True
        
        try:
            # Simple condition evaluation
            # In production, use a safer evaluation method
            condition_str = self._substitute_variables(self.condition, context)
            
            # Basic boolean evaluation (extend as needed)
            if condition_str.lower() in ('true', '1', 'yes'):
                return True
            elif condition_str.lower() in ('false', '0', 'no'):
                return False
            else:
                # For complex conditions, could use safe eval or custom parser
                return bool(eval(condition_str))
                
        except Exception as e:
            self.logger.warning(f"Condition evaluation failed: {e}, defaulting to execute")
            return True
    
    def validate_inputs(self, context: 'PipelineContext') -> bool:
        """
        Validate that required inputs are available in the context.
        
        Args:
            context: Pipeline execution context
            
        Returns:
            True if all required inputs are available, False otherwise
        """
        missing_inputs = []
        
        for input_key, input_config in self.inputs.items():
            # Check if input is required
            if input_config.get('required', False):
                # Check if input is available from step outputs or artifacts
                available = False
                
                # Check in step outputs (from dependencies)
                for dep_step_id in self.dependencies:
                    if context.get_step_output(dep_step_id, input_key) is not None:
                        available = True
                        break
                
                # Check in artifacts
                if not available and context.get_artifact(input_key) is not None:
                    available = True
                
                # Check in context variables
                if not available and context.get_variable(input_key) is not None:
                    available = True
                
                if not available:
                    missing_inputs.append(input_key)
        
        if missing_inputs:
            self.logger.error(f"Missing required inputs: {missing_inputs}")
            return False
        
        return True
    
    def get_input_data(self, 
                      context: 'PipelineContext', 
                      input_key: str, 
                      default: Any = None) -> Any:
        """
        Retrieve input data from context with fallback logic.
        
        Args:
            context: Pipeline execution context
            input_key: Key for the input data
            default: Default value if input not found
            
        Returns:
            Input data value
        """
        # Try step outputs from dependencies first
        for dep_step_id in self.dependencies:
            value = context.get_step_output(dep_step_id, input_key)
            if value is not None:
                return value
        
        # Try artifacts
        value = context.get_artifact(input_key)
        if value is not None:
            return value
        
        # Try context variables
        value = context.get_variable(input_key)
        if value is not None:
            return value
        
        # Try hyperparameters
        value = self.hyperparameters.get(input_key)
        if value is not None:
            return value
        
        return default
    
    def set_output_data(self, 
                       context: 'PipelineContext', 
                       output_key: str, 
                       value: Any,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store output data in context.
        
        Args:
            context: Pipeline execution context
            output_key: Key for the output data
            value: Output value to store
            metadata: Optional metadata about the output
        """
        # Store as step output
        if not hasattr(context, '_temp_step_outputs'):
            context._temp_step_outputs = {}
        
        context._temp_step_outputs[output_key] = value
        
        # Also store as artifact with metadata
        context.set_artifact(output_key, value, metadata)
    
    def _substitute_variables(self, 
                            template: str, 
                            context: 'PipelineContext') -> str:
        """
        Substitute template variables in a string.
        
        Args:
            template: String with template variables
            context: Pipeline execution context
            
        Returns:
            String with variables substituted
        """
        if not isinstance(template, str):
            return template
        
        result = template
        
        # Substitute context variables
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
    
    def pre_execute(self, context: 'PipelineContext') -> None:
        """
        Pre-execution hook. Override in subclasses if needed.
        
        Args:
            context: Pipeline execution context
        """
        self.logger.info(f"Starting execution of {self.step_type} step: {self.step_id}")
        
        # Log resource requirements
        resources = self.get_resource_requirements()
        self.logger.debug(f"Resource requirements: {resources}")
    
    def post_execute(self, 
                    context: 'PipelineContext', 
                    result: StepResult) -> None:
        """
        Post-execution hook. Override in subclasses if needed.
        
        Args:
            context: Pipeline execution context
            result: Step execution result
        """
        if result.status == 'success':
            self.logger.info(f"Successfully completed {self.step_type} step: {self.step_id}")
            if result.execution_time:
                self.logger.info(f"Execution time: {result.execution_time:.2f} seconds")
        elif result.status == 'failed':
            self.logger.error(f"Failed {self.step_type} step: {self.step_id}")
            if result.error_message:
                self.logger.error(f"Error: {result.error_message}")
        else:
            self.logger.info(f"Skipped {self.step_type} step: {self.step_id}")
    
    def execute_with_hooks(self, context: 'PipelineContext') -> StepResult:
        """
        Execute step with pre/post hooks and error handling.
        
        This method wraps the execute() method with standard hooks and error handling.
        It should be called by the orchestrator rather than execute() directly.
        
        Args:
            context: Pipeline execution context
            
        Returns:
            StepResult with execution status and outputs
        """
        start_time = time.time()
        
        try:
            # Pre-execution hook
            self.pre_execute(context)
            
            # Check if step should execute
            if not self.should_execute(context):
                result = StepResult(
                    status='skipped',
                    metadata={'reason': 'condition_not_met', 'step_id': self.step_id}
                )
                self.post_execute(context, result)
                return result
            
            # Validate inputs
            if not self.validate_inputs(context):
                result = StepResult(
                    status='failed',
                    error_message='Input validation failed',
                    metadata={'step_id': self.step_id}
                )
                self.post_execute(context, result)
                return result
            
            # Execute main logic
            result = self.execute(context)
            
            # Ensure result is a StepResult
            if not isinstance(result, StepResult):
                # Convert dict result to StepResult for backward compatibility
                if isinstance(result, dict):
                    result = StepResult(
                        status=result.get('status', 'success'),
                        outputs=result.get('outputs', {}),
                        metadata=result.get('metadata', {}),
                        error_message=result.get('error_message', result.get('error'))
                    )
                else:
                    result = StepResult(status='failed', error_message='Invalid result type')
            
            # Add execution time
            result.execution_time = time.time() - start_time
            result.metadata['step_id'] = self.step_id
            result.metadata['step_type'] = self.step_type
            
            # Store outputs in context
            if result.status == 'success' and result.outputs:
                context.set_step_output(self.step_id, result.outputs)
                
                # Also set individual outputs as artifacts
                for output_key, output_value in result.outputs.items():
                    context.set_artifact(f"{self.step_id}_{output_key}", output_value)
            
            # Post-execution hook
            self.post_execute(context, result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            self.logger.error(f"Step execution failed: {error_message}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            result = StepResult(
                status='failed',
                error_message=error_message,
                execution_time=execution_time,
                metadata={
                    'step_id': self.step_id,
                    'step_type': self.step_type,
                    'exception_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                }
            )
            
            # Post-execution hook even on failure
            try:
                self.post_execute(context, result)
            except Exception as hook_error:
                self.logger.error(f"Post-execution hook failed: {hook_error}")
            
            return result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.step_id}, type={self.step_type})"
    
    def __str__(self) -> str:
        return f"{self.step_type}:{self.step_id}"


class MockStep(BaseStep):
    """
    Mock step implementation for testing and fail-fast development.
    
    This step simulates execution without performing actual operations,
    useful for testing pipeline structure and data flow.
    """
    
    def execute(self, context: 'PipelineContext') -> StepResult:
        """Execute mock step with simulated processing."""
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
        
        # Store outputs in context
        for output_key, output_value in mock_outputs.items():
            self.set_output_data(context, output_key, output_value)
        
        return StepResult(
            status='success',
            outputs=mock_outputs,
            metadata={
                'mock_execution': True,
                'processing_time': processing_time,
                'hyperparameters': dict(self.hyperparameters)
            }
        )


# Utility functions for step development
def create_mock_step(step_id: str, 
                    step_type: str = 'mock',
                    **hyperparameters) -> MockStep:
    """
    Create a mock step for testing.
    
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


if __name__ == "__main__":
    # Quick test of the BaseStep implementation
    import tempfile
    from pathlib import Path
    
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
                return StepResult(status='success')
        
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
    
    print("BaseStep test completed!")
