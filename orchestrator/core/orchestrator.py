"""
Minimal Modular Pipeline Orchestrator - Core Implementation
Fail-fast approach focusing on essential functionality for architecture validation.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import traceback


@dataclass
class ExecutionContext:
    """Lightweight context for pipeline execution state."""
    
    # Core context data
    variables: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    step_outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    pipeline_id: str = ""
    start_time: Optional[datetime] = None
    current_step: str = ""
    
    def get_variable(self, key: str, default=None):
        """Get variable with template support."""
        value = self.variables.get(key, default)
        if isinstance(value, str):
            # Simple template variable substitution
            for var_key, var_value in self.variables.items():
                value = value.replace(f"{{{var_key}}}", str(var_value))
        return value
    
    def set_artifact(self, key: str, value: Any):
        """Store execution artifact."""
        self.artifacts[key] = value
    
    def get_artifact(self, key: str, default=None):
        """Retrieve execution artifact."""
        return self.artifacts.get(key, default)


class BaseStep:
    """Minimal base class for pipeline steps."""
    
    def __init__(self, step_id: str, step_config: Dict[str, Any]):
        self.step_id = step_id
        self.step_type = step_config.get('type', 'unknown')
        self.hyperparameters = step_config.get('hyperparameters', {})
        self.inputs = step_config.get('inputs', {})
        self.outputs = step_config.get('outputs', {})
        self.dependencies = step_config.get('dependencies', [])
        self.condition = step_config.get('condition')
        
    def should_execute(self, context: ExecutionContext) -> bool:
        """Check if step should execute based on condition."""
        if not self.condition:
            return True
        
        # Simple condition evaluation (extend as needed)
        try:
            # Replace variables in condition
            condition_str = self.condition
            for key, value in context.variables.items():
                condition_str = condition_str.replace(f"{{{key}}}", str(value))
            
            # Basic evaluation (in production, use safer evaluation)
            return eval(condition_str)
        except Exception as e:
            logging.warning(f"Condition evaluation failed for step {self.step_id}: {e}")
            return True
    
    def validate_inputs(self, context: ExecutionContext) -> bool:
        """Validate required inputs are available."""
        for input_key, input_config in self.inputs.items():
            if input_config.get('required', False):
                if input_key not in context.step_outputs:
                    logging.error(f"Required input '{input_key}' not found for step {self.step_id}")
                    return False
        return True
    
    def execute(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute the step. Must be implemented by subclasses."""
        raise NotImplementedError(f"Step {self.step_type} must implement execute method")


class StepRegistry:
    """Simple registry for step types."""
    
    _steps = {}
    
    @classmethod
    def register(cls, step_type: str, step_class):
        """Register a step type."""
        cls._steps[step_type] = step_class
        logging.info(f"Registered step type: {step_type}")
    
    @classmethod
    def create_step(cls, step_id: str, step_config: Dict[str, Any]) -> BaseStep:
        """Create step instance from configuration."""
        step_type = step_config.get('type')
        if step_type in cls._steps:
            return cls._steps[step_type](step_id, step_config)
        else:
            # Return a mock step for unknown types (fail-fast testing)
            return MockStep(step_id, step_config)
    
    @classmethod
    def get_registered_types(cls) -> List[str]:
        """Get list of registered step types."""
        return list(cls._steps.keys())


class MockStep(BaseStep):
    """Mock step for testing and unknown step types."""
    
    def execute(self, context: ExecutionContext) -> Dict[str, Any]:
        """Mock execution that simulates success."""
        logging.info(f"Mock execution of step {self.step_id} (type: {self.step_type})")
        
        # Simulate some processing time
        import time
        time.sleep(0.1)
        
        # Create mock outputs based on configuration
        mock_outputs = {}
        for output_key, output_config in self.outputs.items():
            if output_config.get('type') == 'file':
                mock_outputs[output_key] = f"/mock/path/{output_key}.tif"
            elif output_config.get('type') == 'data':
                mock_outputs[output_key] = {"mock": "data", "shape": [100, 100]}
            else:
                mock_outputs[output_key] = f"mock_{output_key}_result"
        
        return {
            'status': 'success',
            'outputs': mock_outputs,
            'metadata': {
                'execution_time': 0.1,
                'step_id': self.step_id,
                'step_type': self.step_type
            }
        }


class DependencyResolver:
    """Simple dependency resolver for step execution order."""
    
    @staticmethod
    def resolve_execution_order(steps: Dict[str, BaseStep]) -> List[str]:
        """Resolve step execution order based on dependencies."""
        # Topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        execution_order = []
        
        def visit(step_id: str):
            if step_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving step: {step_id}")
            if step_id in visited:
                return
            
            temp_visited.add(step_id)
            
            # Visit dependencies first
            step = steps[step_id]
            for dep_id in step.dependencies:
                if dep_id in steps:
                    visit(dep_id)
                else:
                    logging.warning(f"Dependency '{dep_id}' not found for step '{step_id}'")
            
            temp_visited.remove(step_id)
            visited.add(step_id)
            execution_order.append(step_id)
        
        # Visit all steps
        for step_id in steps:
            if step_id not in visited:
                visit(step_id)
        
        return execution_order


class ModularOrchestrator:
    """Minimal modular pipeline orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        self.steps = {}
        self.process_config = {}
        self.context = ExecutionContext()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup basic logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_process(self, process_path: Union[str, Path, Dict]) -> None:
        """Load process definition from JSON file or dict."""
        try:
            if isinstance(process_path, (str, Path)):
                with open(process_path, 'r') as f:
                    self.process_config = json.load(f)
                self.logger.info(f"Loaded process from {process_path}")
            elif isinstance(process_path, dict):
                self.process_config = process_path
                self.logger.info("Loaded process from dictionary")
            else:
                raise ValueError("Process path must be string, Path, or dict")
            
            # Initialize context with process variables
            self.context.variables.update(self.process_config.get('variables', {}))
            self.context.pipeline_id = self.process_config.get('name', 'unnamed_pipeline')
            
            # Load steps
            self._load_steps()
            
        except Exception as e:
            self.logger.error(f"Failed to load process: {e}")
            raise
    
    def _load_steps(self) -> None:
        """Load and instantiate steps from process configuration."""
        steps_config = self.process_config.get('steps', [])
        
        for step_config in steps_config:
            step_id = step_config.get('id')
            if not step_id:
                raise ValueError("Step configuration missing 'id' field")
            
            # Create step instance
            step = StepRegistry.create_step(step_id, step_config)
            self.steps[step_id] = step
            self.logger.info(f"Loaded step: {step_id} (type: {step.step_type})")
    
    def execute_process(self, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the loaded process."""
        if not self.process_config:
            raise ValueError("No process loaded. Call load_process() first.")
        
        try:
            # Update context variables
            if variables:
                self.context.variables.update(variables)
            
            self.context.start_time = datetime.now()
            execution_results = {'steps': {}, 'status': 'success', 'errors': []}
            
            # Resolve execution order
            execution_order = DependencyResolver.resolve_execution_order(self.steps)
            self.logger.info(f"Execution order: {execution_order}")
            
            # Execute steps in order
            for step_id in execution_order:
                try:
                    result = self._execute_step(step_id)
                    execution_results['steps'][step_id] = result
                    
                    if result['status'] != 'success':
                        execution_results['status'] = 'failed'
                        execution_results['errors'].append(f"Step {step_id} failed")
                        break
                        
                except Exception as e:
                    error_msg = f"Step {step_id} execution failed: {str(e)}"
                    self.logger.error(error_msg)
                    execution_results['status'] = 'failed'
                    execution_results['errors'].append(error_msg)
                    execution_results['steps'][step_id] = {
                        'status': 'failed',
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
                    break
            
            # Finalize results
            execution_results['artifacts'] = dict(self.context.artifacts)
            execution_results['execution_time'] = (
                datetime.now() - self.context.start_time
            ).total_seconds()
            
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Process execution failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _execute_step(self, step_id: str) -> Dict[str, Any]:
        """Execute a single step."""
        step = self.steps[step_id]
        self.context.current_step = step_id
        
        self.logger.info(f"Executing step: {step_id}")
        
        # Check execution condition
        if not step.should_execute(self.context):
            self.logger.info(f"Skipping step {step_id} due to condition")
            return {'status': 'skipped', 'reason': 'condition_not_met'}
        
        # Validate inputs
        if not step.validate_inputs(self.context):
            return {'status': 'failed', 'reason': 'input_validation_failed'}
        
        # Execute step
        start_time = datetime.now()
        result = step.execute(self.context)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Store step outputs in context
        if 'outputs' in result:
            self.context.step_outputs[step_id] = result['outputs']
        
        # Add execution metadata
        result['execution_time'] = execution_time
        result['timestamp'] = datetime.now().isoformat()
        
        self.logger.info(f"Step {step_id} completed in {execution_time:.2f}s")
        
        return result
    
    def get_step_registry(self) -> List[str]:
        """Get list of registered step types."""
        return StepRegistry.get_registered_types()
    
    def validate_process(self) -> Dict[str, Any]:
        """Validate the loaded process configuration."""
        if not self.process_config:
            return {'valid': False, 'errors': ['No process loaded']}
        
        errors = []
        warnings = []
        
        # Check required fields
        if 'steps' not in self.process_config:
            errors.append("Process missing 'steps' field")
        
        # Validate steps
        for step_config in self.process_config.get('steps', []):
            if 'id' not in step_config:
                errors.append("Step missing 'id' field")
            if 'type' not in step_config:
                errors.append(f"Step {step_config.get('id', 'unknown')} missing 'type' field")
        
        # Check dependencies
        step_ids = {s.get('id') for s in self.process_config.get('steps', [])}
        for step_config in self.process_config.get('steps', []):
            for dep in step_config.get('dependencies', []):
                if dep not in step_ids:
                    errors.append(f"Step {step_config.get('id')} has unknown dependency: {dep}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


# Register the mock step as default
StepRegistry.register('mock', MockStep)


if __name__ == "__main__":
    # Quick test
    test_process = {
        "name": "test_pipeline",
        "variables": {
            "input_path": "/test/input",
            "output_path": "/test/output"
        },
        "steps": [
            {
                "id": "step1",
                "type": "mock",
                "outputs": {"result": {"type": "data"}}
            },
            {
                "id": "step2",
                "type": "mock",
                "dependencies": ["step1"],
                "inputs": {"data": {"required": True}},
                "outputs": {"final_result": {"type": "file"}}
            }
        ]
    }
    
    orchestrator = ModularOrchestrator()
    orchestrator.load_process(test_process)
    
    validation = orchestrator.validate_process()
    print(f"Process validation: {validation}")
    
    if validation['valid']:
        result = orchestrator.execute_process()
        print(f"Execution result: {result['status']}")
        print(f"Steps executed: {list(result['steps'].keys())}")
    else:
        print(f"Process validation failed: {validation['errors']}")
