"""
Fixed ModularOrchestrator - Core Implementation for _terralux
===========================================================

This implementation fixes the method signature issues, uses existing components
properly, and integrates with the _terralux project structure correctly.
"""

import json
import logging
import warnings
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

# Import existing _terralux components
from .process_loader import ProcessLoader
from .context_manager import PipelineContext
from ..steps.base.step_registry import StepRegistry
from ..steps.base.base_step import BaseStep, MockStep


class ModularOrchestrator:
    """
    Fixed modular pipeline orchestrator with proper method signatures
    and integration with existing _terralux components.
    """
    
    def __init__(self, schema_path: Optional[str] = None):
        """Initialize the orchestrator."""
        self.logger = logging.getLogger('orchestrator.core.orchestrator')
        
        # Initialize components using existing _terralux infrastructure
        self.process_loader = ProcessLoader(enable_strict_validation=False)
        self.context = PipelineContext()
        
        # State management
        self.process_config: Optional[Dict[str, Any]] = None
        self.steps: Dict[str, BaseStep] = {}
        
        # Register available steps
        self._register_available_steps()
        
        self.logger.info("ModularOrchestrator initialized")
    
    def _register_available_steps(self):
        """Register all available step types with error handling."""
        step_import_errors = []
        real_steps = 0
        
        # List of step modules to try importing with corrected paths
        step_modules = [
            ('sentinel_hub_acquisition', 'orchestrator.steps.data_acquisition.sentinel_hub_step', 'SentinelHubStep'),
            ('dem_acquisition', 'orchestrator.steps.data_acquisition.dem_acquisition_step', 'DEMAcquisitionStep'),
            ('local_files_discovery', 'orchestrator.steps.data_acquisition.local_files_step', 'LocalFilesStep'),
            ('spectral_indices_extraction', 'orchestrator.steps.feature_extraction.spectral_indices_step', 'SpectralIndicesStep'),
            ('data_validation', 'orchestrator.steps.preprocessing.data_validation_step', 'DataValidationStep'),
            ('inventory_generation', 'orchestrator.steps.preprocessing.inventory_generation_step', 'InventoryGenerationStep'),
        ]
        
        for step_type, module_path, class_name in step_modules:
            # Skip if already registered to avoid warnings
            if StepRegistry.is_registered(step_type):
                self.logger.debug(f"Step {step_type} already registered, skipping")
                real_steps += 1
                continue
            
            try:
                # Try to import the module and class
                module = __import__(module_path, fromlist=[class_name])
                step_class = getattr(module, class_name)
                StepRegistry.register(step_type, step_class)
                real_steps += 1
                self.logger.debug(f"✓ Registered real step: {step_type}")
            except ImportError as e:
                step_import_errors.append(f"Real {step_type.replace('_', ' ').title()} step: ImportError - {e}")
                self.logger.warning(f"⚠ Real {step_type.replace('_', ' ').title()} step import failed: {e}")
            except Exception as e:
                step_import_errors.append(f"Real {step_type.replace('_', ' ').title()} step: {type(e).__name__} - {e}")
                self.logger.warning(f"⚠ Real {step_type.replace('_', ' ').title()} step registration failed: {e}")
        
        # Log registration summary
        available_types = StepRegistry.get_registered_types()
        self.logger.info(f"Step registration complete: {len(available_types)} types registered")
        self.logger.info(f"Available step types: {available_types}")
        
        # Show import failures
        if step_import_errors:
            self.logger.warning("Some step imports failed (will use mock implementations):")
            for error in step_import_errors:
                self.logger.warning(f"  - {error}")
            
            # Only show the first few errors to avoid spam
            if len(step_import_errors) > 6:
                self.logger.warning(f"  ... and {len(step_import_errors) - 6} more")
        
        # Warning if no real steps available
        if real_steps == 0:
            warnings.warn(
                "No real step implementations are available. "
                "Using mock implementations only. "
                "Install step dependencies for full functionality.",
                RuntimeWarning
            )
        elif real_steps < 3:
            self.logger.warning(f"Only {real_steps} real step implementations available. "
                              f"Consider installing missing dependencies for full functionality.")
    
    def load_process(self, 
                    process_path: Union[str, Path, Dict[str, Any]], 
                    template_variables: Optional[Dict[str, Any]] = None) -> None:
        """
        Load process definition from JSON file or dictionary.
        
        FIXED: Now accepts both process_path and template_variables as separate parameters.
        
        Args:
            process_path: Path to JSON file or process dictionary
            template_variables: Variables for template substitution
        """
        try:
            # Use ProcessLoader to load and validate the process
            self.process_config = self.process_loader.load_process(
                process_path, 
                template_variables
            )
            
            # Initialize context with process variables
            if 'global_config' in self.process_config:
                self.context.variables.update(self.process_config['global_config'])
            
            if template_variables:
                self.context.variables.update(template_variables)
            
            # Set pipeline metadata
            process_info = self.process_config.get('process_info', {})
            self.context.pipeline_id = process_info.get('name', 'unnamed_pipeline')
            
            # Load and instantiate steps
            self._load_steps()
            
            self.logger.info(f"Successfully loaded process: {process_info.get('name', 'Unknown')}")
            
        except Exception as e:
            self.logger.error(f"Failed to load process: {e}")
            raise
    
    def _load_steps(self) -> None:
        """Load and instantiate steps from process configuration."""
        if not self.process_config:
            raise ValueError("No process configuration loaded")
        
        steps_config = self.process_config.get('steps', [])
        
        for step_config in steps_config:
            step_id = step_config.get('id')
            if not step_id:
                raise ValueError("Step configuration missing 'id' field")
            
            try:
                # Create step instance using StepRegistry
                step = StepRegistry.create_step(step_id, step_config)
                self.steps[step_id] = step
                
                # Log whether we're using real or mock implementation
                if isinstance(step, MockStep):
                    self.logger.warning(f"Loaded step: {step_id} (type: {step.step_type}) [MOCK]")
                else:
                    self.logger.info(f"Loaded step: {step_id} (type: {step.step_type}) [REAL]")
                    
            except Exception as e:
                self.logger.error(f"Failed to create step {step_id}: {e}")
                # Create a mock step as fallback
                step = MockStep(step_id, step_config)
                self.steps[step_id] = step
                self.logger.warning(f"Using mock fallback for step: {step_id}")
    
    def execute_process(self, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the loaded process.
        
        Args:
            variables: Additional runtime variables
            
        Returns:
            Execution results dictionary
        """
        if not self.process_config:
            raise ValueError("No process loaded. Call load_process() first.")
        
        if not self.steps:
            raise ValueError("No steps loaded. Process may be invalid.")
        
        # Update context with runtime variables
        if variables:
            self.context.variables.update(variables)
        
        # Initialize execution tracking
        start_time = datetime.now()
        results = {
            'status': 'running',
            'start_time': start_time.isoformat(),
            'step_results': {},
            'artifacts': {},
            'errors': []
        }
        
        try:
            # Resolve execution order using simple dependency resolution
            execution_order = self._resolve_execution_order()
            
            # Execute steps in order
            for step_id in execution_order:
                if step_id not in self.steps:
                    error_msg = f"Step {step_id} not found in loaded steps"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
                    continue
                
                # Execute step
                self.logger.info(f"Executing step: {step_id}")
                step = self.steps[step_id]
                
                try:
                    step_result = step.execute(self.context)
                    
                    # Handle step result
                    if isinstance(step_result, dict):
                        if step_result.get('status') == 'success':
                            self.logger.info(f"✓ Step {step_id} completed successfully")
                            
                            # Update context with step outputs
                            if 'outputs' in step_result:
                                for key, value in step_result['outputs'].items():
                                    self.context.set_artifact(f"{step_id}_{key}", value)
                        else:
                            self.logger.warning(f"⚠ Step {step_id} completed with status: {step_result.get('status')}")
                    
                    results['step_results'][step_id] = step_result
                    
                except Exception as step_error:
                    error_msg = f"Step {step_id} failed: {step_error}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
                    results['step_results'][step_id] = {
                        'status': 'failed',
                        'error': str(step_error)
                    }
                    
                    # For fail-fast, continue with other steps
                    continue
            
            # Finalize results
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            results.update({
                'status': 'success' if not results['errors'] else 'completed_with_errors',
                'end_time': end_time.isoformat(),
                'total_execution_time': execution_time,
                'artifacts': dict(self.context.artifacts)
            })
            
            # Success/failure logging
            successful_steps = len([r for r in results['step_results'].values() 
                                  if r.get('status') == 'success'])
            total_steps = len(results['step_results'])
            
            self.logger.info(f"Process execution completed: {successful_steps}/{total_steps} steps successful")
            
            if results['errors']:
                self.logger.warning(f"Process completed with {len(results['errors'])} errors")
                for error in results['errors']:
                    self.logger.warning(f"  - {error}")
            
            return results
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            error_result = {
                'status': 'failed',
                'error': str(e),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_execution_time': execution_time,
                'step_results': results.get('step_results', {}),
                'artifacts': dict(self.context.artifacts)
            }
            
            self.logger.error(f"Process execution failed: {e}")
            return error_result
    
    def _resolve_execution_order(self) -> List[str]:
        """Simple dependency resolution for step execution order."""
        # Get steps from config to preserve order while handling dependencies
        steps_config = self.process_config.get('steps', [])
        
        # For now, use simple order from config
        # TODO: Implement proper topological sort for dependencies
        execution_order = []
        
        for step_config in steps_config:
            step_id = step_config.get('id')
            if step_id and step_id in self.steps:
                execution_order.append(step_id)
        
        self.logger.debug(f"Execution order: {execution_order}")
        return execution_order
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of loaded steps for execution planning."""
        if not self.steps:
            return {
                'total_steps': 0,
                'real_steps': 0,
                'mock_steps': 0,
                'step_types': {}
            }
        
        step_types = {}
        mock_count = 0
        real_count = 0
        
        for step_id, step in self.steps.items():
            step_type = getattr(step, 'step_type', 'unknown')
            step_types[step_type] = step_types.get(step_type, 0) + 1
            
            if isinstance(step, MockStep):
                mock_count += 1
            else:
                real_count += 1
        
        return {
            'total_steps': len(self.steps),
            'real_steps': real_count,
            'mock_steps': mock_count,
            'step_types': step_types
        }
    
    def get_process_info(self) -> Dict[str, Any]:
        """Get information about the loaded process."""
        if not self.process_config:
            return {'name': 'No process loaded'}
        
        return self.process_config.get('process_info', {'name': 'Unknown process'})
    
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
            
            # Check if step type is available (will it use real or mock?)
            step_type = step_config.get('type')
            if step_type and not StepRegistry.is_step_type_available(step_type):
                warnings.append(f"Step type '{step_type}' not registered, will use mock implementation")
        
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
    
    def get_step_registry(self) -> List[str]:
        """Get list of registered step types."""
        return StepRegistry.get_registered_types()
    
    def list_available_step_types(self) -> List[str]:
        """List all available step types."""
        return StepRegistry.get_registered_types()


# Simple dependency resolver class
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
            for dep_id in getattr(step, 'dependencies', []):
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


if __name__ == "__main__":
    # Test the corrected implementation
    test_process = {
        "process_info": {
            "name": "test_pipeline",
            "version": "1.0.0",
            "description": "Test pipeline for validation"
        },
        "global_config": {
            "input_path": "/test/input",
            "output_path": "/test/output"
        },
        "steps": [
            {
                "id": "step1",
                "type": "sentinel_hub_acquisition",
                "hyperparameters": {
                    "bbox": [85.3, 27.6, 85.4, 27.7]
                },
                "outputs": {"result": {"type": "data"}}
            },
            {
                "id": "step2",
                "type": "data_validation",
                "dependencies": ["step1"],
                "hyperparameters": {
                    "checks": ["format", "completeness"]
                },
                "outputs": {"validation_report": {"type": "file"}}
            }
        ]
    }
    
    # Test orchestrator
    orchestrator = ModularOrchestrator()
    orchestrator.load_process(test_process, {"area_name": "test_area"})
    
    # Show execution summary
    summary = orchestrator.get_execution_summary()
    print(f"Execution Summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Real steps: {summary['real_steps']}")
    print(f"  Mock steps: {summary['mock_steps']}")
    
    # Validate process
    validation = orchestrator.validate_process()
    print(f"\nProcess validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    # Execute if valid
    if validation['valid']:
        result = orchestrator.execute_process()
        print(f"\nExecution result: {result['status']}")
        print(f"Steps executed: {list(result['step_results'].keys())}")
        
        # Show which steps were real vs mock
        for step_id, step_result in result['step_results'].items():
            status = step_result.get('status', 'unknown')
            print(f"  - {step_id}: {status}")
    else:
        print(f"Process validation failed, skipping execution")
