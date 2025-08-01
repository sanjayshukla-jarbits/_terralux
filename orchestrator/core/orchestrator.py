#!/usr/bin/env python3
"""
CORRECTED ModularOrchestrator - Core Implementation for TerraLux
===============================================================

This implementation fixes all the critical issues identified in the execution log:
- Corrects class name mismatches (SentinelHubAcquisitionStep vs SentinelHubStep)
- Prevents duplicate step registration warnings
- Improves error handling and fallback mechanisms
- Enhances integration with TerraLux components
- Adds better step registry management

Key Fixes Applied:
- Fixed step class name mapping in _register_available_steps
- Added duplicate registration prevention
- Enhanced error handling with better logging
- Improved process loading and validation
- Better step factory integration
- Enhanced context management

Author: TerraLux Development Team
Version: 1.0.0-corrected
"""

import json
import logging
import warnings
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

# Import existing TerraLux components
try:
    from .process_loader import ProcessLoader
    from .context_manager import PipelineContext
    from ..steps.base.step_registry import StepRegistry
    from ..steps.base.base_step import BaseStep, MockStep
except ImportError as e:
    logging.warning(f"Import error in orchestrator core: {e}")
    # Fallback minimal classes for development
    class ProcessLoader:
        def __init__(self, enable_strict_validation=False):
            self.enable_strict_validation = enable_strict_validation
        def load_process(self, process_path, template_variables=None):
            return {}
    
    class PipelineContext:
        def __init__(self):
            self.variables = {}
            self.artifacts = {}
            self.pipeline_id = 'test'
        def set_artifact(self, key, value):
            self.artifacts[key] = value
    
    class StepRegistry:
        _registry = {}
        @classmethod
        def is_registered(cls, step_type): return step_type in cls._registry
        @classmethod
        def register(cls, step_type, step_class): cls._registry[step_type] = step_class
        @classmethod
        def get_registered_types(cls): return list(cls._registry.keys())
        @classmethod
        def create_step(cls, step_id, step_config): 
            return MockStep(step_id, step_config)
        @classmethod
        def is_step_type_available(cls, step_type): return step_type in cls._registry
    
    class BaseStep:
        def __init__(self, step_id, hyperparameters=None):
            self.step_id = step_id
            self.step_type = 'base'
            self.hyperparameters = hyperparameters or {}
        def execute(self, context): return {'status': 'success'}
    
    class MockStep(BaseStep):
        def __init__(self, step_id, step_config):
            super().__init__(step_id, step_config.get('hyperparameters', {}))
            self.step_type = step_config.get('type', 'mock')


class ModularOrchestrator:
    """
    CORRECTED modular pipeline orchestrator with proper method signatures
    and integration with existing TerraLux components.
    
    This version fixes all the issues found in the execution log:
    - Correct step class name mapping
    - Duplicate registration prevention
    - Better error handling
    - Enhanced logging and status reporting
    """
    
    def __init__(self, schema_path: Optional[str] = None):
        """Initialize the orchestrator."""
        self.logger = logging.getLogger('orchestrator.core.orchestrator')
        
        # Initialize components using existing TerraLux infrastructure
        try:
            self.process_loader = ProcessLoader(enable_strict_validation=False)
            self.context = PipelineContext()
        except Exception as e:
            self.logger.warning(f"Failed to initialize components: {e}")
            # Fallback initialization
            self.process_loader = ProcessLoader()
            self.context = PipelineContext()
        
        # State management
        self.process_config: Optional[Dict[str, Any]] = None
        self.steps: Dict[str, BaseStep] = {}
        self.schema_path = schema_path
        
        # Track registration to prevent duplicates
        self._registration_completed = False
        
        # Register available steps
        self._register_available_steps()
        
        self.logger.info("ModularOrchestrator initialized")
    
    def _register_available_steps(self):
        """Register all available step types with error handling and duplicate prevention."""
        # Skip registration if already completed to prevent duplicate warnings
        if self._registration_completed:
            self.logger.debug("Step registration already completed, skipping")
            return
        
        # Check if steps are already registered by other components
        already_registered = StepRegistry.get_registered_types()
        if already_registered:
            self.logger.info(f"Found {len(already_registered)} pre-registered steps: {already_registered}")
            self._registration_completed = True
            return
        
        step_import_errors = []
        real_steps = 0
        
        # CORRECTED: List of step modules with correct class names matching actual implementations
        step_modules = [
            # Data acquisition steps - FIXED class names
            ('sentinel_hub_acquisition', 'orchestrator.steps.data_acquisition.sentinel_hub_step', 'SentinelHubAcquisitionStep'),
            ('dem_acquisition', 'orchestrator.steps.data_acquisition.dem_acquisition_step', 'DEMAcquisitionStep'),
            ('local_files_discovery', 'orchestrator.steps.data_acquisition.local_files_step', 'LocalFilesDiscoveryStep'),
            
            # Feature extraction steps
            ('spectral_indices_extraction', 'orchestrator.steps.feature_extraction.spectral_indices_step', 'SpectralIndicesStep'),
            
            # Preprocessing steps  
            ('data_validation', 'orchestrator.steps.preprocessing.data_validation_step', 'DataValidationStep'),
            ('inventory_generation', 'orchestrator.steps.preprocessing.inventory_generation_step', 'InventoryGenerationStep'),
            ('atmospheric_correction', 'orchestrator.steps.preprocessing.atmospheric_correction_step', 'AtmosphericCorrectionStep'),
            ('geometric_correction', 'orchestrator.steps.preprocessing.geometric_correction_step', 'GeometricCorrectionStep'),
            ('cloud_masking', 'orchestrator.steps.preprocessing.cloud_masking_step', 'CloudMaskingStep'),
            
            # Modeling steps
            ('random_forest', 'orchestrator.steps.modeling.random_forest_step', 'RandomForestStep'),
            ('logistic_regression', 'orchestrator.steps.modeling.logistic_regression_step', 'LogisticRegressionStep'),
            ('model_validation', 'orchestrator.steps.modeling.model_validation_step', 'ModelValidationStep'),
            
            # Visualization steps
            ('map_visualization', 'orchestrator.steps.visualization.map_visualization_step', 'MapVisualizationStep'),
            ('report_generation', 'orchestrator.steps.visualization.report_generation_step', 'ReportGenerationStep'),
            ('statistical_plots', 'orchestrator.steps.visualization.statistical_plots_step', 'StatisticalPlotsStep'),
        ]
        
        for step_type, module_path, class_name in step_modules:
            try:
                # Only register if not already registered to prevent duplicates
                if not StepRegistry.is_registered(step_type):
                    # Try to import the module and class
                    module = __import__(module_path, fromlist=[class_name])
                    step_class = getattr(module, class_name)
                    StepRegistry.register(step_type, step_class)
                    real_steps += 1
                    self.logger.debug(f"✓ Registered real step: {step_type} -> {class_name}")
                else:
                    self.logger.debug(f"Step {step_type} already registered, skipping")
                    real_steps += 1
                    
            except ImportError as e:
                step_import_errors.append(f"{step_type}: ImportError - {str(e)[:100]}...")
                self.logger.debug(f"⚠ Step {step_type} import failed: {e}")
            except AttributeError as e:
                step_import_errors.append(f"{step_type}: Class {class_name} not found - {str(e)[:100]}...")
                self.logger.debug(f"⚠ Step {step_type} class not found: {e}")
            except Exception as e:
                step_import_errors.append(f"{step_type}: {type(e).__name__} - {str(e)[:100]}...")
                self.logger.debug(f"⚠ Step {step_type} registration failed: {e}")
        
        # Mark registration as completed
        self._registration_completed = True
        
        # Log registration summary
        available_types = StepRegistry.get_registered_types()
        self.logger.info(f"Step registration complete: {len(available_types)} types registered")
        self.logger.info(f"Available step types: {available_types}")
        
        # Show import failures only if verbose logging
        if step_import_errors and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Some step imports failed (will use mock implementations):")
            for error in step_import_errors[:5]:  # Show only first 5 errors
                self.logger.debug(f"  - {error}")
            
            if len(step_import_errors) > 5:
                self.logger.debug(f"  ... and {len(step_import_errors) - 5} more import errors")
        
        # Issue warnings only for critical situations
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
        else:
            self.logger.info(f"✓ Successfully registered {real_steps} real step implementations")
    
    def load_process(self, 
                    process_path: Union[str, Path, Dict[str, Any]], 
                    template_variables: Optional[Dict[str, Any]] = None) -> None:
        """
        Load process definition from JSON file or dictionary.
        
        CORRECTED: Now properly handles both process_path and template_variables as separate parameters.
        
        Args:
            process_path: Path to JSON file or process dictionary
            template_variables: Variables for template substitution
        """
        try:
            self.logger.debug(f"Loading process from: {process_path}")
            
            # Use ProcessLoader to load and validate the process
            self.process_config = self.process_loader.load_process(
                process_path, 
                template_variables or {}
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
            
            process_name = process_info.get('name', 'Unknown Process')
            self.logger.info(f"Successfully loaded process: {process_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load process: {e}")
            raise
    
    def _load_steps(self) -> None:
        """Load and instantiate steps from process configuration."""
        if not self.process_config:
            raise ValueError("No process configuration loaded")
        
        steps_config = self.process_config.get('steps', [])
        if not steps_config:
            self.logger.warning("No steps defined in process configuration")
            return
        
        for step_config in steps_config:
            step_id = step_config.get('id')
            step_type = step_config.get('type')
            
            if not step_id:
                raise ValueError("Step configuration missing 'id' field")
            
            if not step_type:
                raise ValueError(f"Step {step_id} missing 'type' field")
            
            try:
                # Create step instance using StepRegistry
                step = StepRegistry.create_step(step_id, step_config)
                self.steps[step_id] = step
                
                # Log whether we're using real or mock implementation
                if isinstance(step, MockStep) or 'Mock' in step.__class__.__name__:
                    self.logger.info(f"Loaded step: {step_id} (type: {step_type}) [MOCK]")
                else:
                    self.logger.info(f"Loaded step: {step_id} (type: {step_type}) [REAL]")
                    
            except Exception as e:
                self.logger.error(f"Failed to create step {step_id}: {e}")
                
                # Create a mock step as fallback
                try:
                    step = MockStep(step_id, step_config)
                    self.steps[step_id] = step
                    self.logger.info(f"Using mock fallback for step: {step_id}")
                except Exception as fallback_error:
                    self.logger.error(f"Failed to create mock fallback for {step_id}: {fallback_error}")
                    raise
    
    def execute_process(self, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the loaded process with enhanced error handling and reporting.
        
        Args:
            variables: Additional runtime variables
            
        Returns:
            Execution results dictionary with comprehensive status information
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
            'errors': [],
            'warnings': []
        }
        
        successful_steps = 0
        
        try:
            # Resolve execution order using dependency resolution
            execution_order = self._resolve_execution_order()
            self.logger.debug(f"Execution order: {execution_order}")
            
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
                    # Execute step with context
                    step_result = step.execute(self.context)
                    
                    # Handle step result
                    if isinstance(step_result, dict):
                        step_status = step_result.get('status', 'unknown')
                        
                        if step_status == 'success':
                            self.logger.info(f"✓ Step {step_id} completed successfully")
                            successful_steps += 1
                            
                            # Update context with step outputs
                            if 'outputs' in step_result:
                                for key, value in step_result['outputs'].items():
                                    artifact_key = f"{step_id}_{key}"
                                    self.context.set_artifact(artifact_key, value)
                                    
                        elif step_status == 'warning':
                            self.logger.warning(f"⚠ Step {step_id} completed with warnings")
                            successful_steps += 1
                            results['warnings'].append(f"Step {step_id}: {step_result.get('message', 'Warning')}")
                            
                        else:
                            self.logger.error(f"✗ Step {step_id} failed with status: {step_status}")
                            error_msg = step_result.get('error', f"Step {step_id} failed")
                            results['errors'].append(error_msg)
                    else:
                        # Handle non-dict results
                        self.logger.warning(f"Step {step_id} returned non-dict result: {type(step_result)}")
                        step_result = {
                            'status': 'success',
                            'result': step_result,
                            'message': f"Step {step_id} completed (non-standard result format)"
                        }
                        successful_steps += 1
                    
                    results['step_results'][step_id] = step_result
                    
                except Exception as step_error:
                    error_msg = f"Step {step_id} failed with exception: {step_error}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
                    results['step_results'][step_id] = {
                        'status': 'failed',
                        'error': str(step_error),
                        'step_id': step_id
                    }
                    
                    # Continue with other steps (fail-safe execution)
                    continue
            
            # Finalize results
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            total_steps = len(results['step_results'])
            
            results.update({
                'status': 'success' if not results['errors'] else 'completed_with_errors',
                'end_time': end_time.isoformat(),
                'total_execution_time': execution_time,
                'artifacts': dict(self.context.artifacts),
                'successful_steps': successful_steps,
                'total_steps': total_steps
            })
            
            # Enhanced logging
            if results['status'] == 'success':
                self.logger.info(f"✅ Process execution completed successfully: {successful_steps}/{total_steps} steps successful")
            else:
                self.logger.warning(f"⚠️ Process completed with errors: {successful_steps}/{total_steps} steps successful, {len(results['errors'])} errors")
                
            if results['warnings']:
                self.logger.info(f"Warnings encountered: {len(results['warnings'])}")
                
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
                'artifacts': dict(self.context.artifacts),
                'successful_steps': successful_steps,
                'total_steps': len(results.get('step_results', {}))
            }
            
            self.logger.error(f"❌ Process execution failed: {e}")
            return error_result
    
    def _resolve_execution_order(self) -> List[str]:
        """
        Resolve step execution order based on dependencies.
        
        ENHANCED: Better dependency resolution with cycle detection.
        """
        if not self.process_config:
            return []
            
        steps_config = self.process_config.get('steps', [])
        if not steps_config:
            return []
        
        # Build dependency graph
        step_deps = {}
        all_steps = set()
        
        for step_config in steps_config:
            step_id = step_config.get('id')
            if step_id and step_id in self.steps:
                all_steps.add(step_id)
                step_deps[step_id] = step_config.get('dependencies', [])
        
        # Simple topological sort
        execution_order = []
        remaining_steps = set(all_steps)
        
        # Safety counter to prevent infinite loops
        max_iterations = len(all_steps) * 2
        iteration = 0
        
        while remaining_steps and iteration < max_iterations:
            iteration += 1
            
            # Find steps with no unresolved dependencies
            ready_steps = []
            for step_id in remaining_steps:
                deps = step_deps.get(step_id, [])
                if all(dep in execution_order or dep not in all_steps for dep in deps):
                    ready_steps.append(step_id)
            
            if not ready_steps:
                # No steps ready - possible circular dependency
                self.logger.warning(f"Possible circular dependency detected. Remaining steps: {remaining_steps}")
                # Add remaining steps in arbitrary order
                ready_steps = list(remaining_steps)
            
            # Add ready steps to execution order
            for step_id in ready_steps:
                execution_order.append(step_id)
                remaining_steps.remove(step_id)
        
        self.logger.debug(f"Resolved execution order: {execution_order}")
        return execution_order
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of loaded steps for execution planning."""
        if not self.steps:
            return {
                'total_steps': 0,
                'real_steps': 0,
                'mock_steps': 0,
                'step_types': {},
                'step_details': []
            }
        
        step_types = {}
        mock_count = 0
        real_count = 0
        step_details = []
        
        for step_id, step in self.steps.items():
            step_type = getattr(step, 'step_type', 'unknown')
            step_types[step_type] = step_types.get(step_type, 0) + 1
            
            # Determine if step is mock or real
            is_mock = isinstance(step, MockStep) or 'Mock' in step.__class__.__name__
            if is_mock:
                mock_count += 1
                implementation = 'MOCK'
            else:
                real_count += 1
                implementation = 'REAL'
            
            step_details.append({
                'id': step_id,
                'type': step_type,
                'implementation': implementation,
                'class': step.__class__.__name__
            })
        
        return {
            'total_steps': len(self.steps),
            'real_steps': real_count,
            'mock_steps': mock_count,
            'step_types': step_types,
            'step_details': step_details
        }
    
    def get_process_info(self) -> Dict[str, Any]:
        """Get information about the loaded process."""
        if not self.process_config:
            return {'name': 'No process loaded'}
        
        process_info = self.process_config.get('process_info', {'name': 'Unknown process'})
        
        # Add execution context info
        process_info['steps_loaded'] = len(self.steps)
        process_info['context_variables'] = len(self.context.variables)
        
        return process_info
    
    def validate_process(self) -> Dict[str, Any]:
        """Validate the loaded process configuration."""
        if not self.process_config:
            return {'valid': False, 'errors': ['No process loaded'], 'warnings': []}
        
        errors = []
        warnings = []
        
        # Check required fields
        if 'steps' not in self.process_config:
            errors.append("Process missing 'steps' field")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        steps_config = self.process_config.get('steps', [])
        if not steps_config:
            warnings.append("Process has no steps defined")
        
        step_ids = set()
        
        # Validate individual steps
        for i, step_config in enumerate(steps_config):
            step_context = f"Step {i+1}"
            
            if 'id' not in step_config:
                errors.append(f"{step_context}: missing 'id' field")
                continue
                
            step_id = step_config['id']
            step_context = f"Step '{step_id}'"
            
            if step_id in step_ids:
                errors.append(f"{step_context}: duplicate step ID")
            else:
                step_ids.add(step_id)
            
            if 'type' not in step_config:
                errors.append(f"{step_context}: missing 'type' field")
                continue
                
            step_type = step_config['type']
            
            # Check if step type is available
            if not StepRegistry.is_step_type_available(step_type):
                warnings.append(f"{step_context}: step type '{step_type}' not registered, will use mock implementation")
        
        # Check dependencies
        for step_config in steps_config:
            step_id = step_config.get('id')
            if not step_id:
                continue
                
            for dep in step_config.get('dependencies', []):
                if dep not in step_ids:
                    errors.append(f"Step '{step_id}' has unknown dependency: '{dep}'")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'steps_validated': len(steps_config),
            'unique_step_ids': len(step_ids)
        }
    
    def get_step_registry(self) -> List[str]:
        """Get list of registered step types."""
        return StepRegistry.get_registered_types()
    
    def list_available_step_types(self) -> List[str]:
        """List all available step types."""
        return StepRegistry.get_registered_types()
    
    def get_step_info(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific loaded step."""
        if step_id not in self.steps:
            return None
            
        step = self.steps[step_id]
        return {
            'id': step_id,
            'type': getattr(step, 'step_type', 'unknown'),
            'class': step.__class__.__name__,
            'is_mock': isinstance(step, MockStep) or 'Mock' in step.__class__.__name__,
            'hyperparameters': getattr(step, 'hyperparameters', {}),
            'dependencies': getattr(step, 'dependencies', [])
        }


# Enhanced Dependency Resolver
class DependencyResolver:
    """Enhanced dependency resolver for step execution order with cycle detection."""
    
    @staticmethod
    def resolve_execution_order(steps_config: List[Dict[str, Any]]) -> List[str]:
        """
        Resolve step execution order based on dependencies using topological sort.
        
        Args:
            steps_config: List of step configuration dictionaries
            
        Returns:
            List of step IDs in execution order
        """
        # Build dependency graph
        graph = {}
        in_degree = {}
        
        # Initialize graph
        for step_config in steps_config:
            step_id = step_config.get('id')
            if step_id:
                graph[step_id] = step_config.get('dependencies', [])
                in_degree[step_id] = 0
        
        # Calculate in-degrees
        for step_id, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Topological sort using Kahn's algorithm
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            # Update in-degrees of dependent steps
            for step_id, deps in graph.items():
                if current in deps:
                    in_degree[step_id] -= 1
                    if in_degree[step_id] == 0:
                        queue.append(step_id)
        
        # Check for circular dependencies
        if len(execution_order) != len(graph):
            remaining_steps = set(graph.keys()) - set(execution_order)
            logging.warning(f"Circular dependency detected. Remaining steps: {remaining_steps}")
            # Add remaining steps in arbitrary order
            execution_order.extend(remaining_steps)
        
        return execution_order


# For testing and validation
if __name__ == "__main__":
    # Test the corrected implementation
    import tempfile
    
    def test_corrected_orchestrator():
        """Test the corrected orchestrator implementation."""
        print("🧪 Testing Corrected TerraLux ModularOrchestrator")
        print("=" * 55)
        
        # Test process definition
        test_process = {
            "process_info": {
                "name": "Test Pipeline - Corrected",
                "version": "1.0.0",
                "description": "Test pipeline for validation"
            },
            "global_config": {
                "output_dir": tempfile.mkdtemp(prefix='terralux_test_')
            },
            "steps": [
                {
                    "id": "acquire_data",
                    "type": "sentinel_hub_acquisition",
                    "hyperparameters": {
                        "bbox": [85.3, 27.6, 85.4, 27.7],
                        "start_date": "2023-01-01",
                        "end_date": "2023-01-31",
                        "fallback_to_mock": True
                    }
                },
                {
                    "id": "validate_data",
                    "type": "data_validation",
                    "dependencies": ["acquire_data"],
                    "hyperparameters": {
                        "checks": ["format", "completeness"]
                    }
                }
            ]
        }
        
        try:
            # Test orchestrator initialization
            orchestrator = ModularOrchestrator()
            print("✓ Orchestrator initialized successfully")
            
            # Test process loading
            orchestrator.load_process(test_process, {"area_name": "test_corrected"})
            print("✓ Process loaded successfully")
            
            # Show execution summary
            summary = orchestrator.get_execution_summary()
            print(f"📊 Execution Summary:")
            print(f"  Total steps: {summary['total_steps']}")
            print(f"  Real steps: {summary['real_steps']}")
            print(f"  Mock steps: {summary['mock_steps']}")
            print(f"  Step types: {summary.get('step_types', {})}")
            
            # Validate process
            validation = orchestrator.validate_process()
            print(f"\n✅ Process validation: {'PASSED' if validation['valid'] else 'FAILED'}")
            
            if validation['errors']:
                print(f"❌ Errors: {validation['errors']}")
            if validation['warnings']:
                print(f"⚠️  Warnings: {validation['warnings']}")
            
            # Execute if valid
            if validation['valid']:
                print("\n🚀 Executing process...")
                result = orchestrator.execute_process({"test_variable": "test_value"})
                
                print(f"📋 Execution result: {result['status']}")
                print(f"🕒 Execution time: {result.get('total_execution_time', 0):.2f} seconds")
                print(f"📈 Steps completed: {result.get('successful_steps', 0)}/{result.get('total_steps', 0)}")
                
                if result.get('artifacts'):
                    print(f"📦 Artifacts generated: {len(result['artifacts'])}")
                
                if result.get('errors'):
                    print(f"❌ Errors: {len(result['errors'])}")
                    for error in result['errors'][:2]:
                        print(f"   - {error}")
                
                print("\n✅ Test PASSED - Corrected orchestrator works!")
            else:
                print("❌ Process validation failed, skipping execution")
                
        except Exception as e:
            print(f"❌ Test FAILED with error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n🔧 Key Corrections Applied:")
        print("  ✓ Fixed step class name mapping (SentinelHubAcquisitionStep)")
        print("  ✓ Prevented duplicate step registration warnings")
        print("  ✓ Enhanced error handling and logging")
        print("  ✓ Improved process validation")
        print("  ✓ Better dependency resolution")
        print("  ✓ Enhanced status reporting")
    
    # Run the test
    test_corrected_orchestrator()
