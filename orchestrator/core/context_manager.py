"""
Basic Context Manager for Modular Pipeline Orchestrator
Fail-fast implementation focusing on essential state management functionality.
"""

import os
import json
import logging
import tempfile
import shutil
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading


@dataclass
class StepExecution:
    """Metadata for individual step execution."""
    step_id: str
    step_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, skipped
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    artifacts_created: List[str] = field(default_factory=list)


class PipelineContext:
    """
    Basic context manager for pipeline execution state.
    Manages shared data, artifacts, and execution metadata.
    """
    
    def __init__(self, 
                 pipeline_id: Optional[str] = None,
                 base_output_dir: Optional[Union[str, Path]] = None,
                 global_config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline context.
        
        Args:
            pipeline_id: Unique identifier for this pipeline execution
            base_output_dir: Base directory for outputs
            global_config: Global configuration parameters
        """
        self.pipeline_id = pipeline_id or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.global_config = global_config or {}
        
        # Logger setup
        self.logger = logging.getLogger(f'PipelineContext.{self.pipeline_id}')

        # Setup directories
        self._setup_directories(base_output_dir)
        
        # Core data storage
        self.variables: Dict[str, Any] = {}
        self.artifacts: Dict[str, Any] = {}
        self.step_outputs: Dict[str, Dict[str, Any]] = {}
        
        # Execution tracking
        self.execution_history: List[StepExecution] = []
        self.current_step: Optional[str] = None
        self.start_time = datetime.now()
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info(f"Initialized pipeline context: {self.pipeline_id}")
        
    def _setup_directories(self, base_output_dir: Optional[Union[str, Path]]):
        """Setup working directories for the pipeline."""
        if base_output_dir:
            self.base_output_dir = Path(base_output_dir)
        else:
            self.base_output_dir = Path.cwd() / "outputs"
        
        # Create pipeline-specific directories
        self.output_dir = self.base_output_dir / self.pipeline_id
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f'pipeline_{self.pipeline_id}_'))
        self.cache_dir = self.output_dir / "cache"
        self.logs_dir = self.output_dir / "logs"
        
        # Create directories
        for directory in [self.output_dir, self.cache_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Working directories created:")
        self.logger.info(f"  Output: {self.output_dir}")
        self.logger.info(f"  Temp: {self.temp_dir}")
        self.logger.info(f"  Cache: {self.cache_dir}")
    
    def set_variable(self, key: str, value: Any) -> None:
        """Set a pipeline variable."""
        with self._lock:
            self.variables[key] = value
            self.logger.debug(f"Set variable: {key} = {value}")
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """
        Get a pipeline variable with template substitution support.
        
        Args:
            key: Variable key to retrieve
            default: Default value if key not found
            
        Returns:
            Variable value with template substitution applied
        """
        with self._lock:
            value = self.variables.get(key, default)
            
            # Apply template substitution for string values
            if isinstance(value, str):
                value = self._substitute_templates(value)
            
            return value
    
    def update_variables(self, variables: Dict[str, Any]) -> None:
        """Update multiple variables at once."""
        with self._lock:
            self.variables.update(variables)
            self.logger.debug(f"Updated {len(variables)} variables")
    
    def _substitute_templates(self, template_string: str) -> str:
        """
        Simple template variable substitution.
        Replaces {variable_name} with actual values.
        """
        result = template_string
        for key, value in self.variables.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result
    
    def set_artifact(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store an execution artifact.
        
        Args:
            key: Artifact key
            value: Artifact value (can be file path, data, etc.)
            metadata: Optional metadata about the artifact
        """
        with self._lock:
            artifact_data = {
                'value': value,
                'timestamp': datetime.now(),
                'metadata': metadata or {},
                'created_by_step': self.current_step
            }
            self.artifacts[key] = artifact_data
            
            # Track artifact creation in current step
            if self.current_step and self.execution_history:
                current_execution = self.execution_history[-1]
                if current_execution.step_id == self.current_step:
                    current_execution.artifacts_created.append(key)
            
            self.logger.debug(f"Set artifact: {key}")
    
    def get_artifact(self, key: str, default: Any = None) -> Any:
        """Get an artifact value."""
        with self._lock:
            artifact_data = self.artifacts.get(key)
            if artifact_data:
                return artifact_data['value']
            return default
    
    def get_artifact_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get artifact metadata."""
        with self._lock:
            artifact_data = self.artifacts.get(key)
            if artifact_data:
                return {
                    'timestamp': artifact_data['timestamp'],
                    'metadata': artifact_data['metadata'],
                    'created_by_step': artifact_data['created_by_step']
                }
            return None
    
    def list_artifacts(self) -> List[str]:
        """Get list of all artifact keys."""
        with self._lock:
            return list(self.artifacts.keys())
    
    def set_step_output(self, step_id: str, outputs: Dict[str, Any]) -> None:
        """Store outputs from a step execution."""
        with self._lock:
            self.step_outputs[step_id] = {
                'outputs': outputs,
                'timestamp': datetime.now()
            }
            self.logger.debug(f"Stored outputs for step: {step_id}")
    
    def get_step_output(self, step_id: str, output_key: Optional[str] = None) -> Any:
        """
        Get output from a specific step.
        
        Args:
            step_id: ID of the step that produced the output
            output_key: Specific output key, if None returns all outputs
            
        Returns:
            Step output(s)
        """
        with self._lock:
            step_data = self.step_outputs.get(step_id)
            if not step_data:
                return None
            
            outputs = step_data['outputs']
            if output_key:
                return outputs.get(output_key)
            return outputs
    
    def get_step_dependencies_outputs(self, dependencies: List[str]) -> Dict[str, Any]:
        """Get outputs from multiple dependency steps."""
        dependency_outputs = {}
        for dep_step_id in dependencies:
            outputs = self.get_step_output(dep_step_id)
            if outputs:
                dependency_outputs[dep_step_id] = outputs
        return dependency_outputs
    
    @contextmanager
    def step_execution(self, step_id: str, step_type: str):
        """
        Context manager for step execution tracking.
        
        Usage:
            with context.step_execution('step1', 'data_acquisition'):
                # Execute step logic
                pass
        """
        execution = StepExecution(
            step_id=step_id,
            step_type=step_type,
            start_time=datetime.now()
        )
        
        with self._lock:
            self.current_step = step_id
            self.execution_history.append(execution)
        
        self.logger.info(f"Started step execution: {step_id} ({step_type})")
        
        try:
            yield execution
            
            # Mark as completed
            with self._lock:
                execution.end_time = datetime.now()
                execution.execution_time = (execution.end_time - execution.start_time).total_seconds()
                execution.status = "completed"
                self.current_step = None
                
            self.logger.info(f"Completed step: {step_id} in {execution.execution_time:.2f}s")
            
        except Exception as e:
            # Mark as failed
            with self._lock:
                execution.end_time = datetime.now()
                execution.execution_time = (execution.end_time - execution.start_time).total_seconds()
                execution.status = "failed"
                execution.error_message = str(e)
                self.current_step = None
                
            self.logger.error(f"Step failed: {step_id} - {str(e)}")
            raise
    
    def skip_step(self, step_id: str, step_type: str, reason: str = "condition not met"):
        """Mark a step as skipped."""
        execution = StepExecution(
            step_id=step_id,
            step_type=step_type,
            start_time=datetime.now(),
            end_time=datetime.now(),
            status="skipped",
            error_message=reason,
            execution_time=0.0
        )
        
        with self._lock:
            self.execution_history.append(execution)
        
        self.logger.info(f"Skipped step: {step_id} - {reason}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution."""
        with self._lock:
            total_time = (datetime.now() - self.start_time).total_seconds()
            
            # Count step statuses
            status_counts = {}
            for execution in self.execution_history:
                status = execution.status
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Calculate total step execution time
            total_step_time = sum(
                exec.execution_time for exec in self.execution_history 
                if exec.execution_time is not None
            )
            
            return {
                'pipeline_id': self.pipeline_id,
                'start_time': self.start_time,
                'total_execution_time': total_time,
                'total_step_time': total_step_time,
                'steps_executed': len(self.execution_history),
                'status_counts': status_counts,
                'artifacts_created': len(self.artifacts),
                'current_step': self.current_step,
                'output_directory': str(self.output_dir)
            }
    
    def get_execution_timeline(self) -> List[Dict[str, Any]]:
        """Get detailed execution timeline."""
        with self._lock:
            timeline = []
            for execution in self.execution_history:
                timeline.append({
                    'step_id': execution.step_id,
                    'step_type': execution.step_type,
                    'start_time': execution.start_time,
                    'end_time': execution.end_time,
                    'status': execution.status,
                    'execution_time': execution.execution_time,
                    'error_message': execution.error_message,
                    'artifacts_created': execution.artifacts_created
                })
            return timeline
    
    def save_execution_report(self, filename: Optional[str] = None) -> Path:
        """Save execution report to file."""
        if not filename:
            filename = f"execution_report_{self.pipeline_id}.json"
        
        report_path = self.output_dir / filename
        
        report_data = {
            'summary': self.get_execution_summary(),
            'timeline': self.get_execution_timeline(),
            'variables': dict(self.variables),
            'artifacts': {
                key: {
                    'value': str(data['value']) if not isinstance(data['value'], (str, int, float, bool)) else data['value'],
                    'timestamp': data['timestamp'].isoformat(),
                    'created_by_step': data['created_by_step'],
                    'metadata': data['metadata']
                }
                for key, data in self.artifacts.items()
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Execution report saved: {report_path}")
        return report_path
    
    def cleanup(self, keep_outputs: bool = True) -> None:
        """
        Cleanup pipeline resources.
        
        Args:
            keep_outputs: Whether to keep output files
        """
        try:
            # Remove temporary directory
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            
            # Optionally remove outputs
            if not keep_outputs and self.output_dir.exists():
                shutil.rmtree(self.output_dir)
                self.logger.info(f"Cleaned up output directory: {self.output_dir}")
                
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # Save execution report on exit
        try:
            self.save_execution_report()
        except Exception as e:
            self.logger.warning(f"Failed to save execution report: {e}")
        
        # Cleanup temporary resources
        self.cleanup(keep_outputs=True)
    
    def __repr__(self) -> str:
        return f"PipelineContext(id={self.pipeline_id}, steps={len(self.execution_history)})"


# Utility functions for context management
def create_context(pipeline_id: Optional[str] = None, 
                  base_output_dir: Optional[Union[str, Path]] = None,
                  **config_vars) -> PipelineContext:
    """
    Convenience function to create a pipeline context.
    
    Args:
        pipeline_id: Unique pipeline identifier
        base_output_dir: Base output directory
        **config_vars: Additional configuration variables
        
    Returns:
        Configured PipelineContext instance
    """
    context = PipelineContext(
        pipeline_id=pipeline_id,
        base_output_dir=base_output_dir,
        global_config=config_vars
    )
    
    # Set common template variables
    context.update_variables({
        'pipeline_id': context.pipeline_id,
        'output_dir': str(context.output_dir),
        'temp_dir': str(context.temp_dir),
        'cache_dir': str(context.cache_dir),
        **config_vars
    })
    
    return context


if __name__ == "__main__":
    # Quick test of context manager
    import time
    
    print("Testing PipelineContext...")
    
    with create_context(pipeline_id="test_pipeline") as context:
        # Test variable management
        context.set_variable("test_var", "test_value")
        context.set_variable("bbox", [85.3, 27.6, 85.4, 27.7])
        
        print(f"Variable: {context.get_variable('test_var')}")
        print(f"Template test: {context.get_variable('output_path', '{output_dir}/results.tif')}")
        
        # Test step execution tracking
        with context.step_execution("test_step", "data_acquisition") as step_exec:
            context.set_artifact("test_data", {"shape": [100, 100], "type": "raster"})
            time.sleep(0.1)  # Simulate processing
        
        context.set_step_output("test_step", {"output_file": "/path/to/output.tif"})
        
        # Test execution summary
        summary = context.get_execution_summary()
        print(f"Execution summary: {summary}")
        
        # Test artifact retrieval
        artifact = context.get_artifact("test_data")
        print(f"Artifact: {artifact}")
    
    print("Context manager test completed!")
