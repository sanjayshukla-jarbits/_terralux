"""
JSON Process Loader with Minimal Validation
Fail-fast implementation focusing on essential process loading functionality.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import copy


class ProcessValidationError(Exception):
    """Custom exception for process validation errors."""
    pass


class ProcessLoader:
    """
    Loads and validates JSON process definitions with minimal validation
    for fail-fast development approach.
    """
    
    def __init__(self, enable_strict_validation: bool = False):
        """
        Initialize process loader.
        
        Args:
            enable_strict_validation: Enable comprehensive validation (slower)
        """
        self.logger = logging.getLogger('ProcessLoader')
        self.enable_strict_validation = enable_strict_validation
        
        # Template variable pattern (e.g., {variable_name})
        self.template_pattern = re.compile(r'\{([^}]+)\}')
        
        # Cache for loaded processes
        self._process_cache = {}
    
    def load_process(self, 
                    process_path: Union[str, Path, Dict[str, Any]], 
                    template_variables: Optional[Dict[str, Any]] = None,
                    use_cache: bool = True) -> Dict[str, Any]:
        """
        Load process definition from JSON file or dictionary.
        
        Args:
            process_path: Path to JSON file or process dictionary
            template_variables: Variables for template substitution
            use_cache: Whether to use cached processes
            
        Returns:
            Validated and processed process definition
        """
        # Handle different input types
        if isinstance(process_path, dict):
            process_definition = copy.deepcopy(process_path)
            cache_key = f"dict_{hash(str(process_path))}"
        else:
            process_path = Path(process_path)
            cache_key = str(process_path.absolute())
            
            # Check cache first
            if use_cache and cache_key in self._process_cache:
                self.logger.debug(f"Using cached process: {cache_key}")
                process_definition = copy.deepcopy(self._process_cache[cache_key])
            else:
                process_definition = self._load_from_file(process_path)
                if use_cache:
                    self._process_cache[cache_key] = copy.deepcopy(process_definition)
        
        # Apply template variable substitution
        if template_variables:
            process_definition = self._substitute_template_variables(
                process_definition, template_variables
            )
        
        # Validate the process
        self._validate_process(process_definition)
        
        # Add metadata
        process_definition = self._add_metadata(process_definition, process_path)
        
        self.logger.info(f"Successfully loaded process: {process_definition['process_info']['name']}")
        return process_definition
    
    def _load_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Load process definition from JSON file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Process file not found: {file_path}")
        
        if not file_path.suffix.lower() == '.json':
            raise ValueError(f"Process file must be JSON format: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                process_definition = json.load(f)
            
            self.logger.debug(f"Loaded process from file: {file_path}")
            return process_definition
            
        except json.JSONDecodeError as e:
            raise ProcessValidationError(f"Invalid JSON in process file {file_path}: {e}")
        except Exception as e:
            raise ProcessValidationError(f"Failed to load process file {file_path}: {e}")
    
    def _substitute_template_variables(self, 
                                     obj: Any, 
                                     variables: Dict[str, Any]) -> Any:
        """
        Recursively substitute template variables in the process definition.
        
        Supports:
        - Simple substitution: {variable_name}
        - Nested object traversal
        - Type preservation for non-string values
        """
        if isinstance(obj, dict):
            return {
                key: self._substitute_template_variables(value, variables) 
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [
                self._substitute_template_variables(item, variables) 
                for item in obj
            ]
        elif isinstance(obj, str):
            return self._substitute_string_template(obj, variables)
        else:
            return obj
    
    def _substitute_string_template(self, 
                                  template_string: str, 
                                  variables: Dict[str, Any]) -> Any:
        """
        Substitute variables in a template string.
        
        Examples:
        - "{bbox}" -> [85.3, 27.6, 85.4, 27.7]
        - "/data/{area_name}/output.tif" -> "/data/nepal_test/output.tif"
        - "{start_date}_to_{end_date}" -> "2023-01-01_to_2023-12-31"
        """
        # Find all template variables in the string
        template_vars = self.template_pattern.findall(template_string)
        
        if not template_vars:
            return template_string
        
        # If the entire string is a single template variable, return the actual value
        if len(template_vars) == 1 and template_string == f"{{{template_vars[0]}}}":
            var_name = template_vars[0]
            if var_name in variables:
                return variables[var_name]
            else:
                self.logger.warning(f"Template variable '{var_name}' not found in variables")
                return template_string
        
        # Multiple variables or partial substitution - do string replacement
        result = template_string
        for var_name in template_vars:
            if var_name in variables:
                placeholder = f"{{{var_name}}}"
                value = variables[var_name]
                # Convert value to string for substitution
                result = result.replace(placeholder, str(value))
            else:
                self.logger.warning(f"Template variable '{var_name}' not found in variables")
        
        return result
    
    def _validate_process(self, process_definition: Dict[str, Any]) -> None:
        """
        Validate process definition with minimal validation for fail-fast approach.
        """
        # Essential validation
        self._validate_required_fields(process_definition)
        self._validate_process_info(process_definition)
        self._validate_steps(process_definition)
        
        # Optional strict validation
        if self.enable_strict_validation:
            self._validate_dependencies(process_definition)
            self._validate_step_configurations(process_definition)
    
    def _validate_required_fields(self, process_def: Dict[str, Any]) -> None:
        """Validate that required top-level fields are present."""
        required_fields = ['process_info', 'steps']
        
        for field in required_fields:
            if field not in process_def:
                raise ProcessValidationError(f"Missing required field: '{field}'")
            
        if not isinstance(process_def['steps'], list):
            raise ProcessValidationError("'steps' must be a list")
        
        if len(process_def['steps']) == 0:
            raise ProcessValidationError("Process must contain at least one step")
    
    def _validate_process_info(self, process_def: Dict[str, Any]) -> None:
        """Validate process_info section."""
        process_info = process_def['process_info']
        
        if not isinstance(process_info, dict):
            raise ProcessValidationError("'process_info' must be a dictionary")
        
        # Name is required
        if 'name' not in process_info:
            raise ProcessValidationError("'process_info.name' is required")
        
        if not isinstance(process_info['name'], str) or not process_info['name'].strip():
            raise ProcessValidationError("'process_info.name' must be a non-empty string")
    
    def _validate_steps(self, process_def: Dict[str, Any]) -> None:
        """Validate steps array."""
        steps = process_def['steps']
        step_ids = set()
        
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ProcessValidationError(f"Step {i} must be a dictionary")
            
            # Required fields for each step
            required_step_fields = ['id', 'type']
            for field in required_step_fields:
                if field not in step:
                    raise ProcessValidationError(f"Step {i} missing required field: '{field}'")
            
            # Validate step ID
            step_id = step['id']
            if not isinstance(step_id, str) or not step_id.strip():
                raise ProcessValidationError(f"Step {i} 'id' must be a non-empty string")
            
            if step_id in step_ids:
                raise ProcessValidationError(f"Duplicate step ID: '{step_id}'")
            step_ids.add(step_id)
            
            # Validate step type
            if not isinstance(step['type'], str) or not step['type'].strip():
                raise ProcessValidationError(f"Step '{step_id}' 'type' must be a non-empty string")
    
    def _validate_dependencies(self, process_def: Dict[str, Any]) -> None:
        """Validate step dependencies (strict validation only)."""
        steps = process_def['steps']
        step_ids = {step['id'] for step in steps}
        
        for step in steps:
            dependencies = step.get('dependencies', [])
            if not isinstance(dependencies, list):
                raise ProcessValidationError(
                    f"Step '{step['id']}' dependencies must be a list"
                )
            
            for dep in dependencies:
                if not isinstance(dep, str):
                    raise ProcessValidationError(
                        f"Step '{step['id']}' dependency must be a string: {dep}"
                    )
                
                if dep not in step_ids:
                    raise ProcessValidationError(
                        f"Step '{step['id']}' depends on non-existent step: '{dep}'"
                    )
                
                if dep == step['id']:
                    raise ProcessValidationError(
                        f"Step '{step['id']}' cannot depend on itself"
                    )
        
        # Check for circular dependencies
        if self._has_circular_dependencies(steps):
            raise ProcessValidationError("Circular dependencies detected in process")
    
    def _validate_step_configurations(self, process_def: Dict[str, Any]) -> None:
        """Validate step-specific configurations (strict validation only)."""
        for step in process_def['steps']:
            step_id = step['id']
            
            # Validate hyperparameters if present
            if 'hyperparameters' in step:
                hyperparams = step['hyperparameters']
                if not isinstance(hyperparams, dict):
                    raise ProcessValidationError(
                        f"Step '{step_id}' hyperparameters must be a dictionary"
                    )
            
            # Validate inputs/outputs if present
            for io_type in ['inputs', 'outputs']:
                if io_type in step:
                    io_config = step[io_type]
                    if not isinstance(io_config, dict):
                        raise ProcessValidationError(
                            f"Step '{step_id}' {io_type} must be a dictionary"
                        )
    
    def _has_circular_dependencies(self, steps: List[Dict[str, Any]]) -> bool:
        """Check for circular dependencies using DFS."""
        # Build adjacency list
        graph = {}
        for step in steps:
            step_id = step['id']
            dependencies = step.get('dependencies', [])
            graph[step_id] = dependencies
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for step_id in graph:
            if step_id not in visited:
                if has_cycle(step_id):
                    return True
        
        return False
    
    def _add_metadata(self, 
                     process_def: Dict[str, Any], 
                     source_path: Union[str, Path, Dict]) -> Dict[str, Any]:
        """Add metadata to the process definition."""
        # Add loading metadata
        if 'metadata' not in process_def:
            process_def['metadata'] = {}
        
        process_def['metadata'].update({
            'loaded_at': datetime.now().isoformat(),
            'source': str(source_path) if not isinstance(source_path, dict) else 'dictionary',
            'loader_version': '1.0.0'
        })
        
        # Ensure process_info has required fields
        if 'version' not in process_def['process_info']:
            process_def['process_info']['version'] = '1.0.0'
        
        if 'description' not in process_def['process_info']:
            process_def['process_info']['description'] = 'No description provided'
        
        return process_def
    
    def validate_template_variables(self, 
                                  process_def: Dict[str, Any],
                                  variables: Dict[str, Any]) -> List[str]:
        """
        Validate that all required template variables are provided.
        
        Returns:
            List of missing template variables
        """
        required_vars = self._extract_template_variables(process_def)
        provided_vars = set(variables.keys())
        missing_vars = required_vars - provided_vars
        
        return list(missing_vars)
    
    def _extract_template_variables(self, obj: Any) -> set:
        """Extract all template variables from the process definition."""
        variables = set()
        
        if isinstance(obj, dict):
            for value in obj.values():
                variables.update(self._extract_template_variables(value))
        elif isinstance(obj, list):
            for item in obj:
                variables.update(self._extract_template_variables(item))
        elif isinstance(obj, str):
            template_vars = self.template_pattern.findall(obj)
            variables.update(template_vars)
        
        return variables
    
    def get_process_summary(self, process_def: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of the process definition."""
        steps = process_def['steps']
        
        # Count step types
        step_types = {}
        for step in steps:
            step_type = step['type']
            step_types[step_type] = step_types.get(step_type, 0) + 1
        
        # Extract template variables
        template_vars = self._extract_template_variables(process_def)
        
        return {
            'name': process_def['process_info']['name'],
            'version': process_def['process_info'].get('version', 'unknown'),
            'description': process_def['process_info'].get('description', ''),
            'total_steps': len(steps),
            'step_types': step_types,
            'template_variables': list(template_vars),
            'has_global_config': 'global_config' in process_def
        }
    
    def clear_cache(self) -> None:
        """Clear the process cache."""
        self._process_cache.clear()
        self.logger.debug("Process cache cleared")


# Utility functions
def load_process_simple(process_path: Union[str, Path], 
                       **template_vars) -> Dict[str, Any]:
    """
    Convenience function to load a process with template variables.
    
    Args:
        process_path: Path to the process JSON file
        **template_vars: Template variables as keyword arguments
        
    Returns:
        Loaded and validated process definition
    """
    loader = ProcessLoader()
    return loader.load_process(process_path, template_vars)


def validate_process_file(process_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a process file and return validation results.
    
    Args:
        process_path: Path to the process JSON file
        
    Returns:
        Validation results with status and any errors
    """
    try:
        loader = ProcessLoader(enable_strict_validation=True)
        process_def = loader.load_process(process_path)
        
        return {
            'valid': True,
            'process_summary': loader.get_process_summary(process_def),
            'errors': [],
            'warnings': []
        }
    
    except Exception as e:
        return {
            'valid': False,
            'errors': [str(e)],
            'warnings': []
        }


if __name__ == "__main__":
    # Quick test of the process loader
    import tempfile
    
    # Create a test process definition
    test_process = {
        "process_info": {
            "name": "test_data_acquisition",
            "version": "1.0.0",
            "description": "Test process for data acquisition"
        },
        "global_config": {
            "output_directory": "{output_dir}",
            "temp_directory": "{temp_dir}"
        },
        "steps": [
            {
                "id": "sentinel_acquisition",
                "type": "sentinel_hub_acquisition",
                "hyperparameters": {
                    "bbox": "{bbox}",
                    "date_range": ["{start_date}", "{end_date}"],
                    "collection": "sentinel-2-l2a"
                },
                "outputs": {
                    "imagery_data": {
                        "type": "raster",
                        "path": "{output_dir}/sentinel_data.tif"
                    }
                }
            },
            {
                "id": "dem_acquisition", 
                "type": "dem_acquisition",
                "dependencies": ["sentinel_acquisition"],
                "hyperparameters": {
                    "bbox": "{bbox}",
                    "resolution": 30,
                    "source": "SRTM"
                },
                "outputs": {
                    "dem_data": {
                        "type": "raster", 
                        "path": "{output_dir}/dem_data.tif"
                    }
                }
            }
        ]
    }
    
    print("Testing ProcessLoader...")
    
    # Test 1: Load from dictionary
    loader = ProcessLoader()
    
    template_vars = {
        "bbox": [85.3, 27.6, 85.4, 27.7],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "output_dir": "/tmp/test_output",
        "temp_dir": "/tmp/test_temp"
    }
    
    try:
        loaded_process = loader.load_process(test_process, template_vars)
        print("✓ Successfully loaded process from dictionary")
        
        # Check template substitution
        sentinel_step = loaded_process['steps'][0]
        bbox_value = sentinel_step['hyperparameters']['bbox']
        print(f"✓ Template substitution: bbox = {bbox_value}")
        
        # Get process summary
        summary = loader.get_process_summary(loaded_process)
        print(f"✓ Process summary: {summary['total_steps']} steps, {len(summary['step_types'])} types")
        
    except Exception as e:
        print(f"✗ Error loading process: {e}")
    
    # Test 2: Save to file and reload
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_process, f, indent=2)
            temp_file = f.name
        
        # Load from file
        file_loaded = loader.load_process(temp_file, template_vars)
        print("✓ Successfully loaded process from file")
        
        # Test validation
        validation_result = validate_process_file(temp_file)
        if validation_result['valid']:
            print("✓ Process file validation passed")
        else:
            print(f"✗ Process file validation failed: {validation_result['errors']}")
        
        # Cleanup
        import os
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"✗ Error with file operations: {e}")
    
    print("ProcessLoader test completed!")
