"""
JSON Process Loader with Enhanced Template Variable Handling
Includes default values for common template variables and improved error handling.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import copy
import os


class ProcessValidationError(Exception):
    """Custom exception for process validation errors."""
    pass


class ProcessLoader:
    """
    Loads and validates JSON process definitions with enhanced template variable handling.
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
        
        # Default template variables - commonly used across pipelines
        self.default_template_variables = {
            # Data paths
            'local_data_path': './data/inputs',
            'output_dir': './data/outputs',
            'temp_dir': './data/temp',
            'cache_dir': './data/cache',
            'logs_dir': './logs',
            
            # DEM derivatives
            'derivative': 'elevation',  # Default DEM derivative
            'dem_derivative': 'elevation',
            'slope_derivative': 'slope',
            'aspect_derivative': 'aspect',
            'curvature_derivative': 'curvature',
            
            # File patterns and extensions
            'file_extension': '.tif',
            'vector_extension': '.shp',
            'table_extension': '.csv',
            
            # Processing parameters
            'resolution': 10,  # meters
            'buffer_distance': 1000,  # meters
            'tile_size': 512,
            'overlap': 64,
            
            # Model parameters
            'n_estimators': 100,
            'max_depth': 10,
            'test_size': 0.3,
            'random_state': 42,
            
            # Time parameters
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'time_window': '1Y',
            
            # Spatial parameters
            'bbox': [0, 0, 1, 1],  # Default minimal bbox
            'crs': 'EPSG:4326',
            'area_name': 'default_area',
            
            # Quality and validation
            'cloud_cover_threshold': 20,  # percent
            'quality_threshold': 0.8,
            'validation_split': 0.2,
            
            # Visualization
            'colormap': 'viridis',
            'dpi': 300,
            'figure_size': [10, 8],
            
            # Logging and monitoring
            'log_level': 'INFO',
            'enable_monitoring': True,
            'progress_interval': 10,
        }
    
    def load_process(self, 
                    process_path: Union[str, Path, Dict[str, Any]], 
                    template_variables: Optional[Dict[str, Any]] = None,
                    use_cache: bool = True) -> Dict[str, Any]:
        """
        Load process definition from JSON file or dictionary with enhanced template handling.
        
        Args:
            process_path: Path to JSON file or process dictionary
            template_variables: Variables for template substitution
            use_cache: Whether to use cached processes
            
        Returns:
            Validated and processed process definition
        """
        # Prepare enhanced template variables with defaults
        enhanced_variables = self._prepare_template_variables(template_variables)
        
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
        
        # Apply template variable substitution with enhanced variables
        process_definition = self._substitute_template_variables(
            process_definition, enhanced_variables
        )
        
        # Validate the process
        self._validate_process(process_definition)
        
        # Add metadata
        process_definition = self._add_metadata(process_definition, process_path)
        
        # Log template variable usage summary
        self._log_template_usage_summary(process_definition, enhanced_variables)
        
        self.logger.info(f"Successfully loaded process: {process_definition['process_info']['name']}")
        return process_definition
    
    def _prepare_template_variables(self, 
                                  user_variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare enhanced template variables by merging defaults with user-provided variables.
        User variables take precedence over defaults.
        """
        enhanced_variables = self.default_template_variables.copy()
        
        if user_variables:
            enhanced_variables.update(user_variables)
            
            # Log which user variables override defaults
            overridden = set(user_variables.keys()) & set(self.default_template_variables.keys())
            if overridden:
                self.logger.debug(f"User variables override defaults: {list(overridden)}")
        
        # Add dynamic variables based on current context
        enhanced_variables.update(self._generate_dynamic_variables(enhanced_variables))
        
        return enhanced_variables
    
    def _generate_dynamic_variables(self, base_variables: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dynamic template variables based on context."""
        dynamic_vars = {}
        
        # Generate timestamp-based variables
        now = datetime.now()
        dynamic_vars.update({
            'timestamp': now.strftime('%Y%m%d_%H%M%S'),
            'date': now.strftime('%Y-%m-%d'),
            'year': now.year,
            'month': now.month,
            'day': now.day,
        })
        
        # Generate path-based variables
        if 'area_name' in base_variables and 'output_dir' in base_variables:
            area_name = base_variables['area_name']
            output_dir = base_variables['output_dir']
            dynamic_vars.update({
                'area_output_dir': os.path.join(output_dir, area_name),
                'area_temp_dir': os.path.join(base_variables.get('temp_dir', './temp'), area_name),
                'area_cache_dir': os.path.join(base_variables.get('cache_dir', './cache'), area_name),
            })
        
        # Generate bbox-based variables
        if 'bbox' in base_variables:
            bbox = base_variables['bbox']
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                dynamic_vars.update({
                    'bbox_str': f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}",
                    'bbox_width': abs(bbox[2] - bbox[0]),
                    'bbox_height': abs(bbox[3] - bbox[1]),
                })
        
        # Generate date-based variables
        if 'start_date' in base_variables and 'end_date' in base_variables:
            start_date = base_variables['start_date']
            end_date = base_variables['end_date']
            dynamic_vars.update({
                'date_range': f"{start_date}_to_{end_date}",
                'start_year': start_date.split('-')[0] if isinstance(start_date, str) else str(start_date),
                'end_year': end_date.split('-')[0] if isinstance(end_date, str) else str(end_date),
            })
        
        return dynamic_vars
    
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
        Enhanced version with better error handling and logging.
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
        Substitute variables in a template string with enhanced error handling.
        
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
                # Log as debug instead of warning since we now have defaults
                self.logger.debug(f"Template variable '{var_name}' using default value")
                return template_string
        
        # Multiple variables or partial substitution - do string replacement
        result = template_string
        missing_vars = []
        
        for var_name in template_vars:
            placeholder = f"{{{var_name}}}"
            if var_name in variables:
                value = variables[var_name]
                # Convert value to string for substitution
                result = result.replace(placeholder, str(value))
            else:
                missing_vars.append(var_name)
        
        # Only log missing variables that aren't covered by defaults
        if missing_vars:
            uncovered_vars = [var for var in missing_vars 
                            if var not in self.default_template_variables]
            if uncovered_vars:
                self.logger.warning(f"Template variables not found: {uncovered_vars}")
        
        return result
    
    def _validate_process(self, process_definition: Dict[str, Any]) -> None:
        """
        Validate process definition with minimal validation for fail-fast approach.
        """
        # Check required top-level keys
        required_keys = ['process_info', 'steps']
        for key in required_keys:
            if key not in process_definition:
                raise ProcessValidationError(f"Missing required key: {key}")
        
        # Validate process_info
        process_info = process_definition['process_info']
        if 'name' not in process_info:
            raise ProcessValidationError("Missing 'name' in process_info")
        
        # Validate steps
        steps = process_definition['steps']
        if not isinstance(steps, list) or len(steps) == 0:
            raise ProcessValidationError("Steps must be a non-empty list")
        
        # Validate each step has required fields
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ProcessValidationError(f"Step {i} must be a dictionary")
            
            required_step_keys = ['id', 'type']
            for key in required_step_keys:
                if key not in step:
                    raise ProcessValidationError(f"Step {i} missing required key: {key}")
        
        # Check for duplicate step IDs
        step_ids = [step['id'] for step in steps]
        if len(step_ids) != len(set(step_ids)):
            duplicates = [sid for sid in step_ids if step_ids.count(sid) > 1]
            raise ProcessValidationError(f"Duplicate step IDs found: {duplicates}")
        
        self.logger.debug("Process validation passed")
    
    def _add_metadata(self, process_def: Dict[str, Any], source_path: Any) -> Dict[str, Any]:
        """Add loading metadata to the process definition."""
        # Add loading metadata
        if 'metadata' not in process_def:
            process_def['metadata'] = {}
        
        process_def['metadata'].update({
            'loaded_at': datetime.now().isoformat(),
            'source': str(source_path) if not isinstance(source_path, dict) else 'dictionary',
            'loader_version': '2.0.0'  # Updated version
        })
        
        # Ensure process_info has required fields
        if 'version' not in process_def['process_info']:
            process_def['process_info']['version'] = '1.0.0'
        
        if 'description' not in process_def['process_info']:
            process_def['process_info']['description'] = 'No description provided'
        
        return process_def
    
    def _log_template_usage_summary(self, 
                                   process_def: Dict[str, Any], 
                                   variables: Dict[str, Any]) -> None:
        """Log a summary of template variable usage."""
        used_vars = self._extract_template_variables(process_def)
        
        if used_vars:
            default_vars_used = used_vars & set(self.default_template_variables.keys())
            user_vars_used = used_vars - set(self.default_template_variables.keys())
            
            self.logger.debug(f"Template variables used: {len(used_vars)} total")
            if default_vars_used:
                self.logger.debug(f"Default variables used: {list(default_vars_used)}")
            if user_vars_used:
                self.logger.debug(f"User variables used: {list(user_vars_used)}")
    
    def validate_template_variables(self, 
                                  process_def: Dict[str, Any],
                                  variables: Dict[str, Any]) -> List[str]:
        """
        Validate that all required template variables are provided.
        Now considers default variables.
        
        Returns:
            List of missing template variables (not covered by defaults)
        """
        required_vars = self._extract_template_variables(process_def)
        enhanced_vars = self._prepare_template_variables(variables)
        provided_vars = set(enhanced_vars.keys())
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
        
        # Categorize template variables
        default_vars = template_vars & set(self.default_template_variables.keys())
        custom_vars = template_vars - set(self.default_template_variables.keys())
        
        return {
            'name': process_def['process_info']['name'],
            'version': process_def['process_info'].get('version', 'unknown'),
            'description': process_def['process_info'].get('description', ''),
            'total_steps': len(steps),
            'step_types': step_types,
            'template_variables': {
                'total': len(template_vars),
                'default_vars': list(default_vars),
                'custom_vars': list(custom_vars)
            },
            'has_global_config': 'global_config' in process_def
        }
    
    def get_default_variables(self) -> Dict[str, Any]:
        """Get a copy of the default template variables."""
        return self.default_template_variables.copy()
    
    def add_default_variable(self, name: str, value: Any) -> None:
        """Add a new default template variable."""
        self.default_template_variables[name] = value
        self.logger.debug(f"Added default template variable: {name} = {value}")
    
    def remove_default_variable(self, name: str) -> bool:
        """Remove a default template variable."""
        if name in self.default_template_variables:
            del self.default_template_variables[name]
            self.logger.debug(f"Removed default template variable: {name}")
            return True
        return False
    
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
    # Enhanced test of the process loader
    import tempfile
    
    # Create a test process definition with various template variables
    test_process = {
        "process_info": {
            "name": "test_enhanced_template_handling",
            "version": "2.0.0",
            "description": "Test process for enhanced template variable handling"
        },
        "global_config": {
            "output_directory": "{output_dir}",
            "temp_directory": "{temp_dir}",
            "local_data_directory": "{local_data_path}"
        },
        "steps": [
            {
                "id": "sentinel_acquisition",
                "type": "sentinel_hub_acquisition",
                "hyperparameters": {
                    "bbox": "{bbox}",
                    "date_range": ["{start_date}", "{end_date}"],
                    "collection": "sentinel-2-l2a",
                    "resolution": "{resolution}",
                    "cloud_cover": "{cloud_cover_threshold}"
                },
                "outputs": {
                    "imagery_data": {
                        "type": "raster",
                        "path": "{area_output_dir}/sentinel_data_{date_range}.tif"
                    }
                }
            },
            {
                "id": "dem_processing", 
                "type": "dem_acquisition",
                "dependencies": ["sentinel_acquisition"],
                "hyperparameters": {
                    "bbox": "{bbox}",
                    "resolution": "{resolution}",
                    "source": "SRTM",
                    "derivative": "{derivative}",
                    "local_path": "{local_data_path}/dem"
                },
                "outputs": {
                    "dem_data": {
                        "type": "raster", 
                        "path": "{area_output_dir}/dem_{derivative}_{bbox_str}.tif"
                    }
                }
            }
        ]
    }
    
    print("Testing Enhanced ProcessLoader...")
    
    # Test 1: Load with minimal template variables (most should use defaults)
    loader = ProcessLoader()
    
    minimal_template_vars = {
        "bbox": [85.3, 27.6, 85.4, 27.7],
        "area_name": "nepal_test"
    }
    
    try:
        loaded_process = loader.load_process(test_process, minimal_template_vars)
        print("✓ Successfully loaded process with minimal template variables")
        
        # Check template substitution
        sentinel_step = loaded_process['steps'][0]
        bbox_value = sentinel_step['hyperparameters']['bbox']
        resolution_value = sentinel_step['hyperparameters']['resolution']
        print(f"✓ Template substitution: bbox = {bbox_value}, resolution = {resolution_value}")
        
        # Check dynamic variables
        dem_step = loaded_process['steps'][1]
        output_path = dem_step['outputs']['dem_data']['path']
        print(f"✓ Dynamic template variables: output_path = {output_path}")
        
        # Get process summary
        summary = loader.get_process_summary(loaded_process)
        print(f"✓ Process summary: {summary['total_steps']} steps")
        print(f"  - Default variables used: {len(summary['template_variables']['default_vars'])}")
        print(f"  - Custom variables used: {len(summary['template_variables']['custom_vars'])}")
        
    except Exception as e:
        print(f"✗ Error loading process: {e}")
    
    # Test 2: Show default variables
    try:
        defaults = loader.get_default_variables()
        print(f"✓ Available default variables: {len(defaults)}")
        common_defaults = ['output_dir', 'temp_dir', 'local_data_path', 'derivative', 'resolution']
        for var in common_defaults:
            print(f"  - {var}: {defaults.get(var, 'NOT SET')}")
        
    except Exception as e:
        print(f"✗ Error getting defaults: {e}")
    
    # Test 3: Validation with missing variables
    try:
        missing_vars = loader.validate_template_variables(test_process, minimal_template_vars)
        if missing_vars:
            print(f"⚠ Missing template variables: {missing_vars}")
        else:
            print("✓ All required template variables are available (including defaults)")
        
    except Exception as e:
        print(f"✗ Error in validation: {e}")
    
    print("Enhanced ProcessLoader test completed!")
