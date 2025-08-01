"""
Configuration Utilities for Orchestrator
========================================

This module provides configuration management utilities for the orchestrator system,
including loading, saving, merging, and validating configuration files.

Key Features:
- Multi-format configuration support (JSON, YAML, TOML)
- Configuration validation and schema checking
- Configuration merging and inheritance
- Environment variable substitution
- Default configuration templates
- Nested configuration access utilities
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import re
from copy import deepcopy

# Configure logger
logger = logging.getLogger(__name__)

# Optional imports for different config formats
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import toml
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

# Default configuration templates
DEFAULT_ORCHESTRATOR_CONFIG = {
    "version": "1.0.0",
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "logs/orchestrator.log"
    },
    "execution": {
        "parallel_steps": False,
        "max_workers": 4,
        "timeout_seconds": 3600,
        "retry_failed_steps": False,
        "continue_on_error": False
    },
    "storage": {
        "base_output_dir": "outputs",
        "cache_dir": "cache",
        "temp_dir": "temp",
        "cleanup_temp": True
    },
    "monitoring": {
        "track_performance": True,
        "track_resources": True,
        "log_memory_usage": True
    }
}

DEFAULT_STEP_CONFIG = {
    "timeout_seconds": 300,
    "retry_count": 0,
    "ignore_errors": False,
    "cache_results": True,
    "log_level": "INFO"
}


def load_config(config_path: Union[str, Path], 
               format: Optional[str] = None,
               encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        format: Config format ('json', 'yaml', 'toml') - auto-detected if None
        encoding: File encoding
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> config = load_config('config.json')
        >>> print(f"Version: {config.get('version', 'unknown')}")
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    # Auto-detect format from extension
    if format is None:
        format = _detect_config_format(path)
    
    try:
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
        
        # Substitute environment variables
        content = _substitute_env_vars(content)
        
        # Parse based on format
        if format == 'json':
            config = json.loads(content)
        elif format == 'yaml' and YAML_AVAILABLE:
            config = yaml.safe_load(content)
        elif format == 'toml' and TOML_AVAILABLE:
            config = toml.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {format}")
        
        logger.debug(f"Loaded configuration from {path} ({format})")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {path}: {e}")
        raise


def save_config(config: Dict[str, Any], 
               config_path: Union[str, Path],
               format: Optional[str] = None,
               encoding: str = 'utf-8',
               indent: int = 2) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save configuration file
        format: Config format ('json', 'yaml', 'toml') - auto-detected if None
        encoding: File encoding
        indent: Indentation for formatted output
        
    Example:
        >>> config = {"version": "1.0.0", "debug": True}
        >>> save_config(config, 'output_config.json')
    """
    path = Path(config_path)
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect format from extension
    if format is None:
        format = _detect_config_format(path)
    
    try:
        with open(path, 'w', encoding=encoding) as f:
            if format == 'json':
                json.dump(config, f, indent=indent, default=str)
            elif format == 'yaml' and YAML_AVAILABLE:
                yaml.dump(config, f, default_flow_style=False, indent=indent)
            elif format == 'toml' and TOML_AVAILABLE:
                toml.dump(config, f)
            else:
                raise ValueError(f"Unsupported config format: {format}")
        
        logger.debug(f"Saved configuration to {path} ({format})")
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {path}: {e}")
        raise


def merge_configs(*configs: Dict[str, Any], 
                 strategy: str = 'deep') -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        strategy: Merge strategy ('shallow', 'deep', 'override')
        
    Returns:
        Merged configuration dictionary
        
    Example:
        >>> base_config = {"a": 1, "b": {"x": 10}}
        >>> user_config = {"b": {"y": 20}, "c": 3}
        >>> merged = merge_configs(base_config, user_config)
        >>> print(merged)  # {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}
    """
    if not configs:
        return {}
    
    if len(configs) == 1:
        return deepcopy(configs[0])
    
    result = deepcopy(configs[0])
    
    for config in configs[1:]:
        if strategy == 'shallow':
            result.update(config)
        elif strategy == 'deep':
            result = _deep_merge(result, config)
        elif strategy == 'override':
            result = deepcopy(config)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
    
    logger.debug(f"Merged {len(configs)} configurations using {strategy} strategy")
    return result


def validate_config(config: Dict[str, Any], 
                   schema: Optional[Dict[str, Any]] = None,
                   required_keys: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        schema: JSON schema for validation (optional)
        required_keys: List of required keys (optional)
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> config = {"version": "1.0.0", "debug": True}
        >>> is_valid, errors = validate_config(config, required_keys=["version"])
        >>> print(f"Valid: {is_valid}")
    """
    errors = []
    
    if not isinstance(config, dict):
        errors.append("Configuration must be a dictionary")
        return False, errors
    
    # Check required keys
    if required_keys:
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            errors.extend([f"Missing required key: {key}" for key in missing_keys])
    
    # JSON schema validation
    if schema:
        try:
            import jsonschema
            jsonschema.validate(config, schema)
        except ImportError:
            logger.warning("jsonschema not available for validation")
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
    
    # Basic type validation for common orchestrator config keys
    type_checks = {
        'version': str,
        'debug': bool,
        'parallel_steps': bool,
        'max_workers': int,
        'timeout_seconds': (int, float)
    }
    
    for key, expected_type in type_checks.items():
        if key in config:
            if not isinstance(config[key], expected_type):
                errors.append(f"Key '{key}' should be of type {expected_type.__name__}")
    
    is_valid = len(errors) == 0
    if is_valid:
        logger.debug("Configuration validation passed")
    else:
        logger.warning(f"Configuration validation failed: {errors}")
    
    return is_valid, errors


def get_config_value(config: Dict[str, Any], 
                    key_path: str, 
                    default: Any = None,
                    separator: str = '.') -> Any:
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'logging.level')
        default: Default value if key not found
        separator: Key path separator
        
    Returns:
        Configuration value or default
        
    Example:
        >>> config = {"logging": {"level": "INFO", "file": "app.log"}}
        >>> level = get_config_value(config, "logging.level")
        >>> print(level)  # "INFO"
    """
    keys = key_path.split(separator)
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        logger.debug(f"Config key not found: {key_path}, using default: {default}")
        return default


def set_config_value(config: Dict[str, Any], 
                    key_path: str, 
                    value: Any,
                    separator: str = '.') -> None:
    """
    Set nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary to modify
        key_path: Dot-separated key path (e.g., 'logging.level')
        value: Value to set
        separator: Key path separator
        
    Example:
        >>> config = {"logging": {}}
        >>> set_config_value(config, "logging.level", "DEBUG")
        >>> print(config)  # {"logging": {"level": "DEBUG"}}
    """
    keys = key_path.split(separator)
    current = config
    
    # Navigate to parent of target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value
    logger.debug(f"Set config value: {key_path} = {value}")


def create_default_config(config_type: str = 'orchestrator') -> Dict[str, Any]:
    """
    Create default configuration for specified type.
    
    Args:
        config_type: Type of configuration ('orchestrator', 'step')
        
    Returns:
        Default configuration dictionary
        
    Example:
        >>> default_config = create_default_config('orchestrator')
        >>> print(default_config['version'])
    """
    if config_type == 'orchestrator':
        return deepcopy(DEFAULT_ORCHESTRATOR_CONFIG)
    elif config_type == 'step':
        return deepcopy(DEFAULT_STEP_CONFIG)
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def expand_config_paths(config: Dict[str, Any], 
                       base_path: Union[str, Path] = '.') -> Dict[str, Any]:
    """
    Expand relative paths in configuration to absolute paths.
    
    Args:
        config: Configuration dictionary
        base_path: Base path for relative path expansion
        
    Returns:
        Configuration with expanded paths
        
    Example:
        >>> config = {"output_dir": "outputs", "log_file": "logs/app.log"}
        >>> expanded = expand_config_paths(config, "/home/user/project")
        >>> print(expanded["output_dir"])  # "/home/user/project/outputs"
    """
    result = deepcopy(config)
    base = Path(base_path).resolve()
    
    def expand_paths(obj, current_path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                path_key = f"{current_path}.{key}" if current_path else key
                
                # Common path keys that should be expanded
                path_keys = ['dir', 'path', 'file', 'output', 'input', 'cache', 'temp', 'log']
                
                if any(path_key_part in key.lower() for path_key_part in path_keys):
                    if isinstance(value, str) and not Path(value).is_absolute():
                        obj[key] = str(base / value)
                elif isinstance(value, (dict, list)):
                    expand_paths(value, path_key)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    expand_paths(item, f"{current_path}[{i}]")
    
    expand_paths(result)
    logger.debug(f"Expanded configuration paths relative to: {base}")
    return result


def validate_orchestrator_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate orchestrator-specific configuration.
    
    Args:
        config: Orchestrator configuration to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> config = load_config('orchestrator_config.json')
        >>> is_valid, errors = validate_orchestrator_config(config)
        >>> if not is_valid:
        ...     print("Configuration errors:", errors)
    """
    errors = []
    
    # Required sections
    required_sections = ['version']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate execution settings
    if 'execution' in config:
        execution = config['execution']
        
        if 'max_workers' in execution:
            max_workers = execution['max_workers']
            if not isinstance(max_workers, int) or max_workers < 1:
                errors.append("max_workers must be a positive integer")
        
        if 'timeout_seconds' in execution:
            timeout = execution['timeout_seconds']
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                errors.append("timeout_seconds must be a positive number")
    
    # Validate storage settings
    if 'storage' in config:
        storage = config['storage']
        
        # Check that directories don't conflict
        dirs = ['base_output_dir', 'cache_dir', 'temp_dir']
        dir_values = [storage.get(d) for d in dirs if d in storage]
        
        if len(dir_values) != len(set(dir_values)):
            errors.append("Storage directories should not overlap")
    
    # Validate logging settings
    if 'logging' in config:
        logging_config = config['logging']
        
        if 'level' in logging_config:
            level = logging_config['level']
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if level not in valid_levels:
                errors.append(f"Invalid logging level: {level}. Must be one of {valid_levels}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def _detect_config_format(path: Path) -> str:
    """Detect configuration format from file extension."""
    suffix = path.suffix.lower()
    
    if suffix == '.json':
        return 'json'
    elif suffix in ['.yaml', '.yml']:
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for YAML configuration files")
        return 'yaml'
    elif suffix == '.toml':
        if not TOML_AVAILABLE:
            raise ImportError("toml required for TOML configuration files")
        return 'toml'
    else:
        # Default to JSON
        return 'json'


def _substitute_env_vars(content: str) -> str:
    """Substitute environment variables in configuration content."""
    # Pattern: ${VAR_NAME} or ${VAR_NAME:default_value}
    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
    
    def replace_var(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) is not None else ''
        
        env_value = os.getenv(var_name)
        if env_value is not None:
            return env_value
        elif default_value:
            return default_value
        else:
            logger.warning(f"Environment variable {var_name} not found and no default provided")
            return match.group(0)  # Return original if not found
    
    return re.sub(pattern, replace_var, content)


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = deepcopy(dict1)
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    
    return result


# Export main functions
__all__ = [
    'load_config', 'save_config', 'merge_configs', 'validate_config',
    'get_config_value', 'set_config_value', 'create_default_config',
    'expand_config_paths', 'validate_orchestrator_config'
]
