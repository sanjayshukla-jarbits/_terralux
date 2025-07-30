"""
Step Registry System for Modular Pipeline Orchestrator
=====================================================

This module provides the central registry system for managing pipeline step types,
designed with fail-fast principles for rapid development and testing.

Key Features:
- Global step type registration and discovery
- Step factory pattern for instance creation
- Plugin system support
- Validation and introspection capabilities
- Thread-safe operations
- Development and debugging utilities
"""

import logging
import threading
import inspect
from typing import Dict, Type, Any, List, Optional, Set, Callable, Union
from datetime import datetime
from collections import defaultdict
import warnings

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .base_step import BaseStep


class StepRegistrationError(Exception):
    """Custom exception for step registration errors."""
    pass


class StepRegistry:
    """
    Global registry for pipeline step types with fail-fast capabilities.
    
    This registry manages all available step types in the pipeline system,
    providing a central location for step registration, discovery, and instantiation.
    
    Features:
    - Thread-safe registration and access
    - Step validation and introspection
    - Plugin system support
    - Development utilities and debugging
    - Category-based organization
    """
    
    # Class-level storage for registered steps
    _steps: Dict[str, Type['BaseStep']] = {}
    _step_metadata: Dict[str, Dict[str, Any]] = {}
    _categories: Dict[str, Set[str]] = defaultdict(set)
    _aliases: Dict[str, str] = {}
    
    # Thread safety
    _lock = threading.RLock()
    
    # Registry metadata
    _initialized_at = datetime.now()
    _registration_count = 0
    
    # Logger
    _logger = logging.getLogger('StepRegistry')
    
    @classmethod
    def register(cls, 
                step_type: str, 
                step_class: Type['BaseStep'],
                category: Optional[str] = None,
                aliases: Optional[List[str]] = None,
                override: bool = False,
                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a step class with the registry.
        
        Args:
            step_type: Unique identifier for the step type
            step_class: Step class that inherits from BaseStep
            category: Optional category for organization (e.g., 'data_acquisition')
            aliases: Optional list of alternative names for this step type
            override: Whether to override existing registration
            metadata: Optional metadata about the step
            
        Raises:
            StepRegistrationError: If registration fails validation
        """
        with cls._lock:
            # Validate inputs
            cls._validate_registration(step_type, step_class, override)
            
            # Check for existing registration
            if step_type in cls._steps and not override:
                cls._logger.warning(
                    f"Step type '{step_type}' already registered. "
                    f"Use override=True to replace."
                )
                return
            
            # Validate step class
            validation_errors = cls._validate_step_class(step_class)
            if validation_errors:
                raise StepRegistrationError(
                    f"Step class validation failed: {validation_errors}"
                )
            
            # Register the step
            old_class = cls._steps.get(step_type)
            cls._steps[step_type] = step_class
            cls._registration_count += 1
            
            # Store metadata
            step_metadata = {
                'class_name': step_class.__name__,
                'module': step_class.__module__,
                'registered_at': datetime.now(),
                'category': category,
                'aliases': aliases or [],
                'override': override,
                'replaced_class': old_class.__name__ if old_class else None
            }
            
            if metadata:
                step_metadata.update(metadata)
            
            cls._step_metadata[step_type] = step_metadata
            
            # Add to category
            if category:
                cls._categories[category].add(step_type)
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    if alias in cls._aliases and not override:
                        cls._logger.warning(f"Alias '{alias}' already exists")
                    else:
                        cls._aliases[alias] = step_type
            
            # Log registration
            action = "Overrode" if old_class else "Registered"
            cls._logger.info(
                f"{action} step type: '{step_type}' -> {step_class.__name__}"
                f"{f' (category: {category})' if category else ''}"
            )
            
            if aliases:
                cls._logger.debug(f"Registered aliases for '{step_type}': {aliases}")
    
    @classmethod  
    def _validate_registration(cls, 
                             step_type: str, 
                             step_class: Type['BaseStep'], 
                             override: bool) -> None:
        """Validate registration parameters."""
        if not step_type or not isinstance(step_type, str):
            raise StepRegistrationError("Step type must be a non-empty string")
        
        if not inspect.isclass(step_class):
            raise StepRegistrationError("step_class must be a class")
        
        # Check for reserved names
        reserved_names = {'base', 'registry', 'mock', '__base__', '__registry__'}
        if step_type.lower() in reserved_names:
            raise StepRegistrationError(f"Step type '{step_type}' is reserved")
        
        # Validate step_type format (basic alphanumeric + underscore)
        if not step_type.replace('_', '').replace('-', '').isalnum():
            raise StepRegistrationError(
                f"Step type '{step_type}' contains invalid characters"
            )
    
    @classmethod
    def _validate_step_class(cls, step_class: Type['BaseStep']) -> List[str]:
        """
        Validate that a step class properly implements the BaseStep interface.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Import here to avoid circular imports
        try:
            from .base_step import BaseStep
        except ImportError:
            # If BaseStep not available, skip validation
            return []
        
        # Check inheritance
        if not issubclass(step_class, BaseStep):
            errors.append("Step class must inherit from BaseStep")
            return errors  # Return early if not a BaseStep
        
        # Check required methods
        required_methods = ['execute']
        for method_name in required_methods:
            if not hasattr(step_class, method_name):
                errors.append(f"Step class must implement {method_name}() method")
            elif not callable(getattr(step_class, method_name)):
                errors.append(f"{method_name}() must be callable")
        
        # Check method signatures
        try:
            execute_method = getattr(step_class, 'execute')
            sig = inspect.signature(execute_method)
            params = list(sig.parameters.keys())
            
            # Should have at least 'self' and 'context'
            if len(params) < 2:
                errors.append("execute() method must accept 'context' parameter")
            elif 'context' not in params:
                errors.append("execute() method must have 'context' parameter")
                
        except Exception as e:
            errors.append(f"Could not validate execute() signature: {e}")
        
        return errors
    
    @classmethod
    def unregister(cls, step_type: str) -> bool:
        """
        Unregister a step type.
        
        Args:
            step_type: Step type to unregister
            
        Returns:
            True if step was unregistered, False if not found
        """
        with cls._lock:
            if step_type not in cls._steps:
                cls._logger.warning(f"Step type '{step_type}' not found for unregistration")
                return False
            
            # Remove from main registry
            step_class = cls._steps.pop(step_type)
            metadata = cls._step_metadata.pop(step_type, {})
            
            # Remove from category
            category = metadata.get('category')
            if category and category in cls._categories:
                cls._categories[category].discard(step_type)
                if not cls._categories[category]:  # Remove empty category
                    del cls._categories[category]
            
            # Remove aliases
            aliases_to_remove = []
            for alias, target in cls._aliases.items():
                if target == step_type:
                    aliases_to_remove.append(alias)
            
            for alias in aliases_to_remove:
                del cls._aliases[alias]
            
            cls._logger.info(f"Unregistered step type: '{step_type}' ({step_class.__name__})")
            return True
    
    @classmethod
    def get_step_class(cls, step_type: str) -> Type['BaseStep']:
        """
        Get step class by type identifier.
        
        Args:
            step_type: Step type identifier or alias
            
        Returns:
            Step class
            
        Raises:
            ValueError: If step type not found
        """
        with cls._lock:
            # Check direct registration
            if step_type in cls._steps:
                return cls._steps[step_type]
            
            # Check aliases
            if step_type in cls._aliases:
                actual_type = cls._aliases[step_type]
                return cls._steps[actual_type]
            
            # Not found
            available_types = list(cls._steps.keys())
            available_aliases = list(cls._aliases.keys())
            
            raise ValueError(
                f"Unknown step type: '{step_type}'. "
                f"Available types: {available_types}. "
                f"Available aliases: {available_aliases}"
            )
    
    @classmethod
    def create_step(cls, step_config: Dict[str, Any]) -> 'BaseStep':
        """
        Factory method to create step instance from configuration.
        
        Args:
            step_config: Step configuration dictionary with 'id', 'type', etc.
            
        Returns:
            Instantiated step object
            
        Raises:
            ValueError: If configuration is invalid
            StepRegistrationError: If step type not found
        """
        # Validate configuration
        required_fields = ['id', 'type']
        for field in required_fields:
            if field not in step_config:
                raise ValueError(f"Step configuration missing required field: '{field}'")
        
        step_type = step_config['type']
        step_id = step_config['id']
        
        # Get step class
        try:
            step_class = cls.get_step_class(step_type)
        except ValueError as e:
            raise StepRegistrationError(str(e))
        
        # Create instance
        try:
            step_instance = step_class(step_id, step_config)
            cls._logger.debug(f"Created step instance: {step_id} ({step_type})")
            return step_instance
            
        except Exception as e:
            cls._logger.error(f"Failed to create step '{step_id}' of type '{step_type}': {e}")
            raise StepRegistrationError(f"Step creation failed: {e}") from e
    
    @classmethod
    def is_registered(cls, step_type: str) -> bool:
        """Check if a step type is registered."""
        with cls._lock:
            return step_type in cls._steps or step_type in cls._aliases
    
    @classmethod
    def get_registered_types(cls) -> List[str]:
        """Get list of all registered step types."""
        with cls._lock:
            return list(cls._steps.keys())
    
    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """Get mapping of aliases to step types."""
        with cls._lock:
            return dict(cls._aliases)
    
    @classmethod
    def get_categories(cls) -> Dict[str, List[str]]:
        """Get step types organized by category."""
        with cls._lock:
            return {cat: list(steps) for cat, steps in cls._categories.items()}
    
    @classmethod
    def get_steps_in_category(cls, category: str) -> List[str]:
        """Get all step types in a specific category."""
        with cls._lock:
            return list(cls._categories.get(category, set()))
    
    @classmethod
    def get_step_metadata(cls, step_type: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific step type."""
        with cls._lock:
            # Handle aliases
            if step_type in cls._aliases:
                step_type = cls._aliases[step_type]
            
            return cls._step_metadata.get(step_type)
    
    @classmethod
    def get_step_info(cls, step_type: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a step type.
        
        Args:
            step_type: Step type identifier
            
        Returns:
            Dictionary with step information
        """
        with cls._lock:
            if not cls.is_registered(step_type):
                return {'error': f"Step type '{step_type}' not registered"}
            
            # Resolve alias
            resolved_type = step_type
            if step_type in cls._aliases:
                resolved_type = cls._aliases[step_type]
            
            step_class = cls._steps[resolved_type]
            metadata = cls._step_metadata.get(resolved_type, {})
            
            # Get class information
            class_info = {
                'step_type': resolved_type,
                'class_name': step_class.__name__,
                'module': step_class.__module__,
                'docstring': step_class.__doc__,
                'is_alias': step_type != resolved_type,
                'original_type': resolved_type if step_type != resolved_type else None
            }
            
            # Merge with metadata
            info = {**class_info, **metadata}
            
            return info
    
    @classmethod
    def list_available_types(cls, include_aliases: bool = True) -> List[str]:
        """
        Get list of all available step types.
        
        Args:
            include_aliases: Whether to include aliases in the list
            
        Returns:
            List of available step type identifiers
        """
        with cls._lock:
            types = list(cls._steps.keys())
            
            if include_aliases:
                types.extend(cls._aliases.keys())
            
            return sorted(types)
    
    @classmethod
    def find_steps(cls, 
                  pattern: str = None,
                  category: str = None,
                  class_name: str = None) -> List[str]:
        """
        Find step types matching criteria.
        
        Args:
            pattern: Pattern to match in step type name (case insensitive)
            category: Category to filter by
            class_name: Class name to filter by
            
        Returns:
            List of matching step types
        """
        with cls._lock:
            matches = []
            
            for step_type, step_class in cls._steps.items():
                metadata = cls._step_metadata.get(step_type, {})
                
                # Check pattern
                if pattern and pattern.lower() not in step_type.lower():
                    continue
                
                # Check category
                if category and metadata.get('category') != category:
                    continue
                
                # Check class name
                if class_name and step_class.__name__ != class_name:
                    continue
                
                matches.append(step_type)
            
            return sorted(matches)
    
    @classmethod
    def get_registry_stats(cls) -> Dict[str, Any]:
        """Get statistics about the registry."""
        with cls._lock:
            return {
                'total_registered': len(cls._steps),
                'total_aliases': len(cls._aliases),
                'total_categories': len(cls._categories),
                'registration_count': cls._registration_count,
                'initialized_at': cls._initialized_at,
                'categories': {cat: len(steps) for cat, steps in cls._categories.items()}
            }
    
    @classmethod
    def validate_all(cls) -> Dict[str, List[str]]:
        """
        Validate all registered step classes.
        
        Returns:
            Dictionary mapping step types to validation errors
        """
        with cls._lock:
            validation_results = {}
            
            for step_type, step_class in cls._steps.items():
                errors = cls._validate_step_class(step_class)
                if errors:
                    validation_results[step_type] = errors
            
            return validation_results
    
    @classmethod
    def clear_registry(cls, confirm: bool = False) -> None:
        """
        Clear all registered steps (use with caution).
        
        Args:
            confirm: Must be True to actually clear the registry
        """
        if not confirm:
            cls._logger.warning("clear_registry() called without confirmation")
            return
        
        with cls._lock:
            count = len(cls._steps)
            cls._steps.clear()
            cls._step_metadata.clear()
            cls._categories.clear()
            cls._aliases.clear()
            cls._registration_count = 0
            
            cls._logger.warning(f"Cleared registry: removed {count} step types")
    
    @classmethod
    def export_registry(cls) -> Dict[str, Any]:
        """
        Export registry contents for debugging or persistence.
        
        Returns:
            Dictionary with complete registry state
        """
        with cls._lock:
            return {
                'steps': {
                    step_type: {
                        'class_name': step_class.__name__,
                        'module': step_class.__module__,
                        'metadata': cls._step_metadata.get(step_type, {})
                    }
                    for step_type, step_class in cls._steps.items()
                },
                'aliases': dict(cls._aliases),
                'categories': {cat: list(steps) for cat, steps in cls._categories.items()},
                'stats': cls.get_registry_stats()
            }
    
    @classmethod
    def print_registry(cls, include_metadata: bool = False) -> None:
        """Print formatted registry contents for debugging."""
        with cls._lock:
            print("\n=== Step Registry ===")
            print(f"Total registered: {len(cls._steps)}")
            print(f"Total aliases: {len(cls._aliases)}")
            print(f"Total categories: {len(cls._categories)}")
            
            if cls._categories:
                print("\n--- By Category ---")
                for category, step_types in cls._categories.items():
                    print(f"{category}: {', '.join(sorted(step_types))}")
            
            print("\n--- All Step Types ---")
            for step_type in sorted(cls._steps.keys()):
                step_class = cls._steps[step_type]
                metadata = cls._step_metadata.get(step_type, {})
                category = metadata.get('category', 'uncategorized')
                
                print(f"  {step_type} -> {step_class.__name__} ({category})")
                
                if include_metadata and metadata:
                    for key, value in metadata.items():
                        if key != 'category':
                            print(f"    {key}: {value}")
            
            if cls._aliases:
                print("\n--- Aliases ---")
                for alias, target in sorted(cls._aliases.items()):
                    print(f"  {alias} -> {target}")


# Decorator for easy step registration
def register_step(step_type: str,
                 category: Optional[str] = None,
                 aliases: Optional[List[str]] = None,
                 override: bool = False,
                 **metadata) -> Callable[[Type['BaseStep']], Type['BaseStep']]:
    """
    Decorator for registering step classes.
    
    Args:
        step_type: Step type identifier
        category: Optional category
        aliases: Optional aliases
        override: Whether to override existing registration
        **metadata: Additional metadata
    
    Returns:
        Decorator function
    
    Example:
        @register_step('sentinel_acquisition', category='data_acquisition')
        class SentinelAcquisitionStep(BaseStep):
            def execute(self, context):
                ...
    """
    def decorator(step_class: Type['BaseStep']) -> Type['BaseStep']:
        StepRegistry.register(
            step_type=step_type,
            step_class=step_class,
            category=category,
            aliases=aliases,
            override=override,
            metadata=metadata
        )
        return step_class
    
    return decorator


# Utility functions
def get_available_step_types() -> List[str]:
    """Convenience function to get available step types."""
    return StepRegistry.get_registered_types()


def create_step_from_config(step_config: Dict[str, Any]) -> 'BaseStep':
    """Convenience function to create step from configuration."""
    return StepRegistry.create_step(step_config)


def is_step_type_available(step_type: str) -> bool:
    """Check if a step type is available."""
    return StepRegistry.is_registered(step_type)


if __name__ == "__main__":
    # Quick test of the step registry
    print("Testing StepRegistry...")
    
    # Test 1: Registry initialization
    print(f"✓ Registry initialized with {len(StepRegistry.get_registered_types())} types")
    
    # Test 2: Mock registration
    try:
        from .base_step import MockStep
        
        StepRegistry.register(
            'mock_test',
            MockStep,
            category='testing',
            aliases=['test_mock'],
            metadata={'description': 'Test mock step'}
        )
        
        print("✓ Mock step registration successful")
        
        # Test step creation
        test_config = {
            'id': 'test_step',
            'type': 'mock_test',
            'hyperparameters': {'test_param': 'test_value'}
        }
        
        step_instance = StepRegistry.create_step(test_config)
        print(f"✓ Step creation successful: {step_instance}")
        
        # Test alias resolution
        step_class = StepRegistry.get_step_class('test_mock')
        print(f"✓ Alias resolution successful: {step_class.__name__}")
        
    except ImportError:
        print("⚠ MockStep not available for testing")
    except Exception as e:
        print(f"✗ Registration test failed: {e}")
    
    # Test 3: Registry introspection
    try:
        stats = StepRegistry.get_registry_stats()
        print(f"✓ Registry stats: {stats}")
        
        categories = StepRegistry.get_categories()
        print(f"✓ Categories: {list(categories.keys())}")
        
    except Exception as e:
        print(f"✗ Introspection test failed: {e}")
    
    # Test 4: Print registry contents
    try:
        print("\n--- Registry Contents ---")
        StepRegistry.print_registry(include_metadata=True)
        
    except Exception as e:
        print(f"✗ Print registry failed: {e}")
    
    print("\nStepRegistry test completed!")
