"""
Validation Utilities for Orchestrator
=====================================

This module provides comprehensive validation functions for data integrity,
configuration validation, and schema checking in the orchestrator pipeline.

Key Features:
- JSON schema validation
- Geospatial data validation (bbox, CRS, dates)
- File format validation
- Raster and vector data integrity checks
- Step configuration validation
- Data type and range validation
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, date
import jsonschema

# Configure logger
logger = logging.getLogger(__name__)

# Optional imports for geospatial validation
try:
    import rasterio
    from rasterio.errors import RasterioIOError
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate data against JSON schema.
    
    Args:
        data: Data to validate
        schema: JSON schema to validate against
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> schema = {"type": "object", "required": ["name", "value"]}
        >>> data = {"name": "test", "value": 42}
        >>> is_valid, errors = validate_json_schema(data, schema)
        >>> print(f"Valid: {is_valid}")
    """
    try:
        jsonschema.validate(data, schema)
        logger.debug("JSON schema validation passed")
        return True, []
    except jsonschema.ValidationError as e:
        error_msg = f"Schema validation error: {e.message}"
        logger.warning(error_msg)
        return False, [error_msg]
    except jsonschema.SchemaError as e:
        error_msg = f"Invalid schema: {e.message}"
        logger.error(error_msg)
        return False, [error_msg]
    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        logger.error(error_msg)
        return False, [error_msg]


def validate_bbox(bbox: Union[List[float], Tuple[float, ...]], 
                  crs: str = "EPSG:4326") -> Tuple[bool, List[str]]:
    """
    Validate bounding box coordinates.
    
    Args:
        bbox: Bounding box as [west, south, east, north]
        crs: Coordinate reference system
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> is_valid, errors = validate_bbox([85.3, 27.6, 85.4, 27.7])
        >>> print(f"Valid bbox: {is_valid}")
    """
    errors = []
    
    # Check basic format
    if not isinstance(bbox, (list, tuple)):
        errors.append("Bounding box must be a list or tuple")
        return False, errors
    
    if len(bbox) != 4:
        errors.append("Bounding box must have exactly 4 coordinates [west, south, east, north]")
        return False, errors
    
    try:
        west, south, east, north = [float(coord) for coord in bbox]
    except (ValueError, TypeError):
        errors.append("All bounding box coordinates must be numeric")
        return False, errors
    
    # Validate coordinate ranges for WGS84
    if crs.upper() in ["EPSG:4326", "WGS84"]:
        if not (-180 <= west <= 180):
            errors.append(f"Western longitude {west} is out of range [-180, 180]")
        if not (-180 <= east <= 180):
            errors.append(f"Eastern longitude {east} is out of range [-180, 180]")
        if not (-90 <= south <= 90):
            errors.append(f"Southern latitude {south} is out of range [-90, 90]")
        if not (-90 <= north <= 90):
            errors.append(f"Northern latitude {north} is out of range [-90, 90]")
    
    # Check logical consistency
    if west >= east:
        errors.append(f"Western coordinate {west} must be less than eastern coordinate {east}")
    if south >= north:
        errors.append(f"Southern coordinate {south} must be less than northern coordinate {north}")
    
    # Check minimum size (avoid zero-area bboxes)
    if abs(east - west) < 1e-6:
        errors.append("Bounding box width is too small (may cause processing issues)")
    if abs(north - south) < 1e-6:
        errors.append("Bounding box height is too small (may cause processing issues)")
    
    is_valid = len(errors) == 0
    if is_valid:
        logger.debug(f"Valid bounding box: {bbox}")
    else:
        logger.warning(f"Invalid bounding box: {bbox} - {errors}")
    
    return is_valid, errors


def validate_date_format(date_string: str, 
                        format_string: str = "%Y-%m-%d") -> Tuple[bool, List[str]]:
    """
    Validate date string format.
    
    Args:
        date_string: Date string to validate
        format_string: Expected date format
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> is_valid, errors = validate_date_format("2023-01-01")
        >>> print(f"Valid date: {is_valid}")
    """
    errors = []
    
    if not isinstance(date_string, str):
        errors.append("Date must be a string")
        return False, errors
    
    try:
        parsed_date = datetime.strptime(date_string, format_string)
        
        # Additional checks
        current_date = datetime.now()
        
        # Check if date is not too far in the future (more than 1 year)
        if parsed_date > current_date.replace(year=current_date.year + 1):
            errors.append(f"Date {date_string} is too far in the future")
        
        # Check if date is not too far in the past (before satellite era)
        if parsed_date.year < 1970:
            errors.append(f"Date {date_string} is before satellite era (1970)")
        
        logger.debug(f"Valid date: {date_string}")
        
    except ValueError as e:
        errors.append(f"Invalid date format: {date_string} (expected {format_string})")
        logger.warning(f"Date validation failed: {date_string} - {e}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_file_format(file_path: Union[str, Path], 
                        expected_formats: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """
    Validate file format based on extension and content.
    
    Args:
        file_path: Path to file to validate
        expected_formats: List of expected file extensions (e.g., ['.tif', '.shp'])
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> is_valid, errors = validate_file_format('data.tif', ['.tif', '.tiff'])
        >>> print(f"Valid file format: {is_valid}")
    """
    errors = []
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        errors.append(f"File does not exist: {path}")
        return False, errors
    
    if not path.is_file():
        errors.append(f"Path is not a file: {path}")
        return False, errors
    
    # Check file extension
    file_extension = path.suffix.lower()
    
    if expected_formats:
        expected_formats_lower = [fmt.lower() for fmt in expected_formats]
        if file_extension not in expected_formats_lower:
            errors.append(f"File extension {file_extension} not in expected formats {expected_formats}")
    
    # Basic file accessibility check
    try:
        with open(path, 'rb') as f:
            first_bytes = f.read(1024)  # Read first 1KB
        
        if len(first_bytes) == 0:
            errors.append("File appears to be empty")
        
    except (IOError, OSError) as e:
        errors.append(f"Cannot read file: {e}")
    
    is_valid = len(errors) == 0
    if is_valid:
        logger.debug(f"Valid file format: {path}")
    else:
        logger.warning(f"File format validation failed: {path} - {errors}")
    
    return is_valid, errors


def validate_raster_data(file_path: Union[str, Path]) -> Tuple[bool, List[str]]:
    """
    Validate raster data file using rasterio.
    
    Args:
        file_path: Path to raster file
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> is_valid, errors = validate_raster_data('elevation.tif')
        >>> print(f"Valid raster: {is_valid}")
    """
    errors = []
    
    if not RASTERIO_AVAILABLE:
        errors.append("Rasterio not available for raster validation")
        return False, errors
    
    path = Path(file_path)
    
    if not path.exists():
        errors.append(f"Raster file does not exist: {path}")
        return False, errors
    
    try:
        with rasterio.open(path) as src:
            # Check basic properties
            if src.width <= 0 or src.height <= 0:
                errors.append(f"Invalid raster dimensions: {src.width}x{src.height}")
            
            if src.count <= 0:
                errors.append(f"Invalid number of bands: {src.count}")
            
            # Check CRS
            if src.crs is None:
                errors.append("Raster has no coordinate reference system (CRS)")
            
            # Check bounds
            if src.bounds is None:
                errors.append("Raster has no bounds information")
            else:
                bounds = src.bounds
                if bounds.left >= bounds.right or bounds.bottom >= bounds.top:
                    errors.append(f"Invalid raster bounds: {bounds}")
            
            # Check data types
            dtypes = src.dtypes
            if not dtypes:
                errors.append("Raster has no data type information")
            
            # Try to read a small sample to check data integrity
            try:
                sample = src.read(1, window=rasterio.windows.Window(0, 0, min(100, src.width), min(100, src.height)))
                if NUMPY_AVAILABLE:
                    if np.all(sample == src.nodata):
                        errors.append("Raster appears to contain only nodata values")
            except Exception as e:
                errors.append(f"Cannot read raster data: {e}")
    
    except RasterioIOError as e:
        errors.append(f"Rasterio error reading file: {e}")
    except Exception as e:
        errors.append(f"Error validating raster: {e}")
    
    is_valid = len(errors) == 0
    if is_valid:
        logger.debug(f"Valid raster file: {path}")
    else:
        logger.warning(f"Raster validation failed: {path} - {errors}")
    
    return is_valid, errors


def validate_vector_data(file_path: Union[str, Path]) -> Tuple[bool, List[str]]:
    """
    Validate vector data file using geopandas.
    
    Args:
        file_path: Path to vector file
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> is_valid, errors = validate_vector_data('boundaries.shp')
        >>> print(f"Valid vector: {is_valid}")
    """
    errors = []
    
    if not GEOPANDAS_AVAILABLE:
        errors.append("GeoPandas not available for vector validation")
        return False, errors
    
    path = Path(file_path)
    
    if not path.exists():
        errors.append(f"Vector file does not exist: {path}")
        return False, errors
    
    try:
        gdf = gpd.read_file(path)
        
        # Check if empty
        if len(gdf) == 0:
            errors.append("Vector file contains no features")
        
        # Check geometry column
        if 'geometry' not in gdf.columns:
            errors.append("Vector file has no geometry column")
        else:
            # Check for valid geometries
            invalid_geoms = gdf.geometry.isna().sum()
            if invalid_geoms > 0:
                errors.append(f"Vector file has {invalid_geoms} invalid/null geometries")
            
            # Check geometry types consistency
            geom_types = gdf.geometry.geom_type.unique()
            if len(geom_types) > 1:
                logger.warning(f"Vector file contains mixed geometry types: {geom_types}")
        
        # Check CRS
        if gdf.crs is None:
            errors.append("Vector file has no coordinate reference system (CRS)")
        
        # Check bounds
        try:
            bounds = gdf.total_bounds
            if len(bounds) != 4 or not all(isinstance(b, (int, float)) and not (np.isnan(b) if NUMPY_AVAILABLE else False) for b in bounds):
                errors.append("Vector file has invalid bounds")
        except Exception as e:
            errors.append(f"Cannot calculate vector bounds: {e}")
    
    except Exception as e:
        errors.append(f"Error validating vector file: {e}")
    
    is_valid = len(errors) == 0
    if is_valid:
        logger.debug(f"Valid vector file: {path}")
    else:
        logger.warning(f"Vector validation failed: {path} - {errors}")
    
    return is_valid, errors


def check_data_integrity(data: Any, data_type: str = "unknown") -> Tuple[bool, List[str]]:
    """
    Check data integrity based on type.
    
    Args:
        data: Data to check
        data_type: Type of data ('numpy_array', 'dict', 'list', etc.)
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> import numpy as np
        >>> data = np.array([1, 2, 3, np.nan, 5])
        >>> is_valid, errors = check_data_integrity(data, 'numpy_array')
        >>> print(f"Data integrity: {is_valid}")
    """
    errors = []
    
    if data is None:
        errors.append("Data is None")
        return False, errors
    
    if data_type == "numpy_array" and NUMPY_AVAILABLE:
        if not isinstance(data, np.ndarray):
            errors.append("Expected numpy array but got different type")
        else:
            # Check for NaN values
            if np.any(np.isnan(data)):
                nan_count = np.sum(np.isnan(data))
                total_count = data.size
                nan_percentage = (nan_count / total_count) * 100
                if nan_percentage > 50:
                    errors.append(f"Data has {nan_percentage:.1f}% NaN values (too many)")
                else:
                    logger.warning(f"Data has {nan_percentage:.1f}% NaN values")
            
            # Check for infinite values
            if np.any(np.isinf(data)):
                inf_count = np.sum(np.isinf(data))
                errors.append(f"Data contains {inf_count} infinite values")
            
            # Check array shape
            if data.size == 0:
                errors.append("Array is empty")
            
            if len(data.shape) == 0:
                errors.append("Array has no dimensions")
    
    elif data_type == "dict":
        if not isinstance(data, dict):
            errors.append("Expected dictionary but got different type")
        else:
            if len(data) == 0:
                errors.append("Dictionary is empty")
    
    elif data_type == "list":
        if not isinstance(data, list):
            errors.append("Expected list but got different type")
        else:
            if len(data) == 0:
                errors.append("List is empty")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_step_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate step configuration dictionary.
    
    Args:
        config: Step configuration to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> config = {
        ...     "type": "data_acquisition",
        ...     "hyperparameters": {"bbox": [85.3, 27.6, 85.4, 27.7]}
        ... }
        >>> is_valid, errors = validate_step_config(config)
        >>> print(f"Valid step config: {is_valid}")
    """
    errors = []
    
    # Check required fields
    required_fields = ["type"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate step type
    if "type" in config:
        step_type = config["type"]
        if not isinstance(step_type, str) or not step_type.strip():
            errors.append("Step type must be a non-empty string")
    
    # Validate hyperparameters if present
    if "hyperparameters" in config:
        hyperparams = config["hyperparameters"]
        if not isinstance(hyperparams, dict):
            errors.append("Hyperparameters must be a dictionary")
        else:
            # Check for bbox if present
            if "bbox" in hyperparams:
                bbox_valid, bbox_errors = validate_bbox(hyperparams["bbox"])
                if not bbox_valid:
                    errors.extend([f"Bbox validation: {err}" for err in bbox_errors])
            
            # Check for dates if present
            for date_field in ["start_date", "end_date"]:
                if date_field in hyperparams:
                    date_valid, date_errors = validate_date_format(hyperparams[date_field])
                    if not date_valid:
                        errors.extend([f"{date_field} validation: {err}" for err in date_errors])
    
    # Validate inputs/outputs if present
    for io_field in ["inputs", "outputs"]:
        if io_field in config:
            io_config = config[io_field]
            if not isinstance(io_config, dict):
                errors.append(f"{io_field} must be a dictionary")
    
    # Validate dependencies if present
    if "dependencies" in config:
        dependencies = config["dependencies"]
        if not isinstance(dependencies, list):
            errors.append("Dependencies must be a list")
        else:
            for dep in dependencies:
                if not isinstance(dep, str):
                    errors.append(f"Dependency must be string: {dep}")
    
    is_valid = len(errors) == 0
    if is_valid:
        logger.debug("Step configuration validation passed")
    else:
        logger.warning(f"Step configuration validation failed: {errors}")
    
    return is_valid, errors


def validate_coordinate_system(crs_string: str) -> Tuple[bool, List[str]]:
    """
    Validate coordinate reference system string.
    
    Args:
        crs_string: CRS string to validate (e.g., "EPSG:4326")
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> is_valid, errors = validate_coordinate_system("EPSG:4326")
        >>> print(f"Valid CRS: {is_valid}")
    """
    errors = []
    
    if not isinstance(crs_string, str):
        errors.append("CRS must be a string")
        return False, errors
    
    crs_string = crs_string.strip()
    
    if not crs_string:
        errors.append("CRS string is empty")
        return False, errors
    
    # Check common CRS formats
    epsg_pattern = re.compile(r'^EPSG:\d+$', re.IGNORECASE)
    proj4_pattern = re.compile(r'^\+proj=\w+', re.IGNORECASE)
    wkt_pattern = re.compile(r'^(GEOGCS|PROJCS)\[', re.IGNORECASE)
    
    if epsg_pattern.match(crs_string):
        # Validate EPSG code
        try:
            epsg_code = int(crs_string.split(':')[1])
            if epsg_code <= 0:
                errors.append(f"Invalid EPSG code: {epsg_code}")
            elif epsg_code > 100000:  # Reasonable upper limit
                errors.append(f"EPSG code seems too large: {epsg_code}")
        except ValueError:
            errors.append(f"Cannot parse EPSG code from: {crs_string}")
    
    elif proj4_pattern.match(crs_string):
        # Basic PROJ4 string validation
        if not crs_string.startswith('+'):
            errors.append("PROJ4 string should start with '+'")
    
    elif wkt_pattern.match(crs_string):
        # Basic WKT validation
        if not (crs_string.endswith(']') or crs_string.endswith(']]')):
            errors.append("WKT string appears malformed (missing closing brackets)")
    
    else:
        # Check for some common CRS names
        common_crs = ['WGS84', 'NAD83', 'UTM', 'ETRS89']
        if not any(common in crs_string.upper() for common in common_crs):
            errors.append(f"Unrecognized CRS format: {crs_string}")
    
    is_valid = len(errors) == 0
    if is_valid:
        logger.debug(f"Valid CRS: {crs_string}")
    else:
        logger.warning(f"CRS validation failed: {crs_string} - {errors}")
    
    return is_valid, errors


# Export main functions
__all__ = [
    'validate_json_schema', 'validate_bbox', 'validate_date_format',
    'validate_file_format', 'validate_raster_data', 'validate_vector_data',
    'check_data_integrity', 'validate_step_config', 'validate_coordinate_system'
]
