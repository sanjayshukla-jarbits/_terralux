"""
Geospatial Utilities for Orchestrator
=====================================

This module provides geospatial data processing utilities for the orchestrator system,
including raster operations, coordinate transformations, and spatial analysis functions.

Key Features:
- Raster data information extraction
- Coordinate system validation and transformation
- Bounding box operations and area calculations
- Raster clipping and merging
- Spatial reference system utilities
"""

import logging
import math
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# Optional imports for geospatial operations
try:
    import rasterio
    from rasterio.warp import transform_bounds, reproject, Resampling
    from rasterio.merge import merge
    from rasterio.mask import mask
    from rasterio.crs import CRS
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import box, Point, Polygon
    import pyproj
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def get_raster_info(raster_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive information about a raster file.
    
    Args:
        raster_path: Path to raster file
        
    Returns:
        Dictionary with raster information
        
    Example:
        >>> info = get_raster_info('elevation.tif')
        >>> print(f"Size: {info['width']}x{info['height']}")
        >>> print(f"CRS: {info['crs']}")
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("Rasterio required for raster operations")
    
    path = Path(raster_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")
    
    try:
        with rasterio.open(path) as src:
            info = {
                'path': str(path.absolute()),
                'filename': path.name,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': str(src.dtypes[0]) if src.dtypes else None,
                'crs': str(src.crs) if src.crs else None,
                'bounds': list(src.bounds) if src.bounds else None,
                'transform': list(src.transform) if src.transform else None,
                'nodata': src.nodata,
                'pixel_size': {
                    'x': abs(src.transform[0]) if src.transform else None,
                    'y': abs(src.transform[4]) if src.transform else None
                },
                'area_km2': None,
                'band_names': [src.descriptions[i] or f'Band_{i+1}' for i in range(src.count)]
            }
            
            # Calculate area if bounds are available
            if info['bounds'] and info['crs']:
                area_km2 = calculate_bbox_area(info['bounds'])
                info['area_km2'] = area_km2
            
            # Get sample statistics for first band
            if src.count > 0:
                try:
                    sample = src.read(1, window=Window(0, 0, min(1000, src.width), min(1000, src.height)))
                    if NUMPY_AVAILABLE:
                        valid_data = sample[sample != src.nodata] if src.nodata is not None else sample
                        if len(valid_data) > 0:
                            info['statistics'] = {
                                'min': float(np.min(valid_data)),
                                'max': float(np.max(valid_data)),
                                'mean': float(np.mean(valid_data)),
                                'std': float(np.std(valid_data))
                            }
                except Exception as e:
                    logger.warning(f"Could not calculate statistics: {e}")
        
        logger.debug(f"Retrieved raster info for: {path}")
        return info
        
    except Exception as e:
        logger.error(f"Failed to get raster info for {path}: {e}")
        raise


def calculate_bbox_area(bbox: List[float], crs: str = "EPSG:4326") -> float:
    """
    Calculate area of bounding box in square kilometers.
    
    Args:
        bbox: Bounding box as [west, south, east, north]
        crs: Coordinate reference system
        
    Returns:
        Area in square kilometers
        
    Example:
        >>> area = calculate_bbox_area([85.3, 27.6, 85.4, 27.7])
        >>> print(f"Area: {area:.2f} km²")
    """
    west, south, east, north = bbox
    
    if crs.upper() in ["EPSG:4326", "WGS84"]:
        # Use Haversine formula for geographic coordinates
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth's radius in kilometers
            
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            return R * c
        
        # Calculate width and height in kilometers
        width_km = haversine_distance(south, west, south, east)
        height_km = haversine_distance(south, west, north, west)
        
        area_km2 = width_km * height_km
        
    else:
        # For projected coordinates, assume units are meters
        width_m = abs(east - west)
        height_m = abs(north - south)
        area_km2 = (width_m * height_m) / 1e6  # Convert m² to km²
    
    logger.debug(f"Calculated bbox area: {area_km2:.2f} km²")
    return area_km2


def reproject_bbox(bbox: List[float], 
                  src_crs: str, 
                  dst_crs: str) -> List[float]:
    """
    Reproject bounding box from source CRS to destination CRS.
    
    Args:
        bbox: Bounding box as [west, south, east, north]
        src_crs: Source coordinate reference system
        dst_crs: Destination coordinate reference system
        
    Returns:
        Reprojected bounding box
        
    Example:
        >>> # Convert WGS84 to UTM
        >>> utm_bbox = reproject_bbox([85.3, 27.6, 85.4, 27.7], 
        ...                          "EPSG:4326", "EPSG:32645")
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("Rasterio required for CRS operations")
    
    try:
        src_crs_obj = CRS.from_string(src_crs)
        dst_crs_obj = CRS.from_string(dst_crs)
        
        # Transform bounds
        reprojected_bounds = transform_bounds(src_crs_obj, dst_crs_obj, *bbox)
        
        logger.debug(f"Reprojected bbox from {src_crs} to {dst_crs}")
        return list(reprojected_bounds)
        
    except Exception as e:
        logger.error(f"Failed to reproject bbox: {e}")
        raise


def validate_crs(crs_string: str) -> bool:
    """
    Validate coordinate reference system string.
    
    Args:
        crs_string: CRS string to validate
        
    Returns:
        True if CRS is valid
        
    Example:
        >>> is_valid = validate_crs("EPSG:4326")
        >>> print(f"Valid CRS: {is_valid}")
    """
    if not RASTERIO_AVAILABLE:
        logger.warning("Rasterio not available for CRS validation")
        return False
    
    try:
        crs = CRS.from_string(crs_string)
        return crs.is_valid
    except Exception:
        return False


def get_raster_bounds(raster_path: Union[str, Path], 
                     crs: Optional[str] = None) -> List[float]:
    """
    Get raster bounds, optionally reprojected to specified CRS.
    
    Args:
        raster_path: Path to raster file
        crs: Target CRS for bounds (optional)
        
    Returns:
        Bounds as [west, south, east, north]
        
    Example:
        >>> bounds = get_raster_bounds('elevation.tif', 'EPSG:4326')
        >>> print(f"Bounds: {bounds}")
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("Rasterio required for raster operations")
    
    path = Path(raster_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")
    
    try:
        with rasterio.open(path) as src:
            bounds = list(src.bounds)
            
            # Reproject bounds if target CRS specified
            if crs and str(src.crs) != crs:
                bounds = reproject_bbox(bounds, str(src.crs), crs)
        
        logger.debug(f"Retrieved bounds for: {path}")
        return bounds
        
    except Exception as e:
        logger.error(f"Failed to get raster bounds for {path}: {e}")
        raise


def calculate_pixel_size(raster_path: Union[str, Path]) -> Dict[str, float]:
    """
    Calculate pixel size in map units and meters.
    
    Args:
        raster_path: Path to raster file
        
    Returns:
        Dictionary with pixel size information
        
    Example:
        >>> pixel_size = calculate_pixel_size('elevation.tif')
        >>> print(f"Pixel size: {pixel_size['x_meters']:.2f}m x {pixel_size['y_meters']:.2f}m")
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("Rasterio required for raster operations")
    
    path = Path(raster_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")
    
    try:
        with rasterio.open(path) as src:
            transform = src.transform
            crs = src.crs
            
            pixel_size = {
                'x_map_units': abs(transform[0]),
                'y_map_units': abs(transform[4]),
                'map_units': str(crs.linear_units) if crs and crs.linear_units else 'unknown'
            }
            
            # Convert to meters if possible
            if crs and crs.linear_units:
                if crs.linear_units == 'metre':
                    pixel_size['x_meters'] = pixel_size['x_map_units']
                    pixel_size['y_meters'] = pixel_size['y_map_units']
                elif crs.linear_units == 'foot':
                    pixel_size['x_meters'] = pixel_size['x_map_units'] * 0.3048
                    pixel_size['y_meters'] = pixel_size['y_map_units'] * 0.3048
                else:
                    pixel_size['x_meters'] = None
                    pixel_size['y_meters'] = None
            elif crs and crs.is_geographic:
                # For geographic coordinates, estimate at equator
                pixel_size['x_meters'] = pixel_size['x_map_units'] * 111320  # degrees to meters at equator
                pixel_size['y_meters'] = pixel_size['y_map_units'] * 111320
            else:
                pixel_size['x_meters'] = None
                pixel_size['y_meters'] = None
        
        logger.debug(f"Calculated pixel size for: {path}")
        return pixel_size
        
    except Exception as e:
        logger.error(f"Failed to calculate pixel size for {path}: {e}")
        raise


def clip_raster_to_bbox(raster_path: Union[str, Path], 
                       bbox: List[float],
                       output_path: Union[str, Path],
                       crs: Optional[str] = None) -> Dict[str, Any]:
    """
    Clip raster to bounding box.
    
    Args:
        raster_path: Path to input raster file
        bbox: Bounding box as [west, south, east, north]
        output_path: Path for output clipped raster
        crs: CRS of bounding box (if different from raster CRS)
        
    Returns:
        Dictionary with clipping results
        
    Example:
        >>> result = clip_raster_to_bbox('elevation.tif', 
        ...                             [85.3, 27.6, 85.4, 27.7],
        ...                             'clipped_elevation.tif')
        >>> print(f"Clipped size: {result['width']}x{result['height']}")
    """
    if not RASTERIO_AVAILABLE or not GEOPANDAS_AVAILABLE:
        raise ImportError("Rasterio and GeoPandas required for raster clipping")
    
    input_path = Path(raster_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with rasterio.open(input_path) as src:
            # Create geometry for clipping
            if crs and str(src.crs) != crs:
                # Reproject bbox to raster CRS
                clip_bbox = reproject_bbox(bbox, crs, str(src.crs))
            else:
                clip_bbox = bbox
            
            # Create polygon geometry
            west, south, east, north = clip_bbox
            geometry = box(west, south, east, north)
            
            # Clip raster
            clipped_data, clipped_transform = mask(src, [geometry], crop=True)
            
            # Update metadata
            clipped_meta = src.meta.copy()
            clipped_meta.update({
                'height': clipped_data.shape[1],
                'width': clipped_data.shape[2],
                'transform': clipped_transform
            })
            
            # Write clipped raster
            with rasterio.open(output_path, 'w', **clipped_meta) as dst:
                dst.write(clipped_data)
        
        # Get result information
        result = {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'clip_bbox': clip_bbox,
            'width': clipped_meta['width'],
            'height': clipped_meta['height'],
            'transform': list(clipped_transform)
        }
        
        logger.info(f"Clipped raster: {input_path} -> {output_path}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to clip raster: {e}")
        raise


def merge_raster_files(raster_paths: List[Union[str, Path]], 
                      output_path: Union[str, Path],
                      method: str = 'first') -> Dict[str, Any]:
    """
    Merge multiple raster files into a single file.
    
    Args:
        raster_paths: List of paths to raster files to merge
        output_path: Path for output merged raster
        method: Merge method ('first', 'last', 'min', 'max', 'mean')
        
    Returns:
        Dictionary with merge results
        
    Example:
        >>> files = ['tile1.tif', 'tile2.tif', 'tile3.tif']
        >>> result = merge_raster_files(files, 'merged.tif', method='mean')
        >>> print(f"Merged {result['input_count']} files")
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("Rasterio required for raster merging")
    
    output_path = Path(output_path)
    raster_paths = [Path(p) for p in raster_paths]
    
    # Check that all input files exist
    missing_files = [p for p in raster_paths if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"Input rasters not found: {missing_files}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Open all raster files
        src_files = [rasterio.open(p) for p in raster_paths]
        
        # Merge rasters
        merged_data, merged_transform = merge(src_files, method=method)
        
        # Get metadata from first raster
        merged_meta = src_files[0].meta.copy()
        merged_meta.update({
            'height': merged_data.shape[1],
            'width': merged_data.shape[2],
            'transform': merged_transform
        })
        
        # Write merged raster
        with rasterio.open(output_path, 'w', **merged_meta) as dst:
            dst.write(merged_data)
        
        # Close input files
        for src in src_files:
            src.close()
        
        # Get result information
        result = {
            'input_paths': [str(p) for p in raster_paths],
            'input_count': len(raster_paths),
            'output_path': str(output_path),
            'method': method,
            'width': merged_meta['width'],
            'height': merged_meta['height'],
            'crs': str(merged_meta['crs']) if merged_meta['crs'] else None
        }
        
        logger.info(f"Merged {len(raster_paths)} rasters -> {output_path}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to merge rasters: {e}")
        raise


def calculate_distance(point1: Tuple[float, float], 
                      point2: Tuple[float, float],
                      method: str = 'haversine') -> float:
    """
    Calculate distance between two points.
    
    Args:
        point1: First point as (longitude, latitude) or (x, y)
        point2: Second point as (longitude, latitude) or (x, y)
        method: Distance calculation method ('haversine', 'euclidean')
        
    Returns:
        Distance in kilometers (haversine) or map units (euclidean)
        
    Example:
        >>> dist = calculate_distance((85.3, 27.6), (85.4, 27.7))
        >>> print(f"Distance: {dist:.2f} km")
    """
    lon1, lat1 = point1
    lon2, lat2 = point2
    
    if method == 'haversine':
        # Haversine formula for great circle distance
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        distance = R * c
        
    elif method == 'euclidean':
        # Simple Euclidean distance
        distance = math.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)
    
    else:
        raise ValueError(f"Unknown distance method: {method}")
    
    logger.debug(f"Calculated {method} distance: {distance:.2f}")
    return distance


def create_buffer_geometry(geometry_or_point: Union[Tuple[float, float], Any], 
                          buffer_distance: float,
                          crs: str = "EPSG:4326") -> Any:
    """
    Create buffer geometry around point or existing geometry.
    
    Args:
        geometry_or_point: Point as (lon, lat) tuple or shapely geometry
        buffer_distance: Buffer distance in map units
        crs: Coordinate reference system
        
    Returns:
        Buffered geometry (requires GeoPandas)
        
    Example:
        >>> buffer_geom = create_buffer_geometry((85.35, 27.65), 0.01)
        >>> # Creates 0.01 degree buffer around point
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("GeoPandas required for geometry operations")
    
    # Convert point to geometry if needed
    if isinstance(geometry_or_point, (tuple, list)):
        lon, lat = geometry_or_point
        geometry = Point(lon, lat)
    else:
        geometry = geometry_or_point
    
    # Create buffer
    buffered = geometry.buffer(buffer_distance)
    
    logger.debug(f"Created buffer with distance {buffer_distance}")
    return buffered


# Export main functions
__all__ = [
    'get_raster_info', 'calculate_bbox_area', 'reproject_bbox',
    'validate_crs', 'get_raster_bounds', 'calculate_pixel_size',
    'clip_raster_to_bbox', 'merge_raster_files', 'calculate_distance',
    'create_buffer_geometry'
]
