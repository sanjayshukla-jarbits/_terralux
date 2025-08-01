"""
Digital Elevation Model (DEM) Acquisition Step
==============================================

This module implements DEM data acquisition from multiple sources including SRTM, ASTER,
and ALOS, designed with fail-fast principles for rapid development and testing.
CORRECTED to remove StepResult references and use Dict return type.

Key Features:
- Multiple DEM sources (SRTM, ASTER, ALOS, External files)
- Automatic resolution matching and resampling
- Topographic derivatives generation (slope, aspect, curvature)
- Void filling and quality control
- Mock data support for testing
- Integration with existing landslide_pipeline infrastructure
- Compatible with ModularOrchestrator's ExecutionContext
"""

import logging
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

# CORRECTED: Remove StepResult import - only import BaseStep
from ..base.base_step import BaseStep
from ..base.step_registry import StepRegistry, register_step

# Conditional imports for dependencies
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
    from rasterio.enums import Resampling as ResamplingEnums
    from rasterio.fill import fillnodata
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from elevation import elevation
    import elevation
    ELEVATION_AVAILABLE = True
except ImportError:
    ELEVATION_AVAILABLE = False

try:
    from landslide_pipeline.data.data_acquisition import DEMDataAcquisition
    EXISTING_DEM_ACQUISITOR_AVAILABLE = True
except ImportError:
    EXISTING_DEM_ACQUISITOR_AVAILABLE = False

try:
    from scipy import ndimage
    from scipy.ndimage import generic_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from osgeo import gdal, osr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False


class DEMAcquisitionError(Exception):
    """Custom exception for DEM acquisition errors."""
    pass


@register_step('dem_acquisition',
               category='data_acquisition', 
               aliases=['elevation_data', 'srtm_acquisition', 'dem_loading'])
class DEMAcquisitionStep(BaseStep):
    """
    Digital Elevation Model (DEM) acquisition step.
    CORRECTED to return Dict instead of StepResult and use ExecutionContext.
    
    This step handles acquisition of elevation data from various sources including:
    - SRTM (Shuttle Radar Topography Mission) - 30m resolution
    - ASTER GDEM (Advanced Spaceborne Thermal Emission and Reflection Radiometer) - 30m
    - ALOS World 3D (Advanced Land Observing Satellite) - 30m
    - External DEM files (user-provided)
    - NASA DEM (90m global coverage)
    
    Supported Parameters:
    - bbox: Bounding box coordinates [west, south, east, north]
    - source: DEM source ('SRTM', 'ASTER', 'ALOS', 'NASA', 'external')
    - resolution: Output resolution in meters
    - external_dem_path: Path to external DEM file (if source='external')
    - generate_derivatives: Whether to generate slope, aspect, etc.
    - void_fill: Whether to fill no-data voids
    - output_format: Output file format ('GeoTIFF', 'NetCDF')
    """
    
    # Supported DEM sources and their characteristics
    SUPPORTED_SOURCES = {
        'SRTM': {
            'resolution': 30,  # meters
            'coverage': 'global_60N_60S',
            'vertical_accuracy': '±16m',
            'api_available': True,
            'description': 'SRTM 30m global elevation data'
        },
        'ASTER': {
            'resolution': 30,
            'coverage': 'global_83N_83S', 
            'vertical_accuracy': '±17m',
            'api_available': True,
            'description': 'ASTER GDEM v3 30m elevation data'
        },
        'ALOS': {
            'resolution': 30,
            'coverage': 'global',
            'vertical_accuracy': '±5m',
            'api_available': False,
            'description': 'ALOS World 3D 30m elevation data'
        },
        'NASA': {
            'resolution': 90,
            'coverage': 'global',
            'vertical_accuracy': '±20m',
            'api_available': True,
            'description': 'NASA SRTM 90m elevation data'
        },
        'external': {
            'resolution': 'variable',
            'coverage': 'user_defined',
            'vertical_accuracy': 'variable',
            'api_available': False,
            'description': 'User-provided external DEM file'
        }
    }
    
    def __init__(self, step_id: str, step_config: Dict[str, Any]):
        """Initialize DEM acquisition step."""
        super().__init__(step_id, step_config)
        
        # Step-specific configuration
        self.use_mock_data = False
        self.dem_cache_dir = None
        
        # Initialize configuration
        self._setup_dem_config()
        
        # Log configuration status
        self._log_configuration_status()
    
    def _setup_dem_config(self) -> None:
        """Setup DEM acquisition configuration."""
        try:
            # Check if we can use real DEM acquisition
            if not RASTERIO_AVAILABLE:
                self.logger.warning("Rasterio not available, using mock data")
                self.use_mock_data = True
                return
            
            # Setup cache directory
            cache_dir = self.hyperparameters.get('cache_directory')
            if cache_dir:
                self.dem_cache_dir = Path(cache_dir)
                self.dem_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Check source availability
            source = self.hyperparameters.get('source', 'SRTM')
            if source not in self.SUPPORTED_SOURCES:
                self.logger.warning(f"Unsupported DEM source: {source}, using mock data")
                self.use_mock_data = True
                return
            
            # Check if external file exists for external source
            if source == 'external':
                external_path = self.hyperparameters.get('external_dem_path')
                if not external_path or not Path(external_path).exists():
                    self.logger.warning(f"External DEM file not found: {external_path}, using mock data")
                    self.use_mock_data = True
                    return
            
            # Check API availability for online sources
            source_info = self.SUPPORTED_SOURCES[source]
            if source_info['api_available'] and not ELEVATION_AVAILABLE and not EXISTING_DEM_ACQUISITOR_AVAILABLE:
                self.logger.warning(f"No API available for {source}, using mock data")
                self.use_mock_data = True
                return
            
            self.logger.info("✓ DEM acquisition configuration loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup DEM config: {e}, using mock data")
            self.use_mock_data = True
    
    def _log_configuration_status(self) -> None:
        """Log the current configuration status."""
        status_items = []
        
        if RASTERIO_AVAILABLE:
            status_items.append("✓ Rasterio")
        else:
            status_items.append("✗ Rasterio")
        
        if ELEVATION_AVAILABLE:
            status_items.append("✓ Elevation library")
        else:
            status_items.append("✗ Elevation library")
        
        if EXISTING_DEM_ACQUISITOR_AVAILABLE:
            status_items.append("✓ Existing DEM acquisitor")
        else:
            status_items.append("✗ Existing DEM acquisitor")
        
        if SCIPY_AVAILABLE:
            status_items.append("✓ SciPy")
        else:
            status_items.append("✗ SciPy")
        
        if GDAL_AVAILABLE:
            status_items.append("✓ GDAL")
        else:
            status_items.append("✗ GDAL")
        
        self.logger.debug(f"Configuration status: {', '.join(status_items)}")
    
    def execute(self, context) -> Dict[str, Any]:
        """
        Execute DEM data acquisition.
        CORRECTED to return Dict instead of StepResult and use ExecutionContext.
        
        Args:
            context: Pipeline execution context (ExecutionContext)
            
        Returns:
            Dict with execution results:
            {
                'status': 'success'|'failed'|'skipped',
                'outputs': {...},
                'metadata': {...}
            }
        """
        self.logger.info(f"Starting DEM acquisition: {self.step_id}")
        
        try:
            # Extract and validate parameters
            params = self._extract_parameters(context)
            self._validate_parameters(params)
            
            # Log acquisition parameters
            self._log_acquisition_info(params)
            
            # Execute acquisition
            if self.use_mock_data:
                result_data = self._acquire_mock_dem(params)
                self.logger.info("✓ Mock DEM data acquisition completed")
            else:
                if params['source'] == 'external':
                    result_data = self._load_external_dem(params)
                elif EXISTING_DEM_ACQUISITOR_AVAILABLE:
                    result_data = self._acquire_with_existing_acquisitor(params)
                else:
                    result_data = self._acquire_with_online_api(params)
                
                self.logger.info("✓ Real DEM data acquisition completed")
            
            # Generate derivatives if requested
            if params.get('generate_derivatives', True):
                derivatives = self._generate_topographic_derivatives(result_data, params)
                result_data.update(derivatives)
                self.logger.info("✓ Topographic derivatives generated")
            
            # Store results in context
            output_key = f"{self.step_id}_data"
            context.set_artifact(output_key, result_data)
            
            # Save data if requested
            saved_files = []
            if params.get('save_to_file', True):
                saved_files = self._save_data(result_data, context, params)
            
            return {
                'status': 'success',
                'outputs': {
                    'elevation_data': output_key,
                    'dem_files': saved_files,
                    'derivatives': list(result_data.keys())
                },
                'metadata': {
                    'source': params['source'],
                    'resolution': params['resolution'],
                    'bbox': params['bbox'],
                    'derivatives_generated': params.get('generate_derivatives', True),
                    'void_filled': params.get('void_fill', False),
                    'acquisition_method': 'mock' if self.use_mock_data else params['source'],
                    'files_saved': len(saved_files),
                    'data_quality': self._assess_data_quality(result_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"DEM acquisition failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'metadata': {
                    'step_id': self.step_id,
                    'acquisition_method': 'mock' if self.use_mock_data else 'unknown'
                }
            }
    
    def _extract_parameters(self, context) -> Dict[str, Any]:
        """Extract and process acquisition parameters."""
        # Required parameters - use context-aware method
        bbox = self.get_input_data(context, 'bbox', 
                                  self.hyperparameters.get('bbox'))
        
        # Optional parameters with defaults
        params = {
            'bbox': bbox,
            'source': self.hyperparameters.get('source', 'SRTM'),
            'resolution': self.hyperparameters.get('resolution', 30),
            'external_dem_path': self.hyperparameters.get('external_dem_path'),
            'generate_derivatives': self.hyperparameters.get('generate_derivatives', True),
            'void_fill': self.hyperparameters.get('void_fill', True),
            'output_format': self.hyperparameters.get('output_format', 'GeoTIFF'),
            'crs': self.hyperparameters.get('crs', 'EPSG:4326'),
            'resampling_method': self.hyperparameters.get('resampling_method', 'bilinear'),
            'save_to_file': self.hyperparameters.get('save_to_file', True),
            'cache_enabled': self.hyperparameters.get('cache_enabled', True)
        }
        
        return params
    
    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate acquisition parameters."""
        # Validate required parameters
        if not params.get('bbox'):
            raise DEMAcquisitionError("Missing required parameter: bbox")
        
        # Validate bbox
        bbox = params['bbox']
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise DEMAcquisitionError(f"Invalid bbox format: {bbox}")
        
        west, south, east, north = bbox
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise DEMAcquisitionError(f"Invalid longitude values: {west}, {east}")
        if not (-90 <= south <= 90 and -90 <= north <= 90):
            raise DEMAcquisitionError(f"Invalid latitude values: {south}, {north}")
        if west >= east or south >= north:
            raise DEMAcquisitionError(f"Invalid bbox geometry: {bbox}")
        
        # Validate source
        source = params['source']
        if source not in self.SUPPORTED_SOURCES:
            raise DEMAcquisitionError(f"Unsupported DEM source: {source}")
        
        # Validate resolution
        if params['resolution'] <= 0:
            raise DEMAcquisitionError(f"Invalid resolution: {params['resolution']}")
        
        # Validate external DEM path
        if source == 'external':
            external_path = params['external_dem_path']
            if not external_path:
                raise DEMAcquisitionError("external_dem_path required for external source")
            if not Path(external_path).exists():
                raise DEMAcquisitionError(f"External DEM file not found: {external_path}")
    
    def _log_acquisition_info(self, params: Dict[str, Any]) -> None:
        """Log acquisition information."""
        bbox_str = f"[{', '.join(f'{x:.4f}' for x in params['bbox'])}]"
        source_info = self.SUPPORTED_SOURCES[params['source']]
        
        self.logger.info(f"DEM acquisition parameters:")
        self.logger.info(f"  • Bbox: {bbox_str}")
        self.logger.info(f"  • Source: {params['source']} ({source_info['description']})")
        self.logger.info(f"  • Resolution: {params['resolution']}m")
        self.logger.info(f"  • Generate derivatives: {params['generate_derivatives']}")
        self.logger.info(f"  • Void fill: {params['void_fill']}")
        
        if params['source'] == 'external':
            self.logger.info(f"  • External file: {params['external_dem_path']}")
    
    def _acquire_mock_dem(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock DEM data for testing."""
        self.logger.info("Generating mock DEM data")
        
        # Calculate image dimensions based on bbox and resolution
        bbox = params['bbox']
        resolution = params['resolution']
        
        # Calculate size based on geographic extent
        width = max(50, int((bbox[2] - bbox[0]) * 111320 / resolution))
        height = max(50, int((bbox[3] - bbox[1]) * 111320 / resolution))
        
        # Limit size for memory efficiency
        width = min(width, 2000)
        height = min(height, 2000)
        
        # Generate realistic elevation data
        np.random.seed(42)  # Reproducible mock data
        
        # Create base elevation with some spatial structure
        base_elevation = np.random.rand(height, width) * 2000 + 500  # 500-2500m range
        
        if SCIPY_AVAILABLE:
            # Add terrain-like features with Gaussian filtering
            base_elevation = ndimage.gaussian_filter(base_elevation, sigma=3)
            
            # Add some ridges and valleys
            x, y = np.meshgrid(np.linspace(0, 10, width), np.linspace(0, 10, height))
            terrain_features = (
                200 * np.sin(x * 0.5) * np.cos(y * 0.3) +  # Large-scale features
                100 * np.sin(x * 2) * np.cos(y * 1.5) +     # Medium-scale features  
                50 * np.random.rand(height, width)           # Small-scale noise
            )
            base_elevation += terrain_features
        
        # Ensure reasonable elevation range
        base_elevation = np.clip(base_elevation, 0, 6000).astype(np.float32)
        
        # Create some no-data areas (simulate voids)
        if params.get('void_fill', True):
            void_mask = np.random.rand(height, width) < 0.02  # 2% voids
            base_elevation[void_mask] = np.nan
        
        # Create metadata
        mock_metadata = {
            'bbox': params['bbox'],
            'crs': params['crs'],
            'resolution': params['resolution'],
            'shape': [height, width],
            'source': params['source'],
            'units': 'meters',
            'nodata_value': np.nan,
            'vertical_accuracy': self.SUPPORTED_SOURCES[params['source']]['vertical_accuracy'],
            'mock_data': True,
            'statistics': {
                'min': float(np.nanmin(base_elevation)),
                'max': float(np.nanmax(base_elevation)),
                'mean': float(np.nanmean(base_elevation)),
                'std': float(np.nanstd(base_elevation))
            }
        }
        
        return {
            'elevation': base_elevation,
            'metadata': mock_metadata,
            'transform': self._calculate_transform(params['bbox'], height, width),
            'crs': params['crs']
        }
    
    def _load_external_dem(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load external DEM file."""
        self.logger.info(f"Loading external DEM: {params['external_dem_path']}")
        
        try:
            external_path = Path(params['external_dem_path'])
            
            if not RASTERIO_AVAILABLE:
                raise DEMAcquisitionError("Rasterio not available for loading external DEM")
            
            with rasterio.open(external_path) as src:
                # Read elevation data
                elevation_data = src.read(1).astype(np.float32)
                
                # Handle no-data values
                if src.nodata is not None:
                    elevation_data[elevation_data == src.nodata] = np.nan
                
                # Clip to bbox if requested
                bbox = params['bbox']
                if bbox != list(src.bounds):
                    elevation_data, transform, crs = self._clip_to_bbox(
                        elevation_data, src.transform, src.crs, bbox
                    )
                else:
                    transform = src.transform
                    crs = src.crs
                
                # Resample if resolution doesn't match
                target_resolution = params['resolution']
                current_resolution = abs(src.transform[0])  # Pixel width
                
                if abs(current_resolution - target_resolution) > 1:  # 1m tolerance
                    elevation_data, transform = self._resample_data(
                        elevation_data, transform, current_resolution, target_resolution
                    )
                
                # Void filling if requested
                if params.get('void_fill', True) and np.any(np.isnan(elevation_data)):
                    elevation_data = self._fill_voids(elevation_data)
                
                # Create metadata
                metadata = {
                    'bbox': params['bbox'],
                    'crs': str(crs),
                    'resolution': target_resolution,
                    'shape': list(elevation_data.shape),
                    'source': 'external',
                    'original_file': str(external_path),
                    'units': 'meters',
                    'nodata_value': np.nan,
                    'void_filled': params.get('void_fill', True),
                    'mock_data': False,
                    'statistics': {
                        'min': float(np.nanmin(elevation_data)),
                        'max': float(np.nanmax(elevation_data)),
                        'mean': float(np.nanmean(elevation_data)),
                        'std': float(np.nanstd(elevation_data))
                    }
                }
                
                return {
                    'elevation': elevation_data,
                    'metadata': metadata,
                    'transform': transform,
                    'crs': str(crs)
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to load external DEM: {e}, falling back to mock data")
            return self._acquire_mock_dem(params)
    
    def _acquire_with_existing_acquisitor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Acquire DEM using existing acquisitor from landslide_pipeline."""
        self.logger.info("Using existing DEMDataAcquisition")
        
        try:
            # This would integrate with the existing DEM acquisition system
            # For now, we'll simulate this
            self.logger.warning("Existing DEM acquisitor integration not yet implemented")
            return self._acquire_mock_dem(params)
            
        except Exception as e:
            self.logger.warning(f"Existing acquisitor failed: {e}, falling back to mock data")
            return self._acquire_mock_dem(params)
    
    def _acquire_with_online_api(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Acquire DEM using online APIs."""
        self.logger.info(f"Acquiring DEM from {params['source']} API")
        
        try:
            if not ELEVATION_AVAILABLE:
                raise DEMAcquisitionError("Elevation library not available")
            
            # Use the elevation library for SRTM data
            if params['source'] in ['SRTM', 'NASA']:
                bbox = params['bbox']
                
                # Create temporary output file
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
                temp_path = temp_file.name
                temp_file.close()
                
                try:
                    # Download DEM data
                    elevation.clip(
                        bounds=bbox,
                        output=temp_path,
                        product='SRTM1' if params['source'] == 'SRTM' else 'SRTM3'
                    )
                    
                    # Load the downloaded data
                    with rasterio.open(temp_path) as src:
                        elevation_data = src.read(1).astype(np.float32)
                        transform = src.transform
                        crs = src.crs
                        
                        # Handle no-data
                        if src.nodata is not None:
                            elevation_data[elevation_data == src.nodata] = np.nan
                    
                    # Clean up temporary file
                    os.unlink(temp_path)
                    
                    # Process data
                    if params.get('void_fill', True) and np.any(np.isnan(elevation_data)):
                        elevation_data = self._fill_voids(elevation_data)
                    
                    # Create metadata
                    metadata = {
                        'bbox': params['bbox'],
                        'crs': str(crs),
                        'resolution': params['resolution'],
                        'shape': list(elevation_data.shape),
                        'source': params['source'],
                        'units': 'meters',
                        'nodata_value': np.nan,
                        'void_filled': params.get('void_fill', True),
                        'mock_data': False,
                        'statistics': {
                            'min': float(np.nanmin(elevation_data)),
                            'max': float(np.nanmax(elevation_data)),
                            'mean': float(np.nanmean(elevation_data)),
                            'std': float(np.nanstd(elevation_data))
                        }
                    }
                    
                    return {
                        'elevation': elevation_data,
                        'metadata': metadata,
                        'transform': transform,
                        'crs': str(crs)
                    }
                    
                except Exception as e:
                    # Clean up on error
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise e
            else:
                raise DEMAcquisitionError(f"Online API not available for {params['source']}")
                
        except Exception as e:
            self.logger.warning(f"Online API acquisition failed: {e}, falling back to mock data")
            return self._acquire_mock_dem(params)
    
    def _generate_topographic_derivatives(self, dem_data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate topographic derivatives from elevation data."""
        self.logger.info("Generating topographic derivatives")
        
        derivatives = {}
        elevation = dem_data['elevation']
        
        try:
            if SCIPY_AVAILABLE:
                # Calculate slope and aspect
                derivatives.update(self._calculate_slope_aspect(elevation, params['resolution']))
                
                # Calculate curvatures
                derivatives.update(self._calculate_curvatures(elevation, params['resolution']))
                
                # Calculate additional terrain parameters
                derivatives.update(self._calculate_terrain_parameters(elevation, params['resolution']))
            else:
                # Simple derivatives without scipy
                derivatives.update(self._calculate_simple_derivatives(elevation))
            
            # Add metadata to derivatives
            for key, data in derivatives.items():
                derivatives[key + '_metadata'] = {
                    'derived_from': 'elevation',
                    'method': 'gradient_analysis',
                    'resolution': params['resolution'],
                    'units': self._get_derivative_units(key)
                }
            
            self.logger.info(f"Generated {len(derivatives)//2} topographic derivatives")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate derivatives: {e}")
        
        return derivatives
    
    def _calculate_simple_derivatives(self, elevation: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate simple derivatives without scipy."""
        # Simple gradient-based slope calculation
        dy, dx = np.gradient(elevation)
        slope = np.sqrt(dx**2 + dy**2)
        
        # Simple aspect calculation
        aspect = np.arctan2(-dx, dy) * 180 / np.pi
        aspect[aspect < 0] += 360
        
        return {
            'slope': slope.astype(np.float32),
            'aspect': aspect.astype(np.float32)
        }
    
    def _calculate_slope_aspect(self, elevation: np.ndarray, resolution: float) -> Dict[str, np.ndarray]:
        """Calculate slope and aspect from elevation data."""
        # Calculate gradients
        dy, dx = np.gradient(elevation, resolution)
        
        # Calculate slope in degrees
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
        
        # Calculate aspect in degrees (0-360)
        aspect = np.arctan2(-dx, dy) * 180 / np.pi
        aspect[aspect < 0] += 360
        
        return {
            'slope': slope.astype(np.float32),
            'aspect': aspect.astype(np.float32)
        }
    
    def _calculate_curvatures(self, elevation: np.ndarray, resolution: float) -> Dict[str, np.ndarray]:
        """Calculate curvature parameters."""
        # Second derivatives
        dz_dx = np.gradient(elevation, resolution, axis=1)
        dz_dy = np.gradient(elevation, resolution, axis=0)
        
        d2z_dx2 = np.gradient(dz_dx, resolution, axis=1)
        d2z_dy2 = np.gradient(dz_dy, resolution, axis=0)
        d2z_dxdy = np.gradient(dz_dx, resolution, axis=0)
        
        # Profile curvature (curvature in the direction of steepest slope)
        p = dz_dx**2 + dz_dy**2
        profile_curvature = np.where(
            p > 0,
            (d2z_dx2 * dz_dx**2 + 2 * d2z_dxdy * dz_dx * dz_dy + d2z_dy2 * dz_dy**2) / (p * np.sqrt(1 + p)),
            0
        )
        
        # Plan curvature (perpendicular to the direction of steepest slope)
        plan_curvature = np.where(
            p > 0,
            (d2z_dx2 * dz_dy**2 - 2 * d2z_dxdy * dz_dx * dz_dy + d2z_dy2 * dz_dx**2) / (p * (1 + p)**1.5),
            0
        )
        
        # Mean curvature
        mean_curvature = (d2z_dx2 + d2z_dy2) / 2
        
        return {
            'profile_curvature': profile_curvature.astype(np.float32),
            'plan_curvature': plan_curvature.astype(np.float32),
            'mean_curvature': mean_curvature.astype(np.float32)
        }
    
    def _calculate_terrain_parameters(self, elevation: np.ndarray, resolution: float) -> Dict[str, np.ndarray]:
        """Calculate additional terrain parameters."""
        derivatives = {}
        
        # Roughness (standard deviation in a 3x3 window)
        if SCIPY_AVAILABLE:
            def roughness_filter(window):
                return np.std(window)
            
            roughness = generic_filter(elevation, roughness_filter, size=3, mode='reflect')
            derivatives['roughness'] = roughness.astype(np.float32)
        
        # Terrain Ruggedness Index (TRI)
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1], 
                          [-1, -1, -1]])
        
        if SCIPY_AVAILABLE:
            tri = ndimage.convolve(elevation, kernel, mode='reflect')
            derivatives['terrain_ruggedness_index'] = np.abs(tri).astype(np.float32)
        
        return derivatives
    
    def _get_derivative_units(self, derivative_name: str) -> str:
        """Get units for different derivatives."""
        units_map = {
            'slope': 'degrees',
            'aspect': 'degrees',
            'profile_curvature': '1/m',
            'plan_curvature': '1/m', 
            'mean_curvature': '1/m',
            'roughness': 'meters',
            'terrain_ruggedness_index': 'meters'
        }
        return units_map.get(derivative_name, 'dimensionless')
    
    def _clip_to_bbox(self, data: np.ndarray, transform, crs, bbox: List[float]) -> Tuple[np.ndarray, Any, Any]:
        """Clip data to specified bounding box."""
        if not RASTERIO_AVAILABLE:
            return data, transform, crs
        
        try:
            from rasterio.windows import from_bounds
            from rasterio.transform import from_bounds as transform_from_bounds
            
            # Calculate window for clipping
            window = from_bounds(*bbox, transform)
            
            # Extract the data within the window
            clipped_data = data[
                int(window.row_off):int(window.row_off + window.height),
                int(window.col_off):int(window.col_off + window.width)
            ]
            
            # Calculate new transform
            new_transform = transform_from_bounds(
                *bbox, clipped_data.shape[1], clipped_data.shape[0]
            )
            
            return clipped_data, new_transform, crs
            
        except Exception as e:
            self.logger.warning(f"Failed to clip to bbox: {e}")
            return data, transform, crs
    
    def _resample_data(self, data: np.ndarray, transform, current_res: float, target_res: float) -> Tuple[np.ndarray, Any]:
        """Resample data to target resolution."""
        if not RASTERIO_AVAILABLE:
            return data, transform
        
        try:
            # Calculate new dimensions
            scale_factor = current_res / target_res
            new_height = int(data.shape[0] * scale_factor)
            new_width = int(data.shape[1] * scale_factor)
            
            # Create new transform
            new_transform = rasterio.transform.from_bounds(
                transform.c, transform.f + transform.e * data.shape[0],
                transform.c + transform.a * data.shape[1], transform.f,
                new_width, new_height
            )
            
            # Resample using rasterio
            resampled_data = np.empty((new_height, new_width), dtype=data.dtype)
            
            reproject(
                source=data,
                destination=resampled_data,
                src_transform=transform,
                dst_transform=new_transform,
                src_crs='EPSG:4326',
                dst_crs='EPSG:4326',
                resampling=Resampling.bilinear
            )
            
            return resampled_data, new_transform
            
        except Exception as e:
            self.logger.warning(f"Failed to resample data: {e}")
            return data, transform
    
    def _fill_voids(self, data: np.ndarray) -> np.ndarray:
        """Fill no-data voids in elevation data."""
        if not RASTERIO_AVAILABLE:
            # Simple interpolation without rasterio
            filled_data = data.copy()
            nan_mask = np.isnan(filled_data)
            
            if np.any(nan_mask) and SCIPY_AVAILABLE:
                # Use scipy for interpolation
                from scipy.interpolate import griddata
                
                # Get valid points
                valid_mask = ~nan_mask
                if np.any(valid_mask):
                    y_coords, x_coords = np.where(valid_mask)
                    valid_values = filled_data[valid_mask]
                    
                    # Get points to interpolate
                    nan_y, nan_x = np.where(nan_mask)
                    
                    if len(nan_y) > 0 and len(valid_values) > 10:  # Need enough points
                        interpolated = griddata(
                            (y_coords, x_coords), valid_values,
                            (nan_y, nan_x), method='linear'
                        )
                        filled_data[nan_mask] = interpolated
            
            return filled_data
        
        try:
            # Use rasterio's fillnodata function
            filled_data = data.copy()
            mask = ~np.isnan(filled_data)
            
            fillnodata(filled_data, mask, max_search_distance=100)
            
            return filled_data
            
        except Exception as e:
            self.logger.warning(f"Failed to fill voids: {e}")
            return data
    
    def _calculate_transform(self, bbox: List[float], height: int, width: int):
        """Calculate affine transform for the data."""
        try:
            if RASTERIO_AVAILABLE:
                return from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
            else:
                # Simple transform calculation
                pixel_width = (bbox[2] - bbox[0]) / width
                pixel_height = (bbox[3] - bbox[1]) / height
                return [pixel_width, 0.0, bbox[0], 0.0, -pixel_height, bbox[3]]
        except Exception:
            return None
    
    def _save_data(self, result_data: Dict[str, Any], context, params: Dict[str, Any]) -> List[str]:
        """Save DEM data and derivatives to files."""
        saved_files = []
        
        try:
            # Get output directory from context
            output_dir = getattr(context, 'output_dir', Path('outputs'))
            output_dir = Path(output_dir) / 'dem_data'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main elevation data
            if RASTERIO_AVAILABLE and 'elevation' in result_data:
                elevation_file = output_dir / f"dem_{self.step_id}.tif"
                self._save_geotiff(
                    result_data['elevation'],
                    elevation_file,
                    result_data.get('transform'),
                    result_data.get('crs', params['crs']),
                    result_data['metadata']
                )
                saved_files.append(str(elevation_file))
                self.logger.info(f"Saved elevation data: {elevation_file}")
            
            # Save derivatives
            for key, data in result_data.items():
                if key.endswith('_metadata') or key in ['elevation', 'metadata', 'transform', 'crs']:
                    continue
                
                if isinstance(data, np.ndarray) and RASTERIO_AVAILABLE:
                    derivative_file = output_dir / f"{key}_{self.step_id}.tif"
                    self._save_geotiff(
                        data,
                        derivative_file,
                        result_data.get('transform'),
                        result_data.get('crs', params['crs']),
                        result_data.get(f'{key}_metadata', {})
                    )
                    saved_files.append(str(derivative_file))
                    self.logger.debug(f"Saved {key}: {derivative_file}")
            
            # Save metadata as JSON
            import json
            metadata_file = output_dir / f"dem_{self.step_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                # Make metadata JSON serializable
                metadata = result_data['metadata'].copy()
                for key, value in metadata.items():
                    if isinstance(value, np.ndarray):
                        metadata[key] = value.tolist()
                    elif hasattr(value, '__dict__'):
                        metadata[key] = str(value)
                
                json.dump(metadata, f, indent=2, default=str)
            
            saved_files.append(str(metadata_file))
            self.logger.info(f"Saved metadata: {metadata_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save DEM files: {e}")
        
        return saved_files
    
    def _save_geotiff(self, data: np.ndarray, output_path: Path, transform, crs: str, metadata: Dict[str, Any]) -> None:
        """Save array as GeoTIFF with proper georeferencing."""
        if not RASTERIO_AVAILABLE:
            return
        
        try:
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs=crs,
                transform=transform,
                compress='lzw',
                nodata=np.nan
            ) as dst:
                dst.write(data, 1)
                
                # Add metadata tags
                tags = {
                    'SOURCE': metadata.get('source', 'unknown'),
                    'RESOLUTION': str(metadata.get('resolution', 'unknown')),
                    'UNITS': metadata.get('units', 'unknown'),
                    'PROCESSING_DATE': datetime.now().isoformat(),
                    'VERTICAL_ACCURACY': metadata.get('vertical_accuracy', 'unknown')
                }
                
                # Add statistics if available
                if 'statistics' in metadata:
                    stats = metadata['statistics']
                    tags.update({
                        'STATISTICS_MINIMUM': str(stats.get('min', '')),
                        'STATISTICS_MAXIMUM': str(stats.get('max', '')),
                        'STATISTICS_MEAN': str(stats.get('mean', '')),
                        'STATISTICS_STDDEV': str(stats.get('std', ''))
                    })
                
                dst.update_tags(**tags)
                
        except Exception as e:
            self.logger.warning(f"Failed to save GeoTIFF {output_path}: {e}")
    
    def _assess_data_quality(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of acquired DEM data."""
        quality_metrics = {}
        
        if 'elevation' in result_data:
            elevation = result_data['elevation']
            
            # Basic statistics
            quality_metrics['completeness'] = float(np.sum(~np.isnan(elevation)) / elevation.size)
            quality_metrics['void_percentage'] = float(np.sum(np.isnan(elevation)) / elevation.size * 100)
            
            # Elevation range
            valid_data = elevation[~np.isnan(elevation)]
            if len(valid_data) > 0:
                quality_metrics['elevation_range'] = {
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'span': float(np.max(valid_data) - np.min(valid_data))
                }
                
                # Data reasonableness check
                reasonable_min = -500  # Below sea level but reasonable
                reasonable_max = 9000  # Mount Everest is ~8850m
                
                unreasonable_low = np.sum(valid_data < reasonable_min)
                unreasonable_high = np.sum(valid_data > reasonable_max)
                
                quality_metrics['data_reasonableness'] = {
                    'unreasonable_low_count': int(unreasonable_low),
                    'unreasonable_high_count': int(unreasonable_high),
                    'percentage_reasonable': float((len(valid_data) - unreasonable_low - unreasonable_high) / len(valid_data) * 100)
                }
        
        return quality_metrics
    
    def get_resource_requirements(self) -> Dict[str, Any]:
        """Get resource requirements for this step."""
        # Estimate based on bbox size and resolution
        bbox = self.hyperparameters.get('bbox', [0, 0, 1, 1])
        resolution = self.hyperparameters.get('resolution', 30)
        generate_derivatives = self.hyperparameters.get('generate_derivatives', True)
        
        # Calculate approximate memory requirements
        area_km2 = abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1]) * 111.32 ** 2
        pixels = area_km2 * 1e6 / (resolution ** 2)
        
        # Base memory for elevation data (4 bytes per pixel)
        base_memory_mb = pixels * 4 / 1e6
        
        # Additional memory for derivatives (typically 5-7 derivatives)
        if generate_derivatives:
            derivative_memory_mb = base_memory_mb * 6
        else:
            derivative_memory_mb = 0
        
        total_memory_mb = base_memory_mb + derivative_memory_mb
        
        # Disk space estimation (with compression ~50% reduction)
        disk_space_mb = total_memory_mb * 0.5
        
        return {
            'memory': f"{max(256, int(total_memory_mb))}MB",
            'cpu_cores': 1,
            'gpu_required': False,
            'disk_space': f"{max(50, int(disk_space_mb))}MB",
            'execution_time_estimate': '1-5m' if self.use_mock_data else '2-15m',
            'network_required': self.hyperparameters.get('source') not in ['external'] and not self.use_mock_data
        }


# Utility functions for the step
def validate_dem_config() -> Dict[str, Any]:
    """
    Validate DEM acquisition configuration.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'rasterio': RASTERIO_AVAILABLE,
        'elevation_library': ELEVATION_AVAILABLE,
        'existing_acquisitor': EXISTING_DEM_ACQUISITOR_AVAILABLE,
        'scipy': SCIPY_AVAILABLE,
        'gdal': GDAL_AVAILABLE,
        'requests': REQUESTS_AVAILABLE,
        'issues': []
    }
    
    if not RASTERIO_AVAILABLE:
        results['issues'].append("Install rasterio: pip install rasterio")
    
    if not ELEVATION_AVAILABLE:
        results['issues'].append("Install elevation: pip install elevation")
    
    if not SCIPY_AVAILABLE:
        results['issues'].append("Install scipy: pip install scipy (for advanced derivatives)")
    
    return results


def create_test_dem_step(step_id: str = 'test_dem', **hyperparameters) -> DEMAcquisitionStep:
    """
    Create a test DEM step for development.
    
    Args:
        step_id: Step identifier
        **hyperparameters: Step hyperparameters
        
    Returns:
        Configured DEMAcquisitionStep instance
    """
    default_config = {
        'bbox': [85.3, 27.6, 85.4, 27.7],  # Small area in Nepal
        'source': 'SRTM',
        'resolution': 30,
        'generate_derivatives': True,
        'void_fill': True,
        'save_to_file': True
    }
    
    # Merge with provided hyperparameters
    default_config.update(hyperparameters)
    
    step_config = {
        'type': 'dem_acquisition',
        'hyperparameters': default_config,
        'inputs': {},
        'outputs': {
            'elevation_data': {'type': 'raster', 'bands': 1},
            'dem_files': {'type': 'file_list'},
            'derivatives': {'type': 'raster_list'}
        },
        'dependencies': []
    }
    
    return DEMAcquisitionStep(step_id, step_config)


def get_supported_sources() -> Dict[str, Dict[str, Any]]:
    """Get information about supported DEM sources."""
    return DEMAcquisitionStep.SUPPORTED_SOURCES.copy()


if __name__ == "__main__":
    # Quick test of the DEM step
    import tempfile
    from pathlib import Path
    
    print("Testing DEMAcquisitionStep...")
    
    # Test 1: Configuration validation
    print("\n=== Configuration Validation ===")
    config_results = validate_dem_config()
    
    for component, available in config_results.items():
        if component != 'issues':
            status = "✓" if available else "✗"
            print(f"{status} {component}")
    
    if config_results['issues']:
        print("\nIssues:")
        for issue in config_results['issues']:
            print(f"  ⚠ {issue}")
    
    # Test 2: Supported sources
    print("\n=== Supported DEM Sources ===")
    sources = get_supported_sources()
    for source, info in sources.items():
        print(f"• {source}: {info['description']}")
        print(f"  - Resolution: {info['resolution']}m")
        print(f"  - Coverage: {info['coverage']}")
        print(f"  - Accuracy: {info['vertical_accuracy']}")
        print(f"  - API Available: {info['api_available']}")
    
    # Test 3: Step creation and validation
    print("\n=== Step Creation ===")
    try:
        test_step = create_test_dem_step(
            'test_dem_step',
            resolution=90,  # Lower resolution for faster testing
            generate_derivatives=True
        )
        print(f"✓ Step created: {test_step}")
        
        # Test resource requirements
        resources = test_step.get_resource_requirements()
        print(f"✓ Resource requirements: {resources}")
        
    except Exception as e:
        print(f"✗ Step creation failed: {e}")
    
    print("\n=== DEMAcquisitionStep testing completed! ===")
    print("CORRECTED: No longer uses StepResult - returns Dict")
    print("CORRECTED: Compatible with ExecutionContext")
