"""
Local Files Discovery and Loading Step
======================================

This module implements local file discovery and loading for geospatial data,
designed with fail-fast principles for rapid development and testing.

Key Features:
- Flexible file pattern matching (glob patterns)
- Recursive directory traversal
- Multiple geospatial format support
- File validation and metadata extraction
- Integration with existing landslide_pipeline file management
- Mock file generation for testing
- Comprehensive file filtering and sorting
"""

import logging
import os
import glob
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from pathlib import Path
import re

# Import base step infrastructure
from ..base.base_step import BaseStep, StepResult
from ..base.step_registry import StepRegistry, register_step

# Conditional imports for dependencies
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
    from osgeo import gdal, ogr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import netCDF4
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

try:
    from landslide_pipeline.utils.file_path_manager import FilePathManager
    EXISTING_FILE_MANAGER_AVAILABLE = True
except ImportError:
    EXISTING_FILE_MANAGER_AVAILABLE = False


class LocalFilesDiscoveryError(Exception):
    """Custom exception for local file discovery errors."""
    pass


@register_step('local_files_discovery',
               category='data_acquisition',
               aliases=['local_files', 'file_discovery', 'local_data_loading'])
class LocalFilesStep(BaseStep):
    """
    Local files discovery and loading step.
    
    This step handles discovery and loading of local geospatial files including:
    - Raster formats: GeoTIFF, IMG, JP2, HDF, NetCDF, COG
    - Vector formats: Shapefile, GeoPackage, GeoJSON, KML
    - Satellite data: Sentinel-2, Landsat, MODIS
    - DEM data: SRTM, ASTER, ALOS
    - Custom data formats
    
    Supported Parameters:
    - base_path: Base directory for file search
    - file_patterns: List of glob patterns to search
    - recursive: Whether to search subdirectories
    - file_types: Filter by file types
    - date_filter: Filter files by date range
    - size_filter: Filter files by size range
    - validate_files: Whether to validate file integrity
    - load_metadata: Whether to extract detailed metadata
    - sort_by: How to sort discovered files
    """
    
    # Supported file formats and their characteristics
    SUPPORTED_FORMATS = {
        # Raster formats
        '.tif': {'type': 'raster', 'description': 'GeoTIFF', 'lib': 'rasterio'},
        '.tiff': {'type': 'raster', 'description': 'GeoTIFF', 'lib': 'rasterio'},
        '.img': {'type': 'raster', 'description': 'ERDAS Imagine', 'lib': 'rasterio'},
        '.jp2': {'type': 'raster', 'description': 'JPEG2000', 'lib': 'rasterio'},
        '.hdf': {'type': 'raster', 'description': 'HDF4/5', 'lib': 'h5py'},
        '.h5': {'type': 'raster', 'description': 'HDF5', 'lib': 'h5py'},
        '.nc': {'type': 'raster', 'description': 'NetCDF', 'lib': 'netcdf4'},
        '.grd': {'type': 'raster', 'description': 'Surfer Grid', 'lib': 'rasterio'},
        '.asc': {'type': 'raster', 'description': 'ASCII Grid', 'lib': 'rasterio'},
        
        # Vector formats
        '.shp': {'type': 'vector', 'description': 'Shapefile', 'lib': 'geopandas'},
        '.gpkg': {'type': 'vector', 'description': 'GeoPackage', 'lib': 'geopandas'},
        '.geojson': {'type': 'vector', 'description': 'GeoJSON', 'lib': 'geopandas'},
        '.json': {'type': 'vector', 'description': 'JSON/GeoJSON', 'lib': 'geopandas'},
        '.kml': {'type': 'vector', 'description': 'KML', 'lib': 'geopandas'},
        '.gml': {'type': 'vector', 'description': 'GML', 'lib': 'geopandas'},
        
        # Satellite-specific
        '.SAFE': {'type': 'sentinel2', 'description': 'Sentinel-2 SAFE', 'lib': 'custom'},
        '.zip': {'type': 'archive', 'description': 'Compressed archive', 'lib': 'zipfile'},
        
        # Other formats
        '.txt': {'type': 'text', 'description': 'Text data', 'lib': 'builtin'},
        '.csv': {'type': 'tabular', 'description': 'CSV data', 'lib': 'pandas'},
        '.xlsx': {'type': 'tabular', 'description': 'Excel data', 'lib': 'pandas'}
    }
    
    # Common file patterns for different data types
    COMMON_PATTERNS = {
        'all_raster': ['*.tif', '*.tiff', '*.img', '*.jp2', '*.hdf', '*.h5', '*.nc'],
        'all_vector': ['*.shp', '*.gpkg', '*.geojson', '*.json', '*.kml'],
        'sentinel2': ['*_B??.jp2', '*_B*.jp2', '*.SAFE'],
        'landsat': ['*_B*.TIF', '*_B*.tif', '*LC08*', '*LE07*'],
        'dem': ['*dem*.tif', '*elevation*.tif', '*srtm*.tif', '*aster*.tif'],
        'ndvi': ['*ndvi*.tif', '*NDVI*.tif'],
        'indices': ['*ndvi*.tif', '*ndwi*.tif', '*evi*.tif'],
        'optical': ['*B02*.jp2', '*B03*.jp2', '*B04*.jp2', '*B08*.jp2']
    }
    
    def __init__(self, step_id: str, step_config: Dict[str, Any]):
        """Initialize local files discovery step."""
        super().__init__(step_id, step_config)
        
        # Step-specific configuration
        self.discovered_files = []
        self.file_metadata = {}
        self.validation_results = {}
        
        # Initialize configuration
        self._setup_file_discovery_config()
        
        # Log configuration status
        self._log_configuration_status()
    
    def _setup_file_discovery_config(self) -> None:
        """Setup file discovery configuration."""
        try:
            # Check base path
            base_path = self.hyperparameters.get('base_path', '.')
            base_path = Path(base_path).expanduser().resolve()
            
            if not base_path.exists():
                self.logger.warning(f"Base path does not exist: {base_path}")
                # Create directory if parent exists
                if base_path.parent.exists():
                    base_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created base path: {base_path}")
            
            self.base_path = base_path
            
            # Setup file patterns
            self._setup_file_patterns()
            
            # Check integration with existing file manager
            if EXISTING_FILE_MANAGER_AVAILABLE:
                self.logger.info("✓ FilePathManager integration available")
            
            self.logger.info("✓ Local files discovery configuration loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup file discovery config: {e}")
            self.base_path = Path('.')
    
    def _setup_file_patterns(self) -> None:
        """Setup file search patterns."""
        # Get patterns from hyperparameters
        patterns = self.hyperparameters.get('file_patterns', [])
        
        # Handle common pattern shortcuts
        pattern_shortcuts = self.hyperparameters.get('pattern_shortcuts', [])
        for shortcut in pattern_shortcuts:
            if shortcut in self.COMMON_PATTERNS:
                patterns.extend(self.COMMON_PATTERNS[shortcut])
        
        # Default patterns if none specified
        if not patterns:
            patterns = ['*.tif', '*.shp']
        
        self.file_patterns = patterns
        self.logger.debug(f"File search patterns: {self.file_patterns}")
    
    def _log_configuration_status(self) -> None:
        """Log the current configuration status."""
        status_items = []
        
        if RASTERIO_AVAILABLE:
            status_items.append("✓ Rasterio")
        else:
            status_items.append("✗ Rasterio")
        
        if GEOPANDAS_AVAILABLE:
            status_items.append("✓ GeoPandas")
        else:
            status_items.append("✗ GeoPandas")
        
        if GDAL_AVAILABLE:
            status_items.append("✓ GDAL")
        else:
            status_items.append("✗ GDAL")
        
        if EXISTING_FILE_MANAGER_AVAILABLE:
            status_items.append("✓ Existing file manager")
        else:
            status_items.append("✗ Existing file manager")
        
        self.logger.debug(f"Configuration status: {', '.join(status_items)}")
    
    def execute(self, context) -> StepResult:
        """
        Execute local files discovery.
        
        Args:
            context: Pipeline execution context
            
        Returns:
            StepResult with discovery status and found files
        """
        self.logger.info(f"Starting local files discovery: {self.step_id}")
        
        try:
            # Extract and validate parameters
            params = self._extract_parameters()
            self._validate_parameters(params)
            
            # Log discovery parameters
            self._log_discovery_info(params)
            
            # Execute file discovery
            discovered_files = self._discover_files(params)
            
            if not discovered_files:
                # Generate mock files if no real files found
                if params.get('generate_mock_if_empty', False):
                    discovered_files = self._generate_mock_files(params)
                    self.logger.info("✓ Generated mock files for testing")
                else:
                    return StepResult(
                        status='failed',
                        error_message="No files found matching search criteria",
                        metadata={'search_patterns': params['file_patterns']}
                    )
            
            # Filter and sort files
            filtered_files = self._filter_files(discovered_files, params)
            sorted_files = self._sort_files(filtered_files, params)
            
            # Validate files if requested
            if params.get('validate_files', True):
                validation_results = self._validate_files(sorted_files, params)
                valid_files = [f for f, v in validation_results.items() if v.get('valid', False)]
            else:
                valid_files = sorted_files
                validation_results = {}
            
            # Extract metadata if requested
            if params.get('load_metadata', True):
                file_metadata = self._extract_metadata(valid_files, params)
            else:
                file_metadata = {}
            
            # Organize results
            file_results = self._organize_results(valid_files, file_metadata, validation_results, params)
            
            # Store results in context
            output_key = f"{self.step_id}_files"
            context.set_artifact(output_key, file_results, metadata={
                'discovery_method': 'local_files',
                'parameters': params,
                'timestamp': datetime.now().isoformat()
            })
            
            # Create file inventory
            inventory = self._create_file_inventory(file_results, params)
            
            return StepResult(
                status='success',
                outputs={
                    'files_data': output_key,
                    'file_inventory': inventory,
                    'file_count': len(valid_files)
                },
                metadata={
                    'base_path': str(params['base_path']),
                    'patterns_searched': params['file_patterns'],
                    'files_discovered': len(discovered_files),
                    'files_filtered': len(filtered_files),
                    'files_valid': len(valid_files),
                    'validation_enabled': params.get('validate_files', True),
                    'metadata_loaded': params.get('load_metadata', True),
                    'file_types': self._get_file_type_summary(valid_files),
                    'total_size_mb': self._calculate_total_size(valid_files)
                },
                warnings=self._get_warnings(params, discovered_files, valid_files)
            )
            
        except Exception as e:
            self.logger.error(f"Local files discovery failed: {e}")
            return StepResult(
                status='failed',
                error_message=str(e),
                metadata={
                    'step_id': self.step_id,
                    'base_path': str(self.base_path)
                }
            )
    
    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract and process discovery parameters."""
        # Base parameters
        base_path = self.get_input_data(context=None, input_key='base_path',
                                       default=self.hyperparameters.get('base_path', '.'))
        
        params = {
            'base_path': Path(base_path).expanduser().resolve(),
            'file_patterns': self.hyperparameters.get('file_patterns', self.file_patterns),
            'recursive': self.hyperparameters.get('recursive', True),
            'follow_symlinks': self.hyperparameters.get('follow_symlinks', False),
            
            # Filtering options
            'file_types': self.hyperparameters.get('file_types', []),
            'exclude_patterns': self.hyperparameters.get('exclude_patterns', []),
            'min_size_mb': self.hyperparameters.get('min_size_mb', 0),
            'max_size_mb': self.hyperparameters.get('max_size_mb', float('inf')),
            
            # Date filtering
            'date_filter': self.hyperparameters.get('date_filter', {}),
            
            # Processing options
            'validate_files': self.hyperparameters.get('validate_files', True),
            'load_metadata': self.hyperparameters.get('load_metadata', True),
            'sort_by': self.hyperparameters.get('sort_by', 'name'),
            'max_files': self.hyperparameters.get('max_files', None),
            
            # Mock data options
            'generate_mock_if_empty': self.hyperparameters.get('generate_mock_if_empty', False),
            'mock_file_count': self.hyperparameters.get('mock_file_count', 5)
        }
        
        return params
    
    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate discovery parameters."""
        # Validate base path
        base_path = params['base_path']
        if not base_path.exists():
            if not params.get('generate_mock_if_empty', False):
                raise LocalFilesDiscoveryError(f"Base path does not exist: {base_path}")
        
        # Validate file patterns
        if not params['file_patterns']:
            raise LocalFilesDiscoveryError("No file patterns specified")
        
        # Validate size filters
        if params['min_size_mb'] < 0:
            raise LocalFilesDiscoveryError("min_size_mb must be non-negative")
        
        if params['max_size_mb'] <= params['min_size_mb']:
            raise LocalFilesDiscoveryError("max_size_mb must be greater than min_size_mb")
        
        # Validate sort options
        valid_sort_options = ['name', 'size', 'date', 'type', 'path']
        if params['sort_by'] not in valid_sort_options:
            raise LocalFilesDiscoveryError(f"Invalid sort_by option: {params['sort_by']}")
    
    def _log_discovery_info(self, params: Dict[str, Any]) -> None:
        """Log discovery information."""
        self.logger.info(f"File discovery parameters:")
        self.logger.info(f"  • Base path: {params['base_path']}")
        self.logger.info(f"  • Patterns: {params['file_patterns']}")
        self.logger.info(f"  • Recursive: {params['recursive']}")
        self.logger.info(f"  • File types filter: {params['file_types'] or 'None'}")
        self.logger.info(f"  • Size range: {params['min_size_mb']}-{params['max_size_mb']} MB")
        self.logger.info(f"  • Validate files: {params['validate_files']}")
        self.logger.info(f"  • Load metadata: {params['load_metadata']}")
    
    def _discover_files(self, params: Dict[str, Any]) -> List[Path]:
        """Discover files based on patterns and criteria."""
        discovered_files = []
        base_path = params['base_path']
        
        if not base_path.exists():
            self.logger.warning(f"Base path does not exist: {base_path}")
            return discovered_files
        
        for pattern in params['file_patterns']:
            try:
                if params['recursive']:
                    # Recursive search with ** pattern
                    search_pattern = str(base_path / '**' / pattern)
                    found_files = glob.glob(search_pattern, recursive=True)
                else:
                    # Non-recursive search
                    search_pattern = str(base_path / pattern)
                    found_files = glob.glob(search_pattern)
                
                # Convert to Path objects and filter
                pattern_files = []
                for file_path in found_files:
                    path_obj = Path(file_path)
                    
                    # Skip if not a file
                    if not path_obj.is_file():
                        continue
                    
                    # Skip symlinks if not following them
                    if path_obj.is_symlink() and not params['follow_symlinks']:
                        continue
                    
                    pattern_files.append(path_obj)
                
                discovered_files.extend(pattern_files)
                self.logger.debug(f"Pattern '{pattern}' found {len(pattern_files)} files")
                
            except Exception as e:
                self.logger.warning(f"Error searching pattern '{pattern}': {e}")
        
        # Remove duplicates while preserving order
        unique_files = []
        seen = set()
        for file_path in discovered_files:
            if file_path not in seen:
                unique_files.append(file_path)
                seen.add(file_path)
        
        self.logger.info(f"Discovered {len(unique_files)} unique files")
        return unique_files
    
    def _filter_files(self, files: List[Path], params: Dict[str, Any]) -> List[Path]:
        """Filter files based on criteria."""
        filtered_files = []
        
        for file_path in files:
            try:
                # File type filter
                if params['file_types']:
                    file_ext = file_path.suffix.lower()
                    if file_ext not in params['file_types']:
                        continue
                
                # Exclude patterns filter
                if params['exclude_patterns']:
                    skip_file = False
                    for exclude_pattern in params['exclude_patterns']:
                        if file_path.match(exclude_pattern):
                            skip_file = True
                            break
                    if skip_file:
                        continue
                
                # Size filter
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb < params['min_size_mb'] or file_size_mb > params['max_size_mb']:
                    continue
                
                # Date filter
                if params['date_filter']:
                    if not self._check_date_filter(file_path, params['date_filter']):
                        continue
                
                filtered_files.append(file_path)
                
            except Exception as e:
                self.logger.warning(f"Error filtering file {file_path}: {e}")
        
        self.logger.info(f"Filtered to {len(filtered_files)} files")
        return filtered_files
    
    def _check_date_filter(self, file_path: Path, date_filter: Dict[str, Any]) -> bool:
        """Check if file meets date filter criteria."""
        try:
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            # Check date range
            if 'start_date' in date_filter:
                start_date = datetime.fromisoformat(date_filter['start_date'])
                if file_mtime < start_date:
                    return False
            
            if 'end_date' in date_filter:
                end_date = datetime.fromisoformat(date_filter['end_date'])
                if file_mtime > end_date:
                    return False
            
            # Check age
            if 'max_age_days' in date_filter:
                max_age = timedelta(days=date_filter['max_age_days'])
                if datetime.now() - file_mtime > max_age:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error checking date filter for {file_path}: {e}")
            return True  # Include file if date check fails
    
    def _sort_files(self, files: List[Path], params: Dict[str, Any]) -> List[Path]:
        """Sort files based on specified criteria."""
        sort_by = params['sort_by']
        
        try:
            if sort_by == 'name':
                sorted_files = sorted(files, key=lambda f: f.name.lower())
            elif sort_by == 'size':
                sorted_files = sorted(files, key=lambda f: f.stat().st_size, reverse=True)
            elif sort_by == 'date':
                sorted_files = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)
            elif sort_by == 'type':
                sorted_files = sorted(files, key=lambda f: (f.suffix.lower(), f.name.lower()))
            elif sort_by == 'path':
                sorted_files = sorted(files, key=lambda f: str(f).lower())
            else:
                sorted_files = files
            
            # Limit number of files if specified
            if params['max_files']:
                sorted_files = sorted_files[:params['max_files']]
            
            self.logger.debug(f"Sorted files by '{sort_by}', limited to {len(sorted_files)}")
            return sorted_files
            
        except Exception as e:
            self.logger.warning(f"Error sorting files: {e}")
            return files
    
    def _validate_files(self, files: List[Path], params: Dict[str, Any]) -> Dict[Path, Dict[str, Any]]:
        """Validate file integrity and readability."""
        validation_results = {}
        
        for file_path in files:
            try:
                validation = {'valid': True, 'errors': [], 'warnings': []}
                
                # Basic file checks
                if not file_path.exists():
                    validation['valid'] = False
                    validation['errors'].append("File does not exist")
                    validation_results[file_path] = validation
                    continue
                
                if not file_path.is_file():
                    validation['valid'] = False
                    validation['errors'].append("Not a regular file")
                    validation_results[file_path] = validation
                    continue
                
                # Check file size
                if file_path.stat().st_size == 0:
                    validation['valid'] = False
                    validation['errors'].append("File is empty")
                    validation_results[file_path] = validation
                    continue
                
                # Format-specific validation
                file_ext = file_path.suffix.lower()
                
                if file_ext in ['.tif', '.tiff'] and RASTERIO_AVAILABLE:
                    validation.update(self._validate_raster_file(file_path))
                elif file_ext == '.shp' and GEOPANDAS_AVAILABLE:
                    validation.update(self._validate_vector_file(file_path))
                elif file_ext in ['.hdf', '.h5'] and H5PY_AVAILABLE:
                    validation.update(self._validate_hdf_file(file_path))
                elif file_ext == '.nc' and NETCDF4_AVAILABLE:
                    validation.update(self._validate_netcdf_file(file_path))
                
                validation_results[file_path] = validation
                
            except Exception as e:
                validation_results[file_path] = {
                    'valid': False,
                    'errors': [f"Validation error: {str(e)}"],
                    'warnings': []
                }
        
        valid_count = sum(1 for v in validation_results.values() if v.get('valid', False))
        self.logger.info(f"Validated files: {valid_count}/{len(files)} valid")
        
        return validation_results
    
    def _validate_raster_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate raster file using rasterio."""
        validation = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            with rasterio.open(file_path) as src:
                # Check basic properties
                if src.width <= 0 or src.height <= 0:
                    validation['errors'].append("Invalid raster dimensions")
                    validation['valid'] = False
                
                if src.count <= 0:
                    validation['errors'].append("No bands in raster")
                    validation['valid'] = False
                
                # Check CRS
                if not src.crs:
                    validation['warnings'].append("No CRS defined")
                
                # Check for reasonable data range (basic sanity check)
                try:
                    sample_data = src.read(1, window=rasterio.windows.Window(0, 0, min(100, src.width), min(100, src.height)))
                    if sample_data.size > 0:
                        if src.nodata is not None:
                            valid_data = sample_data[sample_data != src.nodata]
                        else:
                            valid_data = sample_data
                        
                        if len(valid_data) == 0:
                            validation['warnings'].append("Sample area contains only nodata")
                        elif len(np.unique(valid_data)) == 1:
                            validation['warnings'].append("Sample area has constant values")
                            
                except Exception:
                    validation['warnings'].append("Could not read sample data")
                
        except RasterioIOError as e:
            validation['valid'] = False
            validation['errors'].append(f"Rasterio read error: {str(e)}")
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(f"Raster validation error: {str(e)}")
        
        return validation
    
    def _validate_vector_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate vector file using geopandas."""
        validation = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            # Try to read just the first few features for validation
            gdf = gpd.read_file(file_path, rows=5)
            
            if len(gdf) == 0:
                validation['warnings'].append("Vector file is empty")
            
            if gdf.crs is None:
                validation['warnings'].append("No CRS defined")
            
            # Check geometry validity
            if hasattr(gdf, 'geometry'):
                invalid_geoms = gdf.geometry.isna().sum()
                if invalid_geoms > 0:
                    validation['warnings'].append(f"{invalid_geoms} features have invalid geometry")
                    
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(f"Vector validation error: {str(e)}")
        
        return validation
    
    def _validate_hdf_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate HDF file using h5py."""
        validation = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            with h5py.File(file_path, 'r') as f:
                if len(f.keys()) == 0:
                    validation['warnings'].append("HDF file contains no datasets")
                    
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(f"HDF validation error: {str(e)}")
        
        return validation
    
    def _validate_netcdf_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate NetCDF file using netCDF4."""
        validation = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            with netCDF4.Dataset(file_path, 'r') as nc:
                if len(nc.variables) == 0:
                    validation['warnings'].append("NetCDF file contains no variables")
                    
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(f"NetCDF validation error: {str(e)}")
        
        return validation
    
    def _extract_metadata(self, files: List[Path], params: Dict[str, Any]) -> Dict[Path, Dict[str, Any]]:
        """Extract metadata from files."""
        metadata_results = {}
        
        for file_path in files:
            try:
                metadata = self._extract_file_metadata(file_path)
                metadata_results[file_path] = metadata
                
            except Exception as e:
                self.logger.warning(f"Error extracting metadata from {file_path}: {e}")
                metadata_results[file_path] = {'error': str(e)}
        
        self.logger.debug(f"Extracted metadata for {len(metadata_results)} files
