"""
Local Files Discovery and Loading Step
======================================

This module implements local file discovery and loading for geospatial data,
designed with fail-fast principles for rapid development and testing.
CORRECTED for compatibility with ModularOrchestrator ExecutionContext.

Key Features:
- Flexible file pattern matching (glob patterns)
- Recursive directory traversal
- Multiple geospatial format support
- File validation and metadata extraction
- Mock file generation for testing
- Comprehensive file filtering and sorting
- Compatible with ModularOrchestrator ExecutionContext
- Returns Dict instead of deprecated StepResult
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

# CORRECTED: Import base step infrastructure
try:
    from ..base.base_step import BaseStep
except ImportError:
    # Fallback for standalone testing
    logging.warning("BaseStep not available, using fallback implementation")
    
    class BaseStep:
        def __init__(self, step_id: str, step_config: Dict[str, Any]):
            self.step_id = step_id
            self.step_config = step_config
            self.step_type = step_config.get('type', 'unknown')
            self.hyperparameters = step_config.get('hyperparameters', {})
            self.logger = logging.getLogger(f"Step.{self.step_type}.{step_id}")
        
        def execute(self, context) -> Dict[str, Any]:
            """Abstract execute method"""
            raise NotImplementedError("Subclasses must implement execute method")

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


class LocalFilesStep(BaseStep):
    """
    Local files discovery and loading step.
    CORRECTED for compatibility with ModularOrchestrator ExecutionContext
    and to return Dict instead of StepResult.
    
    This step discovers and validates geospatial files in local directories
    with support for various file formats and filtering options.
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        'raster': ['.tif', '.tiff', '.jp2', '.img', '.hdf', '.nc', '.grd'],
        'vector': ['.shp', '.gpkg', '.geojson', '.kml', '.gml'],
        'other': ['.txt', '.csv', '.json', '.xml']
    }
    
    # Common file patterns for different data types
    COMMON_PATTERNS = {
        'all_geospatial': ['*.tif', '*.tiff', '*.shp', '*.gpkg', '*.geojson'],
        'raster': ['*.tif', '*.tiff', '*.jp2', '*.img', '*.hdf', '*.nc'],
        'vector': ['*.shp', '*.gpkg', '*.geojson', '*.kml'],
        'sentinel2': ['*.jp2', '*_B*.jp2', '*.SAFE'],
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
                # Create directory if possible
                try:
                    base_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created base path: {base_path}")
                except Exception as e:
                    self.logger.warning(f"Could not create base path: {e}")
            
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
            patterns = ['*.tif', '*.tiff', '*.shp', '*.gpkg', '*.geojson']
        
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
    
    def execute(self, context) -> Dict[str, Any]:
        """
        Execute local files discovery.
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
        self.logger.info(f"Starting local files discovery: {self.step_id}")
        
        try:
            # Extract and validate parameters
            params = self._extract_parameters(context)
            self._validate_parameters(params)
            
            # Log discovery parameters
            self._log_discovery_info(params)
            
            # Discover files
            discovered_files = self._discover_files(params)
            
            # Filter files
            filtered_files = self._filter_files(discovered_files, params)
            
            # Validate files if requested
            valid_files = []
            if params.get('validate_files', True):
                valid_files = self._validate_files(filtered_files, params)
                self.logger.debug(f"Validated files: {len(valid_files)}/{len(filtered_files)} valid")
            else:
                valid_files = filtered_files
            
            # Generate mock files if no files found and enabled
            if not valid_files and params.get('generate_mock_if_empty', False):
                mock_files = self._generate_mock_files(params)
                valid_files.extend(mock_files)
                self.logger.info("✓ Generated mock files for testing")
            
            # Sort and limit files
            valid_files = self._sort_and_limit_files(valid_files, params)
            
            # Extract metadata if requested
            if params.get('load_metadata', True):
                self._extract_file_metadata(valid_files, params)
                self.logger.debug(f"Extracted metadata for {len(valid_files)} files")
            
            # Create file inventory
            inventory = self._create_file_inventory(valid_files, params)
            
            # Store results in context
            output_key = f"{self.step_id}_files"
            context.set_artifact(output_key, valid_files)
            
            return {
                'status': 'success',
                'outputs': {
                    'files_data': output_key,
                    'file_inventory': inventory,
                    'file_count': len(valid_files)
                },
                'metadata': {
                    'base_path': str(params['base_path']),
                    'patterns_searched': params['file_patterns'],
                    'files_discovered': len(discovered_files),
                    'files_filtered': len(filtered_files),
                    'files_valid': len(valid_files),
                    'validation_enabled': params.get('validate_files', True),
                    'metadata_loaded': params.get('load_metadata', True),
                    'file_types': self._get_file_type_summary(valid_files),
                    'total_size_mb': self._calculate_total_size(valid_files),
                    'step_id': self.step_id,
                    'step_type': self.step_type,
                    'warnings': self._get_warnings(params, discovered_files, valid_files)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Local files discovery failed: {e}")
            return {
                'status': 'failed',
                'outputs': {},
                'metadata': {
                    'error': str(e),
                    'step_id': self.step_id,
                    'step_type': self.step_type,
                    'base_path': str(self.hyperparameters.get('base_path', '.'))
                }
            }
    
    def _extract_parameters(self, context) -> Dict[str, Any]:
        """
        Extract and process discovery parameters.
        CORRECTED to use ExecutionContext properly.
        """
        # Base parameters - get from context variables or hyperparameters
        base_path = context.get_variable('local_data_path', self.hyperparameters.get('base_path', '.'))
        
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
            'mock_file_count': self.hyperparameters.get('mock_file_count', 3)
        }
        
        return params
    
    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate discovery parameters."""
        # Validate base path
        base_path = params['base_path']
        if not base_path.exists() and not params.get('generate_mock_if_empty', False):
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
        self.logger.info("File discovery parameters:")
        self.logger.info(f"  • Base path: {params['base_path']}")
        self.logger.info(f"  • Patterns: {params['file_patterns']}")
        self.logger.info(f"  • Recursive: {params['recursive']}")
        self.logger.info(f"  • File types filter: {params['file_types'] or 'None'}")
        self.logger.info(f"  • Size range: {params['min_size_mb']}-{params['max_size_mb']} MB")
        self.logger.info(f"  • Validate files: {params['validate_files']}")
        self.logger.info(f"  • Load metadata: {params['load_metadata']}")
    
    def _discover_files(self, params: Dict[str, Any]) -> List[Path]:
        """Discover files based on patterns."""
        discovered_files = []
        base_path = params['base_path']
        
        for pattern in params['file_patterns']:
            if params['recursive']:
                # Use recursive glob
                pattern_path = base_path / '**' / pattern
                found_files = list(base_path.glob(f"**/{pattern}"))
            else:
                # Use non-recursive glob
                found_files = list(base_path.glob(pattern))
            
            # Filter symlinks if not following them
            if not params['follow_symlinks']:
                found_files = [f for f in found_files if not f.is_symlink()]
            
            # Only include files (not directories)
            found_files = [f for f in found_files if f.is_file()]
            
            discovered_files.extend(found_files)
            self.logger.debug(f"Pattern '{pattern}' found {len(found_files)} files")
        
        # Remove duplicates while preserving order
        unique_files = []
        seen = set()
        for f in discovered_files:
            if f not in seen:
                unique_files.append(f)
                seen.add(f)
        
        self.logger.info(f"Discovered {len(unique_files)} unique files")
        return unique_files
    
    def _filter_files(self, files: List[Path], params: Dict[str, Any]) -> List[Path]:
        """Apply filtering criteria to discovered files."""
        filtered_files = files.copy()
        
        # Filter by file types
        if params['file_types']:
            type_extensions = []
            for file_type in params['file_types']:
                if file_type in self.SUPPORTED_EXTENSIONS:
                    type_extensions.extend(self.SUPPORTED_EXTENSIONS[file_type])
            
            if type_extensions:
                filtered_files = [f for f in filtered_files if f.suffix.lower() in type_extensions]
                self.logger.debug(f"File type filter: {len(filtered_files)} files remain")
        
        # Filter by exclude patterns
        if params['exclude_patterns']:
            for pattern in params['exclude_patterns']:
                filtered_files = [f for f in filtered_files if not f.match(pattern)]
            self.logger.debug(f"Exclude patterns filter: {len(filtered_files)} files remain")
        
        # Filter by file size
        size_filtered = []
        min_size_bytes = params['min_size_mb'] * 1024 * 1024
        max_size_bytes = params['max_size_mb'] * 1024 * 1024
        
        for file_path in filtered_files:
            try:
                file_size = file_path.stat().st_size
                if min_size_bytes <= file_size <= max_size_bytes:
                    size_filtered.append(file_path)
            except (OSError, IOError) as e:
                self.logger.warning(f"Could not get size for {file_path}: {e}")
        
        filtered_files = size_filtered
        self.logger.debug(f"Size filter: {len(filtered_files)} files remain")
        
        # Filter by date if specified
        if params['date_filter']:
            filtered_files = self._apply_date_filter(filtered_files, params['date_filter'])
            self.logger.debug(f"Date filter: {len(filtered_files)} files remain")
        
        return filtered_files
    
    def _apply_date_filter(self, files: List[Path], date_filter: Dict[str, Any]) -> List[Path]:
        """Apply date-based filtering."""
        if not date_filter:
            return files
        
        filtered_files = []
        
        # Parse date filter options
        after_date = date_filter.get('after')
        before_date = date_filter.get('before')
        
        if after_date:
            after_date = datetime.fromisoformat(after_date) if isinstance(after_date, str) else after_date
        if before_date:
            before_date = datetime.fromisoformat(before_date) if isinstance(before_date, str) else before_date
        
        for file_path in files:
            try:
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                # Check date constraints
                if after_date and file_mtime < after_date:
                    continue
                if before_date and file_mtime > before_date:
                    continue
                
                filtered_files.append(file_path)
                
            except (OSError, IOError) as e:
                self.logger.warning(f"Could not get modification time for {file_path}: {e}")
        
        return filtered_files
    
    def _validate_files(self, files: List[Path], params: Dict[str, Any]) -> List[Path]:
        """Validate files and return only valid ones."""
        valid_files = []
        
        for file_path in files:
            try:
                # Basic file validation
                if not file_path.exists():
                    continue
                
                if not file_path.is_file():
                    continue
                
                # File format validation based on extension
                file_ext = file_path.suffix.lower()
                
                if file_ext in ['.tif', '.tiff'] and RASTERIO_AVAILABLE:
                    if self._validate_raster_file(file_path):
                        valid_files.append(file_path)
                elif file_ext in ['.shp', '.gpkg', '.geojson'] and GEOPANDAS_AVAILABLE:
                    if self._validate_vector_file(file_path):
                        valid_files.append(file_path)
                else:
                    # For other file types, just check if they're readable
                    if self._validate_generic_file(file_path):
                        valid_files.append(file_path)
                
            except Exception as e:
                self.logger.warning(f"Validation failed for {file_path}: {e}")
        
        return valid_files
    
    def _validate_raster_file(self, file_path: Path) -> bool:
        """Validate raster file using rasterio."""
        try:
            with rasterio.open(file_path) as src:
                # Check if file has valid CRS and bounds
                return src.crs is not None and src.bounds is not None
        except Exception:
            return False
    
    def _validate_vector_file(self, file_path: Path) -> bool:
        """Validate vector file using geopandas."""
        try:
            gdf = gpd.read_file(file_path)
            return not gdf.empty and gdf.geometry is not None
        except Exception:
            return False
    
    def _validate_generic_file(self, file_path: Path) -> bool:
        """Generic file validation."""
        try:
            # Check if file is readable and not empty
            return file_path.stat().st_size > 0
        except Exception:
            return False
    
    def _sort_and_limit_files(self, files: List[Path], params: Dict[str, Any]) -> List[Path]:
        """Sort files and apply limits."""
        sort_by = params['sort_by']
        
        # Sort files
        if sort_by == 'name':
            sorted_files = sorted(files, key=lambda f: f.name)
        elif sort_by == 'size':
            sorted_files = sorted(files, key=lambda f: f.stat().st_size, reverse=True)
        elif sort_by == 'date':
            sorted_files = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)
        elif sort_by == 'type':
            sorted_files = sorted(files, key=lambda f: f.suffix)
        elif sort_by == 'path':
            sorted_files = sorted(files, key=lambda f: str(f))
        else:
            sorted_files = files
        
        self.logger.debug(f"Sorted files by '{sort_by}', limited to {params.get('max_files', 'unlimited')}")
        
        # Apply limit
        max_files = params.get('max_files')
        if max_files and max_files > 0:
            sorted_files = sorted_files[:max_files]
        
        return sorted_files
    
    def _extract_file_metadata(self, files: List[Path], params: Dict[str, Any]) -> None:
        """Extract metadata from files."""
        for file_path in files:
            try:
                metadata = {
                    'path': str(file_path),
                    'name': file_path.name,
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'extension': file_path.suffix.lower()
                }
                
                # Add format-specific metadata
                if file_path.suffix.lower() in ['.tif', '.tiff'] and RASTERIO_AVAILABLE:
                    metadata.update(self._get_raster_metadata(file_path))
                elif file_path.suffix.lower() in ['.shp', '.gpkg', '.geojson'] and GEOPANDAS_AVAILABLE:
                    metadata.update(self._get_vector_metadata(file_path))
                
                self.file_metadata[str(file_path)] = metadata
                
            except Exception as e:
                self.logger.warning(f"Failed to extract metadata for {file_path}: {e}")
    
    def _get_raster_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get raster-specific metadata."""
        try:
            with rasterio.open(file_path) as src:
                return {
                    'type': 'raster',
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'crs': str(src.crs) if src.crs else None,
                    'bounds': list(src.bounds) if src.bounds else None,
                    'dtype': str(src.dtypes[0]) if src.dtypes else None
                }
        except Exception:
            return {'type': 'raster', 'error': 'Could not read raster metadata'}
    
    def _get_vector_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get vector-specific metadata."""
        try:
            gdf = gpd.read_file(file_path)
            return {
                'type': 'vector',
                'feature_count': len(gdf),
                'geometry_type': gdf.geometry.geom_type.iloc[0] if not gdf.empty else None,
                'crs': str(gdf.crs) if gdf.crs else None,
                'bounds': list(gdf.total_bounds) if not gdf.empty else None,
                'columns': list(gdf.columns)
            }
        except Exception:
            return {'type': 'vector', 'error': 'Could not read vector metadata'}
    
    def _create_file_inventory(self, files: List[Path], params: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive file inventory."""
        inventory = {
            'total_files': len(files),
            'base_path': str(params['base_path']),
            'search_patterns': params['file_patterns'],
            'discovery_time': datetime.now().isoformat(),
            'files': []
        }
        
        for file_path in files:
            file_info = {
                'path': str(file_path),
                'relative_path': str(file_path.relative_to(params['base_path'])),
                'name': file_path.name,
                'extension': file_path.suffix.lower(),
                'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2)
            }
            
            # Add metadata if available
            if str(file_path) in self.file_metadata:
                file_info.update(self.file_metadata[str(file_path)])
            
            inventory['files'].append(file_info)
        
        return inventory
    
    def _get_file_type_summary(self, files: List[Path]) -> Dict[str, int]:
        """Get summary of file types."""
        type_counts = {}
        for file_path in files:
            ext = file_path.suffix.lower()
            type_counts[ext] = type_counts.get(ext, 0) + 1
        return type_counts
    
    def _calculate_total_size(self, files: List[Path]) -> float:
        """Calculate total size of files in MB."""
        total_size = 0
        for file_path in files:
            try:
                total_size += file_path.stat().st_size
            except (OSError, IOError):
                pass
        return round(total_size / (1024 * 1024), 2)
    
    def _get_warnings(self, params: Dict[str, Any], discovered_files: List[Path], valid_files: List[Path]) -> List[str]:
        """Get warnings about the discovery process."""
        warnings = []
        
        if not discovered_files:
            warnings.append("No files found matching the specified patterns")
        
        invalid_count = len(discovered_files) - len(valid_files)
        if invalid_count > 0:
            warnings.append(f"{invalid_count} files failed validation")
        
        if not params['base_path'].exists():
            warnings.append(f"Base path does not exist: {params['base_path']}")
        
        if not RASTERIO_AVAILABLE:
            warnings.append("Rasterio not available - raster file validation limited")
        
        if not GEOPANDAS_AVAILABLE:
            warnings.append("GeoPandas not available - vector file validation limited")
        
        return warnings
    
    def _generate_mock_files(self, params: Dict[str, Any]) -> List[Path]:
        """Generate mock files for testing when no real files are found."""
        self.logger.info("Generating mock files for testing")
        
        mock_files = []
        base_path = params['base_path']
        base_path.mkdir(parents=True, exist_ok=True)
        
        mock_count = params.get('mock_file_count', 3)
        
        for i in range(mock_count):
            # Create mock file names based on patterns
            if '*.tif' in params['file_patterns'] or '*.tiff' in params['file_patterns']:
                mock_file = base_path / f"mock_raster_{i:02d}.tif"
            elif '*.shp' in params['file_patterns']:
                mock_file = base_path / f"mock_vector_{i:02d}.shp"
            elif '*.geojson' in params['file_patterns']:
                mock_file = base_path / f"mock_vector_{i:02d}.geojson"
            else:
                mock_file = base_path / f"mock_file_{i:02d}.txt"
            
            # Create empty mock file
            mock_file.touch()
            mock_files.append(mock_file)
        
        self.logger.info(f"Generated {len(mock_files)} mock files for testing")
        return mock_files


# Register the step (if registry is available)
try:
    from ..base.step_registry import StepRegistry
    StepRegistry.register('local_files_discovery', LocalFilesStep)
    logging.info("✓ Registered LocalFilesStep")
except ImportError:
    logging.warning("StepRegistry not available - step not auto-registered")


# Utility functions
def validate_local_files_config() -> Dict[str, Any]:
    """Validate local files discovery configuration."""
    return {
        'rasterio': RASTERIO_AVAILABLE,
        'geopandas': GEOPANDAS_AVAILABLE,
        'gdal': GDAL_AVAILABLE,
        'h5py': H5PY_AVAILABLE,
        'netcdf4': NETCDF4_AVAILABLE,
        'existing_file_manager': EXISTING_FILE_MANAGER_AVAILABLE,
        'issues': []
    }


def create_test_local_files_step(step_id: str, **hyperparameters) -> LocalFilesStep:
    """
    Create a local files step for testing.
    
    Args:
        step_id: Step identifier
        **hyperparameters: Step hyperparameters
        
    Returns:
        Configured LocalFilesStep instance
    """
    default_config = {
        'base_path': '.',
        'file_patterns': ['*.tif', '*.shp', '*.geojson'],
        'recursive': True,
        'validate_files': True,
        'load_metadata': True,
        'generate_mock_if_empty': True,
        'mock_file_count': 3
    }
    
    # Merge with provided hyperparameters
    default_config.update(hyperparameters)
    
    step_config = {
        'type': 'local_files_discovery',
        'hyperparameters': default_config,
        'inputs': {},
        'outputs': {
            'files_data': {'type': 'file_list'},
            'file_inventory': {'type': 'metadata'},
            'file_count': {'type': 'integer'}
        },
        'dependencies': []
    }
    
    return LocalFilesStep(step_id, step_config)


if __name__ == "__main__":
    # Test the step
    import tempfile
    
    print("Testing LocalFilesStep...")
    
    # Test 1: Configuration validation
    print("\n=== Configuration Validation ===")
    config_results = validate_local_files_config()
    
    for component, available in config_results.items():
        if component != 'issues':
            status = "✓" if available else "✗"
            print(f"{status} {component}")
    
    # Test 2: Step creation
    print("\n=== Step Creation ===")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_step = create_test_local_files_step(
                'test_local_files',
                base_path=temp_dir,
                file_patterns=['*.tif', '*.txt'],
                generate_mock_if_empty=True
            )
            print(f"✓ Step created: {test_step.step_id}")
            print(f"✓ Base path: {test_step.base_path}")
            print(f"✓ File patterns: {test_step.file_patterns}")
            
    except Exception as e:
        print(f"✗ Step creation failed: {e}")
    
    # Test 3: Mock execution
    print("\n=== Mock Execution Test ===")
    try:
        # Create simple context mock
        class SimpleContext:
            def __init__(self):
                self.variables = {'local_data_path': '.', 'output_dir': 'outputs'}
                self.artifacts = {}
            
            def get_variable(self, key, default=None):
                return self.variables.get(key, default)
            
            def set_artifact(self, key, value):
                self.artifacts[key] = value
        
        context = SimpleContext()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_step = create_test_local_files_step(
                'test_execution',
                base_path=temp_dir,
                generate_mock_if_empty=True,
                mock_file_count=2
            )
            
            result = test_step.execute(context)
            
            if result['status'] == 'success':
                print("✓ Mock execution successful")
                print(f"  Files found: {result['outputs']['file_count']}")
                print(f"  File types: {result['metadata']['file_types']}")
                print(f"  Total size: {result['metadata']['total_size_mb']} MB")
            else:
                print(f"✗ Mock execution failed: {result['metadata'].get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"✗ Mock execution test failed: {e}")
    
    print("\n=== LocalFilesStep testing completed! ===")
    print("✅ CORRECTED: Uses proper execute(self, context) signature")
    print("✅ CORRECTED: Returns Dict instead of StepResult")
    print("✅ CORRECTED: Compatible with ExecutionContext")
    print("✅ CORRECTED: Fixed missing context helper methods")
