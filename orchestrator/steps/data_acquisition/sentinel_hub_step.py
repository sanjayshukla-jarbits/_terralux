#!/usr/bin/env python3
"""
Real Sentinel Hub Acquisition Step - CORRECTED FOR _TERRALUX
===========================================================

This is the corrected implementation that fixes the method signature issue
and integrates properly with the ModularOrchestrator's ExecutionContext.

Key Fixes:
- Renamed from sentinel_hub_step.py to real_sentinel_hub_step.py
- Correct execute(self, context) method signature
- Proper context parameter handling
- Consistent Dict return format
- Template variable integration
- Fallback mechanisms when API unavailable

Author: Pipeline Development Team
Version: 1.0.0-corrected-terralux
"""

import os
import json
import logging
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import time

# Sentinel Hub API imports
try:
    from sentinelhub import (
        SHConfig, 
        BBox, 
        CRS, 
        DataCollection,
        SentinelHubRequest,
        bbox_to_dimensions,
        MimeType
    )
    SENTINELHUB_AVAILABLE = True
except ImportError:
    SENTINELHUB_AVAILABLE = False
    logging.warning("SentinelHub library not available. Install with: pip install sentinelhub")

# Geospatial processing imports
try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logging.warning("Rasterio not available. Install with: pip install rasterio")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available. Install with: pip install numpy")

# Base step import - CORRECTED for TerraLux
try:
    from ..base.base_step import BaseStep
except ImportError:
    # Fallback base class for testing
    class BaseStep:
        def __init__(self, step_id: str, step_config: Dict[str, Any]):
            self.step_id = step_id
            self.step_config = step_config
            self.step_type = step_config.get('type', 'unknown')
            self.hyperparameters = step_config.get('hyperparameters', {})
            self.logger = logging.getLogger(f"Step.{self.step_type}.{self.step_id}")
        
        def execute(self, context) -> Dict[str, Any]:
            raise NotImplementedError("Subclasses must implement execute method")


class TerraluxCacheManager:
    """Cache manager for TerraLux Sentinel Hub data"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        # Use TerraLux-specific cache directory
        self.cache_dir = cache_dir or Path.home() / ".terralux_sentinel_hub" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("TerraluxCacheManager")
    
    def generate_cache_key(self, request_params: Dict[str, Any]) -> str:
        """Generate cache key from request parameters"""
        cache_data = {
            'bbox': request_params.get('bbox'),
            'start_date': request_params.get('start_date'),
            'end_date': request_params.get('end_date'),
            'data_collection': request_params.get('data_collection'),
            'resolution': request_params.get('resolution'),
            'bands': request_params.get('bands', [])
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def is_cached(self, cache_key: str) -> bool:
        """Check if data is cached"""
        cache_file = self.cache_dir / f"{cache_key}.tif"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"
        return cache_file.exists() and metadata_file.exists()
    
    def get_cached_data(self, cache_key: str) -> Tuple[Path, Dict[str, Any]]:
        """Get cached data and metadata"""
        cache_file = self.cache_dir / f"{cache_key}.tif"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return cache_file, metadata
    
    def cache_data(self, cache_key: str, data_path: Path, metadata: Dict[str, Any]) -> None:
        """Cache data and metadata"""
        cache_file = self.cache_dir / f"{cache_key}.tif"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"
        
        # Copy data file
        if data_path != cache_file:
            import shutil
            shutil.copy2(data_path, cache_file)
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


class SentinelHubAcquisitionStep(BaseStep):
    """
    Real Sentinel Hub data acquisition step.
    CORRECTED for TerraLux orchestrator system.
    
    This step acquires satellite imagery from Sentinel Hub API with fallback
    to mock data when API is unavailable or credentials are missing.
    """
    
    def __init__(self, step_id: str, step_config: Dict[str, Any]):
        super().__init__(step_id, step_config)
        
        # Initialize cache manager
        cache_dir = self.hyperparameters.get('cache_directory')
        if cache_dir:
            cache_dir = Path(cache_dir)
        self.cache_manager = TerraluxCacheManager(cache_dir)
        
        # Configuration
        self.use_real_api = self.hyperparameters.get('use_real_api', True)
        self.fallback_to_mock = self.hyperparameters.get('fallback_to_mock', True)
        
        # API configuration
        self.client_id = self.hyperparameters.get('client_id') or os.getenv('SENTINEL_HUB_CLIENT_ID')
        self.client_secret = self.hyperparameters.get('client_secret') or os.getenv('SENTINEL_HUB_CLIENT_SECRET')
        
        # Data acquisition parameters
        self.data_collection = self.hyperparameters.get('data_collection', 'SENTINEL-2-L2A')
        self.resolution = self.hyperparameters.get('resolution', 10)
        self.bands = self.hyperparameters.get('bands', ['B02', 'B03', 'B04', 'B08'])
        self.max_cloud_coverage = self.hyperparameters.get('max_cloud_coverage', 50)
        
        self.logger.debug(f"Initialized {self.__class__.__name__}: {self.step_id}")
    
    def execute(self, context) -> Dict[str, Any]:
        """
        Execute Sentinel Hub data acquisition.
        CORRECTED: Uses context parameter and returns Dict.
        
        Args:
            context: ExecutionContext with variables and artifact storage
            
        Returns:
            Dict with status, outputs, and metadata
        """
        try:
            self.logger.info(f"Starting Sentinel Hub acquisition: {self.step_id}")
            
            # Extract parameters from context and hyperparameters
            params = self._extract_parameters(context)
            self._validate_parameters(params)
            
            # Log acquisition info
            self._log_acquisition_info(params)
            
            # Check if we should use real API
            if self.use_real_api and self._check_api_availability():
                # Try real API acquisition
                try:
                    return self._execute_real_acquisition(params, context)
                except Exception as e:
                    self.logger.warning(f"Real API acquisition failed: {e}")
                    if not self.fallback_to_mock:
                        raise
                    self.logger.info("Falling back to mock data")
            
            # Use mock data
            if self.fallback_to_mock:
                return self._execute_mock_acquisition(params, context)
            else:
                return {
                    'status': 'failed',
                    'outputs': {},
                    'metadata': {
                        'error': 'Real API unavailable and mock fallback disabled',
                        'step_id': self.step_id,
                        'step_type': self.step_type,
                        'timestamp': datetime.now().isoformat()
                    }
                }
                
        except Exception as e:
            error_message = f"Sentinel Hub acquisition failed: {str(e)}"
            self.logger.error(error_message)
            return {
                'status': 'failed',
                'outputs': {},
                'metadata': {
                    'step_id': self.step_id,
                    'error': error_message,
                    'timestamp': datetime.now().isoformat(),
                    'mock': False
                }
            }
    
    def _extract_parameters(self, context) -> Dict[str, Any]:
        """Extract and process acquisition parameters from context"""
        # Get bbox from context variables or hyperparameters
        bbox = context.get_variable('bbox', self.hyperparameters.get('bbox'))
        start_date = context.get_variable('start_date', self.hyperparameters.get('start_date'))
        end_date = context.get_variable('end_date', self.hyperparameters.get('end_date'))
        output_dir = context.get_variable('output_dir', self.hyperparameters.get('output_dir', 'outputs'))
        
        return {
            'bbox': bbox,
            'start_date': start_date,
            'end_date': end_date,
            'output_dir': Path(output_dir),
            'data_collection': self.data_collection,
            'resolution': self.resolution,
            'bands': self.bands,
            'max_cloud_coverage': self.max_cloud_coverage
        }
    
    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate acquisition parameters"""
        if not params.get('bbox'):
            raise ValueError("Missing required parameter: bbox")
        
        bbox = params['bbox']
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat]")
        
        if not params.get('start_date') or not params.get('end_date'):
            raise ValueError("Missing required parameters: start_date, end_date")
    
    def _check_api_availability(self) -> bool:
        """Check if Sentinel Hub API is available"""
        if not SENTINELHUB_AVAILABLE:
            self.logger.warning("SentinelHub library not available")
            return False
        
        if not self.client_id or not self.client_secret:
            self.logger.warning("Sentinel Hub credentials not configured")
            return False
        
        return True
    
    def _log_acquisition_info(self, params: Dict[str, Any]) -> None:
        """Log acquisition information"""
        self.logger.info(f"Acquisition parameters:")
        self.logger.info(f"  - BBox: {params['bbox']}")
        self.logger.info(f"  - Date range: {params['start_date']} to {params['end_date']}")
        self.logger.info(f"  - Collection: {params['data_collection']}")
        self.logger.info(f"  - Resolution: {params['resolution']}m")
        self.logger.info(f"  - Bands: {params['bands']}")
        self.logger.info(f"  - Max cloud: {params['max_cloud_coverage']}%")
    
    def _execute_real_acquisition(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Execute real Sentinel Hub API acquisition"""
        self.logger.info("Using real Sentinel Hub API")
        
        # Check cache first
        cache_key = self.cache_manager.generate_cache_key(params)
        if self.cache_manager.is_cached(cache_key):
            self.logger.info(f"Using cached data: {cache_key}")
            cached_file, cached_metadata = self.cache_manager.get_cached_data(cache_key)
            
            # Store in context
            context.set_artifact('sentinel_imagery', str(cached_file))
            context.set_artifact('sentinel_metadata', cached_metadata)
            
            return {
                'status': 'success',
                'outputs': {
                    'imagery_data': str(cached_file),
                    'metadata': cached_metadata
                },
                'metadata': {
                    'step_id': self.step_id,
                    'step_type': self.step_type,
                    'cache_used': True,
                    'cache_key': cache_key,
                    'timestamp': datetime.now().isoformat(),
                    'mock': False
                }
            }
        
        # Configure Sentinel Hub
        config = SHConfig()
        config.sh_client_id = self.client_id
        config.sh_client_secret = self.client_secret
        
        # Create bbox and dimensions
        bbox = BBox(params['bbox'], crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=params['resolution'])
        
        # Create evalscript
        evalscript = self._create_evalscript(params['bands'])
        
        # Create request
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self._get_data_collection(params['data_collection']),
                    time_interval=(params['start_date'], params['end_date']),
                    maxcc=params['max_cloud_coverage'] / 100.0
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
            ],
            bbox=bbox,
            size=size,
            config=config
        )
        
        # Execute request
        self.logger.info("Executing Sentinel Hub request...")
        start_time = time.time()
        
        data = request.get_data()
        
        execution_time = time.time() - start_time
        self.logger.info(f"API request completed in {execution_time:.2f} seconds")
        
        if not data or len(data) == 0:
            raise RuntimeError("No data returned from Sentinel Hub API")
        
        # Save data
        output_file = params['output_dir'] / f"sentinel_{self.step_id}_{cache_key[:8]}.tif"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save imagery data
        if RASTERIO_AVAILABLE:
            self._save_imagery_data(data[0], output_file, params)
        else:
            # Fallback: save as numpy array
            np.save(output_file.with_suffix('.npy'), data[0])
            output_file = output_file.with_suffix('.npy')
        
        # Create metadata
        metadata = {
            'bbox': params['bbox'],
            'start_date': params['start_date'],
            'end_date': params['end_date'],
            'data_collection': params['data_collection'],
            'resolution': params['resolution'],
            'bands': params['bands'],
            'max_cloud_coverage': params['max_cloud_coverage'],
            'shape': data[0].shape,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat(),
            'source': 'sentinel_hub_api'
        }
        
        # Cache the data
        self.cache_manager.cache_data(cache_key, output_file, metadata)
        
        # Store in context
        context.set_artifact('sentinel_imagery', str(output_file))
        context.set_artifact('sentinel_metadata', metadata)
        
        return {
            'status': 'success',
            'outputs': {
                'imagery_data': str(output_file),
                'metadata': metadata
            },
            'metadata': {
                'step_id': self.step_id,
                'step_type': self.step_type,
                'cache_used': False,
                'cache_key': cache_key,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'mock': False
            }
        }
    
    def _execute_mock_acquisition(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Execute mock data acquisition"""
        self.logger.info("Using mock Sentinel Hub data")
        
        # Generate mock data
        if NUMPY_AVAILABLE:
            # Create realistic mock imagery data
            height, width = 100, 100
            num_bands = len(params['bands'])
            
            # Generate mock data with some realistic patterns
            mock_data = np.random.rand(height, width, num_bands) * 4000  # Typical DN range
            mock_data = mock_data.astype(np.uint16)
        else:
            mock_data = f"mock_sentinel_data_{params['data_collection'].lower()}"
        
        # Create output file
        output_file = params['output_dir'] / f"mock_sentinel_{self.step_id}.tif"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save mock data
        if NUMPY_AVAILABLE and RASTERIO_AVAILABLE:
            self._save_imagery_data(mock_data, output_file, params)
        else:
            # Simple text file for basic mock
            with open(output_file.with_suffix('.txt'), 'w') as f:
                f.write(f"Mock Sentinel Hub data\n")
                f.write(f"BBox: {params['bbox']}\n")
                f.write(f"Date: {params['start_date']} to {params['end_date']}\n")
                f.write(f"Collection: {params['data_collection']}\n")
            output_file = output_file.with_suffix('.txt')
        
        # Create metadata
        metadata = {
            'bbox': params['bbox'],
            'start_date': params['start_date'],
            'end_date': params['end_date'],
            'data_collection': params['data_collection'],
            'resolution': params['resolution'],
            'bands': params['bands'],
            'max_cloud_coverage': params['max_cloud_coverage'],
            'shape': mock_data.shape if NUMPY_AVAILABLE else 'mock',
            'timestamp': datetime.now().isoformat(),
            'source': 'mock_data'
        }
        
        # Store in context
        context.set_artifact('sentinel_imagery', str(output_file))
        context.set_artifact('sentinel_metadata', metadata)
        
        return {
            'status': 'success',
            'outputs': {
                'imagery_data': str(output_file),
                'metadata': metadata
            },
            'metadata': {
                'step_id': self.step_id,
                'step_type': self.step_type,
                'cache_used': False,
                'timestamp': datetime.now().isoformat(),
                'mock': True
            }
        }
    
    def _create_evalscript(self, bands: List[str]) -> str:
        """Create evalscript for Sentinel Hub request"""
        band_setup = []
        band_return = []
        
        for band in bands:
            band_setup.append(f'"{band}"')
            band_return.append(band)
        
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: [{{{','.join(band_setup)}}}],
                output: {{
                    bands: {len(bands)},
                    sampleType: "UINT16"
                }}
            }};
        }}
        
        function evaluatePixel(sample) {{
            return [{','.join([f'sample.{band}' for band in band_return])}];
        }}
        """
    
    def _save_imagery_data(self, data: np.ndarray, output_file: Path, params: Dict[str, Any]) -> None:
        """Save imagery data using rasterio"""
        if not RASTERIO_AVAILABLE:
            raise RuntimeError("Rasterio not available for saving imagery data")
        
        # Handle different data shapes
        if data.ndim == 3:
            height, width, bands = data.shape
            data = np.transpose(data, (2, 0, 1))  # Change to (bands, height, width)
        elif data.ndim == 2:
            bands, height, width = 1, data.shape[0], data.shape[1]
            data = data.reshape(1, height, width)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
        
        # Calculate transform from bbox
        bbox = params['bbox']
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
        
        # Save with rasterio
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(data)


# Register the step (if registry is available)
try:
    from ..base.step_registry import StepRegistry
    StepRegistry.register('sentinel_hub_acquisition', SentinelHubAcquisitionStep)
    logging.info("✓ Registered SentinelHubAcquisitionStep for TerraLux")
except ImportError:
    logging.warning("StepRegistry not available - step not auto-registered")


# For testing
if __name__ == "__main__":
    # Simple test execution
    import tempfile
    
    # Mock context for testing
    class SimpleContext:
        def __init__(self):
            self.variables = {}
            self.artifacts = {}
        
        def get_variable(self, key: str, default=None):
            return self.variables.get(key, default)
        
        def set_artifact(self, key: str, value):
            self.artifacts[key] = value
    
    # Test configuration
    test_config = {
        'type': 'sentinel_hub_acquisition',
        'hyperparameters': {
            'bbox': [85.30, 27.60, 85.32, 27.62],
            'start_date': '2023-06-01',
            'end_date': '2023-06-07',
            'data_collection': 'SENTINEL-2-L2A',
            'resolution': 60,
            'bands': ['B02', 'B03', 'B04', 'B08'],
            'fallback_to_mock': True,
            'cache_directory': tempfile.mkdtemp(prefix='terralux_sentinel_cache_')
        }
    }
    
    # Create and test step
    step = SentinelHubAcquisitionStep('test_sentinel', test_config)
    
    # Create test context
    context = SimpleContext()
    context.variables = {
        'bbox': [85.30, 27.60, 85.32, 27.62],
        'start_date': '2023-06-01',
        'end_date': '2023-06-07',
        'output_dir': 'outputs'
    }
    
    print("🧪 Testing Corrected TerraLux Sentinel Hub Step")
    print("=" * 50)
    
    result = step.execute(context)
    
    print(f"Status: {result['status']}")
    print(f"Mock used: {result['metadata'].get('mock', False)}")
    print(f"Outputs: {list(result['outputs'].keys())}")
    print(f"Data path: {result['outputs'].get('imagery_data', 'N/A')}")
    
    print("\n✅ Test completed - Step uses correct method signature!")
