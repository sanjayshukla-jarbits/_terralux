"""
Continuous probability surface generation step for landslide and mineral mapping.

This step generates smooth probability surfaces from trained models, with advanced
interpolation, smoothing, and risk/prospectivity zone delineation capabilities.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy import ndimage
from scipy.interpolation import griddata, RBFInterpolator
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import joblib
from typing import Dict, Any, Optional, Tuple, List, Union
import logging

from ..base.base_step import BaseStep

logger = logging.getLogger(__name__)


class ProbabilityMappingStep(BaseStep):
    """
    Continuous probability surface generation step.
    
    Creates smooth probability maps from model predictions with interpolation,
    smoothing, and zone delineation for landslide susceptibility and mineral
    prospectivity mapping.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None
        self.probability_surface_: Optional[np.ndarray] = None
        
    def get_step_type(self) -> str:
        return "probability_mapping"
    
    def get_required_inputs(self) -> list:
        return ['trained_model', 'prediction_features']
    
    def get_outputs(self) -> list:
        return ['probability_surface', 'risk_zones', 'surface_statistics']
    
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """Validate probability mapping hyperparameters."""
        # Validate application type
        application_type = hyperparameters.get('application_type', 'landslide_susceptibility')
        valid_types = ['landslide_susceptibility', 'mineral_targeting', 'generic']
        if application_type not in valid_types:
            logger.error(f"Invalid application_type: {application_type}. Must be one of {valid_types}")
            return False
        
        # Validate interpolation method
        interpolation_method = hyperparameters.get('interpolation_method', 'kriging')
        valid_methods = ['kriging', 'rbf', 'idw', 'linear', 'cubic', 'nearest']
        if interpolation_method not in valid_methods:
            logger.error(f"Invalid interpolation_method: {interpolation_method}. Must be one of {valid_methods}")
            return False
        
        # Validate smoothing parameters
        smoothing_sigma = hyperparameters.get('smoothing_sigma', 1.0)
        if not isinstance(smoothing_sigma, (int, float)) or smoothing_sigma < 0:
            logger.error("smoothing_sigma must be a non-negative number")
            return False
        
        # Validate resolution
        output_resolution = hyperparameters.get('output_resolution', 30.0)
        if not isinstance(output_resolution, (int, float)) or output_resolution <= 0:
            logger.error("output_resolution must be a positive number")
            return False
        
        # Validate probability thresholds
        risk_thresholds = hyperparameters.get('risk_thresholds', [0.2, 0.4, 0.6, 0.8])
        if not isinstance(risk_thresholds, list) or not all(0 <= t <= 1 for t in risk_thresholds):
            logger.error("risk_thresholds must be a list of values between 0 and 1")
            return False
        
        return True
    
    def execute(self, context) -> Dict[str, Any]:
        """
        Execute probability mapping.
        
        Args:
            context: Pipeline context containing inputs and configuration
            
        Returns:
            Dictionary containing probability surface results and outputs
        """
        try:
            hyperparameters = context.get('hyperparameters', {})
            
            # Validate hyperparameters
            if not self.validate_hyperparameters(hyperparameters):
                return {
                    'status': 'failed',
                    'error': 'Invalid hyperparameters',
                    'outputs': {}
                }
            
            # Load trained model
            model_info = context['inputs']['trained_model']
            model, scaler, feature_names = self._load_trained_model(model_info)
            
            if model is None:
                return {
                    'status': 'failed',
                    'error': 'Failed to load trained model',
                    'outputs': {}
                }
            
            # Load prediction features and spatial information
            features_path = context['inputs']['prediction_features']
            features_data, spatial_info = self._load_prediction_data(
                features_path, hyperparameters
            )
            
            if features_data is None:
                return {
                    'status': 'failed',
                    'error': 'Failed to load prediction features',
                    'outputs': {}
                }
            
            # Prepare features for prediction
            X_pred, coordinates, valid_mask = self._prepare_prediction_data(
                features_data, feature_names, scaler, spatial_info, hyperparameters
            )
            
            # Generate probability predictions
            logger.info(f"Generating probabilities for {X_pred.shape[0]} samples")
            probabilities = self._generate_probabilities(model, X_pred, hyperparameters)
            
            # Create spatial grid for interpolation
            grid_bounds, grid_transform, grid_shape = self._create_spatial_grid(
                coordinates, spatial_info, hyperparameters
            )
            
            # Interpolate probability surface
            probability_surface = self._interpolate_probability_surface(
                coordinates, probabilities, grid_bounds, grid_shape, 
                grid_transform, hyperparameters
            )
            
            # Apply smoothing if requested
            if hyperparameters.get('apply_smoothing', True):
                probability_surface = self._apply_smoothing(
                    probability_surface, hyperparameters
                )
            
            # Generate risk/prospectivity zones
            risk_zones = self._generate_risk_zones(
                probability_surface, hyperparameters
            )
            
            # Calculate surface statistics
            surface_statistics = self._calculate_surface_statistics(
                probability_surface, risk_zones, hyperparameters
            )
            
            # Save spatial outputs
            spatial_outputs = self._save_spatial_outputs(
                probability_surface, risk_zones, grid_transform, 
                spatial_info, hyperparameters, context
            )
            
            # Store results
            self.model = model
            self.scaler = scaler
            self.feature_names_ = feature_names
            self.probability_surface_ = probability_surface
            
            # Prepare outputs
            outputs = {
                'probability_surface': spatial_outputs.get('probability_raster'),
                'risk_zones': spatial_outputs.get('risk_zones_raster'),
                'surface_statistics': surface_statistics,
                'grid_information': {
                    'bounds': grid_bounds,
                    'shape': grid_shape,
                    'transform': grid_transform,
                    'resolution': hyperparameters.get('output_resolution', 30.0)
                }
            }
            
            # Add vector outputs if generated
            if spatial_outputs.get('risk_zones_vector'):
                outputs['risk_zones_vector'] = spatial_outputs['risk_zones_vector']
            
            logger.info("Probability mapping completed successfully")
            logger.info(f"Generated {grid_shape[0]}x{grid_shape[1]} probability surface")
            
            return {
                'status': 'success',
                'message': 'Probability mapping completed successfully',
                'outputs': outputs,
                'metadata': {
                    'execution_time': context.get('execution_time', 0),
                    'grid_size': grid_shape,
                    'resolution': hyperparameters.get('output_resolution', 30.0),
                    'interpolation_method': hyperparameters.get('interpolation_method', 'kriging'),
                    'application_type': hyperparameters.get('application_type', 'generic')
                }
            }
            
        except Exception as e:
            logger.error(f"Probability mapping failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'outputs': {}
            }
    
    def _load_trained_model(self, model_info: Union[str, Dict[str, Any]]) -> Tuple[Any, Optional[StandardScaler], List[str]]:
        """Load trained model from file or dictionary."""
        try:
            if isinstance(model_info, dict) and 'model_object' in model_info:
                model = model_info['model_object']
                scaler = model_info.get('scaler_object', None)
                feature_names = model_info.get('feature_names', [])
            else:
                # Load from file
                model_path = model_info if isinstance(model_info, str) else model_info.get('model_path')
                model_data = joblib.load(model_path)
                model = model_data['model']
                scaler = model_data.get('scaler', None)
                feature_names = model_data.get('feature_names', [])
            
            logger.info(f"Loaded model: {model.__class__.__name__}")
            return model, scaler, feature_names
            
        except Exception as e:
            logger.error(f"Failed to load trained model: {str(e)}")
            return None, None, []
    
    def _load_prediction_data(self, features_path: str, 
                            hyperparameters: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Load prediction data with spatial information."""
        try:
            spatial_info = None
            
            if features_path.endswith('.csv'):
                features_data = pd.read_csv(features_path)
                
                # Extract coordinates
                if all(col in features_data.columns for col in ['x', 'y']):
                    spatial_info = {
                        'type': 'points',
                        'crs': hyperparameters.get('input_crs', 'EPSG:4326'),
                        'bounds': [
                            features_data['x'].min(), features_data['y'].min(),
                            features_data['x'].max(), features_data['y'].max()
                        ]
                    }
                    
            elif features_path.endswith(('.tif', '.tiff')):
                features_data, spatial_info = self._load_raster_data(features_path)
                
            elif features_path.endswith(('.shp', '.geojson')):
                gdf = gpd.read_file(features_path)
                features_data = pd.DataFrame(gdf.drop('geometry', axis=1))
                
                # Extract coordinates from geometry
                centroids = gdf.geometry.centroid
                features_data['x'] = centroids.x
                features_data['y'] = centroids.y
                
                spatial_info = {
                    'type': 'vector',
                    'crs': str(gdf.crs),
                    'bounds': list(gdf.total_bounds)
                }
                
            else:
                logger.error(f"Unsupported file format: {features_path}")
                return None, None
            
            logger.info(f"Loaded prediction data: {features_data.shape}")
            return features_data, spatial_info
            
        except Exception as e:
            logger.error(f"Failed to load prediction data: {str(e)}")
            return None, None
    
    def _load_raster_data(self, raster_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load data from raster file."""
        with rasterio.open(raster_path) as src:
            data = src.read()
            
            spatial_info = {
                'type': 'raster',
                'transform': src.transform,
                'crs': str(src.crs),
                'shape': (src.height, src.width),
                'bounds': list(src.bounds)
            }
            
            # Sample points from raster
            height, width = src.height, src.width
            rows, cols = np.mgrid[0:height:10, 0:width:10]  # Sample every 10th pixel
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            
            # Extract feature values
            feature_data = {}
            for band in range(data.shape[0]):
                band_data = data[band]
                feature_data[f'band_{band+1}'] = band_data[rows, cols].flatten()
            
            feature_data['x'] = np.array(xs).flatten()
            feature_data['y'] = np.array(ys).flatten()
            
            features_df = pd.DataFrame(feature_data)
            
        return features_df, spatial_info
    
    def _prepare_prediction_data(self, features_data: pd.DataFrame, feature_names: List[str],
                               scaler: Optional[StandardScaler], spatial_info: Dict[str, Any],
                               hyperparameters: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare prediction data and extract coordinates."""
        # Select required features
        available_features = [col for col in feature_names if col in features_data.columns]
        missing_features = [col for col in feature_names if col not in features_data.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        if not available_features:
            raise ValueError("No required features found in prediction data")
        
        # Extract feature matrix
        X = features_data[available_features].values
        
        # Extract coordinates
        if 'x' in features_data.columns and 'y' in features_data.columns:
            coordinates = features_data[['x', 'y']].values
        else:
            raise ValueError("Coordinate columns 'x' and 'y' not found in data")
        
        # Handle missing values
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(coordinates).any(axis=1)
        X_clean = X[valid_mask]
        coordinates_clean = coordinates[valid_mask]
        
        # Apply scaling if available
        if scaler is not None:
            X_clean = scaler.transform(X_clean)
        
        logger.info(f"Prepared {X_clean.shape[0]} valid samples from {X.shape[0]} total samples")
        
        return X_clean, coordinates_clean, valid_mask
    
    def _generate_probabilities(self, model, X: np.ndarray, 
                              hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Generate probability predictions from model."""
        application_type = hyperparameters.get('application_type', 'generic')
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            
            # For binary classification, use positive class probability
            if probabilities.shape[1] == 2:
                return probabilities[:, 1]
            # For multi-class, use maximum probability or weighted combination
            elif application_type in ['landslide_susceptibility', 'mineral_targeting']:
                # Weight higher classes more for risk/prospectivity
                weights = np.arange(probabilities.shape[1]) / (probabilities.shape[1] - 1)
                return np.sum(probabilities * weights, axis=1)
            else:
                return np.max(probabilities, axis=1)
                
        elif hasattr(model, 'decision_function'):
            # Convert decision function to probabilities
            decision_scores = model.decision_function(X)
            if decision_scores.ndim == 1:
                # Binary classification - use sigmoid
                return 1 / (1 + np.exp(-decision_scores))
            else:
                # Multi-class - use softmax and take max
                exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                return np.max(probabilities, axis=1)
        else:
            # No probability support - use predictions as binary
            predictions = model.predict(X)
            return predictions.astype(float)
    
    def _create_spatial_grid(self, coordinates: np.ndarray, spatial_info: Dict[str, Any],
                           hyperparameters: Dict[str, Any]) -> Tuple[List[float], Any, Tuple[int, int]]:
        """Create spatial grid for interpolation."""
        # Get bounds
        if 'bounds' in spatial_info:
            bounds = spatial_info['bounds']
        else:
            bounds = [
                coordinates[:, 0].min(), coordinates[:, 1].min(),
                coordinates[:, 0].max(), coordinates[:, 1].max()
            ]
        
        # Add buffer
        buffer_factor = hyperparameters.get('buffer_factor', 0.1)
        x_range = bounds[2] - bounds[0]
        y_range = bounds[3] - bounds[1]
        
        bounds = [
            bounds[0] - x_range * buffer_factor,
            bounds[1] - y_range * buffer_factor,
            bounds[2] + x_range * buffer_factor,
            bounds[3] + y_range * buffer_factor
        ]
        
        # Calculate grid size
        resolution = hyperparameters.get('output_resolution', 30.0)
        width = int((bounds[2] - bounds[0]) / resolution)
        height = int((bounds[3] - bounds[1]) / resolution)
        
        # Create transform
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
        
        logger.info(f"Created spatial grid: {height}x{width} at {resolution}m resolution")
        
        return bounds, transform, (height, width)
    
    def _interpolate_probability_surface(self, coordinates: np.ndarray, probabilities: np.ndarray,
                                       bounds: List[float], grid_shape: Tuple[int, int],
                                       transform: Any, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Interpolate probability surface using specified method."""
        method = hyperparameters.get('interpolation_method', 'kriging')
        
        # Create grid coordinates
        height, width = grid_shape
        x_grid = np.linspace(bounds[0], bounds[2], width)
        y_grid = np.linspace(bounds[1], bounds[3], height)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        logger.info(f"Interpolating probability surface using {method} method")
        
        if method == 'kriging':
            surface = self._kriging_interpolation(
                coordinates, probabilities, grid_points, hyperparameters
            )
        elif method == 'rbf':
            surface = self._rbf_interpolation(
                coordinates, probabilities, grid_points, hyperparameters
            )
        elif method == 'idw':
            surface = self._idw_interpolation(
                coordinates, probabilities, grid_points, hyperparameters
            )
        else:
            # Use scipy griddata for linear, cubic, nearest
            surface = griddata(
                coordinates, probabilities, grid_points, 
                method=method, fill_value=0.0
            )
        
        # Reshape to grid
        surface_grid = surface.reshape(height, width)
        
        # Ensure values are in [0, 1] range
        surface_grid = np.clip(surface_grid, 0.0, 1.0)
        
        return surface_grid
    
    def _kriging_interpolation(self, coordinates: np.ndarray, values: np.ndarray,
                             grid_points: np.ndarray, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Perform kriging interpolation using Gaussian Process."""
        try:
            # Set up Gaussian Process with RBF kernel
            length_scale = hyperparameters.get('kriging_length_scale', 1000.0)
            noise_level = hyperparameters.get('kriging_noise_level', 0.01)
            
            kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, random_state=42)
            
            # Fit and predict
            gp.fit(coordinates, values)
            predictions, _ = gp.predict(grid_points, return_std=False)
            
            return predictions
            
        except Exception as e:
            logger.warning(f"Kriging failed, falling back to RBF: {str(e)}")
            return self._rbf_interpolation(coordinates, values, grid_points, hyperparameters)
    
    def _rbf_interpolation(self, coordinates: np.ndarray, values: np.ndarray,
                          grid_points: np.ndarray, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Perform Radial Basis Function interpolation."""
        try:
            function = hyperparameters.get('rbf_function', 'thin_plate_spline')
            smoothing = hyperparameters.get('rbf_smoothing', 0.0)
            
            rbf = RBFInterpolator(coordinates, values, kernel=function, smoothing=smoothing)
            predictions = rbf(grid_points)
            
            return predictions
            
        except Exception as e:
            logger.warning(f"RBF failed, falling back to linear: {str(e)}")
            return griddata(coordinates, values, grid_points, method='linear', fill_value=0.0)
    
    def _idw_interpolation(self, coordinates: np.ndarray, values: np.ndarray,
                          grid_points: np.ndarray, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Perform Inverse Distance Weighting interpolation."""
        power = hyperparameters.get('idw_power', 2.0)
        max_distance = hyperparameters.get('idw_max_distance', None)
        
        predictions = np.zeros(len(grid_points))
        
        for i, point in enumerate(grid_points):
            # Calculate distances
            distances = np.sqrt(np.sum((coordinates - point) ** 2, axis=1))
            
            # Apply maximum distance filter if specified
            if max_distance is not None:
                valid_mask = distances <= max_distance
                distances = distances[valid_mask]
                local_values = values[valid_mask]
            else:
                local_values = values
            
            if len(distances) == 0:
                predictions[i] = 0.0
                continue
            
            # Avoid division by zero
            distances = np.maximum(distances, 1e-10)
            
            # Calculate weights
            weights = 1.0 / (distances ** power)
            weights /= np.sum(weights)
            
            # Weighted average
            predictions[i] = np.sum(weights * local_values)
        
        return predictions
    
    def _apply_smoothing(self, surface: np.ndarray, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Apply smoothing to probability surface."""
        smoothing_method = hyperparameters.get('smoothing_method', 'gaussian')
        sigma = hyperparameters.get('smoothing_sigma', 1.0)
        
        if smoothing_method == 'gaussian':
            smoothed = ndimage.gaussian_filter(surface, sigma=sigma)
        elif smoothing_method == 'uniform':
            size = int(sigma * 2 + 1)
            smoothed = ndimage.uniform_filter(surface, size=size)
        elif smoothing_method == 'median':
            size = int(sigma * 2 + 1)
            smoothed = ndimage.median_filter(surface, size=size)
        else:
            logger.warning(f"Unknown smoothing method: {smoothing_method}")
            smoothed = surface
        
        # Preserve original range
        smoothed = np.clip(smoothed, 0.0, 1.0)
        
        logger.info(f"Applied {smoothing_method} smoothing with sigma={sigma}")
        return smoothed
    
    def _generate_risk_zones(self, probability_surface: np.ndarray, 
                           hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Generate risk/prospectivity zones from probability surface."""
        application_type = hyperparameters.get('application_type', 'generic')
        
        # Get thresholds
        if application_type == 'landslide_susceptibility':
            default_thresholds = [0.2, 0.4, 0.6, 0.8]
            zone_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        elif application_type == 'mineral_targeting':
            default_thresholds = [0.25, 0.5, 0.75]
            zone_labels = ['Background', 'Low Potential', 'Moderate Potential', 'High Potential']
        else:
            default_thresholds = [0.33, 0.67]
            zone_labels = ['Low', 'Medium', 'High']
        
        thresholds = hyperparameters.get('risk_thresholds', default_thresholds)
        
        # Create zones
        zones = np.zeros_like(probability_surface, dtype=np.int16)
        
        for i, threshold in enumerate(sorted(thresholds)):
            zones[probability_surface >= threshold] = i + 1
        
        logger.info(f"Generated {len(thresholds) + 1} risk zones using thresholds: {thresholds}")
        
        return zones
    
    def _calculate_surface_statistics(self, probability_surface: np.ndarray, 
                                    risk_zones: np.ndarray, 
                                    hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for the probability surface."""
        stats = {
            'probability_statistics': {
                'mean': float(np.mean(probability_surface)),
                'std': float(np.std(probability_surface)),
                'min': float(np.min(probability_surface)),
                'max': float(np.max(probability_surface)),
                'median': float(np.median(probability_surface)),
                'q25': float(np.percentile(probability_surface, 25)),
                'q75': float(np.percentile(probability_surface, 75))
            },
            'zone_statistics': {}
        }
        
        # Zone statistics
        unique_zones, zone_counts = np.unique(risk_zones, return_counts=True)
        total_pixels = risk_zones.size
        
        application_type = hyperparameters.get('application_type', 'generic')
        
        if application_type == 'landslide_susceptibility':
            zone_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        elif application_type == 'mineral_targeting':
            zone_labels = ['Background', 'Low Potential', 'Moderate Potential', 'High Potential']
        else:
            zone_labels = [f'Zone_{i}' for i in range(len(unique_zones))]
        
        for zone_id, count in zip(unique_zones, zone_counts):
            zone_name = zone_labels[zone_id] if zone_id < len(zone_labels) else f'Zone_{zone_id}'
            
            zone_mask = risk_zones == zone_id
            zone_probs = probability_surface[zone_mask]
            
            stats['zone_statistics'][zone_name] = {
                'pixel_count': int(count),
                'area_percentage': float(count / total_pixels * 100),
                'probability_mean': float(np.mean(zone_probs)),
                'probability_std': float(np.std(zone_probs))
            }
        
        return stats
    
    def _save_spatial_outputs(self, probability_surface: np.ndarray, risk_zones: np.ndarray,
                            transform: Any, spatial_info: Dict[str, Any],
                            hyperparameters: Dict[str, Any], 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Save spatial outputs to files."""
        outputs = {}
        
        output_dir = context.get('output_dir', 'outputs/probability_mapping')
        os.makedirs(output_dir, exist_ok=True)
        
        # Get CRS
        output_crs = hyperparameters.get('output_crs', spatial_info.get('crs', 'EPSG:4326'))
        
        try:
            # Save probability surface
            prob_path = os.path.join(output_dir, 'probability_surface.tif')
            with rasterio.open(
                prob_path, 'w',
                driver='GTiff',
                height=probability_surface.shape[0],
                width=probability_surface.shape[1],
                count=1,
                dtype=rasterio.float32,
                crs=output_crs,
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(probability_surface.astype(rasterio.float32), 1)
                dst.set_band_description(1, 'Probability')
            
            outputs['probability_raster'] = prob_path
            
            # Save risk zones
            zones_path = os.path.join(output_dir, 'risk_zones.tif')
            with rasterio.open(
                zones_path, 'w',
                driver='GTiff',
                height=risk_zones.shape[0],
                width=risk_zones.shape[1],
                count=1,
                dtype=rasterio.int16,
                crs=output_crs,
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(risk_zones.astype(rasterio.int16), 1)
                dst.set_band_description(1, 'Risk Zones')
            
            outputs['risk_zones_raster'] = zones_path
            
            # Optionally create vector zones
            if hyperparameters.get('create_vector_zones', False):
                vector_path = self._create_vector_zones(
                    risk_zones, transform, output_crs, output_dir, hyperparameters
                )
                if vector_path:
                    outputs['risk_zones_vector'] = vector_path
            
            logger.info(f"Saved spatial outputs to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save spatial outputs: {str(e)}")
        
        return outputs
    
    def _create_vector_zones(self, zones: np.ndarray, transform: Any, crs: str,
                           output_dir: str, hyperparameters: Dict[str, Any]) -> Optional[str]:
        """Create vector polygons from risk zones."""
        try:
            from rasterio.features import shapes
            import fiona
            
            # Convert zones to polygons
            shapes_gen = shapes(zones.astype(np.int16), transform=transform)
            
            # Collect geometries and properties
            features = []
            application_type = hyperparameters.get('application_type', 'generic')
            
            if application_type == 'landslide_susceptibility':
                zone_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
            elif application_type == 'mineral_targeting':
                zone_labels = ['Background', 'Low Potential', 'Moderate Potential', 'High Potential']
            else:
                zone_labels = [f'Zone_{i}' for i in range(int(zones.max()) + 1)]
            
            for geom, zone_id in shapes_gen:
                if zone_id > 0:  # Skip background (0)
                    zone_name = zone_labels[zone_id] if zone_id < len(zone_labels) else f'Zone_{zone_id}'
                    
                    features.append({
                        'geometry': geom,
                        'properties': {
                            'zone_id': int(zone_id),
                            'zone_name': zone_name,
                            'risk_level': zone_name
                        }
                    })
            
            # Save to GeoJSON
            vector_path = os.path.join(output_dir, 'risk_zones.geojson')
            
            schema = {
                'geometry': 'Polygon',
                'properties': {
                    'zone_id': 'int',
                    'zone_name': 'str',
                    'risk_level': 'str'
                }
            }
            
            with fiona.open(vector_path, 'w', driver='GeoJSON', crs=crs, schema=schema) as dst:
                for feature in features:
                    dst.write(feature)
            
            logger.info(f"Created vector zones: {len(features)} polygons")
            return vector_path
            
        except Exception as e:
            logger.error(f"Failed to create vector zones: {str(e)}")
            return None
    
    def get_probability_surface(self) -> Optional[np.ndarray]:
        """Get the generated probability surface."""
        return self.probability_surface_
    
    def predict_probabilities(self, features_data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities for new data."""
        if self.model is None:
            raise ValueError("Model not loaded. Execute step first or load model.")
        
        # Prepare features
        if isinstance(features_data, pd.DataFrame):
            if self.feature_names_ is None:
                raise ValueError("Feature names not available")
            X = features_data[self.feature_names_].values
        else:
            X = features_data
        
        # Apply scaling if available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Generate probabilities
        return self._generate_probabilities(self.model, X, {})
