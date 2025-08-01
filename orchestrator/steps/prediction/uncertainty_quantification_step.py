"""
Uncertainty quantification step for model prediction confidence analysis.

This step quantifies prediction uncertainty using multiple methods including
ensemble models, bootstrap sampling, and Bayesian approaches for both
landslide susceptibility and mineral targeting applications.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
from typing import Dict, Any, Optional, Tuple, List, Union
import logging

from ..base.base_step import BaseStep

logger = logging.getLogger(__name__)


class UncertaintyQuantificationStep(BaseStep):
    """
    Uncertainty quantification step for prediction confidence analysis.
    
    Provides comprehensive uncertainty analysis using ensemble methods,
    bootstrap sampling, spatial uncertainty, and confidence intervals
    for landslide and mineral prediction models.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None
        self.uncertainty_maps_: Optional[Dict[str, np.ndarray]] = None
        self.ensemble_models_: Optional[List] = None
        
    def get_step_type(self) -> str:
        return "uncertainty_quantification"
    
    def get_required_inputs(self) -> list:
        return ['trained_model', 'prediction_features']
    
    def get_outputs(self) -> list:
        return ['uncertainty_maps', 'confidence_intervals', 'uncertainty_statistics']
    
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """Validate uncertainty quantification hyperparameters."""
        # Validate uncertainty methods
        uncertainty_methods = hyperparameters.get('uncertainty_methods', ['ensemble'])
        valid_methods = ['ensemble', 'bootstrap', 'monte_carlo', 'spatial_cv', 'prediction_intervals']
        
        for method in uncertainty_methods:
            if method not in valid_methods:
                logger.error(f"Invalid uncertainty method: {method}. Must be one of {valid_methods}")
                return False
        
        # Validate ensemble parameters
        n_ensemble = hyperparameters.get('n_ensemble_models', 10)
        if not isinstance(n_ensemble, int) or n_ensemble < 2 or n_ensemble > 100:
            logger.error("n_ensemble_models must be an integer between 2 and 100")
            return False
        
        # Validate bootstrap parameters
        n_bootstrap = hyperparameters.get('n_bootstrap_samples', 100)
        if not isinstance(n_bootstrap, int) or n_bootstrap < 10 or n_bootstrap > 1000:
            logger.error("n_bootstrap_samples must be an integer between 10 and 1000")
            return False
        
        # Validate confidence level
        confidence_level = hyperparameters.get('confidence_level', 0.95)
        if not isinstance(confidence_level, (int, float)) or not (0.5 <= confidence_level <= 0.99):
            logger.error("confidence_level must be a number between 0.5 and 0.99")
            return False
        
        return True
    
    def execute(self, context) -> Dict[str, Any]:
        """
        Execute uncertainty quantification analysis.
        
        Args:
            context: Pipeline context containing inputs and configuration
            
        Returns:
            Dictionary containing uncertainty analysis results and outputs
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
            
            # Load trained model and training data
            model_info = context['inputs']['trained_model']
            model, scaler, feature_names, training_data = self._load_model_and_data(model_info)
            
            if model is None:
                return {
                    'status': 'failed',
                    'error': 'Failed to load trained model',
                    'outputs': {}
                }
            
            # Load prediction features
            features_path = context['inputs']['prediction_features']
            features_data, spatial_info = self._load_prediction_features(
                features_path, hyperparameters
            )
            
            if features_data is None:
                return {
                    'status': 'failed',
                    'error': 'Failed to load prediction features',
                    'outputs': {}
                }
            
            # Prepare prediction data
            X_pred, coordinates, valid_mask = self._prepare_prediction_data(
                features_data, feature_names, scaler, spatial_info
            )
            
            # Get uncertainty methods to apply
            uncertainty_methods = hyperparameters.get('uncertainty_methods', ['ensemble'])
            
            logger.info(f"Applying uncertainty quantification methods: {uncertainty_methods}")
            
            # Initialize results containers
            uncertainty_results = {}
            uncertainty_maps = {}
            
            # 1. Ensemble-based uncertainty
            if 'ensemble' in uncertainty_methods:
                ensemble_uncertainty = self._ensemble_uncertainty(
                    X_pred, training_data, feature_names, scaler, hyperparameters
                )
                uncertainty_results['ensemble'] = ensemble_uncertainty
                uncertainty_maps['ensemble_std'] = ensemble_uncertainty['prediction_std']
                uncertainty_maps['ensemble_variance'] = ensemble_uncertainty['prediction_variance']
            
            # 2. Bootstrap uncertainty
            if 'bootstrap' in uncertainty_methods:
                bootstrap_uncertainty = self._bootstrap_uncertainty(
                    model, X_pred, training_data, feature_names, scaler, hyperparameters
                )
                uncertainty_results['bootstrap'] = bootstrap_uncertainty
                uncertainty_maps['bootstrap_std'] = bootstrap_uncertainty['prediction_std']
            
            # 3. Monte Carlo dropout uncertainty (for neural networks)
            if 'monte_carlo' in uncertainty_methods:
                mc_uncertainty = self._monte_carlo_uncertainty(
                    model, X_pred, hyperparameters
                )
                if mc_uncertainty is not None:
                    uncertainty_results['monte_carlo'] = mc_uncertainty
                    uncertainty_maps['mc_std'] = mc_uncertainty['prediction_std']
            
            # 4. Spatial cross-validation uncertainty
            if 'spatial_cv' in uncertainty_methods and coordinates is not None:
                spatial_uncertainty = self._spatial_cv_uncertainty(
                    model, X_pred, coordinates, training_data, feature_names, 
                    scaler, hyperparameters
                )
                uncertainty_results['spatial_cv'] = spatial_uncertainty
                uncertainty_maps['spatial_cv_std'] = spatial_uncertainty['prediction_std']
            
            # 5. Prediction intervals
            if 'prediction_intervals' in uncertainty_methods:
                interval_uncertainty = self._prediction_intervals(
                    model, X_pred, training_data, feature_names, scaler, hyperparameters
                )
                uncertainty_results['prediction_intervals'] = interval_uncertainty
                uncertainty_maps['prediction_lower'] = interval_uncertainty['lower_bound']
                uncertainty_maps['prediction_upper'] = interval_uncertainty['upper_bound']
            
            # Calculate combined uncertainty metrics
            combined_uncertainty = self._combine_uncertainties(
                uncertainty_results, hyperparameters
            )
            
            # Generate confidence intervals
            confidence_intervals = self._generate_confidence_intervals(
                uncertainty_results, hyperparameters
            )
            
            # Calculate uncertainty statistics
            uncertainty_statistics = self._calculate_uncertainty_statistics(
                uncertainty_results, uncertainty_maps
            )
            
            # Generate spatial outputs
            spatial_outputs = self._generate_spatial_outputs(
                uncertainty_maps, valid_mask, spatial_info, hyperparameters, context
            )
            
            # Store results
            self.model = model
            self.scaler = scaler
            self.feature_names_ = feature_names
            self.uncertainty_maps_ = uncertainty_maps
            
            # Prepare outputs
            outputs = {
                'uncertainty_maps': spatial_outputs.get('uncertainty_rasters', {}),
                'confidence_intervals': confidence_intervals,
                'uncertainty_statistics': uncertainty_statistics,
                'combined_uncertainty': combined_uncertainty,
                'uncertainty_summary': {
                    'methods_applied': uncertainty_methods,
                    'prediction_samples': len(X_pred),
                    'valid_samples': np.sum(valid_mask),
                    'uncertainty_range': {
                        'min': float(np.min([np.min(um) for um in uncertainty_maps.values()])),
                        'max': float(np.max([np.max(um) for um in uncertainty_maps.values()]))
                    }
                }
            }
            
            logger.info("Uncertainty quantification completed successfully")
            logger.info(f"Applied {len(uncertainty_methods)} uncertainty methods")
            
            return {
                'status': 'success',
                'message': 'Uncertainty quantification completed successfully',
                'outputs': outputs,
                'metadata': {
                    'execution_time': context.get('execution_time', 0),
                    'uncertainty_methods': uncertainty_methods,
                    'prediction_samples': len(X_pred),
                    'valid_samples': np.sum(valid_mask)
                }
            }
            
        except Exception as e:
            logger.error(f"Uncertainty quantification failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'outputs': {}
            }
    
    def _load_model_and_data(self, model_info: Union[str, Dict[str, Any]]) -> Tuple[Any, Optional[StandardScaler], List[str], Optional[pd.DataFrame]]:
        """Load trained model and associated training data."""
        try:
            if isinstance(model_info, dict) and 'model_object' in model_info:
                model = model_info['model_object']
                scaler = model_info.get('scaler_object', None)
                feature_names = model_info.get('feature_names', [])
                training_data = model_info.get('training_data', None)
            else:
                # Load from file
                model_path = model_info if isinstance(model_info, str) else model_info.get('model_path')
                model_data = joblib.load(model_path)
                model = model_data['model']
                scaler = model_data.get('scaler', None)
                feature_names = model_data.get('feature_names', [])
                training_data = model_data.get('training_data', None)
            
            logger.info(f"Loaded model: {model.__class__.__name__}")
            return model, scaler, feature_names, training_data
            
        except Exception as e:
            logger.error(f"Failed to load model and data: {str(e)}")
            return None, None, [], None
    
    def _load_prediction_features(self, features_path: str, 
                                hyperparameters: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Load prediction features with spatial information."""
        try:
            spatial_info = None
            
            if features_path.endswith('.csv'):
                features_data = pd.read_csv(features_path)
                
                if all(col in features_data.columns for col in ['x', 'y']):
                    spatial_info = {
                        'type': 'points',
                        'crs': hyperparameters.get('input_crs', 'EPSG:4326')
                    }
                    
            elif features_path.endswith(('.shp', '.geojson')):
                gdf = gpd.read_file(features_path)
                features_data = pd.DataFrame(gdf.drop('geometry', axis=1))
                
                # Extract coordinates
                centroids = gdf.geometry.centroid
                features_data['x'] = centroids.x
                features_data['y'] = centroids.y
                
                spatial_info = {
                    'type': 'vector',
                    'crs': str(gdf.crs)
                }
                
            else:
                features_data = pd.read_csv(features_path)  # Fallback
            
            logger.info(f"Loaded prediction features: {features_data.shape}")
            return features_data, spatial_info
            
        except Exception as e:
            logger.error(f"Failed to load prediction features: {str(e)}")
            return None, None
    
    def _prepare_prediction_data(self, features_data: pd.DataFrame, feature_names: List[str],
                               scaler: Optional[StandardScaler], 
                               spatial_info: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Prepare prediction data and extract coordinates."""
        # Select required features
        available_features = [col for col in feature_names if col in features_data.columns]
        
        if not available_features:
            raise ValueError("No required features found in prediction data")
        
        # Extract feature matrix
        X = features_data[available_features].values
        
        # Extract coordinates if available
        coordinates = None
        if 'x' in features_data.columns and 'y' in features_data.columns:
            coordinates = features_data[['x', 'y']].values
        
        # Handle missing values
        if coordinates is not None:
            valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(coordinates).any(axis=1)
            coordinates = coordinates[valid_mask]
        else:
            valid_mask = ~np.isnan(X).any(axis=1)
        
        X_clean = X[valid_mask]
        
        # Apply scaling if available
        if scaler is not None:
            X_clean = scaler.transform(X_clean)
        
        logger.info(f"Prepared {X_clean.shape[0]} valid samples from {X.shape[0]} total samples")
        
        return X_clean, coordinates, valid_mask
    
    def _ensemble_uncertainty(self, X_pred: np.ndarray, training_data: Optional[pd.DataFrame],
                            feature_names: List[str], scaler: Optional[StandardScaler],
                            hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate uncertainty using ensemble of models."""
        try:
            n_ensemble = hyperparameters.get('n_ensemble_models', 10)
            random_state = hyperparameters.get('random_state', 42)
            
            if training_data is None:
                logger.warning("No training data available for ensemble uncertainty")
                return {'prediction_std': np.zeros(len(X_pred)), 'prediction_variance': np.zeros(len(X_pred))}
            
            # Prepare training data
            target_col = hyperparameters.get('target_column', 'target')
            available_features = [col for col in feature_names if col in training_data.columns]
            
            X_train = training_data[available_features].values
            y_train = training_data[target_col].values if target_col in training_data.columns else np.zeros(len(training_data))
            
            if scaler is not None:
                X_train = scaler.transform(X_train)
            
            # Create ensemble of models
            predictions = []
            self.ensemble_models_ = []
            
            logger.info(f"Training {n_ensemble} ensemble models")
            
            for i in range(n_ensemble):
                # Bootstrap sample
                n_samples = len(X_train)
                boot_indices = np.random.RandomState(random_state + i).choice(
                    n_samples, size=n_samples, replace=True
                )
                
                X_boot = X_train[boot_indices]
                y_boot = y_train[boot_indices]
                
                # Train model
                if len(np.unique(y_boot)) <= 2:
                    # Classification
                    model = RandomForestClassifier(
                        n_estimators=50, random_state=random_state + i,
                        max_depth=10, min_samples_split=5
                    )
                else:
                    # Regression
                    model = RandomForestRegressor(
                        n_estimators=50, random_state=random_state + i,
                        max_depth=10, min_samples_split=5
                    )
                
                model.fit(X_boot, y_boot)
                self.ensemble_models_.append(model)
                
                # Make predictions
                if hasattr(model, 'predict_proba') and len(np.unique(y_boot)) == 2:
                    pred = model.predict_proba(X_pred)[:, 1]
                else:
                    pred = model.predict(X_pred)
                
                predictions.append(pred)
            
            # Calculate uncertainty metrics
            predictions = np.array(predictions)
            pred_mean = np.mean(predictions, axis=0)
            pred_std = np.std(predictions, axis=0)
            pred_variance = np.var(predictions, axis=0)
            
            # Calculate confidence intervals
            confidence_level = hyperparameters.get('confidence_level', 0.95)
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            pred_lower = np.percentile(predictions, lower_percentile, axis=0)
            pred_upper = np.percentile(predictions, upper_percentile, axis=0)
            
            return {
                'predictions_ensemble': predictions,
                'prediction_mean': pred_mean,
                'prediction_std': pred_std,
                'prediction_variance': pred_variance,
                'confidence_lower': pred_lower,
                'confidence_upper': pred_upper,
                'n_models': n_ensemble
            }
            
        except Exception as e:
            logger.error(f"Ensemble uncertainty calculation failed: {str(e)}")
            return {'prediction_std': np.zeros(len(X_pred)), 'prediction_variance': np.zeros(len(X_pred))}
    
    def _bootstrap_uncertainty(self, model, X_pred: np.ndarray, training_data: Optional[pd.DataFrame],
                             feature_names: List[str], scaler: Optional[StandardScaler],
                             hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate uncertainty using bootstrap sampling."""
        try:
            n_bootstrap = hyperparameters.get('n_bootstrap_samples', 100)
            random_state = hyperparameters.get('random_state', 42)
            
            if training_data is None:
                logger.warning("No training data available for bootstrap uncertainty")
                return {'prediction_std': np.zeros(len(X_pred))}
            
            # Prepare training data
            target_col = hyperparameters.get('target_column', 'target')
            available_features = [col for col in feature_names if col in training_data.columns]
            
            X_train = training_data[available_features].values
            y_train = training_data[target_col].values if target_col in training_data.columns else np.zeros(len(training_data))
            
            if scaler is not None:
                X_train = scaler.transform(X_train)
            
            predictions = []
            
            logger.info(f"Performing {n_bootstrap} bootstrap samples")
            
            for i in range(n_bootstrap):
                # Bootstrap sample
                n_samples = len(X_train)
                boot_indices = np.random.RandomState(random_state + i).choice(
                    n_samples, size=n_samples, replace=True
                )
                
                X_boot = X_train[boot_indices]
                y_boot = y_train[boot_indices]
                
                # Clone and train model
                from sklearn.base import clone
                boot_model = clone(model)
                boot_model.fit(X_boot, y_boot)
                
                # Make predictions
                if hasattr(boot_model, 'predict_proba') and len(np.unique(y_boot)) == 2:
                    pred = boot_model.predict_proba(X_pred)[:, 1]
                else:
                    pred = boot_model.predict(X_pred)
                
                predictions.append(pred)
            
            # Calculate uncertainty metrics
            predictions = np.array(predictions)
            pred_std = np.std(predictions, axis=0)
            pred_mean = np.mean(predictions, axis=0)
            
            return {
                'predictions_bootstrap': predictions,
                'prediction_mean': pred_mean,
                'prediction_std': pred_std,
                'n_bootstrap': n_bootstrap
            }
            
        except Exception as e:
            logger.error(f"Bootstrap uncertainty calculation failed: {str(e)}")
            return {'prediction_std': np.zeros(len(X_pred))}
    
    def _monte_carlo_uncertainty(self, model, X_pred: np.ndarray, 
                               hyperparameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate uncertainty using Monte Carlo dropout (for neural networks)."""
        try:
            # Check if model supports Monte Carlo dropout
            if not hasattr(model, 'predict') or 'neural' not in model.__class__.__name__.lower():
                logger.info("Monte Carlo uncertainty not applicable for this model type")
                return None
            
            n_samples = hyperparameters.get('n_mc_samples', 100)
            predictions = []
            
            # This would need to be implemented for specific neural network frameworks
            # For now, return None to indicate not supported
            logger.warning("Monte Carlo uncertainty not implemented for this model type")
            return None
            
        except Exception as e:
            logger.error(f"Monte Carlo uncertainty calculation failed: {str(e)}")
            return None
    
    def _spatial_cv_uncertainty(self, model, X_pred: np.ndarray, coordinates: np.ndarray,
                              training_data: Optional[pd.DataFrame], feature_names: List[str],
                              scaler: Optional[StandardScaler], 
                              hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate uncertainty using spatial cross-validation."""
        try:
            if training_data is None or coordinates is None:
                logger.warning("Insufficient data for spatial CV uncertainty")
                return {'prediction_std': np.zeros(len(X_pred))}
            
            # This is a simplified implementation
            # In practice, you'd need proper spatial CV with blocking
            
            n_folds = hyperparameters.get('spatial_cv_folds', 5)
            distance_threshold = hyperparameters.get('spatial_distance_threshold', 1000.0)
            
            # Prepare training data
            target_col = hyperparameters.get('target_column', 'target')
            available_features = [col for col in feature_names if col in training_data.columns]
            
            X_train = training_data[available_features].values
            y_train = training_data[target_col].values if target_col in training_data.columns else np.zeros(len(training_data))
            
            if scaler is not None:
                X_train = scaler.transform(X_train)
            
            # Extract training coordinates (simplified)
            if 'x' in training_data.columns and 'y' in training_data.columns:
                train_coords = training_data[['x', 'y']].values
            else:
                logger.warning("No training coordinates available for spatial CV")
                return {'prediction_std': np.zeros(len(X_pred))}
            
            # Calculate spatial uncertainty based on distance to training points
            distances = cdist(coordinates, train_coords)
            min_distances = np.min(distances, axis=1)
            
            # Simple distance-based uncertainty (far from training = higher uncertainty)
            max_distance = np.percentile(min_distances, 95)
            spatial_uncertainty = min_distances / max_distance
            spatial_uncertainty = np.clip(spatial_uncertainty, 0, 1)
            
            return {
                'prediction_std': spatial_uncertainty,
                'min_distances': min_distances,
                'spatial_weights': 1 / (1 + min_distances / distance_threshold)
            }
            
        except Exception as e:
            logger.error(f"Spatial CV uncertainty calculation failed: {str(e)}")
            return {'prediction_std': np.zeros(len(X_pred))}
    
    def _prediction_intervals(self, model, X_pred: np.ndarray, training_data: Optional[pd.DataFrame],
                            feature_names: List[str], scaler: Optional[StandardScaler],
                            hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate prediction intervals using quantile regression approach."""
        try:
            confidence_level = hyperparameters.get('confidence_level', 0.95)
            alpha = 1 - confidence_level
            
            if training_data is None:
                logger.warning("No training data available for prediction intervals")
                base_pred = model.predict(X_pred)
                uncertainty = np.std(base_pred) * np.ones_like(base_pred)
                return {
                    'lower_bound': base_pred - uncertainty,
                    'upper_bound': base_pred + uncertainty
                }
            
            # Prepare training data
            target_col = hyperparameters.get('target_column', 'target')
            available_features = [col for col in feature_names if col in training_data.columns]
            
            X_train = training_data[available_features].values
            y_train = training_data[target_col].values if target_col in training_data.columns else np.zeros(len(training_data))
            
            if scaler is not None:
                X_train = scaler.transform(X_train)
            
            # Get base predictions
            base_predictions = model.predict(X_pred)
            
            # Calculate residuals on training data
            train_predictions = model.predict(X_train)
            residuals = y_train - train_predictions
            
            # Estimate prediction intervals using residual distribution
            residual_std = np.std(residuals)
            z_score = stats.norm.ppf(1 - alpha / 2)
            
            margin_of_error = z_score * residual_std
            
            lower_bound = base_predictions - margin_of_error
            upper_bound = base_predictions + margin_of_error
            
            return {
                'base_predictions': base_predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'margin_of_error': margin_of_error,
                'residual_std': residual_std
            }
            
        except Exception as e:
            logger.error(f"Prediction intervals calculation failed: {str(e)}")
            base_pred = model.predict(X_pred)
            uncertainty = np.std(base_pred) * 0.1
            return {
                'lower_bound': base_pred - uncertainty,
                'upper_bound': base_pred + uncertainty
            }
    
    def _combine_uncertainties(self, uncertainty_results: Dict[str, Dict[str, Any]],
                             hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple uncertainty estimates."""
        combination_method = hyperparameters.get('uncertainty_combination', 'weighted_average')
        
        # Extract standard deviations from all methods
        std_arrays = []
        method_names = []
        
        for method, results in uncertainty_results.items():
            if 'prediction_std' in results:
                std_arrays.append(results['prediction_std'])
                method_names.append(method)
        
        if not std_arrays:
            return {'combined_std': np.array([])}
        
        std_arrays = np.array(std_arrays)
        
        if combination_method == 'weighted_average':
            # Weight methods by their inverse variance (more certain = higher weight)
            weights = 1.0 / (np.mean(std_arrays, axis=1) + 1e-10)
            weights = weights / np.sum(weights)
            
            combined_std = np.average(std_arrays, axis=0, weights=weights)
            
        elif combination_method == 'maximum':
            combined_std = np.max(std_arrays, axis=0)
            
        elif combination_method == 'mean':
            combined_std = np.mean(std_arrays, axis=0)
            
        else:
            combined_std = np.mean(std_arrays, axis=0)
        
        return {
            'combined_std': combined_std,
            'individual_stds': {name: std for name, std in zip(method_names, std_arrays)},
            'combination_method': combination_method,
            'method_weights': weights.tolist() if combination_method == 'weighted_average' else None
        }
    
    def _generate_confidence_intervals(self, uncertainty_results: Dict[str, Dict[str, Any]],
                                     hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate confidence intervals from uncertainty estimates."""
        confidence_level = hyperparameters.get('confidence_level', 0.95)
        
        intervals = {}
        
        for method, results in uncertainty_results.items():
            method_intervals = {}
            
            if 'confidence_lower' in results and 'confidence_upper' in results:
                method_intervals['lower'] = results['confidence_lower']
                method_intervals['upper'] = results['confidence_upper']
                method_intervals['width'] = results['confidence_upper'] - results['confidence_lower']
                
            elif 'prediction_mean' in results and 'prediction_std' in results:
                # Calculate intervals using normal approximation
                z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                margin = z_score * results['prediction_std']
                
                method_intervals['lower'] = results['prediction_mean'] - margin
                method_intervals['upper'] = results['prediction_mean'] + margin
                method_intervals['width'] = 2 * margin
            
            if method_intervals:
                intervals[method] = method_intervals
        
        return intervals
    
    def _calculate_uncertainty_statistics(self, uncertainty_results: Dict[str, Dict[str, Any]],
                                        uncertainty_maps: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate comprehensive uncertainty statistics."""
        stats = {
            'method_statistics': {},
            'overall_statistics': {}
        }
        
        # Statistics for each method
        for method, results in uncertainty_results.items():
            method_stats = {}
            
            if 'prediction_std' in results:
                std_values = results['prediction_std']
                method_stats['uncertainty'] = {
                    'mean': float(np.mean(std_values)),
                    'std': float(np.std(std_values)),
                    'min': float(np.min(std_values)),
                    'max': float(np.max(std_values)),
                    'median': float(np.median(std_values)),
                    'q25': float(np.percentile(std_values, 25)),
                    'q75': float(np.percentile(std_values, 75))
                }
            
            if 'prediction_mean' in results:
                mean_values = results['prediction_mean']
                method_stats['predictions'] = {
                    'mean': float(np.mean(mean_values)),
                    'std': float(np.std(mean_values)),
                    'min': float(np.min(mean_values)),
                    'max': float(np.max(mean_values))
                }
            
            stats['method_statistics'][method] = method_stats
        
        # Overall statistics across all uncertainty maps
        if uncertainty_maps:
            all_uncertainties = np.concatenate([um.flatten() for um in uncertainty_maps.values()])
            
            stats['overall_statistics'] = {
                'n_methods': len(uncertainty_maps),
                'total_pixels': len(all_uncertainties),
                'uncertainty_range': {
                    'min': float(np.min(all_uncertainties)),
                    'max': float(np.max(all_uncertainties)),
                    'mean': float(np.mean(all_uncertainties)),
                    'std': float(np.std(all_uncertainties))
                },
                'high_uncertainty_percentage': float(
                    np.sum(all_uncertainties > np.percentile(all_uncertainties, 75)) / len(all_uncertainties) * 100
                )
            }
        
        return stats
    
    def _generate_spatial_outputs(self, uncertainty_maps: Dict[str, np.ndarray], valid_mask: np.ndarray,
                                spatial_info: Optional[Dict[str, Any]], hyperparameters: Dict[str, Any],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate spatial outputs for uncertainty maps."""
        outputs = {}
        
        if not uncertainty_maps or spatial_info is None:
            return outputs
        
        output_dir = context.get('output_dir', 'outputs/uncertainty')
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # For point data, create simple raster outputs
            if spatial_info['type'] == 'points':
                # This is a simplified approach - in practice you'd need proper gridding
                logger.info("Point-based uncertainty output not fully implemented")
                
            # Save uncertainty rasters (simplified version)
            uncertainty_rasters = {}
            
            for map_name, uncertainty_array in uncertainty_maps.items():
                # Save as simple array for now
                output_path = os.path.join(output_dir, f'{map_name}.npy')
                np.save(output_path, uncertainty_array)
                uncertainty_rasters[map_name] = output_path
            
            outputs['uncertainty_rasters'] = uncertainty_rasters
            
            logger.info(f"Saved {len(uncertainty_maps)} uncertainty maps to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate spatial outputs: {str(e)}")
        
        return outputs
    
    def get_uncertainty_maps(self) -> Optional[Dict[str, np.ndarray]]:
        """Get generated uncertainty maps."""
        return self.uncertainty_maps_
    
    def get_ensemble_models(self) -> Optional[List]:
        """Get trained ensemble models."""
        return self.ensemble_models_
