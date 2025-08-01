"""
Multi-class classification prediction step for landslide susceptibility and mineral targeting.

This step applies trained classification models to generate discrete risk/prospectivity
classes with confidence levels and spatial outputs.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import Point
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Tuple, List, Union
import logging

from ..base.base_step import BaseStep

logger = logging.getLogger(__name__)


class ClassificationPredictionStep(BaseStep):
    """
    Multi-class classification prediction step.
    
    Applies trained models to generate discrete classification maps with confidence
    levels for landslide susceptibility and mineral prospectivity mapping.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.class_labels_: Optional[List[str]] = None
        self.feature_names_: Optional[List[str]] = None
        
    def get_step_type(self) -> str:
        return "classification_prediction"
    
    def get_required_inputs(self) -> list:
        return ['trained_model', 'prediction_features']
    
    def get_outputs(self) -> list:
        return ['classification_map', 'confidence_map', 'prediction_statistics']
    
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """Validate classification prediction hyperparameters."""
        # Validate application type
        application_type = hyperparameters.get('application_type', 'landslide_susceptibility')
        valid_types = ['landslide_susceptibility', 'mineral_targeting', 'generic']
        if application_type not in valid_types:
            logger.error(f"Invalid application_type: {application_type}. Must be one of {valid_types}")
            return False
        
        # Validate prediction mode
        prediction_mode = hyperparameters.get('prediction_mode', 'raster')
        valid_modes = ['raster', 'vector', 'both']
        if prediction_mode not in valid_modes:
            logger.error(f"Invalid prediction_mode: {prediction_mode}. Must be one of {valid_modes}")
            return False
        
        # Validate confidence threshold
        confidence_threshold = hyperparameters.get('confidence_threshold', 0.5)
        if not isinstance(confidence_threshold, (int, float)) or not (0.0 <= confidence_threshold <= 1.0):
            logger.error("confidence_threshold must be a float between 0.0 and 1.0")
            return False
        
        # Validate output parameters
        if 'output_crs' in hyperparameters:
            try:
                CRS.from_string(hyperparameters['output_crs'])
            except Exception:
                logger.error(f"Invalid output_crs: {hyperparameters['output_crs']}")
                return False
        
        return True
    
    def execute(self, context) -> Dict[str, Any]:
        """
        Execute classification prediction.
        
        Args:
            context: Pipeline context containing inputs and configuration
            
        Returns:
            Dictionary containing prediction results and outputs
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
            
            # Prepare features for prediction
            X_pred, valid_mask = self._prepare_prediction_features(
                features_data, feature_names, scaler, hyperparameters
            )
            
            # Make predictions
            logger.info(f"Making predictions for {X_pred.shape[0]} samples")
            predictions, probabilities = self._make_predictions(model, X_pred, hyperparameters)
            
            # Apply class labels
            class_labels = self._get_class_labels(hyperparameters)
            
            # Calculate confidence metrics
            confidence_scores = self._calculate_confidence_scores(
                probabilities, hyperparameters
            )
            
            # Generate spatial outputs
            spatial_outputs = self._generate_spatial_outputs(
                predictions, probabilities, confidence_scores, valid_mask,
                spatial_info, hyperparameters, context
            )
            
            # Calculate prediction statistics
            prediction_stats = self._calculate_prediction_statistics(
                predictions, probabilities, confidence_scores, class_labels
            )
            
            # Store model information
            self.model = model
            self.scaler = scaler
            self.feature_names_ = feature_names
            self.class_labels_ = class_labels
            
            # Prepare outputs
            outputs = {
                'classification_map': spatial_outputs.get('classification_map'),
                'confidence_map': spatial_outputs.get('confidence_map'),
                'prediction_statistics': prediction_stats,
                'class_labels': class_labels,
                'prediction_summary': {
                    'total_samples': len(predictions),
                    'valid_predictions': np.sum(valid_mask),
                    'model_type': model.__class__.__name__,
                    'confidence_threshold': hyperparameters.get('confidence_threshold', 0.5)
                }
            }
            
            # Add vector outputs if requested
            if spatial_outputs.get('vector_outputs'):
                outputs['vector_outputs'] = spatial_outputs['vector_outputs']
            
            logger.info("Classification prediction completed successfully")
            logger.info(f"Generated {len(class_labels)} classes with {np.sum(valid_mask)} valid predictions")
            
            return {
                'status': 'success',
                'message': 'Classification prediction completed successfully',
                'outputs': outputs,
                'metadata': {
                    'execution_time': context.get('execution_time', 0),
                    'prediction_samples': len(predictions),
                    'valid_samples': np.sum(valid_mask),
                    'n_classes': len(class_labels),
                    'application_type': hyperparameters.get('application_type', 'generic')
                }
            }
            
        except Exception as e:
            logger.error(f"Classification prediction failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'outputs': {}
            }
    
    def _load_trained_model(self, model_info: Union[str, Dict[str, Any]]) -> Tuple[Any, Optional[StandardScaler], List[str]]:
        """Load trained model from file or dictionary."""
        try:
            if isinstance(model_info, dict) and 'model_object' in model_info:
                # Model passed as dictionary
                model = model_info['model_object']
                scaler = model_info.get('scaler_object', None)
                feature_names = model_info.get('feature_names', [])
            else:
                # Load from file
                if isinstance(model_info, str):
                    model_path = model_info
                else:
                    model_path = model_info.get('model_path')
                
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    return None, None, []
                
                model_data = joblib.load(model_path)
                model = model_data['model']
                scaler = model_data.get('scaler', None)
                feature_names = model_data.get('feature_names', [])
            
            logger.info(f"Loaded model: {model.__class__.__name__} with {len(feature_names)} features")
            return model, scaler, feature_names
            
        except Exception as e:
            logger.error(f"Failed to load trained model: {str(e)}")
            return None, None, []
    
    def _load_prediction_features(self, features_path: str, 
                                hyperparameters: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Load prediction features from file."""
        try:
            spatial_info = None
            
            if features_path.endswith('.csv'):
                features_data = pd.read_csv(features_path)
                
                # Extract spatial information if available
                if all(col in features_data.columns for col in ['x', 'y']):
                    spatial_info = {
                        'type': 'points',
                        'coordinates': features_data[['x', 'y']].values,
                        'crs': hyperparameters.get('input_crs', 'EPSG:4326')
                    }
                    
            elif features_path.endswith('.parquet'):
                features_data = pd.read_parquet(features_path)
                
            elif features_path.endswith(('.tif', '.tiff')):
                # Raster input
                features_data, spatial_info = self._load_raster_features(features_path)
                
            elif features_path.endswith(('.shp', '.geojson')):
                # Vector input
                gdf = gpd.read_file(features_path)
                features_data = pd.DataFrame(gdf.drop('geometry', axis=1))
                
                spatial_info = {
                    'type': 'vector',
                    'geometry': gdf.geometry,
                    'crs': str(gdf.crs)
                }
                
            else:
                logger.error(f"Unsupported file format: {features_path}")
                return None, None
            
            logger.info(f"Loaded prediction features: {features_data.shape[0]} samples, {features_data.shape[1]} features")
            return features_data, spatial_info
            
        except Exception as e:
            logger.error(f"Failed to load prediction features: {str(e)}")
            return None, None
    
    def _load_raster_features(self, raster_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load features from raster file."""
        with rasterio.open(raster_path) as src:
            # Read all bands
            data = src.read()
            
            # Get spatial information
            spatial_info = {
                'type': 'raster',
                'transform': src.transform,
                'crs': str(src.crs),
                'shape': (src.height, src.width),
                'bounds': src.bounds
            }
            
            # Convert to DataFrame
            n_bands, height, width = data.shape
            
            # Create coordinate grids
            rows, cols = np.mgrid[0:height, 0:width]
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            
            # Flatten arrays
            feature_data = []
            for band in range(n_bands):
                feature_data.append(data[band].flatten())
            
            # Create DataFrame
            feature_dict = {f'band_{i+1}': feature_data[i] for i in range(n_bands)}
            feature_dict['x'] = np.array(xs).flatten()
            feature_dict['y'] = np.array(ys).flatten()
            feature_dict['row'] = rows.flatten()
            feature_dict['col'] = cols.flatten()
            
            features_df = pd.DataFrame(feature_dict)
            
        return features_df, spatial_info
    
    def _prepare_prediction_features(self, features_data: pd.DataFrame, feature_names: List[str],
                                   scaler: Optional[StandardScaler], 
                                   hyperparameters: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for prediction."""
        # Select required features
        available_features = [col for col in feature_names if col in features_data.columns]
        missing_features = [col for col in feature_names if col not in features_data.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        if not available_features:
            raise ValueError("No required features found in prediction data")
        
        # Extract feature matrix
        X = features_data[available_features].values
        
        # Handle missing values
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]
        
        # Apply scaling if available
        if scaler is not None:
            X_clean = scaler.transform(X_clean)
        
        logger.info(f"Prepared {X_clean.shape[0]} valid samples from {X.shape[0]} total samples")
        logger.info(f"Using {X_clean.shape[1]} features: {available_features}")
        
        return X_clean, valid_mask
    
    def _make_predictions(self, model, X: np.ndarray, 
                         hyperparameters: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained model."""
        # Get class predictions
        predictions = model.predict(X)
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
        elif hasattr(model, 'decision_function'):
            # For models like SVM that have decision_function
            decision_scores = model.decision_function(X)
            if decision_scores.ndim == 1:
                # Binary classification
                probabilities = np.column_stack([1 - decision_scores, decision_scores])
            else:
                # Multi-class: convert to probabilities using softmax
                exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            # No probability information available
            n_classes = len(np.unique(predictions))
            probabilities = np.eye(n_classes)[predictions]
        
        return predictions, probabilities
    
    def _get_class_labels(self, hyperparameters: Dict[str, Any]) -> List[str]:
        """Get class labels based on application type."""
        application_type = hyperparameters.get('application_type', 'generic')
        
        if application_type == 'landslide_susceptibility':
            return hyperparameters.get('class_labels', [
                'Very Low', 'Low', 'Moderate', 'High', 'Very High'
            ])
        elif application_type == 'mineral_targeting':
            return hyperparameters.get('class_labels', [
                'Background', 'Low Potential', 'Moderate Potential', 'High Potential'
            ])
        else:
            # Generic or custom labels
            n_classes = hyperparameters.get('n_classes', 2)
            return hyperparameters.get('class_labels', [f'Class_{i}' for i in range(n_classes)])
    
    def _calculate_confidence_scores(self, probabilities: np.ndarray, 
                                   hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Calculate confidence scores for predictions."""
        confidence_method = hyperparameters.get('confidence_method', 'max_probability')
        
        if confidence_method == 'max_probability':
            # Maximum probability as confidence
            confidence_scores = np.max(probabilities, axis=1)
        elif confidence_method == 'entropy':
            # Negative entropy (higher = more confident)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
            max_entropy = np.log(probabilities.shape[1])
            confidence_scores = 1 - (entropy / max_entropy)
        elif confidence_method == 'margin':
            # Difference between top two probabilities
            sorted_probs = np.sort(probabilities, axis=1)
            confidence_scores = sorted_probs[:, -1] - sorted_probs[:, -2]
        else:
            # Default to max probability
            confidence_scores = np.max(probabilities, axis=1)
        
        return confidence_scores
    
    def _generate_spatial_outputs(self, predictions: np.ndarray, probabilities: np.ndarray,
                                confidence_scores: np.ndarray, valid_mask: np.ndarray,
                                spatial_info: Optional[Dict[str, Any]], 
                                hyperparameters: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate spatial outputs (rasters and/or vectors)."""
        outputs = {}
        
        if spatial_info is None:
            logger.warning("No spatial information available, skipping spatial outputs")
            return outputs
        
        prediction_mode = hyperparameters.get('prediction_mode', 'raster')
        output_dir = context.get('output_dir', 'outputs/predictions')
        os.makedirs(output_dir, exist_ok=True)
        
        # Raster outputs
        if prediction_mode in ['raster', 'both'] and spatial_info['type'] == 'raster':
            outputs.update(self._create_raster_outputs(
                predictions, probabilities, confidence_scores, valid_mask,
                spatial_info, output_dir, hyperparameters
            ))
        
        # Vector outputs
        if prediction_mode in ['vector', 'both']:
            outputs.update(self._create_vector_outputs(
                predictions, probabilities, confidence_scores, valid_mask,
                spatial_info, output_dir, hyperparameters
            ))
        
        return outputs
    
    def _create_raster_outputs(self, predictions: np.ndarray, probabilities: np.ndarray,
                              confidence_scores: np.ndarray, valid_mask: np.ndarray,
                              spatial_info: Dict[str, Any], output_dir: str,
                              hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create raster outputs."""
        outputs = {}
        
        try:
            height, width = spatial_info['shape']
            transform = spatial_info['transform']
            crs = spatial_info['crs']
            
            # Initialize output arrays
            pred_array = np.full((height, width), -9999, dtype=np.int16)
            conf_array = np.full((height, width), -9999.0, dtype=np.float32)
            
            # Fill valid predictions
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 0:
                # Assuming we have row/col information from raster loading
                # This is a simplified approach - in practice, you'd need proper indexing
                pred_array[valid_mask.reshape(height, width)] = predictions
                conf_array[valid_mask.reshape(height, width)] = confidence_scores
            
            # Save classification raster
            class_raster_path = os.path.join(output_dir, 'classification_map.tif')
            with rasterio.open(
                class_raster_path, 'w',
                driver='GTiff',
                height=height, width=width,
                count=1, dtype=pred_array.dtype,
                crs=crs, transform=transform,
                nodata=-9999
            ) as dst:
                dst.write(pred_array, 1)
            
            outputs['classification_map'] = class_raster_path
            
            # Save confidence raster
            conf_raster_path = os.path.join(output_dir, 'confidence_map.tif')
            with rasterio.open(
                conf_raster_path, 'w',
                driver='GTiff',
                height=height, width=width,
                count=1, dtype=conf_array.dtype,
                crs=crs, transform=transform,
                nodata=-9999.0
            ) as dst:
                dst.write(conf_array, 1)
            
            outputs['confidence_map'] = conf_raster_path
            
            # Save probability rasters if requested
            if hyperparameters.get('save_probabilities', False):
                prob_dir = os.path.join(output_dir, 'probabilities')
                os.makedirs(prob_dir, exist_ok=True)
                
                prob_paths = []
                for i in range(probabilities.shape[1]):
                    prob_array = np.full((height, width), -9999.0, dtype=np.float32)
                    prob_array[valid_mask.reshape(height, width)] = probabilities[:, i]
                    
                    prob_path = os.path.join(prob_dir, f'probability_class_{i}.tif')
                    with rasterio.open(
                        prob_path, 'w',
                        driver='GTiff',
                        height=height, width=width,
                        count=1, dtype=prob_array.dtype,
                        crs=crs, transform=transform,
                        nodata=-9999.0
                    ) as dst:
                        dst.write(prob_array, 1)
                    
                    prob_paths.append(prob_path)
                
                outputs['probability_maps'] = prob_paths
            
            logger.info(f"Created raster outputs in {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create raster outputs: {str(e)}")
        
        return outputs
    
    def _create_vector_outputs(self, predictions: np.ndarray, probabilities: np.ndarray,
                              confidence_scores: np.ndarray, valid_mask: np.ndarray,
                              spatial_info: Dict[str, Any], output_dir: str,
                              hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create vector outputs."""
        outputs = {}
        
        try:
            # Create GeoDataFrame with predictions
            if spatial_info['type'] == 'points':
                # Point data
                coords = spatial_info['coordinates'][valid_mask]
                geometry = [Point(x, y) for x, y in coords]
                
            elif spatial_info['type'] == 'vector':
                # Existing vector data
                geometry = spatial_info['geometry'][valid_mask]
                
            else:
                # Raster to points
                logger.warning("Converting raster to points for vector output")
                # This would require more complex implementation
                return outputs
            
            # Create attribute data
            data = {
                'prediction': predictions,
                'confidence': confidence_scores
            }
            
            # Add probability columns
            class_labels = self._get_class_labels(hyperparameters)
            for i, label in enumerate(class_labels):
                if i < probabilities.shape[1]:
                    data[f'prob_{label.lower().replace(" ", "_")}'] = probabilities[:, i]
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(data, geometry=geometry)
            gdf.crs = spatial_info['crs']
            
            # Apply confidence threshold filtering if requested
            confidence_threshold = hyperparameters.get('confidence_threshold', 0.0)
            if confidence_threshold > 0:
                gdf = gdf[gdf['confidence'] >= confidence_threshold]
            
            # Save to file
            vector_path = os.path.join(output_dir, 'predictions.geojson')
            gdf.to_file(vector_path, driver='GeoJSON')
            
            outputs['vector_outputs'] = {
                'path': vector_path,
                'format': 'GeoJSON',
                'n_features': len(gdf),
                'crs': str(gdf.crs)
            }
            
            logger.info(f"Created vector output with {len(gdf)} features: {vector_path}")
            
        except Exception as e:
            logger.error(f"Failed to create vector outputs: {str(e)}")
        
        return outputs
    
    def _calculate_prediction_statistics(self, predictions: np.ndarray, probabilities: np.ndarray,
                                       confidence_scores: np.ndarray, 
                                       class_labels: List[str]) -> Dict[str, Any]:
        """Calculate prediction statistics."""
        stats = {
            'class_distribution': {},
            'confidence_statistics': {},
            'probability_statistics': {}
        }
        
        # Class distribution
        unique_classes, class_counts = np.unique(predictions, return_counts=True)
        total_predictions = len(predictions)
        
        for class_idx, count in zip(unique_classes, class_counts):
            if class_idx < len(class_labels):
                class_name = class_labels[class_idx]
                stats['class_distribution'][class_name] = {
                    'count': int(count),
                    'percentage': float(count / total_predictions * 100)
                }
        
        # Confidence statistics
        stats['confidence_statistics'] = {
            'mean': float(np.mean(confidence_scores)),
            'std': float(np.std(confidence_scores)),
            'min': float(np.min(confidence_scores)),
            'max': float(np.max(confidence_scores)),
            'median': float(np.median(confidence_scores)),
            'q25': float(np.percentile(confidence_scores, 25)),
            'q75': float(np.percentile(confidence_scores, 75))
        }
        
        # Probability statistics per class
        for i, class_name in enumerate(class_labels):
            if i < probabilities.shape[1]:
                class_probs = probabilities[:, i]
                stats['probability_statistics'][class_name] = {
                    'mean': float(np.mean(class_probs)),
                    'std': float(np.std(class_probs)),
                    'min': float(np.min(class_probs)),
                    'max': float(np.max(class_probs))
                }
        
        return stats
    
    def predict(self, features_data: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data."""
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
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
        
        return predictions, probabilities
