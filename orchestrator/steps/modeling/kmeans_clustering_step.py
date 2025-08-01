"""
K-means clustering step for unsupervised analysis in mineral targeting.

This step implements K-means clustering with automatic cluster number selection,
cluster validation metrics, and geological interpretation support.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from kneed import KneeLocator
from typing import Dict, Any, Optional, Tuple, List
import logging

from ..base.base_step import BaseStep

logger = logging.getLogger(__name__)


class KMeansClusteringStep(BaseStep):
    """
    K-means clustering step for unsupervised analysis.
    
    Specialized for mineral targeting applications with automatic cluster
    number selection, validation metrics, and geological interpretation.
    """
    
    def __init__(self):
        super().__init__()
        self.model: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.feature_names_: Optional[list] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.optimal_k_: Optional[int] = None
        
    def get_step_type(self) -> str:
        return "kmeans_clustering"
    
    def get_required_inputs(self) -> list:
        return ['feature_data']
    
    def get_outputs(self) -> list:
        return ['cluster_model', 'cluster_assignments', 'cluster_analysis', 'validation_metrics']
    
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """Validate K-means hyperparameters."""
        # Validate algorithm
        algorithm = hyperparameters.get('algorithm', 'lloyd')
        valid_algorithms = ['lloyd', 'elkan']
        if algorithm not in valid_algorithms:
            logger.error(f"Invalid algorithm: {algorithm}. Must be one of {valid_algorithms}")
            return False
        
        # Validate init method
        init_method = hyperparameters.get('init', 'k-means++')
        valid_init = ['k-means++', 'random']
        if init_method not in valid_init:
            logger.error(f"Invalid init method: {init_method}. Must be one of {valid_init}")
            return False
        
        # Validate numeric parameters
        numeric_params = {
            'k': (int, 2, 50),
            'max_k': (int, 2, 50),
            'max_iter': (int, 100, 10000),
            'n_init': (int, 1, 100),
            'tol': (float, 1e-10, 1e-2),
            'random_state': (int, 0, 999999)
        }
        
        for param, (param_type, min_val, max_val) in numeric_params.items():
            if param in hyperparameters:
                value = hyperparameters[param]
                if not isinstance(value, param_type):
                    logger.error(f"Parameter {param} must be of type {param_type}")
                    return False
                if not (min_val <= value <= max_val):
                    logger.error(f"Parameter {param} must be between {min_val} and {max_val}")
                    return False
        
        # Validate k vs max_k
        if 'k' in hyperparameters and 'max_k' in hyperparameters:
            if hyperparameters['k'] > hyperparameters['max_k']:
                logger.error("Parameter 'k' cannot be greater than 'max_k'")
                return False
        
        return True
    
    def execute(self, context) -> Dict[str, Any]:
        """
        Execute K-means clustering analysis.
        
        Args:
            context: Pipeline context containing inputs and configuration
            
        Returns:
            Dictionary containing execution results and outputs
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
            
            # Load feature data
            feature_data_path = context['inputs']['feature_data']
            feature_data = self._load_feature_data(feature_data_path)
            
            if feature_data is None:
                return {
                    'status': 'failed',
                    'error': 'Failed to load feature data',
                    'outputs': {}
                }
            
            # Prepare features
            X, feature_names = self._prepare_features(feature_data, hyperparameters)
            
            # Scale features if requested
            if hyperparameters.get('scale_features', True):
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X
                self.scaler = None
            
            # Apply PCA if requested
            if hyperparameters.get('apply_pca', False):
                n_components = hyperparameters.get('pca_components', 0.95)
                self.pca = PCA(n_components=n_components, random_state=hyperparameters.get('random_state', 42))
                X_processed = self.pca.fit_transform(X_scaled)
                logger.info(f"PCA reduced features from {X_scaled.shape[1]} to {X_processed.shape[1]}")
            else:
                X_processed = X_scaled
                self.pca = None
            
            # Determine optimal number of clusters
            if 'k' in hyperparameters:
                optimal_k = hyperparameters['k']
                elbow_analysis = None
            else:
                optimal_k, elbow_analysis = self._find_optimal_k(
                    X_processed, hyperparameters
                )
            
            self.optimal_k_ = optimal_k
            
            # Train K-means model
            model = self._create_model(hyperparameters, optimal_k)
            
            logger.info(f"Training K-means clustering with k={optimal_k}, {X_processed.shape[0]} samples")
            cluster_labels = model.fit_predict(X_processed)
            
            # Calculate validation metrics
            validation_metrics = self._calculate_validation_metrics(
                X_processed, cluster_labels
            )
            
            # Analyze clusters
            cluster_analysis = self._analyze_clusters(
                X, cluster_labels, feature_names, hyperparameters
            )
            
            # Store model and information
            self.model = model
            self.feature_names_ = feature_names
            self.cluster_centers_ = model.cluster_centers_
            
            # Save model if requested
            model_path = None
            if hyperparameters.get('save_model', True):
                model_path = self._save_model(model, context, hyperparameters)
            
            # Prepare cluster assignments
            cluster_assignments = self._prepare_cluster_assignments(
                feature_data, cluster_labels, hyperparameters
            )
            
            # Prepare outputs
            outputs = {
                'cluster_model': {
                    'model_object': model,
                    'scaler_object': self.scaler,
                    'pca_object': self.pca,
                    'model_path': model_path,
                    'n_clusters': optimal_k,
                    'n_features': len(feature_names),
                    'cluster_centers': model.cluster_centers_
                },
                'cluster_assignments': cluster_assignments,
                'cluster_analysis': cluster_analysis,
                'validation_metrics': validation_metrics
            }
            
            # Add elbow analysis if performed
            if elbow_analysis is not None:
                outputs['elbow_analysis'] = elbow_analysis
            
            logger.info(f"K-means clustering completed successfully")
            logger.info(f"Optimal k={optimal_k}, Silhouette score: {validation_metrics['silhouette_score']:.3f}")
            
            return {
                'status': 'success',
                'message': f'K-means clustering completed with k={optimal_k}',
                'outputs': outputs,
                'metadata': {
                    'execution_time': context.get('execution_time', 0),
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1],
                    'optimal_k': optimal_k
                }
            }
            
        except Exception as e:
            logger.error(f"K-means clustering failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'outputs': {}
            }
    
    def _load_feature_data(self, data_path: str) -> Optional[pd.DataFrame]:
        """Load feature data from file."""
        try:
            if data_path.endswith('.csv'):
                return pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                return pd.read_parquet(data_path)
            elif data_path.endswith('.json'):
                return pd.read_json(data_path)
            else:
                logger.error(f"Unsupported file format: {data_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load feature data: {str(e)}")
            return None
    
    def _prepare_features(self, data: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Tuple[np.ndarray, list]:
        """Prepare feature matrix from input data."""
        # Exclude non-feature columns
        exclude_columns = hyperparameters.get('exclude_columns', [])
        exclude_columns.extend(['geometry', 'id', 'sample_id', 'x', 'y', 'lon', 'lat'])
        
        # Select feature columns
        feature_columns = [col for col in data.columns 
                          if col not in exclude_columns and data[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        # Handle missing values
        data_clean = data[feature_columns].dropna()
        
        X = data_clean.values
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"Feature statistics - Mean: {X.mean():.3f}, Std: {X.std():.3f}")
        
        return X, feature_columns
    
    def _find_optimal_k(self, X: np.ndarray, hyperparameters: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Find optimal number of clusters using elbow method and validation metrics."""
        min_k = hyperparameters.get('min_k', 2)
        max_k = hyperparameters.get('max_k', min(20, X.shape[0] // 10))
        
        k_range = range(min_k, max_k + 1)
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        
        logger.info(f"Testing k values from {min_k} to {max_k}")
        
        for k in k_range:
            # Train K-means for this k
            kmeans = KMeans(
                n_clusters=k,
                init=hyperparameters.get('init', 'k-means++'),
                n_init=hyperparameters.get('n_init', 10),
                max_iter=hyperparameters.get('max_iter', 300),
                random_state=hyperparameters.get('random_state', 42),
                algorithm=hyperparameters.get('algorithm', 'lloyd')
            )
            
            labels = kmeans.fit_predict(X)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            
            if k > 1:  # Silhouette score requires at least 2 clusters
                silhouette_scores.append(silhouette_score(X, labels))
                calinski_scores.append(calinski_harabasz_score(X, labels))
                davies_bouldin_scores.append(davies_bouldin_score(X, labels))
            else:
                silhouette_scores.append(0)
                calinski_scores.append(0)
                davies_bouldin_scores.append(np.inf)
        
        # Find optimal k using elbow method
        elbow_k = self._find_elbow_point(k_range, inertias)
        
        # Find optimal k using silhouette score
        if len(silhouette_scores) > 0:
            silhouette_k = k_range[np.argmax(silhouette_scores)]
        else:
            silhouette_k = elbow_k
        
        # Choose final k (prioritize silhouette score if reasonable)
        if abs(elbow_k - silhouette_k) <= 2:
            optimal_k = silhouette_k
            selection_method = 'silhouette'
        else:
            optimal_k = elbow_k
            selection_method = 'elbow'
        
        # Prepare elbow analysis results
        elbow_analysis = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_harabasz_scores': calinski_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'elbow_k': elbow_k,
            'silhouette_k': silhouette_k,
            'optimal_k': optimal_k,
            'selection_method': selection_method
        }
        
        logger.info(f"Optimal k selected: {optimal_k} (method: {selection_method})")
        
        return optimal_k, elbow_analysis
    
    def _find_elbow_point(self, k_range: range, inertias: List[float]) -> int:
        """Find elbow point in the inertia curve."""
        try:
            # Use KneeLocator to find the elbow
            kl = KneeLocator(
                list(k_range), inertias, curve='convex', direction='decreasing'
            )
            
            if kl.elbow is not None:
                return kl.elbow
            else:
                # Fallback: use the point where improvement drops significantly
                improvements = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
                improvement_ratios = [improvements[i] / improvements[i+1] for i in range(len(improvements)-1)]
                elbow_idx = np.argmax(improvement_ratios) + 1
                return list(k_range)[elbow_idx]
        except Exception:
            # Final fallback: return middle value
            return list(k_range)[len(k_range) // 2]
    
    def _create_model(self, hyperparameters: Dict[str, Any], k: int) -> KMeans:
        """Create K-means model."""
        params = {
            'n_clusters': k,
            'init': hyperparameters.get('init', 'k-means++'),
            'n_init': hyperparameters.get('n_init', 10),
            'max_iter': hyperparameters.get('max_iter', 300),
            'tol': hyperparameters.get('tol', 1e-4),
            'random_state': hyperparameters.get('random_state', 42),
            'algorithm': hyperparameters.get('algorithm', 'lloyd')
        }
        
        return KMeans(**params)
    
    def _calculate_validation_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate cluster validation metrics."""
        if len(np.unique(labels)) < 2:
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': np.inf,
                'inertia': 0.0
            }
        
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        # Calculate inertia manually
        inertia = 0.0
        for cluster_id in np.unique(labels):
            cluster_points = X[labels == cluster_id]
            cluster_center = np.mean(cluster_points, axis=0)
            inertia += np.sum((cluster_points - cluster_center) ** 2)
        
        return {
            'silhouette_score': float(silhouette),
            'calinski_harabasz_score': float(calinski_harabasz),
            'davies_bouldin_score': float(davies_bouldin),
            'inertia': float(inertia),
            'n_clusters': int(len(np.unique(labels)))
        }
    
    def _analyze_clusters(self, X: np.ndarray, labels: np.ndarray, 
                         feature_names: list, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cluster characteristics and geological interpretation."""
        n_clusters = len(np.unique(labels))
        cluster_analysis = {
            'n_clusters': n_clusters,
            'cluster_sizes': {},
            'cluster_centers_original': {},
            'cluster_statistics': {},
            'feature_importance': {}
        }
        
        # Analyze each cluster
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]
            
            # Cluster size and percentage
            cluster_size = np.sum(cluster_mask)
            cluster_percentage = cluster_size / len(labels) * 100
            
            cluster_analysis['cluster_sizes'][f'cluster_{cluster_id}'] = {
                'size': int(cluster_size),
                'percentage': float(cluster_percentage)
            }
            
            # Cluster center in original feature space
            cluster_center = np.mean(cluster_data, axis=0)
            cluster_analysis['cluster_centers_original'][f'cluster_{cluster_id}'] = {
                feature: float(value) for feature, value in zip(feature_names, cluster_center)
            }
            
            # Cluster statistics
            cluster_std = np.std(cluster_data, axis=0)
            cluster_analysis['cluster_statistics'][f'cluster_{cluster_id}'] = {
                'mean': cluster_center.tolist(),
                'std': cluster_std.tolist(),
                'feature_stats': {
                    feature: {
                        'mean': float(cluster_center[i]),
                        'std': float(cluster_std[i]),
                        'min': float(np.min(cluster_data[:, i])),
                        'max': float(np.max(cluster_data[:, i]))
                    } for i, feature in enumerate(feature_names)
                }
            }
        
        # Calculate feature importance for cluster separation
        cluster_analysis['feature_importance'] = self._calculate_feature_importance_for_clustering(
            X, labels, feature_names
        )
        
        # Generate geological interpretation if feature names suggest it
        if hyperparameters.get('geological_interpretation', True):
            cluster_analysis['geological_interpretation'] = self._generate_geological_interpretation(
                cluster_analysis, feature_names
            )
        
        return cluster_analysis
    
    def _calculate_feature_importance_for_clustering(self, X: np.ndarray, labels: np.ndarray, 
                                                   feature_names: list) -> Dict[str, Any]:
        """Calculate feature importance for cluster separation."""
        n_features = X.shape[1]
        feature_importance = np.zeros(n_features)
        
        # Calculate between-cluster variance / within-cluster variance ratio
        for i in range(n_features):
            feature_data = X[:, i]
            
            # Between-cluster variance
            overall_mean = np.mean(feature_data)
            between_var = 0
            within_var = 0
            
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_data = feature_data[cluster_mask]
                cluster_mean = np.mean(cluster_data)
                cluster_size = len(cluster_data)
                
                between_var += cluster_size * (cluster_mean - overall_mean) ** 2
                within_var += np.sum((cluster_data - cluster_mean) ** 2)
            
            # F-ratio (between-cluster variance / within-cluster variance)
            if within_var > 0:
                feature_importance[i] = between_var / within_var
            else:
                feature_importance[i] = 0
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Normalize importance scores
        if np.max(feature_importance) > 0:
            importance_df['importance_normalized'] = importance_df['importance'] / np.max(feature_importance)
        else:
            importance_df['importance_normalized'] = 0
        
        return {
            'importance_scores': feature_importance.tolist(),
            'importance_df': importance_df.to_dict('records'),
            'top_features': importance_df.head(10).to_dict('records')
        }
    
    def _generate_geological_interpretation(self, cluster_analysis: Dict[str, Any], 
                                          feature_names: list) -> Dict[str, Any]:
        """Generate geological interpretation based on cluster characteristics."""
        interpretation = {}
        
        # Common geological/geophysical feature patterns
        geological_indicators = {
            'spectral': ['ndvi', 'ndwi', 'savi', 'evi', 'band', 'reflectance'],
            'topographic': ['elevation', 'slope', 'aspect', 'curvature', 'tpi', 'tri'],
            'geophysical': ['magnetic', 'gravity', 'radiometric', 'conductivity'],
            'geochemical': ['au', 'cu', 'pb', 'zn', 'ag', 'fe', 'al', 'si', 'k', 'ca'],
            'structural': ['lineament', 'fault', 'fracture', 'distance'],
            'alteration': ['clay', 'iron', 'carbonate', 'sericite', 'chlorite']
        }
        
        # Classify features by type
        feature_types = {}
        for feature in feature_names:
            feature_lower = feature.lower()
            for category, keywords in geological_indicators.items():
                if any(keyword in feature_lower for keyword in keywords):
                    if category not in feature_types:
                        feature_types[category] = []
                    feature_types[category].append(feature)
                    break
        
        # Analyze each cluster
        for cluster_key, cluster_stats in cluster_analysis['cluster_statistics'].items():
            cluster_id = cluster_key.split('_')[1]
            interpretation[f'cluster_{cluster_id}'] = {
                'dominant_characteristics': [],
                'geological_signature': '',
                'potential_targets': []
            }
            
            feature_stats = cluster_stats['feature_stats']
            
            # Identify dominant characteristics for each feature type
            for category, features in feature_types.items():
                if features:
                    category_values = [feature_stats[feature]['mean'] for feature in features if feature in feature_stats]
                    if category_values:
                        avg_value = np.mean(category_values)
                        interpretation[f'cluster_{cluster_id}']['dominant_characteristics'].append({
                            'category': category,
                            'average_value': float(avg_value),
                            'features': features
                        })
            
            # Generate geological signature based on feature combinations
            signature_parts = []
            
            # Topographic signature
            if 'topographic' in feature_types:
                topo_features = feature_types['topographic']
                if 'elevation' in [f.lower() for f in topo_features]:
                    elev_features = [f for f in topo_features if 'elevation' in f.lower()]
                    if elev_features and elev_features[0] in feature_stats:
                        elev_mean = feature_stats[elev_features[0]]['mean']
                        if elev_mean > 1000:
                            signature_parts.append("high elevation")
                        elif elev_mean < 500:
                            signature_parts.append("low elevation")
                        else:
                            signature_parts.append("moderate elevation")
                
                if 'slope' in [f.lower() for f in topo_features]:
                    slope_features = [f for f in topo_features if 'slope' in f.lower()]
                    if slope_features and slope_features[0] in feature_stats:
                        slope_mean = feature_stats[slope_features[0]]['mean']
                        if slope_mean > 30:
                            signature_parts.append("steep terrain")
                        elif slope_mean < 5:
                            signature_parts.append("flat terrain")
                        else:
                            signature_parts.append("moderate slopes")
            
            # Spectral signature
            if 'spectral' in feature_types:
                spectral_features = feature_types['spectral']
                ndvi_features = [f for f in spectral_features if 'ndvi' in f.lower()]
                if ndvi_features and ndvi_features[0] in feature_stats:
                    ndvi_mean = feature_stats[ndvi_features[0]]['mean']
                    if ndvi_mean > 0.5:
                        signature_parts.append("dense vegetation")
                    elif ndvi_mean < 0.1:
                        signature_parts.append("sparse vegetation/bare rock")
                    else:
                        signature_parts.append("moderate vegetation")
            
            # Geochemical signature
            if 'geochemical' in feature_types:
                geochem_features = feature_types['geochemical']
                high_values = []
                for feature in geochem_features:
                    if feature in feature_stats:
                        # This is a simplified approach - in practice, you'd need
                        # background values for comparison
                        if feature_stats[feature]['mean'] > feature_stats[feature]['std']:
                            high_values.append(feature)
                
                if high_values:
                    signature_parts.append(f"elevated {', '.join(high_values[:3])}")
            
            interpretation[f'cluster_{cluster_id}']['geological_signature'] = '; '.join(signature_parts) if signature_parts else 'undefined signature'
            
            # Suggest potential targets based on signatures
            if 'geochemical' in feature_types and 'alteration' in feature_types:
                interpretation[f'cluster_{cluster_id}']['potential_targets'].append('hydrothermal alteration zone')
            
            if 'structural' in feature_types:
                interpretation[f'cluster_{cluster_id}']['potential_targets'].append('structural control zone')
            
            if not interpretation[f'cluster_{cluster_id}']['potential_targets']:
                interpretation[f'cluster_{cluster_id}']['potential_targets'].append('background/host rock')
        
        return interpretation
    
    def _prepare_cluster_assignments(self, original_data: pd.DataFrame, 
                                   cluster_labels: np.ndarray, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare cluster assignment results."""
        # Create a copy of original data with cluster assignments
        result_data = original_data.copy()
        result_data['cluster'] = cluster_labels
        
        # Add cluster statistics
        cluster_stats = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_stats[f'cluster_{cluster_id}'] = {
                'count': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(cluster_labels) * 100)
            }
        
        # Save assignments if requested
        assignments_path = None
        if hyperparameters.get('save_assignments', True):
            output_dir = hyperparameters.get('output_dir', 'outputs/clustering')
            os.makedirs(output_dir, exist_ok=True)
            
            assignments_filename = hyperparameters.get('assignments_filename', 'cluster_assignments.csv')
            assignments_path = os.path.join(output_dir, assignments_filename)
            
            result_data.to_csv(assignments_path, index=False)
            logger.info(f"Cluster assignments saved to: {assignments_path}")
        
        return {
            'cluster_labels': cluster_labels.tolist(),
            'assignments_data': result_data,
            'assignments_path': assignments_path,
            'cluster_statistics': cluster_stats,
            'n_clusters': int(len(np.unique(cluster_labels)))
        }
    
    def _save_model(self, model: KMeans, context: Dict[str, Any], 
                   hyperparameters: Dict[str, Any]) -> str:
        """Save trained clustering model to disk."""
        try:
            output_dir = context.get('output_dir', 'outputs/models')
            os.makedirs(output_dir, exist_ok=True)
            
            model_filename = hyperparameters.get('model_filename', 'kmeans_clustering_model.pkl')
            model_path = os.path.join(output_dir, model_filename)
            
            # Save model with metadata
            model_data = {
                'model': model,
                'scaler': self.scaler,
                'pca': self.pca,
                'feature_names': self.feature_names_,
                'hyperparameters': hyperparameters,
                'cluster_centers': model.cluster_centers_,
                'n_clusters': model.n_clusters,
                'optimal_k': self.optimal_k_
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to: {model_path}")
            
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call execute() first.")
        
        # Apply same preprocessing as training
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        if self.pca is not None:
            X = self.pca.transform(X)
        
        return self.model.predict(X)
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers in the processed feature space."""
        if self.model is None:
            return None
        return self.cluster_centers_
    
    def get_cluster_centers_original_space(self) -> Optional[Dict[str, np.ndarray]]:
        """Get cluster centers transformed back to original feature space."""
        if self.model is None or self.feature_names_ is None:
            return None
        
        centers = self.cluster_centers_
        
        # Transform back through PCA if applied
        if self.pca is not None:
            centers = self.pca.inverse_transform(centers)
        
        # Transform back through scaling if applied
        if self.scaler is not None:
            centers = self.scaler.inverse_transform(centers)
        
        # Create dictionary with feature names
        centers_dict = {}
        for i, center in enumerate(centers):
            centers_dict[f'cluster_{i}'] = {
                feature: float(value) 
                for feature, value in zip(self.feature_names_, center)
            }
        
        return centers_dict
