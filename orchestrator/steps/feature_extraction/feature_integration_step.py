# orchestrator/steps/feature_extraction/feature_integration_step.py
"""
Feature integration step for multi-source data fusion.

Combines spectral indices, topographic derivatives, texture features,
and absorption features into integrated feature stacks.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from scipy.stats import pearsonr
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd

from ..base.base_step import BaseStep
from ..base.step_registry import StepRegistry
from ...utils.file_utils import ensure_directory, validate_file_exists
from ...utils.geospatial_utils import get_raster_info


class FeatureIntegrationStep(BaseStep):
    """
    Universal feature integration step for multi-source data fusion.
    
    Capabilities:
    - Multi-source feature stacking
    - Spatial resampling and alignment
    - Feature scaling and normalization
    - Correlation analysis and filtering
    - Feature selection and dimensionality reduction
    - Missing data handling
    - Quality assessment
    
    Configuration Examples:
    
    For Landslide Assessment:
    {
        "feature_sources": ["spectral_indices", "topographic_derivatives", "texture_features"],
        "target_resolution": 10,
        "resampling_method": "bilinear",
        "scaling_method": "robust",
        "correlation_threshold": 0.95,
        "feature_selection": "variance",
        "missing_data_strategy": "interpolate"
    }
    
    For Mineral Targeting:
    {
        "feature_sources": ["spectral_indices", "absorption_features", "topographic_derivatives"],
        "target_resolution": 5,
        "scaling_method": "standard", 
        "feature_selection": "correlation",
        "max_features": 50,
        "create_feature_report": true
    }
    """
    
    def __init__(self, step_id: str, hyperparameters: Dict[str, Any]):
        super().__init__(step_id, "feature_integration", hyperparameters)
        
        # Feature sources
        self.feature_sources = hyperparameters.get("feature_sources", ["spectral_indices", "topographic_derivatives"])
        self.primary_source = hyperparameters.get("primary_source", None)  # Reference for geometry
        
        # Spatial processing
        self.target_resolution = hyperparameters.get("target_resolution", 10)
        self.resampling_method = hyperparameters.get("resampling_method", "bilinear")  # nearest, bilinear, cubic
        self.target_crs = hyperparameters.get("target_crs", None)
        self.clip_to_extent = hyperparameters.get("clip_to_extent", True)
        
        # Feature processing
        self.scaling_method = hyperparameters.get("scaling_method", "robust")  # standard, robust, minmax, none
        self.handle_missing_data = hyperparameters.get("handle_missing_data", True)
        self.missing_data_strategy = hyperparameters.get("missing_data_strategy", "interpolate")  # interpolate, fill, remove
        self.fill_value = hyperparameters.get("fill_value", 0)
        
        # Feature selection
        self.feature_selection = hyperparameters.get("feature_selection", "correlation")  # variance, correlation, kbest, none
        self.correlation_threshold = hyperparameters.get("correlation_threshold", 0.95)
        self.variance_threshold = hyperparameters.get("variance_threshold", 0.01)
        self.max_features = hyperparameters.get("max_features", None)
        
        # Quality control
        self.quality_checks = hyperparameters.get("quality_checks", True)
        self.outlier_detection = hyperparameters.get("outlier_detection", False)
        self.outlier_threshold = hyperparameters.get("outlier_threshold", 3.0)
        
        # Output options
        self.output_format = hyperparameters.get("output_format", "multiband")
        self.create_feature_report = hyperparameters.get("create_feature_report", True)
        self.save_feature_names = hyperparameters.get("save_feature_names", True)
        self.data_type = hyperparameters.get("data_type", "float32")
        self.nodata_value = hyperparameters.get("nodata_value", -9999)
        self.compression = hyperparameters.get("compression", "lzw")
        
        self.logger = logging.getLogger(f"FeatureIntegration.{step_id}")
        self.feature_names = []
        self.scaler = None
    
    def execute(self, context) -> Dict[str, Any]:
        """Execute feature integration"""
        try:
            self.logger.info(f"Starting feature integration from {len(self.feature_sources)} sources")
            
            # Get feature data from different sources
            feature_datasets = self._get_feature_datasets(context)
            
            # Validate feature datasets
            self._validate_feature_datasets(feature_datasets)
            
            # Determine reference geometry
            reference_profile, reference_transform = self._determine_reference_geometry(feature_datasets)
            
            # Align and resample all feature datasets
            aligned_datasets = self._align_feature_datasets(feature_datasets, reference_profile, reference_transform)
            
            # Stack features
            integrated_features, feature_names = self._stack_features(aligned_datasets)
            
            # Handle missing data
            if self.handle_missing_data:
                integrated_features = self._handle_missing_data(integrated_features)
            
            # Feature scaling
            if self.scaling_method != "none":
                integrated_features, scaler_info = self._scale_features(integrated_features)
            else:
                scaler_info = None
            
            # Feature selection
            if self.feature_selection != "none":
                integrated_features, selected_features, selection_info = self._select_features(
                    integrated_features, feature_names
                )
                feature_names = selected_features
            else:
                selection_info = None
            
            # Quality assessment
            if self.quality_checks:
                quality_results = self._assess_feature_quality(integrated_features, feature_names)
            else:
                quality_results = {}
            
            # Save integrated features
            output_files = self._save_integrated_features(
                integrated_features, feature_names, reference_profile, context
            )
            
            # Create feature report
            if self.create_feature_report:
                report_file = self._create_integration_report(
                    feature_names, scaler_info, selection_info, quality_results, context
                )
            else:
                report_file = ""
            
            # Store feature names for later use
            self.feature_names = feature_names
            
            # Update context
            context.add_data(f"{self.step_id}_integrated_features", output_files)
            context.add_data(f"{self.step_id}_feature_names", feature_names)
            if scaler_info:
                context.add_data(f"{self.step_id}_scaler_info", scaler_info)
            
            processing_time = quality_results.get("processing_time", 0)
            
            self.logger.info("Feature integration completed successfully")
            return {
                "status": "success",
                "outputs": {
                    "output_files": output_files,
                    "feature_names": feature_names,
                    "scaler_info": scaler_info,
                    "selection_info": selection_info,
                    "quality_results": quality_results,
                    "integration_report": report_file,
                    "processing_time": processing_time
                },
                "metadata": {
                    "feature_sources": self.feature_sources,
                    "total_features": len(feature_names),
                    "target_resolution": self.target_resolution,
                    "scaling_method": self.scaling_method,
                    "feature_selection": self.feature_selection
                }
            }
            
        except Exception as e:
            self.logger.error(f"Feature integration failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _get_feature_datasets(self, context) -> Dict[str, List[str]]:
        """Get feature datasets from context"""
        feature_datasets = {}
        
        for source in self.feature_sources:
            # Try different naming conventions
            possible_keys = [
                f"{source}_step_output_files",
                f"{source}",
                source,
                f"{source}_features",
                f"{source}_data"
            ]
            
            data = None
            for key in possible_keys:
                data = getattr(context, 'get_data', lambda x: None)(key)
                if data:
                    break
            
            if data:
                if isinstance(data, str):
                    feature_datasets[source] = [data]
                elif isinstance(data, list):
                    feature_datasets[source] = data
                else:
                    self.logger.warning(f"Unexpected data type for {source}: {type(data)}")
            else:
                self.logger.warning(f"No data found for feature source: {source}")
        
        return feature_datasets
    
    def _validate_feature_datasets(self, feature_datasets: Dict[str, List[str]]):
        """Validate feature datasets"""
        if not feature_datasets:
            raise ValueError("No feature datasets found")
        
        for source, files in feature_datasets.items():
            if not files:
                raise ValueError(f"No files found for feature source: {source}")
            
            for file_path in files:
                if not validate_file_exists(file_path):
                    raise FileNotFoundError(f"Feature file not found: {file_path}")
    
    def _determine_reference_geometry(self, feature_datasets: Dict[str, List[str]]) -> Tuple[dict, Any]:
        """Determine reference geometry for alignment"""
        
        # Use primary source if specified
        if self.primary_source and self.primary_source in feature_datasets:
            reference_file = feature_datasets[self.primary_source][0]
        else:
            # Use first available dataset
            first_source = list(feature_datasets.keys())[0]
            reference_file = feature_datasets[first_source][0]
        
        self.logger.info(f"Using {reference_file} as reference geometry")
        
        with rasterio.open(reference_file) as src:
            reference_profile = src.profile.copy()
            reference_transform = src.transform
            
            # Adjust resolution if specified
            if self.target_resolution:
                # Calculate new dimensions
                pixel_size_x = abs(reference_transform.a)
                pixel_size_y = abs(reference_transform.e)
                
                if pixel_size_x != self.target_resolution or pixel_size_y != self.target_resolution:
                    scale_x = pixel_size_x / self.target_resolution
                    scale_y = pixel_size_y / self.target_resolution
                    
                    new_width = int(src.width * scale_x)
                    new_height = int(src.height * scale_y)
                    
                    # Update transform
                    new_transform = reference_transform * rasterio.transform.Affine.scale(
                        self.target_resolution / pixel_size_x,
                        self.target_resolution / pixel_size_y
                    )
                    
                    reference_profile.update({
                        'width': new_width,
                        'height': new_height,
                        'transform': new_transform
                    })
                    reference_transform = new_transform
            
            # Update CRS if specified
            if self.target_crs:
                reference_profile['crs'] = self.target_crs
        
        return reference_profile, reference_transform
    
    def _align_feature_datasets(self, feature_datasets: Dict[str, List[str]], 
                               reference_profile: dict, reference_transform) -> Dict[str, np.ndarray]:
        """Align and resample all feature datasets to reference geometry"""
        aligned_datasets = {}
        
        target_width = reference_profile['width']
        target_height = reference_profile['height']
        target_crs = reference_profile['crs']
        
        resampling_method = getattr(Resampling, self.resampling_method)
        
        for source, files in feature_datasets.items():
            self.logger.info(f"Aligning {source} features")
            
            source_features = []
            
            for file_path in files:
                with rasterio.open(file_path) as src:
                    # Read all bands
                    source_data = src.read()
                    
                    # Reproject if necessary
                    if (src.width != target_width or src.height != target_height or 
                        src.crs != target_crs or src.transform != reference_transform):
                        
                        # Create destination array
                        dst_data = np.zeros((src.count, target_height, target_width), dtype=src.dtype)
                        
                        # Reproject each band
                        reproject(
                            source=source_data,
                            destination=dst_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=reference_transform,
                            dst_crs=target_crs,
                            resampling=resampling_method
                        )
                        
                        aligned_data = dst_data
                    else:
                        aligned_data = source_data
                    
                    # Handle nodata values
                    if src.nodata is not None:
                        aligned_data = np.where(aligned_data == src.nodata, np.nan, aligned_data)
                    
                    source_features.append(aligned_data)
            
            # Concatenate all features from this source
            if source_features:
                concatenated_features = np.concatenate(source_features, axis=0)
                aligned_datasets[source] = concatenated_features.astype(np.float32)
        
        return aligned_datasets
    
    def _stack_features(self, aligned_datasets: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """Stack features from all sources"""
        feature_arrays = []
        feature_names = []
        
        for source, features in aligned_datasets.items():
            feature_arrays.append(features)
            
            # Generate feature names
            n_bands = features.shape[0]
            for i in range(n_bands):
                feature_names.append(f"{source}_band_{i+1}")
        
        # Stack all features
        integrated_features = np.concatenate(feature_arrays, axis=0)
        
        self.logger.info(f"Integrated {integrated_features.shape[0]} features from {len(aligned_datasets)} sources")
        
        return integrated_features, feature_names
    
    def _handle_missing_data(self, features: np.ndarray) -> np.ndarray:
        """Handle missing data in feature stack"""
        if self.missing_data_strategy == "fill":
            # Fill with specified value
            features = np.where(np.isnan(features), self.fill_value, features)
            
        elif self.missing_data_strategy == "interpolate":
            # Simple interpolation (nearest neighbor)
            for band_idx in range(features.shape[0]):
                band = features[band_idx]
                if np.any(np.isnan(band)):
                    # Find valid pixels
                    valid_mask = ~np.isnan(band)
                    if np.sum(valid_mask) > 0:
                        # Use median of valid pixels as fill value
                        fill_val = np.median(band[valid_mask])
                        features[band_idx] = np.where(np.isnan(band), fill_val, band)
        
        elif self.missing_data_strategy == "remove":
            # Create mask for pixels with any missing data
            any_missing = np.any(np.isnan(features), axis=0)
            features = np.where(any_missing[np.newaxis, :, :], np.nan, features)
        
        return features
    
    def _scale_features(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Scale features"""
        n_bands, height, width = features.shape
        
        # Reshape for scaling
        features_reshaped = features.reshape(n_bands, -1).T  # (pixels, features)
        
        # Remove pixels with any NaN values for scaling
        valid_mask = ~np.any(np.isnan(features_reshaped), axis=1)
        valid_features = features_reshaped[valid_mask]
        
        if len(valid_features) == 0:
            self.logger.warning("No valid pixels for feature scaling")
            return features, {"method": self.scaling_method, "error": "No valid pixels"}
        
        # Apply scaling
        if self.scaling_method == "standard":
            scaler = StandardScaler()
        elif self.scaling_method == "robust":
            scaler = RobustScaler()
        elif self.scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            return features, {"method": self.scaling_method, "error": "Unknown scaling method"}
        
        # Fit scaler on valid data
        scaled_valid = scaler.fit_transform(valid_features)
        
        # Apply scaling to all data
        scaled_features = np.full_like(features_reshaped, np.nan)
        scaled_features[valid_mask] = scaled_valid
        
        # Reshape back
        scaled_features = scaled_features.T.reshape(n_bands, height, width)
        
        # Store scaler info
        scaler_info = {
            "method": self.scaling_method,
            "n_features": n_bands,
            "n_valid_pixels": len(valid_features),
            "scaler_params": {
                "mean_": getattr(scaler, 'mean_', None),
                "scale_": getattr(scaler, 'scale_', None),
                "center_": getattr(scaler, 'center_', None)
            }
        }
        
        self.scaler = scaler
        
        return scaled_features, scaler_info
    
    def _select_features(self, features: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Select features based on specified criteria"""
        n_bands, height, width = features.shape
        
        if self.feature_selection == "variance":
            selected_features, selected_names, selection_info = self._variance_feature_selection(
                features, feature_names
            )
        elif self.feature_selection == "correlation":
            selected_features, selected_names, selection_info = self._correlation_feature_selection(
                features, feature_names
            )
        elif self.feature_selection == "kbest":
            selected_features, selected_names, selection_info = self._kbest_feature_selection(
                features, feature_names
            )
        else:
            return features, feature_names, {"method": "none"}
        
        self.logger.info(f"Feature selection: {n_bands} -> {len(selected_names)} features")
        
        return selected_features, selected_names, selection_info
    
    def _variance_feature_selection(self, features: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Select features based on variance threshold"""
        n_bands, height, width = features.shape
        
        # Calculate variance for each band
        variances = []
        for i in range(n_bands):
            band = features[i]
            valid_pixels = band[~np.isnan(band)]
            if len(valid_pixels) > 1:
                variance = np.var(valid_pixels)
            else:
                variance = 0
            variances.append(variance)
        
        # Select features above threshold
        selected_indices = [i for i, var in enumerate(variances) if var > self.variance_threshold]
        
        if not selected_indices:
            self.logger.warning("No features passed variance threshold, keeping all")
            selected_indices = list(range(n_bands))
        
        selected_features = features[selected_indices]
        selected_names = [feature_names[i] for i in selected_indices]
        
        selection_info = {
            "method": "variance",
            "threshold": self.variance_threshold,
            "original_features": n_bands,
            "selected_features": len(selected_indices),
            "variances": variances,
            "selected_indices": selected_indices
        }
        
        return selected_features, selected_names, selection_info
    
    def _correlation_feature_selection(self, features: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Select features based on correlation analysis"""
        n_bands, height, width = features.shape
        
        # Reshape for correlation calculation
        features_reshaped = features.reshape(n_bands, -1)
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(features_reshaped)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(n_bands):
            for j in range(i+1, n_bands):
                if abs(correlation_matrix[i, j]) > self.correlation_threshold:
                    high_corr_pairs.append((i, j, abs(correlation_matrix[i, j])))
        
        # Remove features with high correlation (keep first in pair)
        features_to_remove = set()
        for i, j, corr in high_corr_pairs:
            features_to_remove.add(j)  # Remove second feature
        
        selected_indices = [i for i in range(n_bands) if i not in features_to_remove]
        
        # Apply max features limit if specified
        if self.max_features and len(selected_indices) > self.max_features:
            # Keep features with highest variance
            variances = []
            for i in selected_indices:
                band = features[i]
                valid_pixels = band[~np.isnan(band)]
                variance = np.var(valid_pixels) if len(valid_pixels) > 1 else 0
                variances.append((i, variance))
            
            # Sort by variance and select top features
            variances.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [i for i, _ in variances[:self.max_features]]
        
        selected_features = features[selected_indices]
        selected_names = [feature_names[i] for i in selected_indices]
        
        selection_info = {
            "method": "correlation",
            "threshold": self.correlation_threshold,
            "original_features": n_bands,
            "selected_features": len(selected_indices),
            "high_corr_pairs": len(high_corr_pairs),
            "removed_features": len(features_to_remove),
            "selected_indices": selected_indices
        }
        
        return selected_features, selected_names, selection_info
    
    def _kbest_feature_selection(self, features: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Select K best features (placeholder - requires target variable)"""
        # This would require a target variable for supervised feature selection
        # For now, fall back to variance-based selection
        self.logger.warning("K-best selection requires target variable, using variance selection")
        return self._variance_feature_selection(features, feature_names)
    
    def _assess_feature_quality(self, features: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Assess quality of integrated features"""
        processing_start = datetime.now()
        
        n_bands, height, width = features.shape
        total_pixels = height * width
        
        quality_results = {
            "n_features": n_bands,
            "spatial_dimensions": (height, width),
            "total_pixels": total_pixels,
            "feature_statistics": []
        }
        
        # Calculate statistics for each feature
        for i, feature_name in enumerate(feature_names):
            band = features[i]
            valid_pixels = ~np.isnan(band)
            n_valid = np.sum(valid_pixels)
            
            if n_valid > 0:
                valid_data = band[valid_pixels]
                stats = {
                    "feature_name": feature_name,
                    "n_valid_pixels": int(n_valid),
                    "coverage_percent": (n_valid / total_pixels) * 100,
                    "mean": float(np.mean(valid_data)),
                    "std": float(np.std(valid_data)),
                    "min": float(np.min(valid_data)),
                    "max": float(np.max(valid_data)),
                    "median": float(np.median(valid_data))
                }
            else:
                stats = {
                    "feature_name": feature_name,
                    "n_valid_pixels": 0,
                    "coverage_percent": 0,
                    "error": "No valid pixels"
                }
            
            quality_results["feature_statistics"].append(stats)
        
        # Overall quality metrics
        valid_features = [s for s in quality_results["feature_statistics"] if "error" not in s]
        if valid_features:
            quality_results["mean_coverage"] = np.mean([s["coverage_percent"] for s in valid_features])
            quality_results["min_coverage"] = min([s["coverage_percent"] for s in valid_features])
            quality_results["features_with_full_coverage"] = sum(1 for s in valid_features if s["coverage_percent"] > 95)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        quality_results["processing_time"] = processing_time
        
        return quality_results
    
    def _save_integrated_features(self, features: np.ndarray, feature_names: List[str],
                                 reference_profile: dict, context) -> List[str]:
        """Save integrated features to file"""
        output_dir = context.get_temp_dir() / "integrated_features"
        ensure_directory(output_dir)
        
        # Update profile for output
        output_profile = reference_profile.copy()
        output_profile.update({
            'count': features.shape[0],
            'dtype': self.data_type,
            'nodata': self.nodata_value,
            'compress': self.compression
        })
        
        output_file = output_dir / "integrated_features.tif"
        
        with rasterio.open(output_file, 'w', **output_profile) as dst:
            for i in range(features.shape[0]):
                # Handle NaN values
                band_data = np.where(np.isnan(features[i]), self.nodata_value, features[i])
                dst.write(band_data.astype(self.data_type), i + 1)
                
                # Set band description
                if i < len(feature_names):
                    dst.set_band_description(i + 1, feature_names[i])
        
        output_files = [str(output_file)]
        
        # Save feature names separately if requested
        if self.save_feature_names:
            names_file = output_dir / "feature_names.txt"
            with open(names_file, 'w') as f:
                for name in feature_names:
                    f.write(f"{name}\n")
            output_files.append(str(names_file))
        
        return output_files
    
    def _create_integration_report(self, feature_names: List[str], scaler_info: Optional[Dict[str, Any]],
                                  selection_info: Optional[Dict[str, Any]], quality_results: Dict[str, Any],
                                  context) -> str:
        """Create feature integration report"""
        try:
            output_dir = context.get_temp_dir() / "integrated_features"
            ensure_directory(output_dir)
            
            report_content = f"""
Feature Integration Report
=========================

Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Feature Sources: {', '.join(self.feature_sources)}
Target Resolution: {self.target_resolution}m
Resampling Method: {self.resampling_method}

Integration Summary:
-------------------
- Total Features: {len(feature_names)}
- Spatial Dimensions: {quality_results['spatial_dimensions']}
- Total Pixels: {quality_results['total_pixels']:,}
"""
            
            # Scaling information
            if scaler_info:
                report_content += f"""
Feature Scaling:
---------------
- Method: {scaler_info['method']}
- Features Scaled: {scaler_info['n_features']}
- Valid Pixels Used: {scaler_info['n_valid_pixels']:,}
"""
            
            # Feature selection information
            if selection_info and selection_info.get("method") != "none":
                report_content += f"""
Feature Selection:
-----------------
- Method: {selection_info['method']}
- Original Features: {selection_info['original_features']}
- Selected Features: {selection_info['selected_features']}
"""
                
                if selection_info['method'] == 'correlation':
                    report_content += f"- Correlation Threshold: {selection_info['threshold']}\n"
                    report_content += f"- High Correlation Pairs: {selection_info['high_corr_pairs']}\n"
                    report_content += f"- Features Removed: {selection_info['removed_features']}\n"
                elif selection_info['method'] == 'variance':
                    report_content += f"- Variance Threshold: {selection_info['threshold']}\n"
            
            # Quality assessment
            if quality_results:
                report_content += f"""
Quality Assessment:
------------------
- Mean Coverage: {quality_results.get('mean_coverage', 0):.1f}%
- Minimum Coverage: {quality_results.get('min_coverage', 0):.1f}%
- Features with >95% Coverage: {quality_results.get('features_with_full_coverage', 0)}
"""
            
            # Feature list
            report_content += f"""
Feature List ({len(feature_names)} features):
"""
            
            # Group features by source
            feature_groups = {}
            for name in feature_names:
                source = name.split('_')[0] if '_' in name else 'unknown'
                if source not in feature_groups:
                    feature_groups[source] = []
                feature_groups[source].append(name)
            
            for source, names in feature_groups.items():
                report_content += f"\n{source.upper()} ({len(names)} features):\n"
                for name in names[:10]:  # Show first 10
                    report_content += f"- {name}\n"
                if len(names) > 10:
                    report_content += f"... and {len(names) - 10} more\n"
            
            # Feature statistics (top 10 by coverage)
            if quality_results.get('feature_statistics'):
                valid_stats = [s for s in quality_results['feature_statistics'] if 'error' not in s]
                if valid_stats:
                    # Sort by coverage
                    valid_stats.sort(key=lambda x: x['coverage_percent'], reverse=True)
                    
                    report_content += f"""
Top Features by Coverage:
------------------------
"""
                    for stats in valid_stats[:10]:
                        report_content += f"- {stats['feature_name']}: {stats['coverage_percent']:.1f}% coverage, "
                        report_content += f"mean={stats['mean']:.4f}, std={stats['std']:.4f}\n"
            
            # Save report
            report_file = output_dir / f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            return str(report_file)
            
        except Exception as e:
            self.logger.warning(f"Failed to create integration report: {str(e)}")
            return ""


# Register the step
StepRegistry.register("feature_integration", FeatureIntegrationStep)
