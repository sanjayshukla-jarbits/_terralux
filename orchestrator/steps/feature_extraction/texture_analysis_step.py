# orchestrator/steps/feature_extraction/texture_analysis_step.py
"""
Texture analysis step for spatial pattern analysis.

Calculates GLCM, Gabor filters, Local Binary Patterns, and other texture features
from single or multi-band imagery.
"""

import numpy as np
import rasterio
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor_kernel, gabor
from skimage.util import img_as_ubyte
from scipy import ndimage
from scipy.stats import entropy
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from ..base.base_step import BaseStep
from ..base.step_registry import StepRegistry
from ...utils.file_utils import ensure_directory, validate_file_exists


class TextureAnalysisStep(BaseStep):
    """
    Universal texture analysis step for spatial pattern extraction.
    
    Capabilities:
    - Gray Level Co-occurrence Matrix (GLCM) features
    - Gabor filter responses
    - Local Binary Patterns (LBP)
    - Statistical texture measures
    - Multi-band texture analysis
    - Multi-scale analysis
    
    Configuration Examples:
    
    For Landslide Assessment:
    {
        "texture_methods": ["glcm", "lbp"],
        "glcm_features": ["contrast", "dissimilarity", "homogeneity", "energy"],
        "glcm_distances": [1, 2, 3],
        "glcm_angles": [0, 45, 90, 135],
        "target_bands": ["B04", "B08", "NDVI"],
        "window_size": 9
    }
    
    For Mineral Targeting:
    {
        "texture_methods": ["glcm", "gabor"],
        "glcm_features": ["contrast", "correlation", "energy", "homogeneity"],
        "gabor_frequencies": [0.1, 0.3, 0.5],
        "gabor_orientations": [0, 30, 60, 90, 120, 150],
        "target_bands": ["B11", "B12"],
        "multi_scale": true
    }
    """
    
    def __init__(self, step_id: str, hyperparameters: Dict[str, Any]):
        super().__init__(step_id, "texture_analysis", hyperparameters)
        
        # Texture methods to apply
        self.texture_methods = hyperparameters.get("texture_methods", ["glcm"])
        self.target_bands = hyperparameters.get("target_bands", ["all"])
        
        # GLCM parameters
        self.glcm_features = hyperparameters.get("glcm_features", ["contrast", "dissimilarity", "homogeneity", "energy"])
        self.glcm_distances = hyperparameters.get("glcm_distances", [1, 2, 3])
        self.glcm_angles = hyperparameters.get("glcm_angles", [0, 45, 90, 135])
        self.glcm_levels = hyperparameters.get("glcm_levels", 256)
        self.glcm_symmetric = hyperparameters.get("glcm_symmetric", True)
        self.glcm_normed = hyperparameters.get("glcm_normed", True)
        
        # Gabor filter parameters
        self.gabor_frequencies = hyperparameters.get("gabor_frequencies", [0.1, 0.3, 0.5])
        self.gabor_orientations = hyperparameters.get("gabor_orientations", [0, 45, 90, 135])
        self.gabor_sigma_x = hyperparameters.get("gabor_sigma_x", 2.0)
        self.gabor_sigma_y = hyperparameters.get("gabor_sigma_y", 2.0)
        
        # LBP parameters
        self.lbp_radius = hyperparameters.get("lbp_radius", 3)
        self.lbp_n_points = hyperparameters.get("lbp_n_points", 24)
        self.lbp_method = hyperparameters.get("lbp_method", "uniform")
        
        # Processing parameters
        self.window_size = hyperparameters.get("window_size", 9)
        self.multi_scale = hyperparameters.get("multi_scale", False)
        self.scale_factors = hyperparameters.get("scale_factors", [1, 2, 4])
        
        # Statistical measures
        self.statistical_features = hyperparameters.get("statistical_features", ["mean", "std", "skewness", "kurtosis"])
        
        # Output options
        self.output_format = hyperparameters.get("output_format", "multiband")
        self.data_type = hyperparameters.get("data_type", "float32")
        self.nodata_value = hyperparameters.get("nodata_value", -9999)
        self.compression = hyperparameters.get("compression", "lzw")
        
        # Quality control
        self.normalize_input = hyperparameters.get("normalize_input", True)
        self.edge_handling = hyperparameters.get("edge_handling", "reflect")
        
        self.logger = logging.getLogger(f"TextureAnalysis.{step_id}")
    
    def execute(self, context) -> Dict[str, Any]:
        """Execute texture analysis"""
        try:
            self.logger.info(f"Starting texture analysis using {len(self.texture_methods)} methods")
            
            # Get input data
            input_files = self._get_input_files(context)
            
            # Validate inputs
            self._validate_inputs(input_files)
            
            # Process each input file
            processing_results = []
            for input_file in input_files:
                file_results = self._process_single_file(input_file, context)
                processing_results.append(file_results)
            
            # Combine results
            combined_results = self._combine_results(processing_results, context)
            
            # Update context
            context.add_data(f"{self.step_id}_texture_features", combined_results["output_files"])
            context.add_data(f"{self.step_id}_texture_metadata", combined_results["texture_metadata"])
            
            self.logger.info("Texture analysis completed successfully")
            return {
                "status": "success",
                "outputs": combined_results,
                "metadata": {
                    "texture_methods": self.texture_methods,
                    "features_calculated": combined_results.get("total_features", 0),
                    "processing_time": combined_results.get("processing_time")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Texture analysis failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _get_input_files(self, context) -> List[str]:
        """Get input files from context"""
        # Try different possible data keys
        for key in ["spectral_indices", "resampled_images", "corrected_images", "input_images"]:
            data = getattr(context, 'get_data', lambda x: None)(key)
            if data:
                if isinstance(data, str):
                    return [data]
                elif isinstance(data, list):
                    return data
        
        # Check hyperparameters
        if "input_files" in self.hyperparameters:
            input_files = self.hyperparameters["input_files"]
            if isinstance(input_files, str):
                return [input_files]
            return input_files
        
        raise ValueError("No input files found for texture analysis")
    
    def _validate_inputs(self, input_files: List[str]):
        """Validate input files"""
        if not input_files:
            raise ValueError("No input files provided for texture analysis")
        
        for file_path in input_files:
            if not validate_file_exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
    
    def _process_single_file(self, input_file: str, context) -> Dict[str, Any]:
        """Process a single input file"""
        self.logger.info(f"Processing texture features for {Path(input_file).name}")
        processing_start = datetime.now()
        
        with rasterio.open(input_file) as src:
            # Read data
            profile = src.profile.copy()
            
            # Determine which bands to process
            bands_to_process = self._determine_bands_to_process(src)
            
            calculated_features = {}
            texture_metadata = []
            
            # Process each band
            for band_idx in bands_to_process:
                band_data = src.read(band_idx).astype(np.float32)
                
                # Handle nodata values
                if src.nodata is not None:
                    band_data = np.where(band_data == src.nodata, np.nan, band_data)
                
                # Normalize if requested
                if self.normalize_input:
                    band_data = self._normalize_band(band_data)
                
                band_name = f"band_{band_idx}"
                if hasattr(src, 'descriptions') and src.descriptions[band_idx-1]:
                    band_name = src.descriptions[band_idx-1]
                
                # Calculate texture features for this band
                band_features = self._calculate_texture_features(band_data, band_name)
                calculated_features.update(band_features)
                
                # Store metadata
                for feature_name in band_features.keys():
                    texture_metadata.append({
                        "feature_name": feature_name,
                        "source_band": band_name,
                        "method": self._get_method_from_feature_name(feature_name),
                        "valid_pixels": int(np.sum(~np.isnan(band_features[feature_name]))),
                        "mean_value": float(np.nanmean(band_features[feature_name])),
                        "std_value": float(np.nanstd(band_features[feature_name]))
                    })
            
            # Multi-scale analysis if requested
            if self.multi_scale:
                multiscale_features = self._calculate_multiscale_features(src, bands_to_process)
                calculated_features.update(multiscale_features)
            
            # Save results
            output_files = self._save_texture_features(calculated_features, profile, input_file, context)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        return {
            "input_file": input_file,
            "output_files": output_files,
            "calculated_features": calculated_features,
            "texture_metadata": texture_metadata,
            "processing_time": processing_time
        }
    
    def _determine_bands_to_process(self, src) -> List[int]:
        """Determine which bands to process"""
        if self.target_bands == ["all"]:
            return list(range(1, src.count + 1))
        
        bands_to_process = []
        
        for target_band in self.target_bands:
            if isinstance(target_band, int):
                if 1 <= target_band <= src.count:
                    bands_to_process.append(target_band)
            elif isinstance(target_band, str):
                # Try to match band descriptions
                if hasattr(src, 'descriptions'):
                    for i, desc in enumerate(src.descriptions):
                        if desc and target_band.lower() in desc.lower():
                            bands_to_process.append(i + 1)
                            break
                else:
                    # Try to parse as band number (e.g., "B04" -> 4)
                    import re
                    match = re.search(r'\d+', target_band)
                    if match:
                        band_num = int(match.group())
                        if 1 <= band_num <= src.count:
                            bands_to_process.append(band_num)
        
        if not bands_to_process:
            self.logger.warning("No matching bands found, using first band")
            bands_to_process = [1]
        
        return bands_to_process
    
    def _normalize_band(self, band_data: np.ndarray) -> np.ndarray:
        """Normalize band data to 0-255 range for texture analysis"""
        valid_data = band_data[~np.isnan(band_data)]
        
        if len(valid_data) == 0:
            return band_data
        
        min_val, max_val = np.percentile(valid_data, [2, 98])  # Robust normalization
        
        if max_val > min_val:
            normalized = ((band_data - min_val) / (max_val - min_val)) * 255
            normalized = np.clip(normalized, 0, 255)
        else:
            normalized = np.full_like(band_data, 128)  # Mid-gray if no variation
        
        return normalized
    
    def _calculate_texture_features(self, band_data: np.ndarray, band_name: str) -> Dict[str, np.ndarray]:
        """Calculate texture features for a single band"""
        features = {}
        
        for method in self.texture_methods:
            if method == "glcm":
                glcm_features = self._calculate_glcm_features(band_data, band_name)
                features.update(glcm_features)
            
            elif method == "gabor":
                gabor_features = self._calculate_gabor_features(band_data, band_name)
                features.update(gabor_features)
            
            elif method == "lbp":
                lbp_features = self._calculate_lbp_features(band_data, band_name)
                features.update(lbp_features)
            
            elif method == "statistical":
                stat_features = self._calculate_statistical_features(band_data, band_name)
                features.update(stat_features)
            
            else:
                self.logger.warning(f"Unknown texture method: {method}")
        
        return features
    
    def _calculate_glcm_features(self, band_data: np.ndarray, band_name: str) -> Dict[str, np.ndarray]:
        """Calculate GLCM texture features"""
        features = {}
        
        # Convert to uint8 for GLCM calculation
        if np.isnan(band_data).any():
            # Handle NaN values
            valid_mask = ~np.isnan(band_data)
            band_uint8 = np.zeros_like(band_data, dtype=np.uint8)
            if np.sum(valid_mask) > 0:
                band_uint8[valid_mask] = img_as_ubyte(band_data[valid_mask])
        else:
            band_uint8 = img_as_ubyte(band_data)
        
        # Calculate GLCM for each distance and angle
        for distance in self.glcm_distances:
            for angle_deg in self.glcm_angles:
                angle_rad = np.radians(angle_deg)
                
                try:
                    # Calculate GLCM
                    glcm = graycomatrix(
                        band_uint8, distances=[distance], angles=[angle_rad],
                        levels=self.glcm_levels, symmetric=self.glcm_symmetric,
                        normed=self.glcm_normed
                    )
                    
                    # Calculate GLCM properties
                    for feature in self.glcm_features:
                        if feature in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
                            prop_value = graycoprops(glcm, feature)[0, 0]
                            feature_name = f"{band_name}_glcm_{feature}_d{distance}_a{angle_deg}"
                            
                            # Create feature map (simplified - using single value for whole image)
                            feature_map = np.full_like(band_data, prop_value, dtype=np.float32)
                            features[feature_name] = feature_map
                
                except Exception as e:
                    self.logger.warning(f"GLCM calculation failed for {band_name}, d={distance}, a={angle_deg}: {str(e)}")
        
        # Calculate windowed GLCM features for spatial variation
        if self.window_size > 3:
            windowed_features = self._calculate_windowed_glcm(band_uint8, band_name)
            features.update(windowed_features)
        
        return features
    
    def _calculate_windowed_glcm(self, band_uint8: np.ndarray, band_name: str) -> Dict[str, np.ndarray]:
        """Calculate GLCM features in sliding windows"""
        features = {}
        
        # Use only first distance and angle for windowed analysis (computational efficiency)
        distance = self.glcm_distances[0]
        angle_rad = np.radians(self.glcm_angles[0])
        
        # Define window function
        def glcm_window_function(window):
            try:
                if window.size < 9:  # Minimum window size
                    return 0
                
                window_2d = window.reshape(self.window_size, self.window_size)
                
                # Calculate GLCM for window
                glcm = graycomatrix(
                    window_2d.astype(np.uint8), distances=[distance], angles=[angle_rad],
                    levels=64, symmetric=True, normed=True  # Reduced levels for efficiency
                )
                
                # Return first GLCM feature (contrast)
                return graycoprops(glcm, 'contrast')[0, 0]
            
            except:
                return 0
        
        # Apply windowed GLCM
        try:
            windowed_contrast = ndimage.generic_filter(
                band_uint8.astype(np.float64), glcm_window_function,
                size=self.window_size, mode=self.edge_handling
            )
            
            features[f"{band_name}_glcm_windowed_contrast"] = windowed_contrast.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Windowed GLCM calculation failed for {band_name}: {str(e)}")
        
        return features
    
    def _calculate_gabor_features(self, band_data: np.ndarray, band_name: str) -> Dict[str, np.ndarray]:
        """Calculate Gabor filter responses"""
        features = {}
        
        # Normalize data for Gabor filtering
        if np.isnan(band_data).any():
            valid_mask = ~np.isnan(band_data)
            if np.sum(valid_mask) == 0:
                return features
            band_normalized = np.zeros_like(band_data)
            band_normalized[valid_mask] = (band_data[valid_mask] - np.nanmean(band_data)) / (np.nanstd(band_data) + 1e-8)
        else:
            band_normalized = (band_data - np.mean(band_data)) / (np.std(band_data) + 1e-8)
        
        # Apply Gabor filters
        for frequency in self.gabor_frequencies:
            for orientation in self.gabor_orientations:
                try:
                    # Calculate Gabor response
                    real_response, _ = gabor(
                        band_normalized, frequency=frequency, 
                        theta=np.radians(orientation),
                        sigma_x=self.gabor_sigma_x, sigma_y=self.gabor_sigma_y
                    )
                    
                    # Store magnitude of response
                    feature_name = f"{band_name}_gabor_f{frequency:.1f}_o{orientation}"
                    features[feature_name] = np.abs(real_response).astype(np.float32)
                    
                except Exception as e:
                    self.logger.warning(f"Gabor filter failed for {band_name}, f={frequency}, o={orientation}: {str(e)}")
        
        return features
    
    def _calculate_lbp_features(self, band_data: np.ndarray, band_name: str) -> Dict[str, np.ndarray]:
        """Calculate Local Binary Pattern features"""
        features = {}
        
        try:
            # Convert to uint8
            if np.isnan(band_data).any():
                valid_mask = ~np.isnan(band_data)
                band_uint8 = np.zeros_like(band_data, dtype=np.uint8)
                if np.sum(valid_mask) > 0:
                    band_uint8[valid_mask] = img_as_ubyte(band_data[valid_mask])
            else:
                band_uint8 = img_as_ubyte(band_data)
            
            # Calculate LBP
            lbp = local_binary_pattern(
                band_uint8, self.lbp_n_points, self.lbp_radius, method=self.lbp_method
            )
            
            feature_name = f"{band_name}_lbp_r{self.lbp_radius}_p{self.lbp_n_points}"
            features[feature_name] = lbp.astype(np.float32)
            
            # Calculate LBP histogram features in windows
            if self.window_size > self.lbp_radius * 2 + 1:
                lbp_variance = self._calculate_lbp_variance(lbp, band_name)
                features.update(lbp_variance)
        
        except Exception as e:
            self.logger.warning(f"LBP calculation failed for {band_name}: {str(e)}")
        
        return features
    
    def _calculate_lbp_variance(self, lbp: np.ndarray, band_name: str) -> Dict[str, np.ndarray]:
        """Calculate LBP variance in sliding windows"""
        features = {}
        
        try:
            # Calculate LBP variance as texture measure
            lbp_variance = ndimage.generic_filter(
                lbp.astype(np.float64), np.var, size=self.window_size, mode=self.edge_handling
            )
            
            feature_name = f"{band_name}_lbp_variance"
            features[feature_name] = lbp_variance.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"LBP variance calculation failed: {str(e)}")
        
        return features
    
    def _calculate_statistical_features(self, band_data: np.ndarray, band_name: str) -> Dict[str, np.ndarray]:
        """Calculate statistical texture features"""
        features = {}
        
        for stat_feature in self.statistical_features:
            try:
                if stat_feature == "mean":
                    result = ndimage.generic_filter(
                        band_data, np.nanmean, size=self.window_size, mode=self.edge_handling
                    )
                elif stat_feature == "std":
                    result = ndimage.generic_filter(
                        band_data, np.nanstd, size=self.window_size, mode=self.edge_handling
                    )
                elif stat_feature == "variance":
                    result = ndimage.generic_filter(
                        band_data, np.nanvar, size=self.window_size, mode=self.edge_handling
                    )
                elif stat_feature == "skewness":
                    from scipy.stats import skew
                    def nan_skew(x):
                        x_clean = x[~np.isnan(x)]
                        return skew(x_clean) if len(x_clean) > 2 else 0
                    
                    result = ndimage.generic_filter(
                        band_data, nan_skew, size=self.window_size, mode=self.edge_handling
                    )
                elif stat_feature == "kurtosis":
                    from scipy.stats import kurtosis
                    def nan_kurtosis(x):
                        x_clean = x[~np.isnan(x)]
                        return kurtosis(x_clean) if len(x_clean) > 3 else 0
                    
                    result = ndimage.generic_filter(
                        band_data, nan_kurtosis, size=self.window_size, mode=self.edge_handling
                    )
                elif stat_feature == "entropy":
                    def local_entropy(x):
                        x_clean = x[~np.isnan(x)]
                        if len(x_clean) < 2:
                            return 0
                        # Calculate histogram-based entropy
                        hist, _ = np.histogram(x_clean, bins=10, density=True)
                        hist = hist[hist > 0]  # Remove zero entries
                        return entropy(hist) if len(hist) > 0 else 0
                    
                    result = ndimage.generic_filter(
                        band_data, local_entropy, size=self.window_size, mode=self.edge_handling
                    )
                else:
                    continue
                
                feature_name = f"{band_name}_stat_{stat_feature}"
                features[feature_name] = result.astype(np.float32)
                
            except Exception as e:
                self.logger.warning(f"Statistical feature {stat_feature} failed for {band_name}: {str(e)}")
        
        return features
    
    def _calculate_multiscale_features(self, src, bands_to_process: List[int]) -> Dict[str, np.ndarray]:
        """Calculate texture features at multiple scales"""
        multiscale_features = {}
        
        for scale in self.scale_factors:
            if scale == 1:
                continue  # Already calculated at original scale
            
            for band_idx in bands_to_process:
                band_data = src.read(band_idx).astype(np.float32)
                
                # Handle nodata
                if src.nodata is not None:
                    band_data = np.where(band_data == src.nodata, np.nan, band_data)
                
                # Downsample data
                if scale > 1:
                    # Simple downsampling by averaging
                    from skimage.transform import rescale
                    try:
                        downsampled = rescale(band_data, 1.0/scale, anti_aliasing=True, preserve_range=True)
                        # Upsample back to original size
                        upsampled = rescale(downsampled, scale, anti_aliasing=True, preserve_range=True)
                        
                        # Ensure same size as original
                        if upsampled.shape != band_data.shape:
                            from skimage.transform import resize
                            upsampled = resize(upsampled, band_data.shape, preserve_range=True)
                        
                        band_data_scaled = upsampled.astype(np.float32)
                    except Exception as e:
                        self.logger.warning(f"Multiscale processing failed for scale {scale}: {str(e)}")
                        continue
                else:
                    band_data_scaled = band_data
                
                # Normalize
                if self.normalize_input:
                    band_data_scaled = self._normalize_band(band_data_scaled)
                
                band_name = f"band_{band_idx}_scale_{scale}"
                
                # Calculate only GLCM features for multiscale (computational efficiency)
                if "glcm" in self.texture_methods:
                    try:
                        # Simple GLCM calculation for multiscale
                        band_uint8 = img_as_ubyte(band_data_scaled) if not np.isnan(band_data_scaled).all() else np.zeros_like(band_data_scaled, dtype=np.uint8)
                        
                        # Calculate only contrast for efficiency
                        glcm = graycomatrix(
                            band_uint8, distances=[1], angles=[0],
                            levels=64, symmetric=True, normed=True
                        )
                        
                        contrast = graycoprops(glcm, 'contrast')[0, 0]
                        feature_map = np.full_like(band_data, contrast, dtype=np.float32)
                        
                        feature_name = f"band_{band_idx}_glcm_contrast_scale_{scale}"
                        multiscale_features[feature_name] = feature_map
                        
                    except Exception as e:
                        self.logger.warning(f"Multiscale GLCM failed for band {band_idx}, scale {scale}: {str(e)}")
        
        return multiscale_features
    
    def _get_method_from_feature_name(self, feature_name: str) -> str:
        """Extract method name from feature name"""
        if "glcm" in feature_name:
            return "glcm"
        elif "gabor" in feature_name:
            return "gabor"
        elif "lbp" in feature_name:
            return "lbp"
        elif "stat" in feature_name:
            return "statistical"
        else:
            return "unknown"
    
    def _save_texture_features(self, features: Dict[str, np.ndarray], profile: dict,
                              input_file: str, context) -> List[str]:
        """Save calculated texture features to files"""
        output_dir = context.get_temp_dir() / "texture_features"
        ensure_directory(output_dir)
        
        output_files = []
        
        # Update profile for output
        output_profile = profile.copy()
        output_profile.update({
            'dtype': self.data_type,
            'nodata': self.nodata_value,
            'compress': self.compression
        })
        
        if self.output_format == "multiband":
            # Save as single multiband file
            output_profile['count'] = len(features)
            
            output_file = output_dir / f"texture_features_{Path(input_file).stem}.tif"
            
            with rasterio.open(output_file, 'w', **output_profile) as dst:
                for band_idx, (feature_name, feature_array) in enumerate(features.items(), 1):
                    # Handle NaN values
                    output_array = np.where(np.isnan(feature_array), self.nodata_value, feature_array)
                    dst.write(output_array.astype(self.data_type), band_idx)
                    dst.set_band_description(band_idx, feature_name)
            
            output_files.append(str(output_file))
            
        else:  # separate files
            # Save each feature as separate file
            output_profile['count'] = 1
            
            for feature_name, feature_array in features.items():
                safe_name = feature_name.replace(" ", "_").replace("/", "_")
                output_file = output_dir / f"{safe_name}_{Path(input_file).stem}.tif"
                
                with rasterio.open(output_file, 'w', **output_profile) as dst:
                    # Handle NaN values
                    output_array = np.where(np.isnan(feature_array), self.nodata_value, feature_array)
                    dst.write(output_array.astype(self.data_type), 1)
                    dst.set_band_description(1, feature_name)
                
                output_files.append(str(output_file))
        
        return output_files
    
    def _combine_results(self, processing_results: List[Dict[str, Any]], context) -> Dict[str, Any]:
        """Combine results from multiple files"""
        processing_start = datetime.now()
        
        # Flatten output files and metadata
        all_output_files = []
        all_metadata = []
        
        for result in processing_results:
            all_output_files.extend(result["output_files"])
            all_metadata.extend(result["texture_metadata"])
        
        # Calculate combined statistics
        combined_stats = self._calculate_texture_statistics(all_metadata)
        
        # Create summary report
        summary_report = self._create_texture_report(all_metadata, combined_stats, context)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        total_processing_time = sum(r["processing_time"] for r in processing_results) + processing_time
        
        return {
            "output_files": all_output_files,
            "texture_metadata": all_metadata,
            "combined_statistics": combined_stats,
            "summary_report": summary_report,
            "files_processed": len(processing_results),
            "total_features": len(all_metadata),
            "processing_time": total_processing_time
        }
    
    def _calculate_texture_statistics(self, all_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate texture feature statistics"""
        stats = {}
        
        # Group by method
        method_groups = {}
        for metadata in all_metadata:
            method = metadata.get("method", "unknown")
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(metadata)
        
        # Calculate statistics for each method
        for method, metadata_list in method_groups.items():
            valid_features = [m for m in metadata_list if "valid_pixels" in m]
            
            if valid_features:
                stats[method] = {
                    "feature_count": len(metadata_list),
                    "total_valid_pixels": sum(m["valid_pixels"] for m in valid_features),
                    "mean_feature_value": np.mean([m["mean_value"] for m in valid_features]),
                    "feature_names": list(set(m["feature_name"] for m in metadata_list))
                }
        
        return stats
    
    def _create_texture_report(self, all_metadata: List[Dict[str, Any]], 
                              combined_stats: Dict[str, Any], context) -> str:
        """Create texture analysis report"""
        try:
            output_dir = context.get_temp_dir() / "texture_features"
            ensure_directory(output_dir)
            
            report_content = f"""
Texture Analysis Report
======================

Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Texture Methods: {', '.join(self.texture_methods)}
Window Size: {self.window_size}
Multi-scale Analysis: {'Yes' if self.multi_scale else 'No'}

Method Summary:
--------------
"""
            
            for method, stats in combined_stats.items():
                report_content += f"""
{method.upper()} Features:
- Feature Count: {stats['feature_count']}
- Total Valid Pixels: {stats['total_valid_pixels']:,}
- Mean Feature Value: {stats['mean_feature_value']:.4f}
- Features: {', '.join(stats['feature_names'][:5])}{'...' if len(stats['feature_names']) > 5 else ''}
"""
            
            # GLCM specific details
            if "glcm" in self.texture_methods:
                report_content += f"""
GLCM Configuration:
- Features: {', '.join(self.glcm_features)}
- Distances: {self.glcm_distances}
- Angles: {self.glcm_angles}°
- Gray Levels: {self.glcm_levels}
"""
            
            # Gabor specific details
            if "gabor" in self.texture_methods:
                report_content += f"""
Gabor Configuration:
- Frequencies: {self.gabor_frequencies}
- Orientations: {self.gabor_orientations}°
- Sigma X/Y: {self.gabor_sigma_x}/{self.gabor_sigma_y}
"""
            
            # Processing summary
            total_features = len(all_metadata)
            successful_features = len([m for m in all_metadata if "valid_pixels" in m])
            success_rate = (successful_features / total_features) * 100 if total_features > 0 else 0
            
            report_content += f"""
Processing Summary:
------------------
- Total Features Calculated: {total_features}
- Successful Features: {successful_features}
- Success Rate: {success_rate:.1f}%
- Target Bands: {', '.join(map(str, self.target_bands))}

Quality Assessment:
------------------
"""
            
            # Quality metrics
            for metadata in all_metadata[:10]:  # Show first 10 features
                if "valid_pixels" in metadata:
                    report_content += f"- {metadata['feature_name']}: {metadata['valid_pixels']:,} valid pixels\n"
            
            if len(all_metadata) > 10:
                report_content += f"... and {len(all_metadata) - 10} more features\n"
            
            # Save report
            report_file = output_dir / f"texture_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            return str(report_file)
            
        except Exception as e:
            self.logger.warning(f"Failed to create texture report: {str(e)}")
            return ""


# Register the step
StepRegistry.register("texture_analysis", TextureAnalysisStep)
