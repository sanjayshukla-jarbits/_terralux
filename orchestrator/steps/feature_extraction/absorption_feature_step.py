# orchestrator/steps/feature_extraction/absorption_feature_step.py
"""
Spectral absorption feature analysis step.

Analyzes spectral absorption features including continuum removal,
absorption depth, position, and mineral identification features.
"""

import numpy as np
import rasterio
from scipy import interpolate, optimize
from scipy.signal import find_peaks, savgol_filter
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from ..base.base_step import BaseStep
from ..base.step_registry import StepRegistry
from ...utils.file_utils import ensure_directory, validate_file_exists


class AbsorptionFeatureStep(BaseStep):
    """
    Universal spectral absorption feature analysis step.
    
    Capabilities:
    - Continuum removal analysis
    - Absorption depth calculation
    - Absorption position detection
    - Absorption asymmetry analysis
    - Mineral-specific absorption features
    - Multi-band spectral analysis
    
    Configuration Examples:
    
    For Mineral Targeting (Hyperspectral):
    {
        "analysis_type": "continuum_removal",
        "target_minerals": ["kaolinite", "muscovite", "alunite"],
        "wavelength_ranges": {
            "clay": [2100, 2300],
            "carbonates": [2300, 2400],
            "iron_oxides": [400, 900]
        },
        "smoothing": true,
        "derivative_analysis": true
    }
    
    For Alteration Mapping:
    {
        "analysis_type": "absorption_depth",
        "absorption_bands": {
            "OH": [2200, 2210, 2220],
            "Al-OH": [2160, 2170, 2180],
            "Fe-OH": [2240, 2250, 2260]
        },
        "continuum_removal": true,
        "asymmetry_analysis": true
    }
    """
    
    def __init__(self, step_id: str, hyperparameters: Dict[str, Any]):
        super().__init__(step_id, "absorption_features", hyperparameters)
        
        # Analysis configuration
        self.analysis_type = hyperparameters.get("analysis_type", "continuum_removal")
        self.target_minerals = hyperparameters.get("target_minerals", [])
        self.wavelength_ranges = hyperparameters.get("wavelength_ranges", {})
        self.absorption_bands = hyperparameters.get("absorption_bands", {})
        
        # Processing options
        self.continuum_removal = hyperparameters.get("continuum_removal", True)
        self.smoothing = hyperparameters.get("smoothing", True)
        self.smoothing_window = hyperparameters.get("smoothing_window", 5)
        self.smoothing_order = hyperparameters.get("smoothing_order", 2)
        
        # Analysis options
        self.derivative_analysis = hyperparameters.get("derivative_analysis", False)
        self.asymmetry_analysis = hyperparameters.get("asymmetry_analysis", False)
        self.depth_normalization = hyperparameters.get("depth_normalization", True)
        
        # Feature extraction
        self.extract_depth = hyperparameters.get("extract_depth", True)
        self.extract_position = hyperparameters.get("extract_position", True)
        self.extract_width = hyperparameters.get("extract_width", False)
        self.extract_asymmetry = hyperparameters.get("extract_asymmetry", False)
        
        # Band/wavelength configuration
        self.band_centers = hyperparameters.get("band_centers", None)  # Wavelengths in nm
        self.band_widths = hyperparameters.get("band_widths", None)
        self.sensor_type = hyperparameters.get("sensor_type", "multispectral")
        
        # Output options
        self.output_format = hyperparameters.get("output_format", "multiband")
        self.data_type = hyperparameters.get("data_type", "float32")
        self.nodata_value = hyperparameters.get("nodata_value", -9999)
        self.compression = hyperparameters.get("compression", "lzw")
        
        # Quality control
        self.min_valid_bands = hyperparameters.get("min_valid_bands", 3)
        self.outlier_threshold = hyperparameters.get("outlier_threshold", 3.0)
        
        # Predefined mineral absorption features
        self.mineral_features = self._get_mineral_absorption_features()
        
        self.logger = logging.getLogger(f"AbsorptionFeatures.{step_id}")
    
    def execute(self, context) -> Dict[str, Any]:
        """Execute absorption feature analysis"""
        try:
            self.logger.info(f"Starting absorption feature analysis: {self.analysis_type}")
            
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
            context.add_data(f"{self.step_id}_absorption_features", combined_results["output_files"])
            context.add_data(f"{self.step_id}_absorption_metadata", combined_results["absorption_metadata"])
            
            self.logger.info("Absorption feature analysis completed successfully")
            return {
                "status": "success",
                "outputs": combined_results,
                "metadata": {
                    "analysis_type": self.analysis_type,
                    "features_extracted": combined_results.get("total_features", 0),
                    "processing_time": combined_results.get("processing_time")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Absorption feature analysis failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _get_input_files(self, context) -> List[str]:
        """Get input files from context"""
        # Try different possible data keys
        for key in ["corrected_images", "spectral_data", "hyperspectral_data", "resampled_images"]:
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
        
        raise ValueError("No input files found for absorption feature analysis")
    
    def _validate_inputs(self, input_files: List[str]):
        """Validate input files"""
        if not input_files:
            raise ValueError("No input files provided for absorption feature analysis")
        
        for file_path in input_files:
            if not validate_file_exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
            
            # Check band count
            try:
                with rasterio.open(file_path) as src:
                    if src.count < self.min_valid_bands:
                        raise ValueError(f"Insufficient bands in {file_path}: {src.count} < {self.min_valid_bands}")
            except Exception as e:
                raise ValueError(f"Cannot read input file {file_path}: {str(e)}")
    
    def _get_mineral_absorption_features(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined mineral absorption features"""
        return {
            "kaolinite": {
                "absorption_centers": [2206, 2165],  # nm
                "absorption_ranges": [(2190, 2220), (2150, 2180)],
                "description": "Al-OH absorption features"
            },
            "muscovite": {
                "absorption_centers": [2200, 2345],
                "absorption_ranges": [(2180, 2220), (2320, 2360)],
                "description": "Al-OH and OH-stretch features"
            },
            "alunite": {
                "absorption_centers": [2165, 2315],
                "absorption_ranges": [(2150, 2180), (2300, 2330)],
                "description": "Al-OH and sulfate features"
            },
            "calcite": {
                "absorption_centers": [2340],
                "absorption_ranges": [(2320, 2360)],
                "description": "Carbonate absorption"
            },
            "dolomite": {
                "absorption_centers": [2325],
                "absorption_ranges": [(2310, 2340)],
                "description": "Mg-carbonate absorption"
            },
            "goethite": {
                "absorption_centers": [480, 650, 900],
                "absorption_ranges": [(460, 500), (630, 670), (880, 920)],
                "description": "Fe3+ crystal field absorptions"
            },
            "hematite": {
                "absorption_centers": [565, 860],
                "absorption_ranges": [(550, 580), (840, 880)],
                "description": "Fe3+ absorptions"
            },
            "jarosite": {
                "absorption_centers": [430, 2265],
                "absorption_ranges": [(410, 450), (2250, 2280)],
                "description": "Fe3+ and OH features"
            },
            "chlorite": {
                "absorption_centers": [2255, 2330],
                "absorption_ranges": [(2240, 2270), (2315, 2345)],
                "description": "Mg-OH and Fe-OH features"
            },
            "epidote": {
                "absorption_centers": [2338],
                "absorption_ranges": [(2320, 2356)],
                "description": "OH absorption"
            }
        }
    
    def _process_single_file(self, input_file: str, context) -> Dict[str, Any]:
        """Process a single input file"""
        self.logger.info(f"Processing absorption features for {Path(input_file).name}")
        processing_start = datetime.now()
        
        with rasterio.open(input_file) as src:
            # Read spectral data
            spectral_data = src.read().astype(np.float32)
            profile = src.profile.copy()
            
            # Handle nodata values
            if src.nodata is not None:
                spectral_data = np.where(spectral_data == src.nodata, np.nan, spectral_data)
            
            # Get wavelengths/band centers
            wavelengths = self._get_wavelengths(src)
            
            # Calculate absorption features
            calculated_features = {}
            absorption_metadata = []
            
            if self.analysis_type == "continuum_removal":
                features = self._perform_continuum_removal_analysis(spectral_data, wavelengths)
                calculated_features.update(features)
                
            elif self.analysis_type == "absorption_depth":
                features = self._calculate_absorption_depths(spectral_data, wavelengths)
                calculated_features.update(features)
                
            elif self.analysis_type == "mineral_mapping":
                features = self._perform_mineral_mapping(spectral_data, wavelengths)
                calculated_features.update(features)
                
            elif self.analysis_type == "derivative_analysis":
                features = self._perform_derivative_analysis(spectral_data, wavelengths)
                calculated_features.update(features)
            
            # Store metadata for each feature
            for feature_name, feature_array in calculated_features.items():
                absorption_metadata.append({
                    "feature_name": feature_name,
                    "analysis_type": self.analysis_type,
                    "valid_pixels": int(np.sum(~np.isnan(feature_array))),
                    "mean_value": float(np.nanmean(feature_array)),
                    "std_value": float(np.nanstd(feature_array)),
                    "min_value": float(np.nanmin(feature_array)),
                    "max_value": float(np.nanmax(feature_array))
                })
            
            # Save results
            output_files = self._save_absorption_features(calculated_features, profile, input_file, context)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        return {
            "input_file": input_file,
            "output_files": output_files,
            "calculated_features": calculated_features,
            "absorption_metadata": absorption_metadata,
            "processing_time": processing_time,
            "wavelengths": wavelengths
        }
    
    def _get_wavelengths(self, src) -> np.ndarray:
        """Get wavelengths for bands"""
        if self.band_centers is not None:
            return np.array(self.band_centers)
        
        # Try to get from metadata
        if hasattr(src, 'tags') and src.tags():
            tags = src.tags()
            if 'wavelengths' in tags:
                try:
                    wavelengths_str = tags['wavelengths']
                    wavelengths = [float(w.strip()) for w in wavelengths_str.split(',')]
                    return np.array(wavelengths)
                except:
                    pass
        
        # Default wavelengths for common sensors
        if self.sensor_type == "sentinel2":
            # Sentinel-2 band centers (nm)
            return np.array([443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190])
        elif self.sensor_type == "landsat8":
            # Landsat 8 band centers (nm)  
            return np.array([443, 482, 561, 655, 865, 1609, 2201])
        else:
            # Generic wavelengths
            return np.linspace(400, 2500, src.count)
    
    def _perform_continuum_removal_analysis(self, spectral_data: np.ndarray, 
                                           wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform continuum removal analysis"""
        features = {}
        
        rows, cols = spectral_data.shape[1], spectral_data.shape[2]
        n_bands = spectral_data.shape[0]
        
        # Process each pixel
        for target_range_name, wavelength_range in self.wavelength_ranges.items():
            self.logger.info(f"Processing continuum removal for {target_range_name}")
            
            # Find bands within wavelength range
            range_mask = (wavelengths >= wavelength_range[0]) & (wavelengths <= wavelength_range[1])
            range_bands = np.where(range_mask)[0]
            
            if len(range_bands) < 3:
                self.logger.warning(f"Insufficient bands for {target_range_name}: {len(range_bands)}")
                continue
            
            # Initialize output arrays
            continuum_removed = np.full((rows, cols), np.nan, dtype=np.float32)
            absorption_depth = np.full((rows, cols), np.nan, dtype=np.float32)
            absorption_position = np.full((rows, cols), np.nan, dtype=np.float32)
            
            # Process in chunks to manage memory
            chunk_size = 1000
            for i in range(0, rows, chunk_size):
                end_i = min(i + chunk_size, rows)
                
                for j in range(0, cols, chunk_size):
                    end_j = min(j + chunk_size, cols)
                    
                    # Extract chunk
                    chunk = spectral_data[:, i:end_i, j:end_j]
                    
                    # Process each pixel in chunk
                    for r in range(chunk.shape[1]):
                        for c in range(chunk.shape[2]):
                            spectrum = chunk[:, r, c]
                            
                            # Skip invalid pixels
                            if np.any(np.isnan(spectrum)) or np.all(spectrum <= 0):
                                continue
                            
                            # Extract wavelength range
                            range_spectrum = spectrum[range_bands]
                            range_wavelengths = wavelengths[range_bands]
                            
                            # Smooth spectrum if requested
                            if self.smoothing and len(range_spectrum) >= self.smoothing_window:
                                try:
                                    range_spectrum = savgol_filter(
                                        range_spectrum, self.smoothing_window, self.smoothing_order
                                    )
                                except:
                                    pass
                            
                            # Calculate continuum
                            try:
                                continuum = self._calculate_continuum_line(range_wavelengths, range_spectrum)
                                
                                # Continuum removal
                                if continuum is not None and np.all(continuum > 0):
                                    cr_spectrum = range_spectrum / continuum
                                    
                                    # Find absorption minimum
                                    min_idx = np.argmin(cr_spectrum)
                                    absorption_depth[i+r, j+c] = 1 - cr_spectrum[min_idx]
                                    absorption_position[i+r, j+c] = range_wavelengths[min_idx]
                                    
                                    # Store continuum removed spectrum (use mean for single value)
                                    continuum_removed[i+r, j+c] = np.mean(cr_spectrum)
                                    
                            except Exception as e:
                                continue
            
            # Store features
            features[f"{target_range_name}_continuum_removed"] = continuum_removed
            features[f"{target_range_name}_absorption_depth"] = absorption_depth
            features[f"{target_range_name}_absorption_position"] = absorption_position
        
        return features
    
    def _calculate_continuum_line(self, wavelengths: np.ndarray, spectrum: np.ndarray) -> Optional[np.ndarray]:
        """Calculate continuum line for spectrum"""
        try:
            # Find hull points (convex hull of spectrum)
            from scipy.spatial import ConvexHull
            
            # Create points for convex hull
            points = np.column_stack([wavelengths, spectrum])
            
            # Find convex hull
            hull = ConvexHull(points)
            
            # Get upper hull points
            hull_points = points[hull.vertices]
            
            # Sort by wavelength
            sort_idx = np.argsort(hull_points[:, 0])
            hull_points = hull_points[sort_idx]
            
            # Find upper envelope
            upper_hull = []
            for i in range(len(hull_points)):
                point = hull_points[i]
                # Check if point is on upper envelope
                is_upper = True
                for j in range(len(hull_points)):
                    if i != j:
                        other_point = hull_points[j]
                        if (other_point[0] < point[0] and other_point[1] > point[1]) or \
                           (other_point[0] > point[0] and other_point[1] > point[1]):
                            is_upper = False
                            break
                if is_upper:
                    upper_hull.append(point)
            
            upper_hull = np.array(upper_hull)
            
            if len(upper_hull) < 2:
                # Fallback: linear interpolation between endpoints
                continuum = np.linspace(spectrum[0], spectrum[-1], len(spectrum))
            else:
                # Interpolate continuum line
                continuum_interp = interpolate.interp1d(
                    upper_hull[:, 0], upper_hull[:, 1], 
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
                continuum = continuum_interp(wavelengths)
            
            return continuum
            
        except Exception as e:
            # Fallback: linear continuum
            continuum = np.linspace(spectrum[0], spectrum[-1], len(spectrum))
            return continuum
    
    def _calculate_absorption_depths(self, spectral_data: np.ndarray, 
                                   wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate absorption depths for specific bands"""
        features = {}
        
        rows, cols = spectral_data.shape[1], spectral_data.shape[2]
        
        for absorption_name, band_triplet in self.absorption_bands.items():
            self.logger.info(f"Calculating absorption depth for {absorption_name}")
            
            if len(band_triplet) != 3:
                self.logger.warning(f"Absorption band {absorption_name} must have 3 wavelengths")
                continue
            
            # Find closest bands
            left_idx = np.argmin(np.abs(wavelengths - band_triplet[0]))
            center_idx = np.argmin(np.abs(wavelengths - band_triplet[1]))
            right_idx = np.argmin(np.abs(wavelengths - band_triplet[2]))
            
            # Calculate absorption depth
            left_band = spectral_data[left_idx]
            center_band = spectral_data[center_idx]
            right_band = spectral_data[right_idx]
            
            # Linear interpolation for continuum
            continuum_value = (left_band + right_band) / 2.0
            
            # Absorption depth = 1 - (center / continuum)
            with np.errstate(divide='ignore', invalid='ignore'):
                absorption_depth = 1 - (center_band / continuum_value)
                absorption_depth = np.where(
                    (continuum_value > 0) & np.isfinite(absorption_depth),
                    absorption_depth, np.nan
                )
            
            features[f"{absorption_name}_absorption_depth"] = absorption_depth.astype(np.float32)
        
        return features
    
    def _perform_mineral_mapping(self, spectral_data: np.ndarray, 
                               wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform mineral-specific absorption feature mapping"""
        features = {}
        
        for mineral_name in self.target_minerals:
            if mineral_name not in self.mineral_features:
                self.logger.warning(f"Unknown mineral: {mineral_name}")
                continue
            
            self.logger.info(f"Mapping {mineral_name} features")
            
            mineral_info = self.mineral_features[mineral_name]
            absorption_centers = mineral_info["absorption_centers"]
            absorption_ranges = mineral_info["absorption_ranges"]
            
            # Calculate mineral index based on absorption features
            mineral_indices = []
            
            for i, (center, range_bounds) in enumerate(zip(absorption_centers, absorption_ranges)):
                # Find bands for this absorption feature
                center_idx = np.argmin(np.abs(wavelengths - center))
                range_mask = (wavelengths >= range_bounds[0]) & (wavelengths <= range_bounds[1])
                range_indices = np.where(range_mask)[0]
                
                if len(range_indices) < 3:
                    continue
                
                # Calculate absorption feature strength
                center_band = spectral_data[center_idx]
                
                # Find continuum bands (edges of absorption range)
                left_idx = range_indices[0]
                right_idx = range_indices[-1]
                
                left_band = spectral_data[left_idx]
                right_band = spectral_data[right_idx]
                
                # Linear continuum
                continuum_value = (left_band + right_band) / 2.0
                
                # Absorption depth
                with np.errstate(divide='ignore', invalid='ignore'):
                    absorption_feature = 1 - (center_band / continuum_value)
                    absorption_feature = np.where(
                        (continuum_value > 0) & np.isfinite(absorption_feature),
                        absorption_feature, 0
                    )
                
                mineral_indices.append(absorption_feature)
            
            # Combine multiple absorption features
            if mineral_indices:
                combined_index = np.mean(mineral_indices, axis=0)
                features[f"{mineral_name}_index"] = combined_index.astype(np.float32)
        
        return features
    
    def _perform_derivative_analysis(self, spectral_data: np.ndarray, 
                                   wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform spectral derivative analysis"""
        features = {}
        
        rows, cols = spectral_data.shape[1], spectral_data.shape[2]
        n_bands = spectral_data.shape[0]
        
        # Calculate first derivative
        first_derivative = np.zeros_like(spectral_data)
        for i in range(1, n_bands - 1):
            first_derivative[i] = (spectral_data[i+1] - spectral_data[i-1]) / \
                                 (wavelengths[i+1] - wavelengths[i-1])
        
        # Calculate second derivative
        second_derivative = np.zeros_like(spectral_data)
        for i in range(1, n_bands - 1):
            second_derivative[i] = (first_derivative[i+1] - first_derivative[i-1]) / \
                                  (wavelengths[i+1] - wavelengths[i-1])
        
        # Extract derivative features for specific wavelength ranges
        for range_name, wavelength_range in self.wavelength_ranges.items():
            range_mask = (wavelengths >= wavelength_range[0]) & (wavelengths <= wavelength_range[1])
            range_bands = np.where(range_mask)[0]
            
            if len(range_bands) == 0:
                continue
            
            # First derivative features
            range_first_deriv = first_derivative[range_bands]
            first_deriv_min = np.min(range_first_deriv, axis=0)
            first_deriv_max = np.max(range_first_deriv, axis=0)
            
            features[f"{range_name}_first_deriv_min"] = first_deriv_min.astype(np.float32)
            features[f"{range_name}_first_deriv_max"] = first_deriv_max.astype(np.float32)
            
            # Second derivative features
            range_second_deriv = second_derivative[range_bands]
            second_deriv_min = np.min(range_second_deriv, axis=0)
            second_deriv_max = np.max(range_second_deriv, axis=0)
            
            features[f"{range_name}_second_deriv_min"] = second_deriv_min.astype(np.float32)
            features[f"{range_name}_second_deriv_max"] = second_deriv_max.astype(np.float32)
        
        return features
    
    def _save_absorption_features(self, features: Dict[str, np.ndarray], profile: dict,
                                 input_file: str, context) -> List[str]:
        """Save calculated absorption features to files"""
        output_dir = context.get_temp_dir() / "absorption_features"
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
            
            output_file = output_dir / f"absorption_features_{Path(input_file).stem}.tif"
            
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
            all_metadata.extend(result["absorption_metadata"])
        
        # Calculate combined statistics
        combined_stats = self._calculate_absorption_statistics(all_metadata)
        
        # Create summary report
        summary_report = self._create_absorption_report(all_metadata, combined_stats, context)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        total_processing_time = sum(r["processing_time"] for r in processing_results) + processing_time
        
        return {
            "output_files": all_output_files,
            "absorption_metadata": all_metadata,
            "combined_statistics": combined_stats,
            "summary_report": summary_report,
            "files_processed": len(processing_results),
            "total_features": len(all_metadata),
            "processing_time": total_processing_time
        }
    
    def _calculate_absorption_statistics(self, all_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate absorption feature statistics"""
        stats = {}
        
        # Group by analysis type
        analysis_groups = {}
        for metadata in all_metadata:
            analysis_type = metadata.get("analysis_type", "unknown")
            if analysis_type not in analysis_groups:
                analysis_groups[analysis_type] = []
            analysis_groups[analysis_type].append(metadata)
        
        # Calculate statistics for each analysis type
        for analysis_type, metadata_list in analysis_groups.items():
            valid_features = [m for m in metadata_list if "valid_pixels" in m]
            
            if valid_features:
                stats[analysis_type] = {
                    "feature_count": len(metadata_list),
                    "total_valid_pixels": sum(m["valid_pixels"] for m in valid_features),
                    "mean_feature_value": np.mean([m["mean_value"] for m in valid_features]),
                    "feature_names": [m["feature_name"] for m in metadata_list]
                }
        
        return stats
    
    def _create_absorption_report(self, all_metadata: List[Dict[str, Any]], 
                                 combined_stats: Dict[str, Any], context) -> str:
        """Create absorption analysis report"""
        try:
            output_dir = context.get_temp_dir() / "absorption_features"
            ensure_directory(output_dir)
            
            report_content = f"""
Absorption Feature Analysis Report
=================================

Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Type: {self.analysis_type}
Sensor Type: {self.sensor_type}
Continuum Removal: {'Yes' if self.continuum_removal else 'No'}
Smoothing: {'Yes' if self.smoothing else 'No'}

Analysis Summary:
----------------
"""
            
            for analysis_type, stats in combined_stats.items():
                report_content += f"""
{analysis_type.upper()}:
- Feature Count: {stats['feature_count']}
- Total Valid Pixels: {stats['total_valid_pixels']:,}
- Mean Feature Value: {stats['mean_feature_value']:.4f}
- Features: {', '.join(stats['feature_names'][:5])}{'...' if len(stats['feature_names']) > 5 else ''}
"""
            
            # Target minerals if specified
            if self.target_minerals:
                report_content += f"""
Target Minerals:
- {', '.join(self.target_minerals)}
"""
            
            # Wavelength ranges if specified
            if self.wavelength_ranges:
                report_content += f"""
Wavelength Ranges:
"""
                for range_name, wavelength_range in self.wavelength_ranges.items():
                    report_content += f"- {range_name}: {wavelength_range[0]}-{wavelength_range[1]} nm\n"
            
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

Feature Details:
---------------
"""
            
            # Show details for first few features
            for metadata in all_metadata[:10]:
                if "valid_pixels" in metadata:
                    report_content += f"""- {metadata['feature_name']}:
  - Valid pixels: {metadata['valid_pixels']:,}
  - Mean: {metadata['mean_value']:.4f}
  - Range: {metadata['min_value']:.4f} - {metadata['max_value']:.4f}
"""
            
            if len(all_metadata) > 10:
                report_content += f"... and {len(all_metadata) - 10} more features\n"
            
            # Save report
            report_file = output_dir / f"absorption_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            return str(report_file)
            
        except Exception as e:
            self.logger.warning(f"Failed to create absorption report: {str(e)}")
            return ""


# Register the step
StepRegistry.register("absorption_features", AbsorptionFeatureStep)
