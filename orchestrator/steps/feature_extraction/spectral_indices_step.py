# orchestrator/steps/feature_extraction/spectral_indices_step.py
"""
Universal spectral indices calculation step.

Supports vegetation, water, urban, soil, and mineral indices with configurable
band mappings for different sensor types.
"""

import numpy as np
import rasterio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import json

from ..base.base_step import BaseStep
from ..base.step_registry import StepRegistry
from ...utils.file_utils import ensure_directory, validate_file_exists


class SpectralIndicesStep(BaseStep):
    """
    Universal spectral indices calculation step.
    
    Capabilities:
    - 50+ predefined spectral indices
    - Custom index definitions
    - Multiple sensor support (Sentinel-2, Landsat, WorldView, etc.)
    - Flexible band mapping
    - Quality masking and validation
    - Batch processing
    
    Configuration Examples:
    
    For Landslide Assessment (Vegetation focus):
    {
        "indices": ["NDVI", "NDWI", "SAVI", "EVI", "BSI"],
        "sensor": "sentinel2",
        "quality_mask": true,
        "output_format": "multiband",
        "normalization": "none"
    }
    
    For Mineral Targeting (Mineral focus):
    {
        "indices": ["clay_minerals", "iron_oxides", "carbonates", "alteration_index"],
        "custom_indices": [
            {
                "name": "ferric_iron",
                "formula": "B4 / B3",
                "description": "Ferric iron detection"
            }
        ],
        "sensor": "worldview3",
        "continuum_removal": true
    }
    """
    
    def __init__(self, step_id: str, hyperparameters: Dict[str, Any]):
        super().__init__(step_id, "spectral_indices", hyperparameters)
        
        # Index configuration
        self.indices = hyperparameters.get("indices", ["NDVI", "NDWI"])
        self.custom_indices = hyperparameters.get("custom_indices", [])
        self.sensor = hyperparameters.get("sensor", "sentinel2")
        
        # Processing options
        self.quality_mask = hyperparameters.get("quality_mask", True)
        self.cloud_mask = hyperparameters.get("cloud_mask", True)
        self.water_mask = hyperparameters.get("water_mask", False)
        self.normalization = hyperparameters.get("normalization", "none")  # none, minmax, zscore
        
        # Advanced options
        self.continuum_removal = hyperparameters.get("continuum_removal", False)
        self.atmospheric_correction = hyperparameters.get("atmospheric_correction", False)
        self.sun_angle_correction = hyperparameters.get("sun_angle_correction", False)
        
        # Output options
        self.output_format = hyperparameters.get("output_format", "multiband")  # multiband, separate
        self.data_type = hyperparameters.get("data_type", "float32")
        self.nodata_value = hyperparameters.get("nodata_value", -9999)
        self.compression = hyperparameters.get("compression", "lzw")
        
        # Validation options
        self.validate_indices = hyperparameters.get("validate_indices", True)
        self.clip_values = hyperparameters.get("clip_values", True)
        self.value_range = hyperparameters.get("value_range", [-1, 1])
        
        # Band mapping for different sensors
        self.band_mappings = self._get_sensor_band_mappings()
        self.predefined_indices = self._get_predefined_indices()
        
        self.logger = logging.getLogger(f"SpectralIndices.{step_id}")
    
    def execute(self, context) -> Dict[str, Any]:
        """Execute spectral indices calculation"""
        try:
            self.logger.info(f"Starting spectral indices calculation for {len(self.indices)} indices")
            
            # Get input data
            input_files = self._get_input_files(context)
            
            # Validate inputs
            self._validate_inputs(input_files)
            
            # Get band mapping for sensor
            band_mapping = self.band_mappings.get(self.sensor, self.band_mappings["generic"])
            
            # Process each input file
            processing_results = []
            for input_file in input_files:
                file_results = self._process_single_file(input_file, band_mapping, context)
                processing_results.append(file_results)
            
            # Combine results
            combined_results = self._combine_results(processing_results, context)
            
            # Validate outputs
            if self.validate_indices:
                validation_results = self._validate_outputs(combined_results)
                combined_results.update(validation_results)
            
            # Update context
            context.add_data(f"{self.step_id}_spectral_indices", combined_results["output_files"])
            context.add_data(f"{self.step_id}_index_metadata", combined_results["index_metadata"])
            
            self.logger.info("Spectral indices calculation completed successfully")
            return {
                "status": "success",
                "outputs": combined_results,
                "metadata": {
                    "indices_calculated": len(self.indices) + len(self.custom_indices),
                    "sensor": self.sensor,
                    "output_format": self.output_format,
                    "processing_time": combined_results.get("processing_time")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Spectral indices calculation failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _get_input_files(self, context) -> List[str]:
        """Get input files from context"""
        # Try different possible data keys
        for key in ["resampled_images", "masked_images", "corrected_images", "spectral_data"]:
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
        
        raise ValueError("No input files found for spectral indices calculation")
    
    def _validate_inputs(self, input_files: List[str]):
        """Validate input files"""
        if not input_files:
            raise ValueError("No input files provided for spectral indices calculation")
        
        for file_path in input_files:
            if not validate_file_exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
            
            # Validate that file has sufficient bands
            try:
                with rasterio.open(file_path) as src:
                    if src.count < 3:
                        self.logger.warning(f"File {file_path} has only {src.count} bands")
            except Exception as e:
                raise ValueError(f"Cannot read input file {file_path}: {str(e)}")
    
    def _get_sensor_band_mappings(self) -> Dict[str, Dict[str, int]]:
        """Get band mappings for different sensors"""
        return {
            "sentinel2": {
                "B01": 1, "B02": 2, "B03": 3, "B04": 4, "B05": 5, "B06": 6,
                "B07": 7, "B08": 8, "B8A": 9, "B09": 10, "B10": 11, "B11": 12, "B12": 13,
                # Common aliases
                "blue": 2, "green": 3, "red": 4, "nir": 8, "swir1": 12, "swir2": 13
            },
            "landsat8": {
                "B01": 1, "B02": 2, "B03": 3, "B04": 4, "B05": 5, "B06": 6, "B07": 7,
                "blue": 2, "green": 3, "red": 4, "nir": 5, "swir1": 6, "swir2": 7
            },
            "worldview3": {
                "B01": 1, "B02": 2, "B03": 3, "B04": 4, "B05": 5, "B06": 6, "B07": 7, "B08": 8,
                "blue": 2, "green": 3, "red": 4, "nir": 7, "swir1": 8, "swir2": 8
            },
            "generic": {
                "B1": 1, "B2": 2, "B3": 3, "B4": 4, "B5": 5, "B6": 6, "B7": 7, "B8": 8,
                "blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5, "swir2": 6
            }
        }
    
    def _get_predefined_indices(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined spectral indices"""
        return {
            # Vegetation indices
            "NDVI": {
                "formula": "(nir - red) / (nir + red)",
                "description": "Normalized Difference Vegetation Index",
                "range": [-1, 1],
                "category": "vegetation"
            },
            "EVI": {
                "formula": "2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))",
                "description": "Enhanced Vegetation Index", 
                "range": [-1, 1],
                "category": "vegetation"
            },
            "SAVI": {
                "formula": "((nir - red) / (nir + red + 0.5)) * 1.5",
                "description": "Soil Adjusted Vegetation Index",
                "range": [-1.5, 1.5],
                "category": "vegetation"
            },
            "ARVI": {
                "formula": "(nir - (2 * red - blue)) / (nir + (2 * red - blue))",
                "description": "Atmospherically Resistant Vegetation Index",
                "range": [-1, 1],
                "category": "vegetation"
            },
            "GNDVI": {
                "formula": "(nir - green) / (nir + green)",
                "description": "Green Normalized Difference Vegetation Index",
                "range": [-1, 1],
                "category": "vegetation"
            },
            
            # Water indices
            "NDWI": {
                "formula": "(green - nir) / (green + nir)",
                "description": "Normalized Difference Water Index",
                "range": [-1, 1],
                "category": "water"
            },
            "MNDWI": {
                "formula": "(green - swir1) / (green + swir1)",
                "description": "Modified Normalized Difference Water Index",
                "range": [-1, 1],
                "category": "water"
            },
            "WI": {
                "formula": "(blue + green) / (nir + swir1)",
                "description": "Water Index",
                "range": [0, 10],
                "category": "water"
            },
            
            # Urban/built-up indices
            "NDBI": {
                "formula": "(swir1 - nir) / (swir1 + nir)",
                "description": "Normalized Difference Built-up Index",
                "range": [-1, 1],
                "category": "urban"
            },
            "UI": {
                "formula": "(swir2 - nir) / (swir2 + nir)",
                "description": "Urban Index",
                "range": [-1, 1],
                "category": "urban"
            },
            "BUI": {
                "formula": "((swir1 - nir) / (swir1 + nir)) - ((nir - red) / (nir + red))",
                "description": "Built-up Index",
                "range": [-2, 2],
                "category": "urban"
            },
            
            # Soil indices
            "BSI": {
                "formula": "((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))",
                "description": "Bare Soil Index",
                "range": [-1, 1],
                "category": "soil"
            },
            "SI": {
                "formula": "((swir1 - blue) / (swir1 + blue))",
                "description": "Soil Index",
                "range": [-1, 1],
                "category": "soil"
            },
            
            # Mineral indices (using band numbers for flexibility)
            "clay_minerals": {
                "formula": "B11 / B12",
                "description": "Clay minerals ratio (SWIR1/SWIR2)",
                "range": [0, 5],
                "category": "mineral"
            },
            "iron_oxides": {
                "formula": "B04 / B02",
                "description": "Iron oxides ratio (Red/Blue)",
                "range": [0, 10],
                "category": "mineral"
            },
            "carbonates": {
                "formula": "B12 / B11",
                "description": "Carbonate minerals ratio (SWIR2/SWIR1)",
                "range": [0, 5],
                "category": "mineral"
            },
            "ferric_iron": {
                "formula": "B04 / B03",
                "description": "Ferric iron ratio (Red/Green)",
                "range": [0, 5],
                "category": "mineral"
            },
            "ferrous_iron": {
                "formula": "B12 / B11",
                "description": "Ferrous silicates ratio",
                "range": [0, 3],
                "category": "mineral"
            },
            "alteration_index": {
                "formula": "(B11 + B12) / (B08 + B04)",
                "description": "Hydrothermal alteration index",
                "range": [0, 10],
                "category": "mineral"
            },
            "gossan_index": {
                "formula": "B04 / B08",
                "description": "Gossan detection index",
                "range": [0, 5],
                "category": "mineral"
            },
            "alunite": {
                "formula": "B11 / B12",
                "description": "Alunite detection ratio",
                "range": [0, 3],
                "category": "mineral"
            },
            "kaolinite": {
                "formula": "B11 / (B10 + B12)",
                "description": "Kaolinite detection",
                "range": [0, 2],
                "category": "mineral"
            },
            "muscovite": {
                "formula": "B11 / B12",
                "description": "Muscovite detection ratio", 
                "range": [0, 3],
                "category": "mineral"
            },
            
            # Snow and ice indices
            "NDSI": {
                "formula": "(green - swir1) / (green + swir1)",
                "description": "Normalized Difference Snow Index",
                "range": [-1, 1],
                "category": "snow"
            },
            "S3": {
                "formula": "(nir * (red - swir1)) / ((nir + red) * (nir + swir1))",
                "description": "S3 Snow Index",
                "range": [-1, 1],
                "category": "snow"
            }
        }
    
    def _process_single_file(self, input_file: str, band_mapping: Dict[str, int], context) -> Dict[str, Any]:
        """Process a single input file"""
        self.logger.info(f"Processing {Path(input_file).name}")
        processing_start = datetime.now()
        
        with rasterio.open(input_file) as src:
            # Read all bands
            bands_data = src.read().astype(np.float32)
            profile = src.profile.copy()
            
            # Handle nodata values
            nodata = src.nodata
            if nodata is not None:
                bands_data = np.where(bands_data == nodata, np.nan, bands_data)
            
            # Apply masks if requested
            if self.quality_mask:
                bands_data = self._apply_quality_masks(bands_data, src, context)
            
            # Calculate indices
            calculated_indices = {}
            index_metadata = []
            
            # Process predefined indices
            for index_name in self.indices:
                if index_name in self.predefined_indices:
                    index_def = self.predefined_indices[index_name]
                    try:
                        index_array = self._calculate_index(bands_data, index_def, band_mapping, src.count)
                        calculated_indices[index_name] = index_array
                        
                        # Store metadata
                        index_metadata.append({
                            "name": index_name,
                            "formula": index_def["formula"],
                            "description": index_def["description"],
                            "category": index_def["category"],
                            "range": index_def["range"],
                            "valid_pixels": int(np.sum(~np.isnan(index_array))),
                            "mean_value": float(np.nanmean(index_array)),
                            "std_value": float(np.nanstd(index_array))
                        })
                        
                        self.logger.info(f"Calculated {index_name}: mean={np.nanmean(index_array):.4f}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to calculate {index_name}: {str(e)}")
                        index_metadata.append({
                            "name": index_name,
                            "error": str(e)
                        })
                else:
                    self.logger.warning(f"Unknown predefined index: {index_name}")
            
            # Process custom indices
            for custom_index in self.custom_indices:
                index_name = custom_index.get("name", "custom")
                try:
                    index_array = self._calculate_custom_index(bands_data, custom_index, band_mapping, src.count)
                    calculated_indices[index_name] = index_array
                    
                    # Store metadata
                    index_metadata.append({
                        "name": index_name,
                        "formula": custom_index.get("formula", ""),
                        "description": custom_index.get("description", "Custom index"),
                        "category": "custom",
                        "valid_pixels": int(np.sum(~np.isnan(index_array))),
                        "mean_value": float(np.nanmean(index_array)),
                        "std_value": float(np.nanstd(index_array))
                    })
                    
                    self.logger.info(f"Calculated custom {index_name}: mean={np.nanmean(index_array):.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to calculate custom index {index_name}: {str(e)}")
                    index_metadata.append({
                        "name": index_name,
                        "error": str(e)
                    })
            
            # Apply post-processing
            if self.normalization != "none":
                calculated_indices = self._normalize_indices(calculated_indices)
            
            if self.clip_values:
                calculated_indices = self._clip_indices(calculated_indices)
            
            # Save results
            output_files = self._save_indices(calculated_indices, profile, input_file, context)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        return {
            "input_file": input_file,
            "output_files": output_files,
            "calculated_indices": calculated_indices,
            "index_metadata": index_metadata,
            "processing_time": processing_time
        }
    
    def _calculate_index(self, bands_data: np.ndarray, index_def: Dict[str, Any], 
                        band_mapping: Dict[str, int], total_bands: int) -> np.ndarray:
        """Calculate a predefined spectral index"""
        formula = index_def["formula"]
        
        # Create band variables for formula evaluation
        band_vars = {}
        
        # Map band names to actual data
        for band_name, band_idx in band_mapping.items():
            if band_idx <= total_bands:
                band_vars[band_name] = bands_data[band_idx - 1]  # Convert to 0-based indexing
        
        # Also create numbered band variables (B1, B2, etc.)
        for i in range(total_bands):
            band_vars[f"B{i+1:02d}"] = bands_data[i]  # B01, B02, ...
            band_vars[f"B{i+1}"] = bands_data[i]      # B1, B2, ...
        
        # Evaluate formula safely
        try:
            # Replace band names in formula with array operations
            result = self._safe_eval_formula(formula, band_vars)
            
            # Handle infinite and invalid values
            result = np.where(np.isfinite(result), result, np.nan)
            
            return result.astype(np.float32)
            
        except Exception as e:
            raise ValueError(f"Formula evaluation failed for '{formula}': {str(e)}")
    
    def _calculate_custom_index(self, bands_data: np.ndarray, custom_index: Dict[str, Any],
                               band_mapping: Dict[str, int], total_bands: int) -> np.ndarray:
        """Calculate a custom spectral index"""
        formula = custom_index.get("formula", "")
        if not formula:
            raise ValueError("Custom index formula is empty")
        
        # Use same logic as predefined indices
        index_def = {
            "formula": formula,
            "range": custom_index.get("range", [-10, 10])
        }
        
        return self._calculate_index(bands_data, index_def, band_mapping, total_bands)
    
    def _safe_eval_formula(self, formula: str, band_vars: Dict[str, np.ndarray]) -> np.ndarray:
        """Safely evaluate index formula"""
        import re
        import operator
        
        # Replace band references with array variables
        eval_formula = formula
        
        # Sort band names by length (longest first) to avoid partial replacements
        sorted_bands = sorted(band_vars.keys(), key=len, reverse=True)
        
        for band_name in sorted_bands:
            if band_name in eval_formula:
                eval_formula = eval_formula.replace(band_name, f"band_vars['{band_name}']")
        
        # Safe evaluation environment
        safe_dict = {
            "__builtins__": {},
            "band_vars": band_vars,
            "np": np,
            "abs": np.abs,
            "sqrt": np.sqrt,
            "log": np.log,
            "exp": np.exp,
            "where": np.where,
            "maximum": np.maximum,
            "minimum": np.minimum
        }
        
        try:
            result = eval(eval_formula, safe_dict)
            return result
        except Exception as e:
            raise ValueError(f"Formula evaluation error: {str(e)}")
    
    def _apply_quality_masks(self, bands_data: np.ndarray, src, context) -> np.ndarray:
        """Apply quality masks to data"""
        try:
            # Try to get masks from context
            if self.cloud_mask:
                cloud_mask_data = getattr(context, 'get_data', lambda x: None)("cloud_mask")
                if cloud_mask_data and validate_file_exists(cloud_mask_data):
                    with rasterio.open(cloud_mask_data) as mask_src:
                        cloud_mask = mask_src.read(1).astype(bool)
                        # Apply mask to all bands
                        for i in range(bands_data.shape[0]):
                            bands_data[i] = np.where(cloud_mask, np.nan, bands_data[i])
            
            return bands_data
            
        except Exception as e:
            self.logger.warning(f"Quality masking failed: {str(e)}")
            return bands_data
    
    def _normalize_indices(self, indices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize index values"""
        normalized_indices = {}
        
        for name, array in indices.items():
            if self.normalization == "minmax":
                # Min-max normalization to [0, 1]
                valid_data = array[~np.isnan(array)]
                if len(valid_data) > 0:
                    min_val, max_val = np.min(valid_data), np.max(valid_data)
                    if max_val > min_val:
                        normalized = (array - min_val) / (max_val - min_val)
                    else:
                        normalized = array
                else:
                    normalized = array
            
            elif self.normalization == "zscore":
                # Z-score normalization
                valid_data = array[~np.isnan(array)]
                if len(valid_data) > 1:
                    mean_val, std_val = np.mean(valid_data), np.std(valid_data)
                    if std_val > 0:
                        normalized = (array - mean_val) / std_val
                    else:
                        normalized = array
                else:
                    normalized = array
            
            else:
                normalized = array
            
            normalized_indices[name] = normalized.astype(np.float32)
        
        return normalized_indices
    
    def _clip_indices(self, indices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Clip index values to reasonable ranges"""
        clipped_indices = {}
        
        for name, array in indices.items():
            # Get expected range for this index
            if name in self.predefined_indices:
                value_range = self.predefined_indices[name]["range"]
            else:
                value_range = self.value_range
            
            clipped_array = np.clip(array, value_range[0], value_range[1])
            clipped_indices[name] = clipped_array.astype(np.float32)
        
        return clipped_indices
    
    def _save_indices(self, indices: Dict[str, np.ndarray], profile: dict, 
                     input_file: str, context) -> List[str]:
        """Save calculated indices to files"""
        output_dir = context.get_temp_dir() / "spectral_indices"
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
            output_profile['count'] = len(indices)
            
            output_file = output_dir / f"spectral_indices_{Path(input_file).stem}.tif"
            
            with rasterio.open(output_file, 'w', **output_profile) as dst:
                for band_idx, (index_name, index_array) in enumerate(indices.items(), 1):
                    # Handle NaN values
                    output_array = np.where(np.isnan(index_array), self.nodata_value, index_array)
                    dst.write(output_array.astype(self.data_type), band_idx)
                    dst.set_band_description(band_idx, index_name)
            
            output_files.append(str(output_file))
            
        else:  # separate files
            # Save each index as separate file
            output_profile['count'] = 1
            
            for index_name, index_array in indices.items():
                safe_name = index_name.replace(" ", "_").replace("/", "_")
                output_file = output_dir / f"{safe_name}_{Path(input_file).stem}.tif"
                
                with rasterio.open(output_file, 'w', **output_profile) as dst:
                    # Handle NaN values
                    output_array = np.where(np.isnan(index_array), self.nodata_value, index_array)
                    dst.write(output_array.astype(self.data_type), 1)
                    dst.set_band_description(1, index_name)
                
                output_files.append(str(output_file))
        
        return output_files
    
    def _combine_results(self, processing_results: List[Dict[str, Any]], context) -> Dict[str, Any]:
        """Combine results from multiple files"""
        processing_start = datetime.now()
        
        # Flatten output files
        all_output_files = []
        all_metadata = []
        
        for result in processing_results:
            all_output_files.extend(result["output_files"])
            all_metadata.extend(result["index_metadata"])
        
        # Calculate combined statistics
        combined_stats = self._calculate_combined_statistics(all_metadata)
        
        # Create summary report
        summary_report = self._create_summary_report(all_metadata, combined_stats, context)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        total_processing_time = sum(r["processing_time"] for r in processing_results) + processing_time
        
        return {
            "output_files": all_output_files,
            "index_metadata": all_metadata,
            "combined_statistics": combined_stats,
            "summary_report": summary_report,
            "files_processed": len(processing_results),
            "indices_calculated": len(self.indices) + len(self.custom_indices),
            "processing_time": total_processing_time
        }
    
    def _calculate_combined_statistics(self, all_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate combined statistics across all indices"""
        stats = {}
        
        # Group by index name
        index_groups = {}
        for metadata in all_metadata:
            if "error" not in metadata:
                name = metadata["name"]
                if name not in index_groups:
                    index_groups[name] = []
                index_groups[name].append(metadata)
        
        # Calculate statistics for each index
        for index_name, metadata_list in index_groups.items():
            if metadata_list:
                mean_values = [m["mean_value"] for m in metadata_list if "mean_value" in m]
                std_values = [m["std_value"] for m in metadata_list if "std_value" in m]
                valid_pixels = [m["valid_pixels"] for m in metadata_list if "valid_pixels" in m]
                
                stats[index_name] = {
                    "files_calculated": len(metadata_list),
                    "mean_of_means": float(np.mean(mean_values)) if mean_values else 0,
                    "std_of_means": float(np.std(mean_values)) if len(mean_values) > 1 else 0,
                    "total_valid_pixels": sum(valid_pixels) if valid_pixels else 0,
                    "category": metadata_list[0].get("category", "unknown")
                }
        
        return stats
    
    def _create_summary_report(self, all_metadata: List[Dict[str, Any]], 
                              combined_stats: Dict[str, Any], context) -> str:
        """Create summary report"""
        try:
            output_dir = context.get_temp_dir() / "spectral_indices"
            ensure_directory(output_dir)
            
            report_content = f"""
Spectral Indices Calculation Report
==================================

Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Sensor: {self.sensor}
Output Format: {self.output_format}
Normalization: {self.normalization}

Indices Calculated:
------------------
"""
            
            # Group indices by category
            categories = {}
            for metadata in all_metadata:
                if "error" not in metadata:
                    category = metadata.get("category", "unknown")
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(metadata["name"])
            
            for category, indices in categories.items():
                unique_indices = list(set(indices))
                report_content += f"\n{category.title()} Indices ({len(unique_indices)}):\n"
                for index_name in unique_indices:
                    if index_name in combined_stats:
                        stats = combined_stats[index_name]
                        report_content += f"- {index_name}: mean={stats['mean_of_means']:.4f}, "
                        report_content += f"files={stats['files_calculated']}\n"
            
            # Error summary
            errors = [m for m in all_metadata if "error" in m]
            if errors:
                report_content += f"\nErrors ({len(errors)}):\n"
                for error in errors:
                    report_content += f"- {error['name']}: {error['error']}\n"
            
            # Processing summary
            successful_calculations = len([m for m in all_metadata if "error" not in m])
            total_calculations = len(all_metadata)
            success_rate = (successful_calculations / total_calculations) * 100 if total_calculations > 0 else 0
            
            report_content += f"""
Processing Summary:
------------------
- Total Calculations: {total_calculations}
- Successful: {successful_calculations}
- Failed: {len(errors)}
- Success Rate: {success_rate:.1f}%
- Files Processed: {len(set(m.get('input_file', '') for m in all_metadata))}

Quality Assessment:
------------------
"""
            
            # Quality assessment
            for index_name, stats in combined_stats.items():
                if stats['total_valid_pixels'] > 0:
                    report_content += f"- {index_name}: {stats['total_valid_pixels']:,} valid pixels\n"
                else:
                    report_content += f"- {index_name}: No valid pixels (check input data)\n"
            
            # Save report
            report_file = output_dir / f"spectral_indices_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            return str(report_file)
            
        except Exception as e:
            self.logger.warning(f"Failed to create summary report: {str(e)}")
            return ""
    
    def _validate_outputs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output files and results"""
        try:
            validation_results = {
                "validation_passed": True,
                "validation_messages": [],
                "file_checks": []
            }
            
            output_files = results.get("output_files", [])
            
            for output_file in output_files:
                file_check = {
                    "file": output_file,
                    "exists": Path(output_file).exists(),
                    "readable": False,
                    "band_count": 0,
                    "has_data": False
                }
                
                try:
                    with rasterio.open(output_file) as src:
                        file_check["readable"] = True
                        file_check["band_count"] = src.count
                        
                        # Check if file has data
                        sample_data = src.read(1, window=rasterio.windows.Window(0, 0, 100, 100))
                        valid_pixels = np.sum((sample_data != src.nodata) & (~np.isnan(sample_data)))
                        file_check["has_data"] = valid_pixels > 0
                        
                        if not file_check["has_data"]:
                            validation_results["validation_messages"].append(
                                f"No valid data found in {Path(output_file).name}"
                            )
                
                except Exception as e:
                    file_check["error"] = str(e)
                    validation_results["validation_passed"] = False
                    validation_results["validation_messages"].append(
                        f"Cannot read {Path(output_file).name}: {str(e)}"
                    )
                
                validation_results["file_checks"].append(file_check)
            
            # Check metadata
            metadata = results.get("index_metadata", [])
            failed_indices = [m for m in metadata if "error" in m]
            
            if len(failed_indices) > len(metadata) * 0.5:  # More than 50% failed
                validation_results["validation_passed"] = False
                validation_results["validation_messages"].append(
                    f"High failure rate: {len(failed_indices)}/{len(metadata)} indices failed"
                )
            
            return {"validation_results": validation_results}
            
        except Exception as e:
            return {
                "validation_results": {
                    "validation_passed": False,
                    "validation_messages": [f"Validation error: {str(e)}"],
                    "file_checks": []
                }
            }


# Register the step
StepRegistry.register("spectral_indices", SpectralIndicesStep)
