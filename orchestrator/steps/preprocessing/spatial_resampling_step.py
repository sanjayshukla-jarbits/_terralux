# orchestrator/steps/preprocessing/spatial_resampling_step.py
"""
Spatial resampling step for resolution harmonization across multiple data sources.

Supports various resampling methods and handles multi-source data integration
for both landslide assessment and mineral targeting applications.
"""

import rasterio
import rasterio.warp
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling as ResamplingEnum
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime
from scipy.ndimage import zoom

from ..base.base_step import BaseStep
from ..base.step_registry import StepRegistry
from ...utils.file_utils import ensure_directory, validate_file_exists
from ...utils.geospatial_utils import get_raster_info, calculate_pixel_size


class SpatialResamplingStep(BaseStep):
    """
    Universal spatial resampling step for resolution harmonization.
    
    Capabilities:
    - Resolution upsampling and downsampling
    - Multi-source data harmonization
    - Extent alignment
    - Multiple resampling algorithms
    - Quality preservation optimization
    
    Configuration Examples:
    
    For Landslide Assessment (Multi-resolution harmonization):
    {
        "target_resolution": 10,
        "resampling_method": "bilinear",
        "reference_image": "auto",
        "align_pixels": true,
        "preserve_extent": false
    }
    
    For Mineral Targeting (High-resolution preservation):
    {
        "target_resolution": 5,
        "resampling_method": "cubic",
        "reference_image": "highest_resolution",
        "snap_to_grid": true,
        "quality_preservation": "high"
    }
    """
    
    def __init__(self, step_id: str, hyperparameters: Dict[str, Any]):
        super().__init__(step_id, "spatial_resampling", hyperparameters)
        
        # Resolution configuration
        self.target_resolution = hyperparameters.get("target_resolution", 10)
        self.resampling_method = hyperparameters.get("resampling_method", "bilinear")
        self.reference_image = hyperparameters.get("reference_image", "auto")
        
        # Alignment options  
        self.align_pixels = hyperparameters.get("align_pixels", True)
        self.snap_to_grid = hyperparameters.get("snap_to_grid", False)
        self.preserve_extent = hyperparameters.get("preserve_extent", False)
        
        # Quality options
        self.quality_preservation = hyperparameters.get("quality_preservation", "medium")
        self.anti_alias = hyperparameters.get("anti_alias", True)
        self.edge_handling = hyperparameters.get("edge_handling", "reflect")
        
        # Processing options
        self.chunk_size = hyperparameters.get("chunk_size", 1024)
        self.memory_limit = hyperparameters.get("memory_limit", "2GB")
        self.parallel_processing = hyperparameters.get("parallel_processing", True)
        
        # Output options
        self.create_overview = hyperparameters.get("create_overview", True)
        self.compression = hyperparameters.get("compression", "lzw")
        self.validate_output = hyperparameters.get("validate_output", True)
        
        self.logger = logging.getLogger(f"SpatialResampling.{step_id}")
    
    def execute(self, context) -> Dict[str, Any]:
        """Execute spatial resampling"""
        try:
            self.logger.info(f"Starting spatial resampling to {self.target_resolution}m resolution")
            
            # Get input data
            input_files = self._get_input_files(context)
            
            # Validate inputs
            self._validate_inputs(input_files)
            
            # Analyze input data characteristics
            input_info = self._analyze_input_data(input_files)
            
            # Determine reference parameters
            reference_params = self._determine_reference_parameters(input_files, input_info)
            
            # Execute resampling
            outputs = self._execute_resampling(input_files, reference_params, context)
            
            # Validate outputs
            if self.validate_output:
                validation_results = self._validate_outputs(outputs, input_info)
                outputs.update(validation_results)
            
            # Update context
            context.add_data(f"{self.step_id}_resampled_images", outputs["resampled_files"])
            if "harmonization_report" in outputs:
                context.add_data(f"{self.step_id}_harmonization_report", outputs["harmonization_report"])
            
            self.logger.info("Spatial resampling completed successfully")
            return {
                "status": "success", 
                "outputs": outputs,
                "metadata": {
                    "target_resolution": self.target_resolution,
                    "resampling_method": self.resampling_method,
                    "processing_time": outputs.get("processing_time"),
                    "files_processed": len(input_files)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Spatial resampling failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _get_input_files(self, context) -> List[str]:
        """Get input files from context"""
        # Try different possible data keys
        for key in ["masked_images", "corrected_images", "spectral_data", "input_images"]:
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
        
        raise ValueError("No input files found for spatial resampling")
    
    def _validate_inputs(self, input_files: List[str]):
        """Validate input files"""
        if not input_files:
            raise ValueError("No input files provided for spatial resampling")
        
        for file_path in input_files:
            if not validate_file_exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
    
    def _analyze_input_data(self, input_files: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of input data"""
        input_info = {
            "files": [],
            "resolutions": [],
            "extents": [],
            "crs_list": [],
            "dtypes": [],
            "band_counts": []
        }
        
        for file_path in input_files:
            try:
                with rasterio.open(file_path) as src:
                    # Get basic information
                    info = {
                        "file": file_path,
                        "width": src.width,
                        "height": src.height,
                        "count": src.count,
                        "dtype": str(src.dtype),
                        "crs": src.crs,
                        "bounds": src.bounds,
                        "transform": src.transform
                    }
                    
                    # Calculate resolution
                    pixel_size_x, pixel_size_y = calculate_pixel_size(src.transform)
                    info["resolution"] = (abs(pixel_size_x) + abs(pixel_size_y)) / 2
                    
                    input_info["files"].append(info)
                    input_info["resolutions"].append(info["resolution"])
                    input_info["extents"].append(src.bounds)
                    input_info["crs_list"].append(src.crs)
                    input_info["dtypes"].append(src.dtype)
                    input_info["band_counts"].append(src.count)
                    
            except Exception as e:
                self.logger.warning(f"Could not analyze {file_path}: {str(e)}")
        
        # Calculate summary statistics
        if input_info["resolutions"]:
            input_info["min_resolution"] = min(input_info["resolutions"])
            input_info["max_resolution"] = max(input_info["resolutions"])
            input_info["mean_resolution"] = np.mean(input_info["resolutions"])
        
        return input_info
    
    def _determine_reference_parameters(self, input_files: List[str], 
                                      input_info: Dict[str, Any]) -> Dict[str, Any]:
        """Determine reference parameters for resampling"""
        
        # Determine reference image
        if self.reference_image == "auto":
            # Use image closest to target resolution
            target_res = self.target_resolution
            best_idx = 0
            best_diff = float('inf')
            
            for idx, res in enumerate(input_info["resolutions"]):
                diff = abs(res - target_res)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
            
            reference_file = input_files[best_idx]
            
        elif self.reference_image == "highest_resolution":
            # Use highest resolution image
            min_res_idx = input_info["resolutions"].index(input_info["min_resolution"])
            reference_file = input_files[min_res_idx]
            
        elif self.reference_image == "largest_extent":
            # Use image with largest extent
            areas = [(b.right - b.left) * (b.top - b.bottom) for b in input_info["extents"]]
            max_area_idx = areas.index(max(areas))
            reference_file = input_files[max_area_idx]
            
        else:
            # Use specified reference file
            reference_file = self.reference_image
        
        # Get reference parameters
        with rasterio.open(reference_file) as ref_src:
            ref_bounds = ref_src.bounds
            ref_crs = ref_src.crs
            ref_transform = ref_src.transform
        
        # Calculate target transform
        if self.snap_to_grid:
            # Snap to regular grid
            target_transform = self._create_grid_aligned_transform(
                ref_bounds, ref_crs, self.target_resolution
            )
        else:
            # Use reference transform scaled to target resolution
            scale_factor = calculate_pixel_size(ref_transform)[0] / self.target_resolution
            target_transform = ref_transform * rasterio.transform.Affine.scale(scale_factor)
        
        # Calculate extent
        if self.preserve_extent:
            # Use union of all extents
            min_left = min([b.left for b in input_info["extents"]])
            max_right = max([b.right for b in input_info["extents"]])
            min_bottom = min([b.bottom for b in input_info["extents"]])
            max_top = max([b.top for b in input_info["extents"]])
            target_bounds = rasterio.coords.BoundingBox(min_left, min_bottom, max_right, max_top)
        else:
            # Use reference bounds
            target_bounds = ref_bounds
        
        # Calculate dimensions
        target_width = int((target_bounds.right - target_bounds.left) / self.target_resolution)
        target_height = int((target_bounds.top - target_bounds.bottom) / self.target_resolution)
        
        return {
            "reference_file": reference_file,
            "target_crs": ref_crs,
            "target_transform": target_transform,
            "target_bounds": target_bounds,
            "target_width": target_width,
            "target_height": target_height
        }
    
    def _create_grid_aligned_transform(self, bounds, crs, resolution: float):
        """Create grid-aligned transform"""
        # Snap bounds to grid
        left = np.floor(bounds.left / resolution) * resolution
        bottom = np.floor(bounds.bottom / resolution) * resolution
        
        # Create transform
        transform = rasterio.transform.from_origin(left, bottom + 
                   (np.ceil((bounds.top - bottom) / resolution) * resolution), 
                   resolution, resolution)
        
        return transform
    
    def _execute_resampling(self, input_files: List[str], reference_params: Dict[str, Any], 
                           context) -> Dict[str, Any]:
        """Execute the resampling process"""
        processing_start = datetime.now()
        output_dir = context.get_temp_dir() / "spatial_resampling"
        ensure_directory(output_dir)
        
        resampled_files = []
        processing_stats = []
        
        target_crs = reference_params["target_crs"]
        target_transform = reference_params["target_transform"]
        target_width = reference_params["target_width"]
        target_height = reference_params["target_height"]
        
        for input_file in input_files:
            try:
                self.logger.info(f"Resampling {Path(input_file).name}")
                
                resampled_file, stats = self._resample_single_file(
                    input_file, target_crs, target_transform, 
                    target_width, target_height, output_dir
                )
                
                resampled_files.append(resampled_file)
                processing_stats.append(stats)
                
            except Exception as e:
                self.logger.error(f"Failed to resample {input_file}: {str(e)}")
                raise
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        # Create harmonization report
        harmonization_report = self._create_harmonization_report(
            input_files, resampled_files, processing_stats, reference_params
        )
        
        return {
            "resampled_files": resampled_files,
            "processing_time": processing_time,
            "processing_stats": processing_stats,
            "harmonization_report": harmonization_report,
            "reference_parameters": reference_params
        }
    
    def _resample_single_file(self, input_file: str, target_crs, target_transform,
                             target_width: int, target_height: int, 
                             output_dir: Path) -> Tuple[str, Dict[str, Any]]:
        """Resample a single file"""
        
        with rasterio.open(input_file) as src:
            # Prepare output profile
            dst_profile = src.profile.copy()
            dst_profile.update({
                'crs': target_crs,
                'transform': target_transform,
                'width': target_width,
                'height': target_height,
                'compress': self.compression
            })
            
            # Create output file path
            output_file = output_dir / f"resampled_{Path(input_file).name}"
            
            # Get resampling method
            resampling_method = self._get_resampling_method()
            
            # Track processing statistics
            original_resolution = calculate_pixel_size(src.transform)[0]
            scale_factor = original_resolution / self.target_resolution
            
            stats = {
                "input_file": input_file,
                "output_file": str(output_file),
                "original_resolution": original_resolution,
                "target_resolution": self.target_resolution,
                "scale_factor": scale_factor,
                "resampling_method": self.resampling_method,
                "original_dimensions": (src.width, src.height),
                "target_dimensions": (target_width, target_height)
            }
            
            # Perform resampling
            with rasterio.open(output_file, 'w', **dst_profile) as dst:
                # Handle different processing approaches based on data size
                total_pixels = src.width * src.height * src.count
                memory_threshold = 1e8  # 100M pixels
                
                if total_pixels < memory_threshold:
                    # Process all bands at once for smaller files
                    reproject(
                        source=rasterio.band(src, list(range(1, src.count + 1))),
                        destination=rasterio.band(dst, list(range(1, dst.count + 1))),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=target_transform,
                        dst_crs=target_crs,
                        resampling=resampling_method
                    )
                else:
                    # Process band by band for larger files
                    for band_idx in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, band_idx),
                            destination=rasterio.band(dst, band_idx),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=target_transform,
                            dst_crs=target_crs,
                            resampling=resampling_method
                        )
                
                # Create overviews if requested
                if self.create_overview:
                    dst.build_overviews([2, 4, 8, 16], resampling_method)
                    dst.update_tags(ns='gdal', **{'TILED': 'YES'})
        
        return str(output_file), stats
    
    def _get_resampling_method(self) -> Resampling:
        """Get resampling method enum"""
        resampling_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "cubic_spline": Resampling.cubic_spline,
            "lanczos": Resampling.lanczos,
            "average": Resampling.average,
            "mode": Resampling.mode,
            "gauss": Resampling.gauss,
            "max": Resampling.max,
            "min": Resampling.min,
            "med": Resampling.med,
            "q1": Resampling.q1,
            "q3": Resampling.q3
        }
        
        method = resampling_map.get(self.resampling_method, Resampling.bilinear)
        
        # Apply quality preservation logic
        if self.quality_preservation == "high":
            # Use higher quality methods for upsampling
            if self.resampling_method == "bilinear":
                method = Resampling.cubic
            elif self.resampling_method == "nearest":
                method = Resampling.bilinear
        elif self.quality_preservation == "low":
            # Use faster methods
            if self.resampling_method in ["cubic", "lanczos"]:
                method = Resampling.bilinear
        
        return method
    
    def _create_harmonization_report(self, input_files: List[str], resampled_files: List[str],
                                   processing_stats: List[Dict[str, Any]], 
                                   reference_params: Dict[str, Any]) -> str:
        """Create harmonization report"""
        try:
            report_content = f"""
Spatial Resampling Harmonization Report
======================================

Processing Date: {datetime.now().isoformat()}
Target Resolution: {self.target_resolution}m
Resampling Method: {self.resampling_method} 
Quality Preservation: {self.quality_preservation}
Reference Image: {reference_params.get('reference_file', 'N/A')}

Target Grid Parameters:
- CRS: {reference_params['target_crs']}
- Bounds: {reference_params['target_bounds']}
- Dimensions: {reference_params['target_width']} x {reference_params['target_height']}
- Transform: {reference_params['target_transform']}

File Processing Summary:
"""
            
            for i, stats in enumerate(processing_stats):
                report_content += f"""
File {i+1}: {Path(stats['input_file']).name}
- Original Resolution: {stats['original_resolution']:.2f}m
- Scale Factor: {stats['scale_factor']:.3f}x
- Original Dimensions: {stats['original_dimensions']}
- Target Dimensions: {stats['target_dimensions']}
- Output: {Path(stats['output_file']).name}
"""
            
            # Calculate summary statistics
            scale_factors = [s['scale_factor'] for s in processing_stats]
            original_resolutions = [s['original_resolution'] for s in processing_stats]
            
            report_content += f"""

Summary Statistics:
- Files Processed: {len(processing_stats)}
- Resolution Range: {min(original_resolutions):.2f}m - {max(original_resolutions):.2f}m
- Scale Factor Range: {min(scale_factors):.3f}x - {max(scale_factors):.3f}x
- Average Scale Factor: {np.mean(scale_factors):.3f}x

Quality Assessment:
"""
            
            # Assess resampling quality
            upsampled_count = sum(1 for sf in scale_factors if sf > 1)
            downsampled_count = sum(1 for sf in scale_factors if sf < 1)
            unchanged_count = sum(1 for sf in scale_factors if abs(sf - 1) < 0.01)
            
            report_content += f"""- Upsampled Files: {upsampled_count}
- Downsampled Files: {downsampled_count}
- Unchanged Files: {unchanged_count}
- Recommended for Analysis: {'Yes' if max(scale_factors) < 3.0 else 'Caution - high upsampling factor'}
"""
            
            # Write report
            report_file = Path(resampled_files[0]).parent / "harmonization_report.txt"
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            return str(report_file)
            
        except Exception as e:
            self.logger.warning(f"Failed to create harmonization report: {str(e)}")
            return ""
    
    def _validate_outputs(self, outputs: Dict[str, Any], 
                         input_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate resampled outputs"""
        try:
            validation_results = {
                "validation_passed": True,
                "validation_messages": [],
                "quality_scores": []
            }
            
            resampled_files = outputs.get("resampled_files", [])
            
            for i, resampled_file in enumerate(resampled_files):
                try:
                    with rasterio.open(resampled_file) as src:
                        # Check basic properties
                        pixel_size = calculate_pixel_size(src.transform)[0]
                        
                        # Validate resolution
                        resolution_error = abs(pixel_size - self.target_resolution)
                        if resolution_error > self.target_resolution * 0.01:  # 1% tolerance
                            validation_results["validation_messages"].append(
                                f"Resolution mismatch in {Path(resampled_file).name}: "
                                f"expected {self.target_resolution}m, got {pixel_size:.2f}m"
                            )
                        
                        # Check for valid data
                        sample_data = src.read(1, window=rasterio.windows.Window(0, 0, 100, 100))
                        valid_pixels = np.sum(~np.isnan(sample_data) & (sample_data != src.nodata))
                        
                        if valid_pixels == 0:
                            validation_results["validation_passed"] = False
                            validation_results["validation_messages"].append(
                                f"No valid data in {Path(resampled_file).name}"
                            )
                        
                        # Calculate quality score
                        quality_score = min(1.0, valid_pixels / sample_data.size)
                        validation_results["quality_scores"].append(quality_score)
                        
                except Exception as e:
                    validation_results["validation_passed"] = False
                    validation_results["validation_messages"].append(
                        f"Validation failed for {resampled_file}: {str(e)}"
                    )
            
            # Overall quality assessment
            if validation_results["quality_scores"]:
                avg_quality = np.mean(validation_results["quality_scores"])
                validation_results["average_quality_score"] = avg_quality
                
                if avg_quality < 0.5:
                    validation_results["validation_passed"] = False
                    validation_results["validation_messages"].append(
                        f"Low average quality score: {avg_quality:.2f}"
                    )
            
            return {"validation_results": validation_results}
            
        except Exception as e:
            self.logger.warning(f"Output validation failed: {str(e)}")
            return {
                "validation_results": {
                    "validation_passed": False,
                    "validation_messages": [f"Validation error: {str(e)}"],
                    "quality_scores": []
                }
            }
    
    def _calculate_optimal_chunk_size(self, file_size: int, available_memory: str) -> int:
        """Calculate optimal chunk size for processing"""
        try:
            # Convert memory string to bytes
            memory_map = {"GB": 1e9, "MB": 1e6, "KB": 1e3}
            memory_value = float(available_memory[:-2])
            memory_unit = available_memory[-2:]
            memory_bytes = memory_value * memory_map.get(memory_unit, 1e6)
            
            # Calculate chunk size (use 25% of available memory)
            chunk_bytes = memory_bytes * 0.25
            
            # Convert to pixels (assuming 4 bytes per pixel for float32)
            chunk_pixels = int(chunk_bytes / 4)
            
            # Convert to chunk size (square chunks)
            chunk_size = int(np.sqrt(chunk_pixels))
            
            # Ensure reasonable bounds
            chunk_size = max(256, min(chunk_size, 4096))
            
            return chunk_size
            
        except Exception as e:
            self.logger.warning(f"Could not calculate optimal chunk size: {str(e)}")
            return self.chunk_size


# Register the step
StepRegistry.register("spatial_resampling", SpatialResamplingStep)
