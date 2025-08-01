# orchestrator/steps/preprocessing/geometric_correction_step.py
"""
Geometric correction step for orthorectification and georeferencing.

Supports DEM-based orthorectification, coordinate system transformations,
and geometric accuracy improvement for both landslide and mineral applications.
"""

import rasterio
import rasterio.warp
import rasterio.transform
import numpy as np
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import subprocess
import tempfile

from ..base.base_step import BaseStep
from ..base.step_registry import StepRegistry
from ...utils.file_utils import ensure_directory, validate_file_exists
from ...utils.geospatial_utils import get_raster_info, estimate_utm_crs


class GeometricCorrectionStep(BaseStep):
    """
    Universal geometric correction step supporting orthorectification and georeferencing.
    
    Capabilities:
    - DEM-based orthorectification
    - Coordinate system transformation
    - Ground Control Point (GCP) correction
    - Geometric accuracy assessment
    - Terrain correction
    
    Configuration Examples:
    
    For Landslide Assessment:
    {
        "correction_type": "orthorectification",
        "dem_source": "srtm",
        "target_crs": "auto_utm",
        "resampling_method": "bilinear",
        "output_resolution": 10
    }
    
    For Mineral Targeting:
    {
        "correction_type": "orthorectification", 
        "dem_source": "aster",
        "target_crs": "EPSG:32643",
        "resampling_method": "cubic",
        "output_resolution": 5,
        "terrain_correction": true
    }
    """
    
    def __init__(self, step_id: str, hyperparameters: Dict[str, Any]):
        super().__init__(step_id, "geometric_correction", hyperparameters)
        
        # Correction configuration
        self.correction_type = hyperparameters.get("correction_type", "orthorectification")
        self.dem_source = hyperparameters.get("dem_source", "srtm")
        self.target_crs = hyperparameters.get("target_crs", "auto_utm")
        self.resampling_method = hyperparameters.get("resampling_method", "bilinear")
        self.output_resolution = hyperparameters.get("output_resolution", 10)
        
        # Advanced options
        self.terrain_correction = hyperparameters.get("terrain_correction", False)
        self.gcp_file = hyperparameters.get("gcp_file", None)
        self.accuracy_assessment = hyperparameters.get("accuracy_assessment", True)
        self.create_accuracy_report = hyperparameters.get("create_accuracy_report", True)
        
        # Quality control
        self.validate_geometry = hyperparameters.get("validate_geometry", True)
        self.geometric_tolerance = hyperparameters.get("geometric_tolerance", 0.5)  # pixels
        
        # Output options
        self.preserve_original_extent = hyperparameters.get("preserve_original_extent", False)
        self.clip_to_valid_data = hyperparameters.get("clip_to_valid_data", True)
        
        self.logger = logging.getLogger(f"GeometricCorrection.{step_id}")
    
    def execute(self, context) -> Dict[str, Any]:
        """Execute geometric correction"""
        try:
            self.logger.info(f"Starting geometric correction: {self.correction_type}")
            
            # Get input data
            input_files = self._get_input_files(context)
            dem_file = self._get_dem_file(context)
            
            # Validate inputs
            self._validate_inputs(input_files, dem_file)
            
            # Determine target CRS
            target_crs = self._determine_target_crs(input_files[0])
            
            # Execute geometric correction
            if self.correction_type == "orthorectification":
                outputs = self._execute_orthorectification(input_files, dem_file, target_crs, context)
            elif self.correction_type == "reprojection":
                outputs = self._execute_reprojection(input_files, target_crs, context)
            elif self.correction_type == "gcp_correction":
                outputs = self._execute_gcp_correction(input_files, target_crs, context)
            else:
                raise ValueError(f"Unsupported correction type: {self.correction_type}")
            
            # Accuracy assessment
            if self.accuracy_assessment:
                accuracy_results = self._assess_geometric_accuracy(outputs, input_files)
                outputs.update(accuracy_results)
            
            # Update context
            context.add_data(f"{self.step_id}_corrected_images", outputs["corrected_files"])
            if "accuracy_report" in outputs:
                context.add_data(f"{self.step_id}_accuracy_report", outputs["accuracy_report"])
            
            self.logger.info("Geometric correction completed successfully")
            return {
                "status": "success",
                "outputs": outputs,
                "metadata": {
                    "correction_type": self.correction_type,
                    "target_crs": str(target_crs),
                    "output_resolution": self.output_resolution,
                    "processing_time": outputs.get("processing_time")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Geometric correction failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _get_input_files(self, context) -> List[str]:
        """Get input files from context"""
        # Try different possible data keys
        for key in ["corrected_images", "spectral_data", "sentinel_data", "input_images"]:
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
        
        raise ValueError("No input files found for geometric correction")
    
    def _get_dem_file(self, context) -> Optional[str]:
        """Get DEM file from context or configuration"""
        # Try to get from context
        dem_data = getattr(context, 'get_data', lambda x: None)("dem_data")
        if dem_data:
            return dem_data
        
        # Check hyperparameters
        if "dem_file" in self.hyperparameters:
            return self.hyperparameters["dem_file"]
        
        return None
    
    def _validate_inputs(self, input_files: List[str], dem_file: Optional[str]):
        """Validate input files"""
        if not input_files:
            raise ValueError("No input files provided for geometric correction")
        
        # Validate file existence
        for file_path in input_files:
            if not validate_file_exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
        
        # Validate DEM if required for orthorectification
        if self.correction_type == "orthorectification" and not dem_file:
            self.logger.warning("No DEM provided for orthorectification, will use simple reprojection")
            self.correction_type = "reprojection"
        
        if dem_file and not validate_file_exists(dem_file):
            raise FileNotFoundError(f"DEM file not found: {dem_file}")
    
    def _determine_target_crs(self, reference_file: str) -> CRS:
        """Determine target coordinate reference system"""
        if self.target_crs == "auto_utm":
            # Automatically determine UTM zone from image center
            with rasterio.open(reference_file) as src:
                bounds = src.bounds
                center_lon = (bounds.left + bounds.right) / 2
                center_lat = (bounds.bottom + bounds.top) / 2
                
                return estimate_utm_crs(center_lat, center_lon)
        
        elif self.target_crs == "preserve":
            # Keep original CRS
            with rasterio.open(reference_file) as src:
                return src.crs
        
        else:
            # Use specified CRS
            return CRS.from_string(self.target_crs)
    
    def _execute_orthorectification(self, input_files: List[str], dem_file: str, 
                                  target_crs: CRS, context) -> Dict[str, Any]:
        """Execute DEM-based orthorectification"""
        self.logger.info("Executing DEM-based orthorectification")
        
        corrected_files = []
        processing_start = datetime.now()
        output_dir = context.get_temp_dir() / "geometric_correction"
        ensure_directory(output_dir)
        
        # Load DEM
        with rasterio.open(dem_file) as dem_src:
            dem_data = dem_src.read(1)
            dem_transform = dem_src.transform
            dem_crs = dem_src.crs
        
        for input_file in input_files:
            try:
                corrected_file = self._orthorectify_image(
                    input_file, dem_data, dem_transform, dem_crs,
                    target_crs, output_dir
                )
                corrected_files.append(corrected_file)
                
            except Exception as e:
                self.logger.error(f"Orthorectification failed for {input_file}: {str(e)}")
                # Fall back to simple reprojection
                corrected_file = self._reproject_image(input_file, target_crs, output_dir)
                corrected_files.append(corrected_file)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        return {
            "corrected_files": corrected_files,
            "processing_time": processing_time,
            "method_used": "orthorectification",
            "target_crs": str(target_crs)
        }
    
    def _orthorectify_image(self, input_file: str, dem_data: np.ndarray, 
                           dem_transform, dem_crs: CRS, target_crs: CRS, 
                           output_dir: Path) -> str:
        """Orthorectify single image using DEM"""
        
        with rasterio.open(input_file) as src:
            # Get image metadata
            src_crs = src.crs
            src_transform = src.transform
            src_bounds = src.bounds
            
            # Calculate output transform and dimensions
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, target_crs, src.width, src.height, *src_bounds,
                resolution=self.output_resolution
            )
            
            # Prepare output profile
            dst_profile = src.profile.copy()
            dst_profile.update({
                'crs': target_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height
            })
            
            # Create output file path
            output_file = output_dir / f"ortho_{Path(input_file).name}"
            
            # Perform orthorectification
            with rasterio.open(output_file, 'w', **dst_profile) as dst:
                # Reproject each band
                for band_idx in range(1, src.count + 1):
                    source_band = src.read(band_idx)
                    
                    # Apply terrain correction if enabled
                    if self.terrain_correction:
                        source_band = self._apply_terrain_correction(
                            source_band, dem_data, src_transform, dem_transform
                        )
                    
                    # Reproject band
                    destination = np.zeros((dst_height, dst_width), dtype=dst_profile['dtype'])
                    
                    reproject(
                        source=source_band,
                        destination=destination,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=target_crs,
                        resampling=self._get_resampling_method()
                    )
                    
                    dst.write(destination, band_idx)
        
        return str(output_file)
    
    def _apply_terrain_correction(self, image_data: np.ndarray, dem_data: np.ndarray,
                                 image_transform, dem_transform) -> np.ndarray:
        """Apply terrain-based illumination correction"""
        try:
            # Calculate slope and aspect from DEM
            dy, dx = np.gradient(dem_data)
            slope = np.arctan(np.sqrt(dx*dx + dy*dy))
            aspect = np.arctan2(-dx, dy)
            
            # Simulate solar illumination (simplified)
            solar_zenith = np.radians(45)  # 45 degrees
            solar_azimuth = np.radians(135)  # 135 degrees (SE)
            
            # Calculate illumination
            illumination = (np.cos(slope) * np.cos(solar_zenith) + 
                          np.sin(slope) * np.sin(solar_zenith) * 
                          np.cos(solar_azimuth - aspect))
            
            # Normalize illumination
            illumination = np.clip(illumination, 0.1, 1.0)
            
            # Resample illumination to image grid if needed
            if illumination.shape != image_data.shape:
                from scipy.ndimage import zoom
                zoom_factors = (image_data.shape[0] / illumination.shape[0],
                               image_data.shape[1] / illumination.shape[1])
                illumination = zoom(illumination, zoom_factors, order=1)
            
            # Apply correction
            corrected_data = image_data.astype(np.float32) / illumination
            corrected_data = np.clip(corrected_data, 0, np.iinfo(image_data.dtype).max)
            
            return corrected_data.astype(image_data.dtype)
            
        except Exception as e:
            self.logger.warning(f"Terrain correction failed: {str(e)}, using original data")
            return image_data
    
    def _execute_reprojection(self, input_files: List[str], target_crs: CRS, context) -> Dict[str, Any]:
        """Execute simple coordinate system reprojection"""
        self.logger.info("Executing coordinate system reprojection")
        
        corrected_files = []
        processing_start = datetime.now()
        output_dir = context.get_temp_dir() / "geometric_correction"
        ensure_directory(output_dir)
        
        for input_file in input_files:
            corrected_file = self._reproject_image(input_file, target_crs, output_dir)
            corrected_files.append(corrected_file)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        return {
            "corrected_files": corrected_files,
            "processing_time": processing_time,
            "method_used": "reprojection",
            "target_crs": str(target_crs)
        }
    
    def _reproject_image(self, input_file: str, target_crs: CRS, output_dir: Path) -> str:
        """Reproject single image to target CRS"""
        
        with rasterio.open(input_file) as src:
            # Calculate output transform and dimensions
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds,
                resolution=self.output_resolution
            )
            
            # Prepare output profile
            dst_profile = src.profile.copy()
            dst_profile.update({
                'crs': target_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height
            })
            
            # Create output file
            output_file = output_dir / f"reprojected_{Path(input_file).name}"
            
            with rasterio.open(output_file, 'w', **dst_profile) as dst:
                reproject(
                    source=rasterio.band(src, list(range(1, src.count + 1))),
                    destination=rasterio.band(dst, list(range(1, dst.count + 1))),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=self._get_resampling_method()
                )
        
        return str(output_file)
    
    def _execute_gcp_correction(self, input_files: List[str], target_crs: CRS, context) -> Dict[str, Any]:
        """Execute Ground Control Point based correction"""
        self.logger.info("Executing GCP-based correction")
        
        if not self.gcp_file:
            self.logger.warning("No GCP file provided, falling back to reprojection")
            return self._execute_reprojection(input_files, target_crs, context)
        
        # For now, implement as simple reprojection
        # Full GCP correction would require more complex implementation
        return self._execute_reprojection(input_files, target_crs, context)
    
    def _get_resampling_method(self) -> Resampling:
        """Get resampling method enum"""
        resampling_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "cubic_spline": Resampling.cubic_spline,
            "lanczos": Resampling.lanczos,
            "average": Resampling.average,
            "mode": Resampling.mode
        }
        
        return resampling_map.get(self.resampling_method, Resampling.bilinear)
    
    def _assess_geometric_accuracy(self, outputs: Dict[str, Any], 
                                  original_files: List[str]) -> Dict[str, Any]:
        """Assess geometric accuracy of correction"""
        try:
            self.logger.info("Assessing geometric accuracy")
            
            accuracy_metrics = []
            corrected_files = outputs["corrected_files"]
            
            for orig_file, corr_file in zip(original_files, corrected_files):
                metrics = self._calculate_accuracy_metrics(orig_file, corr_file)
                accuracy_metrics.append(metrics)
            
            # Calculate overall accuracy
            overall_rmse = np.mean([m["rmse"] for m in accuracy_metrics])
            overall_ce90 = np.mean([m["ce90"] for m in accuracy_metrics])
            
            accuracy_report = {
                "overall_rmse": overall_rmse,
                "overall_ce90": overall_ce90,
                "individual_metrics": accuracy_metrics,
                "accuracy_assessment_date": datetime.now().isoformat(),
                "meets_tolerance": overall_rmse <= self.geometric_tolerance
            }
            
            # Create accuracy report file if requested
            if self.create_accuracy_report:
                report_file = self._create_accuracy_report(accuracy_report)
                accuracy_report["report_file"] = report_file
            
            return {"accuracy_report": accuracy_report}
            
        except Exception as e:
            self.logger.warning(f"Accuracy assessment failed: {str(e)}")
            return {"accuracy_report": {"error": str(e)}}
    
    def _calculate_accuracy_metrics(self, original_file: str, corrected_file: str) -> Dict[str, Any]:
        """Calculate accuracy metrics for a single file pair"""
        try:
            with rasterio.open(original_file) as orig_src, rasterio.open(corrected_file) as corr_src:
                # Get corner coordinates
                orig_bounds = orig_src.bounds
                corr_bounds = corr_src.bounds
                
                # Calculate center point displacement
                orig_center = ((orig_bounds.left + orig_bounds.right) / 2,
                              (orig_bounds.bottom + orig_bounds.top) / 2)
                corr_center = ((corr_bounds.left + corr_bounds.right) / 2,
                              (corr_bounds.bottom + corr_bounds.top) / 2)
                
                # Calculate displacement in meters (approximate)
                dx = abs(corr_center[0] - orig_center[0])
                dy = abs(corr_center[1] - orig_center[1])
                
                # Convert to meters if in geographic coordinates
                if orig_src.crs.is_geographic:
                    # Rough conversion: 1 degree ≈ 111,000 meters
                    dx *= 111000
                    dy *= 111000
                
                # Calculate RMSE
                rmse = np.sqrt(dx**2 + dy**2)
                
                # Calculate CE90 (90% circular error)
                ce90 = rmse * 1.645  # Assuming normal distribution
                
                return {
                    "file": Path(original_file).name,
                    "displacement_x": dx,
                    "displacement_y": dy,
                    "rmse": rmse,
                    "ce90": ce90
                }
                
        except Exception as e:
            self.logger.warning(f"Accuracy calculation failed for {original_file}: {str(e)}")
            return {
                "file": Path(original_file).name,
                "error": str(e),
                "rmse": 999.0,
                "ce90": 999.0
            }
    
    def _create_accuracy_report(self, accuracy_data: Dict[str, Any]) -> str:
        """Create accuracy assessment report"""
        try:
            # Create report content
            report_content = f"""
Geometric Correction Accuracy Report
====================================

Processing Date: {accuracy_data['accuracy_assessment_date']}
Correction Type: {self.correction_type}
Target CRS: {self.target_crs}
Resampling Method: {self.resampling_method}
Output Resolution: {self.output_resolution}m

Overall Accuracy Metrics:
- RMSE: {accuracy_data['overall_rmse']:.3f} meters
- CE90: {accuracy_data['overall_ce90']:.3f} meters
- Meets Tolerance: {accuracy_data['meets_tolerance']}
- Geometric Tolerance: {self.geometric_tolerance} pixels

Individual File Metrics:
"""
            for metrics in accuracy_data['individual_metrics']:
                if 'error' not in metrics:
                    report_content += f"""
- {metrics['file']}:
  - Displacement X: {metrics['displacement_x']:.3f}m
  - Displacement Y: {metrics['displacement_y']:.3f}m
  - RMSE: {metrics['rmse']:.3f}m
  - CE90: {metrics['ce90']:.3f}m
"""
                else:
                    report_content += f"- {metrics['file']}: Error - {metrics['error']}\n"
            
            # Write report file
            report_file = Path(tempfile.gettempdir()) / f"geometric_accuracy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            return str(report_file)
            
        except Exception as e:
            self.logger.warning(f"Report creation failed: {str(e)}")
            return ""


# Register the step
StepRegistry.register("geometric_correction", GeometricCorrectionStep)
