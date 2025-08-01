# orchestrator/steps/feature_extraction/topographic_derivatives_step.py
"""
Topographic derivatives calculation step.

Calculates slope, aspect, curvature, TPI, TRI, and other terrain features
from Digital Elevation Models (DEM).
"""

import numpy as np
import rasterio
from scipy import ndimage
from scipy.ndimage import generic_filter
from skimage.filters import rank
from skimage.morphology import disk
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from ..base.base_step import BaseStep
from ..base.step_registry import StepRegistry
from ...utils.file_utils import ensure_directory, validate_file_exists


class TopographicDerivativesStep(BaseStep):
    """
    Universal topographic derivatives calculation step.
    
    Capabilities:
    - Slope and aspect calculation
    - Curvature analysis (plan, profile, general)
    - Topographic Position Index (TPI) at multiple scales
    - Terrain Ruggedness Index (TRI)
    - Flow accumulation and drainage analysis
    - Morphometric features
    - Multi-scale analysis
    
    Configuration Examples:
    
    For Landslide Assessment:
    {
        "derivatives": ["slope", "aspect", "curvature", "tpi", "tri", "flow_accumulation"],
        "slope_units": "degrees",
        "aspect_units": "degrees", 
        "tpi_radii": [3, 9, 15],
        "flow_algorithm": "d8",
        "multi_scale": true
    }
    
    For Mineral Targeting:
    {
        "derivatives": ["slope", "aspect", "plan_curvature", "profile_curvature"],
        "slope_units": "percent",
        "edge_enhancement": true,
        "ridge_valley_detection": true,
        "structural_analysis": true
    }
    """
    
    def __init__(self, step_id: str, hyperparameters: Dict[str, Any]):
        super().__init__(step_id, "topographic_derivatives", hyperparameters)
        
        # Derivatives to calculate
        self.derivatives = hyperparameters.get("derivatives", ["slope", "aspect", "curvature"])
        
        # Slope parameters
        self.slope_units = hyperparameters.get("slope_units", "degrees")  # degrees, radians, percent
        self.slope_algorithm = hyperparameters.get("slope_algorithm", "horn")  # horn, simple
        
        # Aspect parameters
        self.aspect_units = hyperparameters.get("aspect_units", "degrees")  # degrees, radians
        self.aspect_algorithm = hyperparameters.get("aspect_algorithm", "horn")
        
        # Curvature parameters
        self.curvature_types = hyperparameters.get("curvature_types", ["plan", "profile", "general"])
        self.curvature_method = hyperparameters.get("curvature_method", "evans")  # evans, zevenbergen
        
        # TPI parameters
        self.tpi_radii = hyperparameters.get("tpi_radii", [3, 9, 15])
        self.tpi_standardize = hyperparameters.get("tpi_standardize", True)
        
        # Flow analysis parameters
        self.flow_algorithm = hyperparameters.get("flow_algorithm", "d8")  # d8, dinf
        self.flow_accumulation = hyperparameters.get("flow_accumulation", True)
        self.stream_power_index = hyperparameters.get("stream_power_index", False)
        self.wetness_index = hyperparameters.get("wetness_index", False)
        
        # Advanced options
        self.multi_scale = hyperparameters.get("multi_scale", False)
        self.scale_factors = hyperparameters.get("scale_factors", [1, 3, 5])
        self.edge_enhancement = hyperparameters.get("edge_enhancement", False)
        self.structural_analysis = hyperparameters.get("structural_analysis", False)
        
        # Processing options
        self.smoothing = hyperparameters.get("smoothing", False)
        self.smoothing_sigma = hyperparameters.get("smoothing_sigma", 1.0)
        self.fill_pits = hyperparameters.get("fill_pits", True)
        
        # Output options
        self.output_format = hyperparameters.get("output_format", "multiband")
        self.data_type = hyperparameters.get("data_type", "float32")
        self.nodata_value = hyperparameters.get("nodata_value", -9999)
        self.compression = hyperparameters.get("compression", "lzw")
        
        self.logger = logging.getLogger(f"TopographicDerivatives.{step_id}")
    
    def execute(self, context) -> Dict[str, Any]:
        """Execute topographic derivatives calculation"""
        try:
            self.logger.info(f"Starting topographic derivatives calculation")
            
            # Get DEM data
            dem_file = self._get_dem_file(context)
            
            # Validate DEM
            self._validate_dem(dem_file)
            
            # Process DEM
            processing_results = self._process_dem(dem_file, context)
            
            # Update context
            context.add_data(f"{self.step_id}_topographic_features", processing_results["output_files"])
            context.add_data(f"{self.step_id}_derivatives_metadata", processing_results["derivatives_metadata"])
            
            self.logger.info("Topographic derivatives calculation completed successfully")
            return {
                "status": "success",
                "outputs": processing_results,
                "metadata": {
                    "derivatives_calculated": len(self.derivatives),
                    "dem_file": dem_file,
                    "processing_time": processing_results.get("processing_time")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Topographic derivatives calculation failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _get_dem_file(self, context) -> str:
        """Get DEM file from context"""
        # Try different possible data keys
        for key in ["dem_data", "elevation_data", "dem_file"]:
            data = getattr(context, 'get_data', lambda x: None)(key)
            if data:
                if isinstance(data, str):
                    return data
        
        # Check hyperparameters
        if "dem_file" in self.hyperparameters:
            return self.hyperparameters["dem_file"]
        
        raise ValueError("No DEM file found in context or configuration")
    
    def _validate_dem(self, dem_file: str):
        """Validate DEM file"""
        if not validate_file_exists(dem_file):
            raise FileNotFoundError(f"DEM file not found: {dem_file}")
        
        try:
            with rasterio.open(dem_file) as src:
                if src.count != 1:
                    raise ValueError(f"DEM must have exactly 1 band, found {src.count}")
                
                # Check for valid elevation data
                sample_data = src.read(1, window=rasterio.windows.Window(0, 0, 100, 100))
                valid_pixels = np.sum((sample_data != src.nodata) & (~np.isnan(sample_data)))
                
                if valid_pixels == 0:
                    raise ValueError("DEM contains no valid elevation data")
                
                self.logger.info(f"DEM validated: {src.width}x{src.height}, {valid_pixels} valid pixels in sample")
                
        except Exception as e:
            raise ValueError(f"Cannot read DEM file {dem_file}: {str(e)}")
    
    def _process_dem(self, dem_file: str, context) -> Dict[str, Any]:
        """Process DEM to calculate derivatives"""
        processing_start = datetime.now()
        
        with rasterio.open(dem_file) as src:
            # Read elevation data
            elevation = src.read(1).astype(np.float64)
            profile = src.profile.copy()
            transform = src.transform
            
            # Get pixel size for calculations
            pixel_size_x = abs(transform.a)
            pixel_size_y = abs(transform.e)
            pixel_size = (pixel_size_x + pixel_size_y) / 2  # Average pixel size
            
            # Handle nodata values
            nodata = src.nodata
            if nodata is not None:
                elevation = np.where(elevation == nodata, np.nan, elevation)
            
            # Preprocessing
            if self.fill_pits:
                elevation = self._fill_pits(elevation)
            
            if self.smoothing:
                elevation = ndimage.gaussian_filter(elevation, sigma=self.smoothing_sigma)
            
            # Calculate derivatives
            calculated_derivatives = {}
            derivatives_metadata = []
            
            for derivative in self.derivatives:
                try:
                    self.logger.info(f"Calculating {derivative}")
                    
                    if derivative == "slope":
                        result = self._calculate_slope(elevation, pixel_size)
                        calculated_derivatives["slope"] = result
                        
                    elif derivative == "aspect":
                        result = self._calculate_aspect(elevation, pixel_size)
                        calculated_derivatives["aspect"] = result
                        
                    elif derivative == "curvature":
                        curvatures = self._calculate_curvature(elevation, pixel_size)
                        calculated_derivatives.update(curvatures)
                        
                    elif derivative == "plan_curvature":
                        result = self._calculate_plan_curvature(elevation, pixel_size)
                        calculated_derivatives["plan_curvature"] = result
                        
                    elif derivative == "profile_curvature":
                        result = self._calculate_profile_curvature(elevation, pixel_size)
                        calculated_derivatives["profile_curvature"] = result
                        
                    elif derivative == "tpi":
                        tpis = self._calculate_tpi(elevation, self.tpi_radii)
                        calculated_derivatives.update(tpis)
                        
                    elif derivative == "tri":
                        result = self._calculate_tri(elevation)
                        calculated_derivatives["tri"] = result
                        
                    elif derivative == "roughness":
                        result = self._calculate_roughness(elevation)
                        calculated_derivatives["roughness"] = result
                        
                    elif derivative == "flow_accumulation":
                        result = self._calculate_flow_accumulation(elevation, pixel_size)
                        calculated_derivatives["flow_accumulation"] = result
                        
                    elif derivative == "stream_power_index":
                        if "flow_accumulation" not in calculated_derivatives:
                            flow_acc = self._calculate_flow_accumulation(elevation, pixel_size)
                            calculated_derivatives["flow_accumulation"] = flow_acc
                        slope_rad = self._calculate_slope(elevation, pixel_size, units="radians")
                        result = self._calculate_stream_power_index(calculated_derivatives["flow_accumulation"], slope_rad)
                        calculated_derivatives["stream_power_index"] = result
                        
                    elif derivative == "wetness_index":
                        if "flow_accumulation" not in calculated_derivatives:
                            flow_acc = self._calculate_flow_accumulation(elevation, pixel_size)
                            calculated_derivatives["flow_accumulation"] = flow_acc
                        slope_rad = self._calculate_slope(elevation, pixel_size, units="radians")
                        result = self._calculate_wetness_index(calculated_derivatives["flow_accumulation"], slope_rad)
                        calculated_derivatives["wetness_index"] = result
                    
                    else:
                        self.logger.warning(f"Unknown derivative: {derivative}")
                        continue
                    
                    # Store metadata for successfully calculated derivatives
                    if derivative in calculated_derivatives or any(k.startswith(derivative) for k in calculated_derivatives.keys()):
                        derivatives_metadata.append({
                            "name": derivative,
                            "description": self._get_derivative_description(derivative),
                            "units": self._get_derivative_units(derivative),
                            "algorithm": getattr(self, f"{derivative}_algorithm", "default"),
                            "processing_date": datetime.now().isoformat()
                        })
                        
                except Exception as e:
                    self.logger.error(f"Failed to calculate {derivative}: {str(e)}")
                    derivatives_metadata.append({
                        "name": derivative,
                        "error": str(e)
                    })
            
            # Multi-scale analysis if requested
            if self.multi_scale:
                multiscale_derivatives = self._calculate_multiscale_derivatives(elevation, pixel_size)
                calculated_derivatives.update(multiscale_derivatives)
            
            # Save results
            output_files = self._save_derivatives(calculated_derivatives, profile, dem_file, context)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        return {
            "output_files": output_files,
            "calculated_derivatives": calculated_derivatives,
            "derivatives_metadata": derivatives_metadata,
            "processing_time": processing_time,
            "pixel_size": pixel_size
        }
    
    def _calculate_slope(self, elevation: np.ndarray, pixel_size: float, units: str = None) -> np.ndarray:
        """Calculate slope"""
        if units is None:
            units = self.slope_units
        
        # Calculate gradients
        if self.slope_algorithm == "horn":
            # Horn's method (3x3 kernel)
            gy, gx = np.gradient(elevation, pixel_size)
        else:
            # Simple method
            gy = np.gradient(elevation, axis=0) / pixel_size
            gx = np.gradient(elevation, axis=1) / pixel_size
        
        # Calculate slope
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        
        if units == "degrees":
            return np.degrees(slope_rad)
        elif units == "percent":
            return np.tan(slope_rad) * 100
        else:  # radians
            return slope_rad
    
    def _calculate_aspect(self, elevation: np.ndarray, pixel_size: float) -> np.ndarray:
        """Calculate aspect"""
        # Calculate gradients
        if self.aspect_algorithm == "horn":
            gy, gx = np.gradient(elevation, pixel_size)
        else:
            gy = np.gradient(elevation, axis=0) / pixel_size
            gx = np.gradient(elevation, axis=1) / pixel_size
        
        # Calculate aspect
        aspect_rad = np.arctan2(-gx, gy)
        
        # Convert to compass bearing (0-360 degrees from North)
        aspect_rad = np.where(aspect_rad < 0, aspect_rad + 2 * np.pi, aspect_rad)
        
        if self.aspect_units == "degrees":
            return np.degrees(aspect_rad)
        else:  # radians
            return aspect_rad
    
    def _calculate_curvature(self, elevation: np.ndarray, pixel_size: float) -> Dict[str, np.ndarray]:
        """Calculate curvature components"""
        curvatures = {}
        
        # First derivatives
        gy, gx = np.gradient(elevation, pixel_size)
        
        # Second derivatives
        gxy = np.gradient(gx, axis=0) / pixel_size
        gxx = np.gradient(gx, axis=1) / pixel_size  
        gyy = np.gradient(gy, axis=0) / pixel_size
        
        # Calculate curvature components
        if "plan" in self.curvature_types:
            # Plan curvature (horizontal curvature)
            plan_curvature = -(gxx * gy**2 - 2 * gxy * gx * gy + gyy * gx**2) / ((gx**2 + gy**2)**(3/2))
            plan_curvature = np.where(np.isfinite(plan_curvature), plan_curvature, 0)
            curvatures["plan_curvature"] = plan_curvature
        
        if "profile" in self.curvature_types:
            # Profile curvature (vertical curvature)
            profile_curvature = -(gxx * gx**2 + 2 * gxy * gx * gy + gyy * gy**2) / ((gx**2 + gy**2) * (1 + gx**2 + gy**2)**(3/2))
            profile_curvature = np.where(np.isfinite(profile_curvature), profile_curvature, 0)
            curvatures["profile_curvature"] = profile_curvature
        
        if "general" in self.curvature_types:
            # General curvature
            general_curvature = -(gxx + gyy)
            curvatures["general_curvature"] = general_curvature
        
        return curvatures
    
    def _calculate_plan_curvature(self, elevation: np.ndarray, pixel_size: float) -> np.ndarray:
        """Calculate plan curvature only"""
        curvatures = self._calculate_curvature(elevation, pixel_size)
        return curvatures.get("plan_curvature", np.zeros_like(elevation))
    
    def _calculate_profile_curvature(self, elevation: np.ndarray, pixel_size: float) -> np.ndarray:
        """Calculate profile curvature only"""
        curvatures = self._calculate_curvature(elevation, pixel_size)
        return curvatures.get("profile_curvature", np.zeros_like(elevation))
    
    def _calculate_tpi(self, elevation: np.ndarray, radii: List[int]) -> Dict[str, np.ndarray]:
        """Calculate Topographic Position Index at multiple scales"""
        tpis = {}
        
        for radius in radii:
            # Create circular kernel
            kernel_size = 2 * radius + 1
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            kernel = x**2 + y**2 <= radius**2
            
            # Calculate mean elevation in neighborhood
            mean_elevation = ndimage.generic_filter(
                elevation, np.mean, footprint=kernel, mode='nearest'
            )
            
            # TPI = elevation - mean neighborhood elevation
            tpi = elevation - mean_elevation
            
            # Standardize if requested
            if self.tpi_standardize:
                std_elevation = ndimage.generic_filter(
                    elevation, np.std, footprint=kernel, mode='nearest'
                )
                tpi = np.where(std_elevation > 0, tpi / std_elevation, 0)
            
            tpis[f"tpi_{radius}"] = tpi
        
        return tpis
    
    def _calculate_tri(self, elevation: np.ndarray) -> np.ndarray:
        """Calculate Terrain Ruggedness Index"""
        # TRI = mean of absolute differences between center cell and neighbors
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1], 
                          [1, 1, 1]])
        
        def tri_function(values):
            center = values[4]  # Center value in 3x3 window
            neighbors = np.concatenate([values[:4], values[5:]])
            return np.mean(np.abs(neighbors - center))
        
        tri = ndimage.generic_filter(elevation, tri_function, size=3, mode='nearest')
        return tri
    
    def _calculate_roughness(self, elevation: np.ndarray) -> np.ndarray:
        """Calculate surface roughness"""
        # Calculate slope
        gy, gx = np.gradient(elevation)
        slope = np.sqrt(gx**2 + gy**2)
        
        # Roughness as standard deviation of slope in 3x3 window
        roughness = ndimage.generic_filter(slope, np.std, size=3, mode='nearest')
        return roughness
    
    def _calculate_flow_accumulation(self, elevation: np.ndarray, pixel_size: float) -> np.ndarray:
        """Calculate flow accumulation using D8 algorithm (simplified)"""
        try:
            # This is a simplified implementation
            # For production use, consider using dedicated hydrological libraries
            
            rows, cols = elevation.shape
            flow_acc = np.ones_like(elevation)
            
            # D8 flow directions (neighbors)
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            
            # Create elevation order (process from highest to lowest)
            valid_mask = ~np.isnan(elevation)
            valid_coords = np.where(valid_mask)
            valid_elevations = elevation[valid_coords]
            sorted_indices = np.argsort(valid_elevations)[::-1]  # Highest first
            
            # Process cells in elevation order
            for idx in sorted_indices:
                r, c = valid_coords[0][idx], valid_coords[1][idx]
                current_elev = elevation[r, c]
                
                # Find steepest descent neighbor
                max_slope = 0
                flow_r, flow_c = r, c
                
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and valid_mask[nr, nc]:
                        neighbor_elev = elevation[nr, nc]
                        if neighbor_elev < current_elev:
                            slope = (current_elev - neighbor_elev) / (pixel_size * np.sqrt(dr**2 + dc**2))
                            if slope > max_slope:
                                max_slope = slope
                                flow_r, flow_c = nr, nc
                
                # Accumulate flow
                if flow_r != r or flow_c != c:  # Flow goes to a neighbor
                    flow_acc[flow_r, flow_c] += flow_acc[r, c]
            
            # Log transform for better visualization
            return np.log10(flow_acc + 1)
            
        except Exception as e:
            self.logger.warning(f"Flow accumulation calculation failed: {str(e)}, returning uniform values")
            return np.ones_like(elevation)
    
    def _calculate_stream_power_index(self, flow_accumulation: np.ndarray, slope_rad: np.ndarray) -> np.ndarray:
        """Calculate Stream Power Index"""
        # SPI = ln(flow_accumulation * tan(slope))
        spi = flow_accumulation * np.tan(slope_rad)
        spi = np.where(spi > 0, np.log(spi), 0)
        return spi
    
    def _calculate_wetness_index(self, flow_accumulation: np.ndarray, slope_rad: np.ndarray) -> np.ndarray:
        """Calculate Topographic Wetness Index"""
        # TWI = ln(flow_accumulation / tan(slope))
        tan_slope = np.tan(slope_rad)
        twi = np.where(tan_slope > 0, np.log(flow_accumulation / tan_slope), 0)
        return twi
    
    def _calculate_multiscale_derivatives(self, elevation: np.ndarray, pixel_size: float) -> Dict[str, np.ndarray]:
        """Calculate derivatives at multiple scales"""
        multiscale_derivatives = {}
        
        for scale in self.scale_factors:
            if scale == 1:
                continue  # Already calculated at original scale
            
            # Smooth elevation at different scales
            sigma = scale * pixel_size
            smoothed_elevation = ndimage.gaussian_filter(elevation, sigma=sigma)
            
            # Calculate slope at this scale
            slope = self._calculate_slope(smoothed_elevation, pixel_size)
            multiscale_derivatives[f"slope_scale_{scale}"] = slope
            
            # Calculate curvature at this scale
            curvatures = self._calculate_curvature(smoothed_elevation, pixel_size)
            for name, curvature in curvatures.items():
                multiscale_derivatives[f"{name}_scale_{scale}"] = curvature
        
        return multiscale_derivatives
    
    def _fill_pits(self, elevation: np.ndarray) -> np.ndarray:
        """Simple pit filling algorithm"""
        try:
            # Use morphological closing to fill small pits
            from skimage.morphology import closing, disk
            
            # Create structuring element
            selem = disk(3)
            
            # Apply closing to fill pits
            filled = closing(elevation, selem)
            
            # Only fill where original was lower (preserve peaks)
            filled_elevation = np.maximum(elevation, filled)
            
            return filled_elevation
            
        except Exception as e:
            self.logger.warning(f"Pit filling failed: {str(e)}, using original elevation")
            return elevation
    
    def _get_derivative_description(self, derivative: str) -> str:
        """Get description for derivative"""
        descriptions = {
            "slope": "Rate of elevation change",
            "aspect": "Direction of steepest descent",
            "curvature": "Surface curvature",
            "plan_curvature": "Horizontal curvature",
            "profile_curvature": "Vertical curvature",
            "general_curvature": "Overall curvature",
            "tpi": "Topographic Position Index",
            "tri": "Terrain Ruggedness Index",
            "roughness": "Surface roughness",
            "flow_accumulation": "Accumulated flow",
            "stream_power_index": "Stream power index",
            "wetness_index": "Topographic wetness index"
        }
        return descriptions.get(derivative, f"{derivative} derivative")
    
    def _get_derivative_units(self, derivative: str) -> str:
        """Get units for derivative"""
        units_map = {
            "slope": self.slope_units,
            "aspect": self.aspect_units,
            "curvature": "1/m",
            "plan_curvature": "1/m",
            "profile_curvature": "1/m",
            "general_curvature": "1/m",
            "tpi": "m" if not self.tpi_standardize else "standardized",
            "tri": "m",
            "roughness": "slope units",
            "flow_accumulation": "log(cells)",
            "stream_power_index": "dimensionless",
            "wetness_index": "dimensionless"
        }
        return units_map.get(derivative, "unknown")
    
    def _save_derivatives(self, derivatives: Dict[str, np.ndarray], profile: dict, 
                         dem_file: str, context) -> List[str]:
        """Save calculated derivatives to files"""
        output_dir = context.get_temp_dir() / "topographic_derivatives"
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
            output_profile['count'] = len(derivatives)
            
            output_file = output_dir / f"topographic_derivatives_{Path(dem_file).stem}.tif"
            
            with rasterio.open(output_file, 'w', **output_profile) as dst:
                for band_idx, (deriv_name, deriv_array) in enumerate(derivatives.items(), 1):
                    # Handle NaN values
                    output_array = np.where(np.isnan(deriv_array), self.nodata_value, deriv_array)
                    dst.write(output_array.astype(self.data_type), band_idx)
                    dst.set_band_description(band_idx, deriv_name)
            
            output_files.append(str(output_file))
            
        else:  # separate files
            # Save each derivative as separate file
            output_profile['count'] = 1
            
            for deriv_name, deriv_array in derivatives.items():
                safe_name = deriv_name.replace(" ", "_").replace("/", "_")
                output_file = output_dir / f"{safe_name}_{Path(dem_file).stem}.tif"
                
                with rasterio.open(output_file, 'w', **output_profile) as dst:
                    # Handle NaN values
                    output_array = np.where(np.isnan(deriv_array), self.nodata_value, deriv_array)
                    dst.write(output_array.astype(self.data_type), 1)
                    dst.set_band_description(1, deriv_name)
                
                output_files.append(str(output_file))
        
        return output_files


# Register the step
StepRegistry.register("topographic_derivatives", TopographicDerivativesStep)
