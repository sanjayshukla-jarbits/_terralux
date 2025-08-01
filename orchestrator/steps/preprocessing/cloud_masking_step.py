# orchestrator/steps/preprocessing/cloud_masking_step.py
"""
Cloud and shadow masking step for multiple sensor types.

Supports cloud detection using multiple algorithms including Fmask, Sen2Cor SCL,
machine learning approaches, and simple threshold-based methods.
"""

import rasterio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
from scipy import ndimage
from sklearn.cluster import KMeans
import cv2

from ..base.base_step import BaseStep
from ..base.step_registry import StepRegistry
from ...utils.file_utils import ensure_directory, validate_file_exists


class CloudMaskingStep(BaseStep):
    """
    Universal cloud and shadow masking step.
    
    Supported Methods:
    - scl: Sentinel-2 Scene Classification Layer
    - fmask: Fmask algorithm (if available)
    - threshold: Simple threshold-based approach
    - ml_clustering: Machine learning clustering approach
    - spectral_indices: Spectral indices based detection
    
    Configuration Examples:
    
    For Landslide Assessment (Sentinel-2):
    {
        "method": "scl",
        "cloud_threshold": 0.1,
        "shadow_threshold": 0.05,
        "buffer_size": 50,
        "fill_small_gaps": true
    }
    
    For Mineral Targeting (Multi-sensor):
    {
        "method": "spectral_indices",
        "cloud_indices": ["NDSI", "BTD"],
        "shadow_detection": true,
        "conservative_masking": false,
        "preserve_bright_surfaces": true
    }
    """
    
    def __init__(self, step_id: str, hyperparameters: Dict[str, Any]):
        super().__init__(step_id, "cloud_masking", hyperparameters)
        
        # Method configuration
        self.method = hyperparameters.get("method", "scl")
        self.sensor = hyperparameters.get("sensor", "sentinel2")
        
        # Threshold parameters
        self.cloud_threshold = hyperparameters.get("cloud_threshold", 0.1)
        self.shadow_threshold = hyperparameters.get("shadow_threshold", 0.05)
        self.cirrus_threshold = hyperparameters.get("cirrus_threshold", 0.02)
        
        # Morphological operations
        self.buffer_size = hyperparameters.get("buffer_size", 50)  # meters
        self.erosion_size = hyperparameters.get("erosion_size", 3)  # pixels
        self.dilation_size = hyperparameters.get("dilation_size", 5)  # pixels
        
        # Processing options
        self.shadow_detection = hyperparameters.get("shadow_detection", True)
        self.cirrus_detection = hyperparameters.get("cirrus_detection", True)
        self.fill_small_gaps = hyperparameters.get("fill_small_gaps", True)
        self.conservative_masking = hyperparameters.get("conservative_masking", True)
        self.preserve_bright_surfaces = hyperparameters.get("preserve_bright_surfaces", False)
        
        # Output options
        self.create_confidence_layer = hyperparameters.get("create_confidence_layer", True)
        self.separate_cloud_shadow = hyperparameters.get("separate_cloud_shadow", True)
        self.output_statistics = hyperparameters.get("output_statistics", True)
        
        self.logger = logging.getLogger(f"CloudMasking.{step_id}")
    
    def execute(self, context) -> Dict[str, Any]:
        """Execute cloud and shadow masking"""
        try:
            self.logger.info(f"Starting cloud masking using {self.method} method")
            
            # Get input data
            input_files = self._get_input_files(context)
            scl_file = self._get_scl_file(context)
            
            # Validate inputs
            self._validate_inputs(input_files)
            
            # Execute cloud masking based on method
            if self.method == "scl" and scl_file:
                outputs = self._execute_scl_masking(input_files, scl_file, context)
            elif self.method == "fmask":
                outputs = self._execute_fmask(input_files, context)
            elif self.method == "spectral_indices":
                outputs = self._execute_spectral_indices_masking(input_files, context)
            elif self.method == "ml_clustering":
                outputs = self._execute_ml_clustering(input_files, context)
            elif self.method == "threshold":
                outputs = self._execute_threshold_masking(input_files, context)
            else:
                raise ValueError(f"Unsupported cloud masking method: {self.method}")
            
            # Post-process masks
            outputs = self._post_process_masks(outputs, context)
            
            # Calculate statistics
            if self.output_statistics:
                stats = self._calculate_masking_statistics(outputs)
                outputs["statistics"] = stats
            
            # Update context
            context.add_data(f"{self.step_id}_cloud_mask", outputs["cloud_mask"])
            if "shadow_mask" in outputs:
                context.add_data(f"{self.step_id}_shadow_mask", outputs["shadow_mask"])
            if "masked_images" in outputs:
                context.add_data(f"{self.step_id}_masked_images", outputs["masked_images"])
            
            self.logger.info("Cloud masking completed successfully")
            return {
                "status": "success",
                "outputs": outputs,
                "metadata": {
                    "method": self.method,
                    "sensor": self.sensor,
                    "processing_time": outputs.get("processing_time"),
                    "cloud_coverage": outputs.get("statistics", {}).get("cloud_coverage", 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Cloud masking failed: {str(e)}")
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
        
        raise ValueError("No input files found for cloud masking")
    
    def _get_scl_file(self, context) -> Optional[str]:
        """Get Scene Classification Layer file"""
        scl_data = getattr(context, 'get_data', lambda x: None)("scl_data")
        if scl_data:
            return scl_data
        
        if "scl_file" in self.hyperparameters:
            return self.hyperparameters["scl_file"]
        
        return None
    
    def _validate_inputs(self, input_files: List[str]):
        """Validate input files"""
        if not input_files:
            raise ValueError("No input files provided for cloud masking")
        
        for file_path in input_files:
            if not validate_file_exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
    
    def _execute_scl_masking(self, input_files: List[str], scl_file: str, context) -> Dict[str, Any]:
        """Execute SCL-based cloud masking for Sentinel-2"""
        self.logger.info("Executing SCL-based cloud masking")
        
        processing_start = datetime.now()
        output_dir = context.get_temp_dir() / "cloud_masking"
        ensure_directory(output_dir)
        
        # Read SCL data
        with rasterio.open(scl_file) as scl_src:
            scl_data = scl_src.read(1)
            scl_profile = scl_src.profile
        
        # Create cloud and shadow masks from SCL
        # SCL values: 0=NO_DATA, 1=SATURATED_DEFECTIVE, 2=DARK_AREA_PIXELS, 3=CLOUD_SHADOWS,
        # 4=VEGETATION, 5=NOT_VEGETATED, 6=WATER, 7=UNCLASSIFIED, 8=CLOUD_MEDIUM_PROBABILITY,
        # 9=CLOUD_HIGH_PROBABILITY, 10=THIN_CIRRUS, 11=SNOW_ICE
        
        cloud_mask = np.isin(scl_data, [8, 9, 10])  # Medium/high prob clouds + cirrus
        shadow_mask = np.isin(scl_data, [3])  # Cloud shadows
        
        # Additional masks
        snow_ice_mask = np.isin(scl_data, [11])
        saturated_mask = np.isin(scl_data, [1])
        
        # Combine masks if conservative masking is enabled
        if self.conservative_masking:
            combined_mask = cloud_mask | shadow_mask | saturated_mask
            if not self.preserve_bright_surfaces:
                combined_mask = combined_mask | snow_ice_mask
        else:
            combined_mask = cloud_mask | shadow_mask
        
        # Save masks
        cloud_mask_file = self._save_mask(cloud_mask, scl_profile, output_dir, "cloud_mask")
        shadow_mask_file = self._save_mask(shadow_mask, scl_profile, output_dir, "shadow_mask")
        combined_mask_file = self._save_mask(combined_mask, scl_profile, output_dir, "combined_mask")
        
        # Apply masks to input images
        masked_images = self._apply_masks_to_images(input_files, combined_mask, output_dir)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        return {
            "cloud_mask": cloud_mask_file,
            "shadow_mask": shadow_mask_file,
            "combined_mask": combined_mask_file,
            "masked_images": masked_images,
            "processing_time": processing_time,
            "method_used": "scl"
        }
    
    def _execute_spectral_indices_masking(self, input_files: List[str], context) -> Dict[str, Any]:
        """Execute spectral indices based cloud masking"""
        self.logger.info("Executing spectral indices based cloud masking")
        
        processing_start = datetime.now()
        output_dir = context.get_temp_dir() / "cloud_masking"
        ensure_directory(output_dir)
        
        # Process first file to create masks
        reference_file = input_files[0]
        
        with rasterio.open(reference_file) as src:
            # Read bands (assuming Sentinel-2 order)
            blue = src.read(1).astype(np.float32)
            green = src.read(2).astype(np.float32)
            red = src.read(3).astype(np.float32)
            nir = src.read(4).astype(np.float32)
            swir1 = src.read(5).astype(np.float32) if src.count > 4 else nir
            swir2 = src.read(6).astype(np.float32) if src.count > 5 else swir1
            
            profile = src.profile
        
        # Create masks for valid data
        valid_mask = (blue > 0) & (green > 0) & (red > 0) & (nir > 0)
        
        # Initialize masks
        cloud_mask = np.zeros_like(blue, dtype=bool)
        shadow_mask = np.zeros_like(blue, dtype=bool)
        
        # Cloud detection using spectral indices
        if np.sum(valid_mask) > 1000:  # Ensure enough valid pixels
            
            # NDSI (Normalized Difference Snow Index) - high for clouds
            ndsi = np.divide(green - swir1, green + swir1, 
                           out=np.zeros_like(green), where=(green + swir1) != 0)
            
            # BTD (Brightness Temperature Difference) - simulated
            btd = (swir1 - swir2) / (swir1 + swir2 + 1e-8)
            
            # Brightness threshold
            brightness = (blue + green + red + nir) / 4
            brightness_threshold = np.percentile(brightness[valid_mask], 95)
            
            # Cloud detection
            cloud_mask = valid_mask & (
                (ndsi > 0.4) |  # High NDSI
                (brightness > brightness_threshold * 0.8) |  # High brightness
                (blue > np.percentile(blue[valid_mask], 90))  # High blue reflectance
            )
            
            # Shadow detection
            if self.shadow_detection:
                shadow_brightness = (blue + green + red) / 3
                shadow_threshold_val = np.percentile(shadow_brightness[valid_mask], 10)
                
                shadow_mask = valid_mask & (
                    (shadow_brightness < shadow_threshold_val) &
                    (nir < np.percentile(nir[valid_mask], 20))
                )
        
        # Apply morphological operations
        cloud_mask = self._apply_morphological_operations(cloud_mask)
        shadow_mask = self._apply_morphological_operations(shadow_mask)
        
        # Combine masks
        combined_mask = cloud_mask | shadow_mask
        
        # Save masks
        cloud_mask_file = self._save_mask(cloud_mask, profile, output_dir, "cloud_mask")
        shadow_mask_file = self._save_mask(shadow_mask, profile, output_dir, "shadow_mask")
        combined_mask_file = self._save_mask(combined_mask, profile, output_dir, "combined_mask")
        
        # Apply masks to input images
        masked_images = self._apply_masks_to_images(input_files, combined_mask, output_dir)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        return {
            "cloud_mask": cloud_mask_file,
            "shadow_mask": shadow_mask_file,
            "combined_mask": combined_mask_file,
            "masked_images": masked_images,
            "processing_time": processing_time,
            "method_used": "spectral_indices"
        }
    
    def _execute_ml_clustering(self, input_files: List[str], context) -> Dict[str, Any]:
        """Execute machine learning clustering based cloud masking"""
        self.logger.info("Executing ML clustering based cloud masking")
        
        processing_start = datetime.now()
        output_dir = context.get_temp_dir() / "cloud_masking"
        ensure_directory(output_dir)
        
        reference_file = input_files[0]
        
        with rasterio.open(reference_file) as src:
            # Read all bands
            data = src.read()
            profile = src.profile
            
            # Reshape for clustering
            original_shape = data.shape
            data_reshaped = data.reshape(data.shape[0], -1).T
            
            # Remove invalid pixels
            valid_pixels = np.all(data_reshaped > 0, axis=1)
            valid_data = data_reshaped[valid_pixels]
            
            if len(valid_data) > 1000:
                # Apply K-means clustering
                n_clusters = min(8, len(valid_data) // 100)  # Adaptive number of clusters
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(valid_data)
                
                # Assign labels back to full image
                full_labels = np.zeros(data_reshaped.shape[0], dtype=int)
                full_labels[valid_pixels] = labels
                cluster_image = full_labels.reshape(original_shape[1], original_shape[2])
                
                # Identify cloud and shadow clusters based on characteristics
                cluster_stats = []
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_image == cluster_id
                    if np.sum(cluster_mask) > 100:  # Minimum cluster size
                        cluster_data = data[:, cluster_mask]
                        mean_values = np.mean(cluster_data, axis=1)
                        
                        # Calculate characteristics
                        brightness = np.mean(mean_values)
                        blue_ratio = mean_values[0] / (np.sum(mean_values) + 1e-8)
                        nir_ratio = mean_values[3] / (np.sum(mean_values) + 1e-8) if len(mean_values) > 3 else 0
                        
                        cluster_stats.append({
                            'id': cluster_id,
                            'brightness': brightness,
                            'blue_ratio': blue_ratio,
                            'nir_ratio': nir_ratio,
                            'size': np.sum(cluster_mask)
                        })
                
                # Sort by brightness and identify cloud/shadow clusters
                cluster_stats.sort(key=lambda x: x['brightness'])
                
                cloud_clusters = []
                shadow_clusters = []
                
                # Identify cloud clusters (bright, high blue ratio)
                for stats in cluster_stats[-2:]:  # Top 2 brightest clusters
                    if stats['brightness'] > np.mean([s['brightness'] for s in cluster_stats]) * 1.2:
                        cloud_clusters.append(stats['id'])
                
                # Identify shadow clusters (dark, low NIR)
                for stats in cluster_stats[:2]:  # Bottom 2 darkest clusters
                    if stats['brightness'] < np.mean([s['brightness'] for s in cluster_stats]) * 0.8:
                        shadow_clusters.append(stats['id'])
                
                # Create masks
                cloud_mask = np.isin(cluster_image, cloud_clusters)
                shadow_mask = np.isin(cluster_image, shadow_clusters)
                
            else:
                # Fallback to simple thresholding
                cloud_mask = np.zeros((original_shape[1], original_shape[2]), dtype=bool)
                shadow_mask = np.zeros((original_shape[1], original_shape[2]), dtype=bool)
        
        # Apply morphological operations
        cloud_mask = self._apply_morphological_operations(cloud_mask)
        shadow_mask = self._apply_morphological_operations(shadow_mask)
        
        # Combine masks
        combined_mask = cloud_mask | shadow_mask
        
        # Save masks
        cloud_mask_file = self._save_mask(cloud_mask, profile, output_dir, "cloud_mask")
        shadow_mask_file = self._save_mask(shadow_mask, profile, output_dir, "shadow_mask")
        combined_mask_file = self._save_mask(combined_mask, profile, output_dir, "combined_mask")
        
        # Apply masks to input images
        masked_images = self._apply_masks_to_images(input_files, combined_mask, output_dir)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        return {
            "cloud_mask": cloud_mask_file,
            "shadow_mask": shadow_mask_file,
            "combined_mask": combined_mask_file,
            "masked_images": masked_images,
            "processing_time": processing_time,
            "method_used": "ml_clustering"
        }
    
    def _execute_threshold_masking(self, input_files: List[str], context) -> Dict[str, Any]:
        """Execute simple threshold-based cloud masking"""
        self.logger.info("Executing threshold-based cloud masking")
        
        processing_start = datetime.now()
        output_dir = context.get_temp_dir() / "cloud_masking"
        ensure_directory(output_dir)
        
        reference_file = input_files[0]
        
        with rasterio.open(reference_file) as src:
            # Read bands
            blue = src.read(1).astype(np.float32)
            green = src.read(2).astype(np.float32) if src.count > 1 else blue
            red = src.read(3).astype(np.float32) if src.count > 2 else green
            nir = src.read(4).astype(np.float32) if src.count > 3 else red
            
            profile = src.profile
        
        # Create valid data mask
        valid_mask = (blue > 0) & (green > 0) & (red > 0) & (nir > 0)
        
        # Simple cloud detection based on brightness
        brightness = (blue + green + red + nir) / 4
        
        if np.sum(valid_mask) > 1000:
            brightness_threshold = np.percentile(brightness[valid_mask], 95)
            cloud_mask = valid_mask & (brightness > brightness_threshold * self.cloud_threshold)
            
            # Simple shadow detection based on darkness
            shadow_threshold_val = np.percentile(brightness[valid_mask], 5)
            shadow_mask = valid_mask & (brightness < shadow_threshold_val * (1 + self.shadow_threshold))
        else:
            cloud_mask = np.zeros_like(blue, dtype=bool)
            shadow_mask = np.zeros_like(blue, dtype=bool)
        
        # Apply morphological operations
        cloud_mask = self._apply_morphological_operations(cloud_mask)
        shadow_mask = self._apply_morphological_operations(shadow_mask)
        
        # Combine masks
        combined_mask = cloud_mask | shadow_mask
        
        # Save masks
        cloud_mask_file = self._save_mask(cloud_mask, profile, output_dir, "cloud_mask")
        shadow_mask_file = self._save_mask(shadow_mask, profile, output_dir, "shadow_mask")
        combined_mask_file = self._save_mask(combined_mask, profile, output_dir, "combined_mask")
        
        # Apply masks to input images
        masked_images = self._apply_masks_to_images(input_files, combined_mask, output_dir)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        return {
            "cloud_mask": cloud_mask_file,
            "shadow_mask": shadow_mask_file,
            "combined_mask": combined_mask_file,
            "masked_images": masked_images,
            "processing_time": processing_time,
            "method_used": "threshold"
        }
    
    def _execute_fmask(self, input_files: List[str], context) -> Dict[str, Any]:
        """Execute Fmask algorithm (placeholder - requires external implementation)"""
        self.logger.warning("Fmask not implemented, falling back to spectral indices method")
        return self._execute_spectral_indices_masking(input_files, context)
    
    def _apply_morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up mask"""
        if not np.any(mask):
            return mask
        
        try:
            # Convert to uint8 for OpenCV
            mask_uint8 = mask.astype(np.uint8) * 255
            
            # Erosion to remove small noise
            if self.erosion_size > 0:
                kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                         (self.erosion_size, self.erosion_size))
                mask_uint8 = cv2.erode(mask_uint8, kernel_erosion, iterations=1)
            
            # Dilation to restore size and fill gaps
            if self.dilation_size > 0:
                kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                          (self.dilation_size, self.dilation_size))
                mask_uint8 = cv2.dilate(mask_uint8, kernel_dilation, iterations=1)
            
            # Fill small gaps if requested
            if self.fill_small_gaps:
                # Close small holes
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
            
            return (mask_uint8 > 127).astype(bool)
            
        except Exception as e:
            self.logger.warning(f"Morphological operations failed: {str(e)}, using original mask")
            return mask
    
    def _save_mask(self, mask: np.ndarray, profile: dict, output_dir: Path, name: str) -> str:
        """Save mask to file"""
        output_file = output_dir / f"{name}.tif"
        
        # Update profile for mask
        mask_profile = profile.copy()
        mask_profile.update({
            'dtype': 'uint8',
            'count': 1,
            'nodata': 0
        })
        
        with rasterio.open(output_file, 'w', **mask_profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
        
        return str(output_file)
    
    def _apply_masks_to_images(self, input_files: List[str], combined_mask: np.ndarray, 
                              output_dir: Path) -> List[str]:
        """Apply combined mask to input images"""
        masked_files = []
        
        for input_file in input_files:
            try:
                with rasterio.open(input_file) as src:
                    data = src.read()
                    profile = src.profile.copy()
                    
                    # Apply mask to all bands
                    masked_data = data.copy()
                    
                    # Resize mask if necessary
                    if combined_mask.shape != data.shape[1:]:
                        from scipy.ndimage import zoom
                        zoom_factors = (data.shape[1] / combined_mask.shape[0],
                                      data.shape[2] / combined_mask.shape[1])
                        resized_mask = zoom(combined_mask.astype(float), zoom_factors, order=0) > 0.5
                    else:
                        resized_mask = combined_mask
                    
                    # Apply mask (set masked pixels to nodata value)
                    nodata_value = profile.get('nodata', 0)
                    for band_idx in range(data.shape[0]):
                        masked_data[band_idx][resized_mask] = nodata_value
                    
                    # Save masked image
                    output_file = output_dir / f"masked_{Path(input_file).name}"
                    with rasterio.open(output_file, 'w', **profile) as dst:
                        dst.write(masked_data)
                    
                    masked_files.append(str(output_file))
                    
            except Exception as e:
                self.logger.error(f"Failed to apply mask to {input_file}: {str(e)}")
                # Copy original file as fallback
                masked_files.append(input_file)
        
        return masked_files
    
    def _post_process_masks(self, outputs: Dict[str, Any], context) -> Dict[str, Any]:
        """Post-process masks for quality improvement"""
        try:
            # Load masks
            cloud_mask_file = outputs.get("cloud_mask")
            shadow_mask_file = outputs.get("shadow_mask")
            
            if not cloud_mask_file or not shadow_mask_file:
                return outputs
            
            with rasterio.open(cloud_mask_file) as cloud_src:
                cloud_mask = cloud_src.read(1).astype(bool)
                profile = cloud_src.profile
            
            with rasterio.open(shadow_mask_file) as shadow_src:
                shadow_mask = shadow_src.read(1).astype(bool)
            
            # Apply buffer around clouds if specified
            if self.buffer_size > 0:
                # Convert buffer size from meters to pixels (approximate)
                # Assuming 10m resolution as default
                buffer_pixels = max(1, int(self.buffer_size / 10))
                
                # Dilate cloud mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (buffer_pixels * 2 + 1, buffer_pixels * 2 + 1))
                cloud_mask_uint8 = (cloud_mask.astype(np.uint8) * 255)
                buffered_cloud = cv2.dilate(cloud_mask_uint8, kernel, iterations=1)
                cloud_mask = (buffered_cloud > 127).astype(bool)
            
            # Update combined mask
            combined_mask = cloud_mask | shadow_mask
            
            # Save updated masks
            output_dir = Path(cloud_mask_file).parent
            
            updated_cloud_mask = self._save_mask(cloud_mask, profile, output_dir, "cloud_mask_processed")
            updated_combined_mask = self._save_mask(combined_mask, profile, output_dir, "combined_mask_processed")
            
            # Update outputs
            outputs["cloud_mask"] = updated_cloud_mask
            outputs["combined_mask"] = updated_combined_mask
            
            # Re-apply masks to images if they exist
            if "masked_images" in outputs:
                updated_masked_images = self._apply_masks_to_images(
                    outputs.get("original_files", []), combined_mask, output_dir
                )
                outputs["masked_images"] = updated_masked_images
            
            return outputs
            
        except Exception as e:
            self.logger.warning(f"Post-processing failed: {str(e)}, using original masks")
            return outputs
    
    def _calculate_masking_statistics(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cloud masking statistics"""
        try:
            stats = {}
            
            # Load masks and calculate coverage
            if "cloud_mask" in outputs:
                with rasterio.open(outputs["cloud_mask"]) as src:
                    cloud_mask = src.read(1).astype(bool)
                    total_pixels = cloud_mask.size
                    cloud_pixels = np.sum(cloud_mask)
                    stats["cloud_coverage"] = (cloud_pixels / total_pixels) * 100
            
            if "shadow_mask" in outputs:
                with rasterio.open(outputs["shadow_mask"]) as src:
                    shadow_mask = src.read(1).astype(bool)
                    shadow_pixels = np.sum(shadow_mask)
                    stats["shadow_coverage"] = (shadow_pixels / total_pixels) * 100
            
            if "combined_mask" in outputs:
                with rasterio.open(outputs["combined_mask"]) as src:
                    combined_mask = src.read(1).astype(bool)
                    masked_pixels = np.sum(combined_mask)
                    stats["total_masked_coverage"] = (masked_pixels / total_pixels) * 100
                    stats["usable_coverage"] = ((total_pixels - masked_pixels) / total_pixels) * 100
            
            # Quality assessment
            cloud_cov = stats.get("cloud_coverage", 0)
            if cloud_cov < 5:
                stats["quality_rating"] = "excellent"
            elif cloud_cov < 15:
                stats["quality_rating"] = "good"
            elif cloud_cov < 30:
                stats["quality_rating"] = "fair"
            else:
                stats["quality_rating"] = "poor"
            
            stats["processing_date"] = datetime.now().isoformat()
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Statistics calculation failed: {str(e)}")
            return {"error": str(e)}


# Register the step
StepRegistry.register("cloud_masking", CloudMaskingStep)
